import os
import sys
import time
import socket
import random
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import configargparse
from tqdm import tqdm

# Working directory setup
working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)
lib_path = os.path.join(working_dir)
sys.path.append(lib_path)
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from lib.geoopt import ManifoldParameter, ManifoldTensor
from lib.lorentz.layers.LBnorm import LorentzBatchNorm, LorentzBatchNorm1d, LorentzBatchNorm2d

# Project imports (assumed available)
try:
    from lib.utils.activation_logger import ActivationLogger
    from utils.initialize import select_dataset, select_model, select_optimizer, load_checkpoint
    from lib.utils.utils import AverageMeter, accuracy, GPUTempMonitor, setup, cleanup, reduce_tensor
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Please ensure all required modules are available in the path")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_port_free(port: int) -> bool:
    """Check if a port is free on localhost."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            return True
    except (OSError, socket.error):
        return False

def update_bn_mixed(loader, model, device=None):
    """
    Update batch normalization statistics for models that might have both 
    standard and Lorentz batch norm layers.
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.train()
    
    # Check if we have any Lorentz batch norm layers
    has_lorentz_bn = any(isinstance(module, (LorentzBatchNorm, LorentzBatchNorm1d, LorentzBatchNorm2d))
                        for module in model.modules())
    
    if has_lorentz_bn:
        # Custom update for mixed models
        bn_layers = []
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                  LorentzBatchNorm, LorentzBatchNorm1d, LorentzBatchNorm2d)):
                bn_layers.append(module)
        
        if not bn_layers:
            return
        
        # Reset running statistics
        for module in bn_layers:
            if hasattr(module, 'running_mean'):
                module.running_mean.zero_()
            if hasattr(module, 'running_var'):
                module.running_var.fill_(1.0)
            if hasattr(module, 'num_batches_tracked'):
                module.num_batches_tracked.zero_()
        
        # Accumulate statistics
        with torch.no_grad():
            for data in loader:
                if isinstance(data, (list, tuple)):
                    inputs = data[0]
                else:
                    inputs = data
                
                inputs = inputs.to(device, non_blocking=True)
                
                try:
                    model(inputs)
                except Exception as e:
                    print(f"Warning: Failed to update BN statistics for batch: {e}")
                    continue
    else:
        # Use standard PyTorch update_bn for pure Euclidean models
        update_bn(loader, model, device)

def stabilize_manifold_params(model):
    """
    Ensure all manifold parameters are properly projected back to their manifolds.
    This is useful after SWA averaging which might move parameters off-manifold.
    """
    for param in model.parameters():
        if isinstance(param, (ManifoldParameter, ManifoldTensor)):
            with torch.no_grad():
                param.copy_(param.manifold.projx(param))

def restore_nmf_params(ddp_model, swa_model):
    """
    Restore NMF parameters in SWA model from the current DDP model.
    Simplified version that handles both Euclidean and manifold parameters.
    """
    ddp_params = dict(ddp_model.named_parameters())
    swa_params = dict(swa_model.named_parameters())
    
    for name, param in ddp_params.items():
        if 'nmf_module' in name and name in swa_params:
            swa_param = swa_params[name]
            with torch.no_grad():
                swa_param.data.copy_(param.data)
                
                # If it's a manifold parameter, project back to manifold
                if isinstance(swa_param, (ManifoldParameter, ManifoldTensor)):
                    swa_param.copy_(swa_param.manifold.projx(swa_param))

def create_swa_components(model, optimizer, swa_lr=0.05):
    """
    Create SWA components. Use standard PyTorch components and handle 
    manifold constraints separately.
    """
    # Always use standard SWA components
    averaged_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
    
    return averaged_model, swa_scheduler, update_bn_mixed


def getArguments():
    """Parses command-line options."""
    parser = configargparse.ArgumentParser(description="Image classification training", add_help=True)

    parser.add_argument("-c", "--config_file", required=False, default=None, is_config_file=True, type=str,
                        help="Path to config file.")

    parser.add_argument("--world_size", default=4, type=int, help="Number of GPUs to use for distributed training")

    parser.add_argument("--exp_name", default="test", type=str, help="Name of the experiment.")
    parser.add_argument("--output_dir", default="classification/output", type=str,
                        help="Path for output files (relative to working directory).")

    # Sparsity reg (external to NMF)
    parser.add_argument("--enable_sparsity", default=False, action="store_true",
                        help="Enable sparsity regularization for Lorentz models.")
    parser.add_argument("--sparsity_weight", default=0.01, type=float, help="Weight for sparsity regularization loss.")
    parser.add_argument("--sparsity_type", default="l1", type=str, choices=["l1", "k_sparse"],
                        help="Type of sparsity regularization.")
    parser.add_argument("--sparsity_schedule", default="annealing", type=str,
                        choices=["constant", "annealing", "warmup"], help="Sparsity weight scheduling strategy.")
    parser.add_argument("--k_ratio", default=0.1, type=float,
                        help="Ratio of elements to keep for k-sparse constraints (e.g., 0.1 = keep 10%).")
    parser.add_argument("--apply_k_sparse_to", default="final_layer", type=str,
                        choices=["final_layer", "all_conv_layers", "conv2_x", "conv3_x", "conv4_x", "conv5_x"],
                        help="Where to apply k-sparse constraints. 'final_layer' applies only before classifier.")




    parser.add_argument("--nmf_rec_decay", default=False, action="store_true",
                        help="Enable decay of NMF reconstruction loss weight over training.")
    parser.add_argument("--nmf_rec_decay_type", default="linear", type=str,
                        choices=["linear", "cosine", "exponential"],
                        help="Type of decay for NMF reconstruction loss weight.")

    # Dist port
    parser.add_argument("--port", default=None, type=str, help="Port for distributed training communication")

    # NMF general toggles
    parser.add_argument("--enable_nmf", default=False, action="store_true",
                        help="Enable Non-negative Matrix Factorization (NMF) regularization.")
    parser.add_argument("--rank_ratio", default=0.35, type=float,
                        help="Ratio of channels to use as rank in NMF (e.g., 0.35 means rank = 0.35 * channels).")
    parser.add_argument("--iters_train", default=5, type=int, help="Number of NMF iterations during training.")
    parser.add_argument("--iters_eval", default=5, type=int, help="Number of NMF iterations during evaluation.")
    parser.add_argument("--per_channel_gate", default=True, type=bool,
                        help="Use per-channel gating in NMF.")
    parser.add_argument("--gate_init", default=-2.0, type=float,
                        help="Initial value for NMF gates (bias toward identity early in training).")
    parser.add_argument("--space", default="hyperbolic", type=str, choices=["hyperbolic", "tangent"],
                        help="Space to perform NMF computation (for Lorentz models): hyperbolic or tangent.")
    parser.add_argument("--nmf_apply_to", default="all_blocks", type=str,
                        choices=["all_blocks", "lorentz_blocks", "euclidean_blocks"],
                        help="Which blocks to apply NMF to (relevant for hybrid models).")
    parser.add_argument("--nmf_loss_weight_rec", default=1e-3, type=float, help="Weight for NMF reconstruction loss.")
    parser.add_argument("--nmf_loss_weight_sparse", default=5e-4, type=float, help="Weight for NMF sparsity loss.")
    parser.add_argument("--nmf_loss_weight_orth", default=1e-4, type=float, help="Weight for NMF orthogonality loss.")

    parser.add_argument("--swa_start_epoch", default=160, type=int, help="Epoch to start SWA.")
    parser.add_argument("--swa_lr", default=0.05, type=float, help="SWA learning rate.")

    # General training
    parser.add_argument("--dtype", default="float32", type=str, choices=["float32", "float64"],
                        help="Set floating point precision.")
    parser.add_argument("--seed", default=1, type=int, help="Set seed for deterministic training.")
    parser.add_argument("--load_checkpoint", default=None, type=str,
                        help="Path to model checkpoint (weights, optimizer, epoch).")
    parser.add_argument("--compile", action="store_true",
                        help="Compile model for faster training (requires PyTorch 2).")

    parser.add_argument("--num_epochs", default=200, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=128, type=int, help="Training batch size per GPU.")
    parser.add_argument("--lr", default=1e-1, type=float, help="Training learning rate.")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay (L2 regularization)")
    parser.add_argument("--optimizer", default="RiemannianSGD", type=str,
                        choices=["RiemannianAdam", "RiemannianSGD", "Adam", "SGD"], help="Optimizer for training.")
    parser.add_argument("--use_lr_scheduler", action="store_true",
                        help="Enable LR scheduler step each epoch.")

    parser.add_argument("--batch_size_test", default=64, type=int, help="Validation/Testing batch size per GPU.")
    parser.add_argument("--save_activations", action="store_true",
                        help="Save layer activations at each epoch during validation.")

    parser.add_argument("--num_layers", default=18, type=int, choices=[18, 34,50], help="Number of layers in ResNet.")
    parser.add_argument("--embedding_dim", default=512, type=int,
                        help="Dimensionality of classification embedding space")

    parser.add_argument("--encoder_manifold", default="lorentz", type=str,
                        choices=["euclidean", "lorentz", "hybrid"], help="Select conv model encoder manifold.")
    parser.add_argument("--decoder_manifold", default="lorentz", type=str,
                        choices=["euclidean", "lorentz", "poincare"], help="Select conv model decoder manifold.")

    parser.add_argument("--learn_k", action="store_true", help="Learnable curvature of hyperbolic geometry.")
    parser.add_argument("--encoder_k", default=1.0, type=float,
                        help="Initial curvature in backbone (geoopt.K=-1/K).")
    parser.add_argument("--decoder_k", default=1.0, type=float,
                        help="Initial curvature in decoder (geoopt.K=-1/K).")
    parser.add_argument("--clip_features", default=1.0, type=float,
                        help="Clipping parameter for hybrid HNNs (Guo et al., 2022)")

    parser.add_argument("--dataset", default="CIFAR-100", type=str,
                        choices=["MNIST", "CIFAR-10", "CIFAR-100", "Tiny-ImageNet",'CUB'], help="Select a dataset.")
    parser.add_argument("--num_runs", default=1, type=int, help="Number of runs to perform for averaging results.")
    
    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments and fill safe defaults."""
    if args.enable_nmf:
        if not (0.0 < args.rank_ratio <= 1.0):
            raise ValueError(f"rank_ratio must be in (0, 1], got {args.rank_ratio}")
        if args.iters_train < 0 or args.iters_eval < 0:
            raise ValueError("NMF iterations must be non-negative")

    if args.enable_sparsity and args.sparsity_weight < 0:
        raise ValueError("Sparsity weight must be non-negative")

    if args.num_epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    
    if args.batch_size <= 0 or args.batch_size_test <= 0:
        raise ValueError("Batch sizes must be positive")
    
    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")

    # Auto-assign a free port if not provided
    if args.port is None:
        max_attempts = 100
        for _ in range(max_attempts):
            port = random.randint(20000, 65535)
            if is_port_free(port):
                args.port = str(port)
                break
        else:
            raise RuntimeError(f"Could not find a free port after {max_attempts} attempts")
    
    return args




def get_nmf_loss_weights(epoch: int, total_epochs: int, args) -> Tuple[float, float, float, float]:
    """Calculate NMF loss weights with optional decay on reconstruction."""
    lambda_rec = args.nmf_loss_weight_rec
    lambda_sparse = args.nmf_loss_weight_sparse
    lambda_orth = args.nmf_loss_weight_orth

    if args.enable_nmf and args.nmf_rec_decay and total_epochs > 0:
        progress = max(0.0, min(1.0, (epoch - 1) / total_epochs))
        
        if args.nmf_rec_decay_type == "linear":
            decay_factor = 1.0 - progress
        elif args.nmf_rec_decay_type == "cosine":
            decay_factor = 0.5 * (1 + np.cos(np.pi * progress))
        elif args.nmf_rec_decay_type == "exponential":
            decay_factor = np.exp(-5 * progress)
        else:
            decay_factor = 1.0
            
        lambda_rec = args.nmf_loss_weight_rec * decay_factor

    return lambda_rec, lambda_sparse, lambda_orth

def create_data_loaders_ddp(args, rank: int, world_size: int):
    """Create data loaders with distributed samplers when world_size > 1."""
    try:
        train_loader, val_loader, test_loader, img_dim, num_classes = select_dataset(args)
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise

    if world_size > 1:
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        test_dataset = test_loader.dataset

        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

        train_loader_ddp = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True
        )
        val_loader_ddp = DataLoader(
            val_dataset, batch_size=args.batch_size_test, sampler=val_sampler,
            num_workers=4, pin_memory=True, drop_last=False, persistent_workers=True
        )
        test_loader_ddp = DataLoader(
            test_dataset, batch_size=args.batch_size_test, sampler=test_sampler,
            num_workers=4, pin_memory=True, drop_last=False, persistent_workers=True
        )
        return train_loader_ddp, val_loader_ddp, test_loader_ddp, img_dim, num_classes, train_sampler
    else:
        # Single GPU: use loaders as-is
        return train_loader, val_loader, test_loader, img_dim, num_classes, None


def main_worker(rank: int, world_size: int, args, shared_dict=None):
    """Main worker function for each process."""
    try:
        setup(rank, world_size, args.port)
    except Exception as e:
        logger.error(f"Failed to setup distributed training: {e}")
        return None

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if rank == 0:
        logger.info(f"Running experiment: {args.exp_name}")
        logger.info(f"Using {world_size} GPUs with DDP")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Global batch size: {args.batch_size * world_size}")

    # Per-rank deterministic seeds
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if rank == 0:
        logger.info("Loading dataset...")
    
    try:
        train_loader, val_loader, test_loader, img_dim, num_classes, train_sampler = create_data_loaders_ddp(
            args, rank, world_size
        )
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        cleanup()
        return None

    if rank == 0:
        logger.info("Creating model...")
    
    try:
        model = select_model(img_dim, num_classes, args)
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        cleanup()
        return None

    # Optional compile BEFORE DDP
    if args.compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")

    model = model.to(device)
    
    # Ensure parameters are contiguous
    for p in model.parameters():
        if p.data is not None:
            p.data = p.data.contiguous()

    if rank == 0:
        logger.info("Creating optimizer...")
    
    try:
        optimizer, lr_scheduler = select_optimizer(model, args)
    except Exception as e:
        logger.error(f"Failed to create optimizer: {e}")
        cleanup()
        return None

    
    param_names_to_average = []  # Remove this list entirely
    for name, param in model.named_parameters():
        if 'nmf_module' not in name:
            param_names_to_average.append(name)  # Remove this entire loop

    swa_model, swa_scheduler, update_bn_func = create_swa_components(
        model, optimizer, swa_lr=args.swa_lr
    )
    # Activation logger (attach to the base model module, not DDP)
    activation_logger = None
    if rank == 0 and args.save_activations and args.output_dir is not None:
        try:
            activation_dir = os.path.join(args.output_dir, "activations", args.exp_name)
            activation_logger = ActivationLogger(
                output_dir=activation_dir, save_every_n_epochs=10, max_samples_per_epoch=50
            )
            activation_logger.set_optimizer(optimizer)
            if hasattr(model, "set_activation_logger"):
                model.set_activation_logger(activation_logger)
            logger.info(f"Activation logging enabled, saving to: {activation_dir}")
        except Exception as e:
            logger.warning(f"Failed to setup activation logger: {e}")

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"-> Number of model params: {total_params:,} (trainable: {trainable_params:,})")

    criterion = nn.CrossEntropyLoss(reduction="mean")

    # Load checkpoint on ALL ranks for robustness
    start_epoch = 0
    if args.load_checkpoint is not None:
        if rank == 0:
            logger.info(f"Loading model checkpoint from {args.load_checkpoint}")
        
        try:
            model, optimizer, lr_scheduler, start_epoch = load_checkpoint(model, optimizer, lr_scheduler, args)
            if activation_logger is not None:
                activation_logger.set_optimizer(optimizer)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            cleanup()
            return None

    # Wrap with DDP
    ddp_kwargs = {}
    if torch.cuda.is_available():
        ddp_kwargs = dict(device_ids=[rank], output_device=rank)
    
    try:
        model = DDP(model, **ddp_kwargs)
    except Exception as e:
        logger.error(f"Failed to wrap model with DDP: {e}")
        cleanup()
        return None

    # GPU temperature monitor
    temp_monitor = None
    if rank == 0:
        try:
            temp_monitor = GPUTempMonitor(temp_threshold=95, check_interval=30)
            temp_monitor.start()
        except Exception as e:
            logger.warning(f"Failed to start GPU temperature monitor: {e}")

    if rank == 0:
        logger.info("Training...")

    global_step = start_epoch * len(train_loader)
    best_acc = 0.0
    best_epoch = 0

    try:
        for epoch in range(start_epoch, args.num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if rank == 0 and activation_logger is not None:
                activation_logger.set_epoch(epoch)

            model.train()

            lambda_rec, lambda_sparse, lambda_orth = get_nmf_loss_weights(
                epoch + 1, args.num_epochs, args
            )

            # Meters
            losses = AverageMeter("Loss", ":.4e")
            cls_losses = AverageMeter("ClsLoss", ":.4e")
            sparsity_losses = AverageMeter("SparsityLoss", ":.4e")

            nmf_total_meter = AverageMeter("NMFLoss", ":.4e")
            nmf_rec_meter = AverageMeter("NMFRecLoss", ":.4e")
            nmf_sparse_meter = AverageMeter("NMFSparseLoss", ":.4e")
            nmf_orth_meter = AverageMeter("NMFOrthLoss", ":.4e")

            acc1_meter = AverageMeter("Acc@1", ":6.2f")
            acc5_meter = AverageMeter("Acc@5", ":6.2f")

            pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=(rank != 0))

            for i, (x, y) in pbar:
                if rank == 0 and temp_monitor is not None:
                    while getattr(temp_monitor, 'pause_execution', False):
                        time.sleep(1.0)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                try:
                    logits = model(x)
                    classification_loss = criterion(logits, y)
                except Exception as e:
                    logger.error(f"Forward pass failed: {e}")
                    continue

                # External sparsity loss
                sparsity_loss = torch.tensor(0.0, device=device)
                if hasattr(model.module, "get_sparsity_loss"):
                    try:
                        sparsity_loss = model.module.get_sparsity_loss()
                    except Exception as e:
                        logger.warning(f"Failed to get sparsity loss: {e}")

                if hasattr(model.module, "update_training_step"):
                    try:
                        model.module.update_training_step()
                    except Exception as e:
                        logger.warning(f"Failed to update training step: {e}")

                # NMF aggregation: single source of truth
                nmf_total_loss = torch.tensor(0.0, device=device)
                total_nmf_rec_loss = torch.tensor(0.0, device=device)
                total_nmf_sparse_loss = torch.tensor(0.0, device=device)
                total_nmf_orth_loss = torch.tensor(0.0, device=device)

                if args.enable_nmf and hasattr(model.module, "get_nmf_losses"):
                    try:
                        detailed_nmf_losses = model.module.get_nmf_losses()
                        if rank == 0 and i == 0 and hasattr(model.module, "get_nmf_stats"):
                            logger.debug(f"NMF Stats: {model.module.get_nmf_stats()}")

                        for k, loss in detailed_nmf_losses.items():
                            if not torch.is_tensor(loss):
                                continue
                            if "rec_loss" in k:
                                total_nmf_rec_loss = total_nmf_rec_loss + loss
                            elif "sparse_loss" in k:
                                total_nmf_sparse_loss = total_nmf_sparse_loss + loss
                            elif "orth_loss" in k:
                                total_nmf_orth_loss = total_nmf_orth_loss + loss

                        # Apply decay to reconstruction term only
                        rec_term = lambda_rec * total_nmf_rec_loss

                        nmf_total_loss = (
                                rec_term
                                + lambda_sparse * total_nmf_sparse_loss
                                + lambda_orth * total_nmf_orth_loss
                            )
                    except Exception as e:
                        logger.warning(f"Failed to compute NMF losses: {e}")

                # Compose total loss (matching mmm.py)
                total_loss = classification_loss + sparsity_loss + nmf_total_loss

                # Check for NaN/Inf in loss
                if not torch.isfinite(total_loss):
                    logger.warning(f"Non-finite loss detected: {total_loss}, skipping batch")
                    continue

                optimizer.zero_grad(set_to_none=True)
                
                try:
                    total_loss.backward()
                except Exception as e:
                    logger.error(f"Backward pass failed: {e}")
                    continue

               
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data = p.grad.data.contiguous()

                try:
                    optimizer.step()
                except Exception as e:
                    logger.error(f"Optimizer step failed: {e}")
                    continue

                for p in model.parameters():
                    if p.data is not None:
                        p.data = p.data.contiguous()

                with torch.no_grad():
                    try:
                        top1, top5 = accuracy(logits, y, topk=(1, 5))
                        losses.update((total_loss), x.size(0))
                        cls_losses.update((classification_loss), x.size(0))
                        sparsity_losses.update((sparsity_loss), x.size(0))

                        if args.enable_nmf and hasattr(model.module, "get_nmf_losses"):
                            nmf_total_meter.update((nmf_total_loss), x.size(0))
                            nmf_rec_meter.update((total_nmf_rec_loss), x.size(0))
                            nmf_sparse_meter.update((total_nmf_sparse_loss), x.size(0))
                            nmf_orth_meter.update((total_nmf_orth_loss), x.size(0))

                        acc1_meter.update((top1), x.size(0))
                        acc5_meter.update((top5), x.size(0))

                        if rank == 0 and activation_logger is not None:
                            try:
                                activation_logger.save_epoch_summary()
                            except Exception as e:
                                logger.warning(f"Failed to save activation summary: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to compute metrics: {e}")

                global_step += 1

            # Reduce across ranks for all displayed components
            def _rt(val):
                if dist.is_initialized() and world_size > 1:
                    try:
                        return reduce_tensor(val.detach().clone().to(device), world_size).item()
                    except Exception:
                        return float(val)
                return float(val)

            avg_loss = _rt(losses.avg)
            avg_cls_loss = _rt(cls_losses.avg)
            avg_sparsity_loss = _rt(sparsity_losses.avg)
            avg_acc1 = _rt(acc1_meter.avg)
            avg_acc5 = _rt(acc5_meter.avg)

            avg_nmf_total = 0.0
            avg_nmf_rec = 0.0
            avg_nmf_sparse = 0.0
            avg_nmf_orth = 0.0

            if args.enable_nmf:
                avg_nmf_total = _rt(nmf_total_meter.avg)
                avg_nmf_rec = _rt(nmf_rec_meter.avg)
                avg_nmf_sparse = _rt(nmf_sparse_meter.avg)
                avg_nmf_orth = _rt(nmf_orth_meter.avg)

            # Scheduler step (per-epoch, matching mmm.py)
            if lr_scheduler is not None and args.use_lr_scheduler:
                try:
                    lr_scheduler.step()
                    current_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") \
                        else optimizer.param_groups[0]["lr"]
                    if rank == 0:
                        logger.info(f"Learning rate updated to: {current_lr:.6f}")
                except Exception as e:
                    logger.warning(f"Failed to step scheduler: {e}")

            # SWA update (added to match mmm.py)
            if epoch >= args.swa_start_epoch:
                swa_model.update_parameters(model)
                restore_nmf_params(model, swa_model)  
                stabilize_manifold_params(swa_model)
                swa_scheduler.step()

            # Validation (use SWA model if applicable, matching mmm.py)
            eval_model = swa_model if epoch >= args.swa_start_epoch else model
            loss_val, acc1_val, acc5_val = evaluate_ddp(eval_model, val_loader, criterion, device, world_size)

            if rank == 0:
                log_msg = (
                    f"Epoch {epoch + 1}/{args.num_epochs}: "
                    f"TotalLoss={avg_loss:.4f}, ClsLoss={avg_cls_loss:.4f}, "
                    f"SparsityLoss={avg_sparsity_loss:.6f}, Acc@1={avg_acc1:.4f}, Acc@5={avg_acc5:.4f}, "
                    f"Validation: Loss={loss_val:.4f}, Acc@1={acc1_val:.4f}, Acc@5={acc5_val:.4f}"
                )
                if args.enable_nmf:
                    log_msg += (f", NMFLoss={avg_nmf_total:.6f}, "
                                f"NMFRecLoss={avg_nmf_rec:.6f}, "
                                f"NMFSparseLoss={avg_nmf_sparse:.6f}, NMFOrthLoss={avg_nmf_orth:.6f}, ")
                
                    log_msg += (f", λ_rec={lambda_rec:.6f}, "
                                    f"λ_sparse={lambda_sparse:.6f}, λ_orth={lambda_orth:.6f}")
                logger.info(log_msg)

                # Save best model (using eval_model, matching mmm.py)
                if acc1_val > best_acc:
                    best_acc = acc1_val
                    best_epoch = epoch + 1
                    if args.output_dir is not None:
                        try:
                            os.makedirs(args.output_dir, exist_ok=True)
                            save_path = os.path.join(args.output_dir, f"best_{args.exp_name}.pth")

                            model_state = eval_model.module.state_dict() if hasattr(eval_model, "module") else eval_model.state_dict()
                            checkpoint_type = "regular"

                          
                            checkpoint = {
                                "model": model_state,
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                                "epoch": epoch,
                                "args": args,
                                "best_acc": best_acc,
                                "model_type": checkpoint_type,
                                "lambda_weights": {
                                    "lambda_rec": lambda_rec,
                                    "lambda_sparse": lambda_sparse,
                                    "lambda_orth": lambda_orth,
                                },
                            }
                            torch.save(checkpoint, save_path)
                            logger.info(f"Saved best {checkpoint_type} model with acc: {best_acc:.4f}")
                        except Exception as e:
                            logger.error(f"Failed to save checkpoint: {e}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        cleanup()
        return None

    # End training
    if rank == 0:
        if temp_monitor is not None:
            try:
                temp_monitor.stop()
            except Exception as e:
                logger.warning(f"Failed to stop temperature monitor: {e}")
        
        logger.info("-" * 50)
        logger.info("Training finished")
        logger.info("-" * 50)
        logger.info(f"Best epoch = {best_epoch}, with Acc@1={best_acc:.4f}")

        # Update BN for SWA and final eval (added to match mmm.py)
        if args.num_epochs >= args.swa_start_epoch:
            logger.info("Updating BatchNorm statistics for SWA model...")
            update_bn_func(train_loader, swa_model, device=device)

            # Final eval with SWA model
            swa_model.eval()
            loss_swa, acc1_swa, acc5_swa = evaluate_ddp(swa_model, test_loader, criterion, device, world_size)
            logger.info(f"SWA test acc: Acc@1={acc1_swa:.2f}%, Acc@5={acc5_swa:.2f}%")

            # Save SWA model
            try:
                save_path = os.path.join(args.output_dir, f"final_{args.exp_name}.pth")
                torch.save(swa_model.state_dict(), save_path)
                logger.info(f"Saved SWA final model to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save SWA final model: {e}")
        else:
            # Save regular model
            try:
                save_path = os.path.join(args.output_dir, f"final_{args.exp_name}.pth")
                torch.save(model.module.state_dict(), save_path)
                logger.info(f"Saved final model to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save final model: {e}")

    if rank == 0 and activation_logger is not None:
        try:
            activation_logger.shutdown()
            logger.info("Activation logger shutdown complete")
        except Exception as e:
            logger.warning(f"Failed to shutdown activation logger: {e}")

    # Test final model
    loss_test, acc1_test, acc5_test = evaluate_ddp(eval_model, test_loader, criterion, device, world_size)

    if rank == 0:
        logger.info(f"Final model results: Loss={loss_test:.4f}, Acc@1={acc1_test:.4f}, Acc@5={acc5_test:.4f}")

        if shared_dict is not None:
            shared_dict["final_acc1"] = acc1_test
            shared_dict["final_acc5"] = acc5_test
            shared_dict["best_acc"] = best_acc

        if args.output_dir is not None:
            logger.info("Testing best saved model...")
            save_path = os.path.join(args.output_dir, f"best_{args.exp_name}.pth")
            try:
                checkpoint = torch.load(save_path, map_location=device)
                eval_model.module.load_state_dict(checkpoint["model"], strict=True) if hasattr(eval_model, "module") else eval_model.load_state_dict(checkpoint["model"], strict=True)
                loss_test_best, acc1_test_best, acc5_test_best = evaluate_ddp(
                    eval_model, test_loader, criterion, device, world_size
                )
                logger.info(f"Best model results: Loss={loss_test_best:.4f}, Acc@1={acc1_test_best:.4f}, Acc@5={acc5_test_best:.4f}")
            except FileNotFoundError:
                logger.warning(f"Best model checkpoint not found at {save_path}")
            except Exception as e:
                logger.error(f"Failed to load and test best model: {e}")

        cleanup()
        return {"final_acc1": acc1_test, "final_acc5": acc5_test, "best_acc": best_acc}

    cleanup()
    return None


@torch.no_grad()
def evaluate_ddp(model, val_loader, criterion, device, world_size: int) -> Tuple[float, float, float]:
    """Evaluate model on validation set with proper handling for DDP/non-DDP."""
    model.eval()
    eval_model = model  # DDP or plain

    losses = AverageMeter("Loss", ":.4e")
    acc1 = AverageMeter("Acc@1", ":6.2f")
    acc5 = AverageMeter("Acc@5", ":6.2f")

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        try:
            logits = eval_model(x)
            loss = criterion(logits, y)
            
            if torch.isfinite(loss):
                top1, top5 = accuracy(logits, y, topk=(1, 5))
                losses.update((loss), x.size(0))
                acc1.update((top1), x.size(0))
                acc5.update((top5), x.size(0))
        except Exception as e:
            logger.warning(f"Error during evaluation: {e}")
            continue

    if dist.is_initialized() and world_size > 1:
        try:
            avg_loss = reduce_tensor(losses.avg.detach().clone().to(device), world_size).item()
            avg_acc1 = reduce_tensor(acc1.avg.detach().clone().to(device), world_size).item()
            avg_acc5 = reduce_tensor(acc5.avg.detach().clone().to(device), world_size).item()
        except Exception as e:
            logger.warning(f"Failed to reduce validation metrics: {e}")
            avg_loss = float(losses.avg)
            avg_acc1 = float(acc1.avg)
            avg_acc5 = float(acc5.avg)
    else:
        avg_loss = float(losses.avg)
        avg_acc1 = float(acc1.avg)
        avg_acc5 = float(acc5.avg)

    return avg_loss, avg_acc1, avg_acc5


def main(args, shared_dict=None) -> Optional[Dict[str, float]]:
    """Main function that spawns distributed processes."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Please ensure you have GPUs available.")
        return None

    available_gpus = torch.cuda.device_count()
    world_size = min(args.world_size, available_gpus)

    if world_size < args.world_size:
        logger.warning(f"Requested {args.world_size} GPUs, but only {available_gpus} available. Using {world_size} GPUs.")
        args.world_size = world_size

    logger.info(f"Starting distributed training with {world_size} GPUs")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = args.port

    if args.output_dir is not None and not os.path.exists(args.output_dir):
        logger.info("Creating missing output directory...")
        os.makedirs(args.output_dir, exist_ok=True)

    if world_size == 1:
        # Single GPU case: do not spawn
        return main_worker(0, 1, args, shared_dict)
    else:
        # Multi-GPU case
        try:
            with mp.Manager() as manager:
                shared = manager.dict() if shared_dict is None else shared_dict
                mp.spawn(main_worker, nprocs=world_size, args=(world_size, args, shared), join=True)
                if shared_dict is None:
                    if shared:
                        return {
                            "final_acc1": shared.get("final_acc1", 0.0),
                            "final_acc5": shared.get("final_acc5", 0.0),
                            "best_acc": shared.get("best_acc", 0.0),
                        }
                return None
        except Exception as e:
            logger.error(f"Error in multiprocessing: {e}")
            return None


if __name__ == "__main__":
    try:
        args = getArguments()
        args = validate_args(args)

        # Global dtype
        if args.dtype == "float64":
            torch.set_default_dtype(torch.float64)
        elif args.dtype == "float32":
            torch.set_default_dtype(torch.float32)
        else:
            raise ValueError(f"Wrong dtype in configuration -> {args.dtype}")

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        if args.num_runs == 1:
            main(args)
        else:
            all_results = []
            original_exp_name = args.exp_name

            for run in range(args.num_runs):
                args.seed = run + 1
                args.exp_name = f"{original_exp_name}_run{run}"

                # Generate new port for each run
                max_attempts = 100
                for _ in range(max_attempts):
                    port = random.randint(20000, 65535)
                    if is_port_free(port):
                        args.port = str(port)
                        break
                else:
                    logger.error(f"Could not find a free port for run {run}")
                    continue

                if args.world_size > 1:
                    with mp.Manager() as manager:
                        shared = manager.dict()
                        main(args, shared)
                        if shared:
                            result = {
                                "final_acc1": shared.get("final_acc1", 0.0),
                                "final_acc5": shared.get("final_acc5", 0.0),
                                "best_acc": shared.get("best_acc", 0.0),
                            }
                            all_results.append(result)
                else:
                    result = main(args)
                    if result:
                        all_results.append(result)

            if all_results:
                final_acc1s = [r["final_acc1"] for r in all_results]
                final_acc5s = [r["final_acc5"] for r in all_results]
                best_accs = [r["best_acc"] for r in all_results]

                mean_final_acc1 = np.mean(final_acc1s)
                std_final_acc1 = np.std(final_acc1s)
                mean_final_acc5 = np.mean(final_acc5s)
                std_final_acc5 = np.std(final_acc5s)
                mean_best_acc = np.mean(best_accs)
                std_best_acc = np.std(best_accs)

                logger.info(f"Summary over {args.num_runs} runs:")
                logger.info(f"Final Acc@1: {mean_final_acc1:.2f} ± {std_final_acc1:.2f}")
                logger.info(f"Final Acc@5: {mean_final_acc5:.2f} ± {std_final_acc5:.2f}")
                logger.info(f"Best Acc@1: {mean_best_acc:.2f} ± {std_best_acc:.2f}")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
