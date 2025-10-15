import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lib.geoopt import ManifoldParameter
from lib.geoopt.optim import RiemannianAdam, RiemannianSGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from classification.models.classifier import ResNetClassifier 

def load_checkpoint(model, optimizer, lr_scheduler, args):
    """ Loads a checkpoint from file-system. """

    checkpoint = torch.load(args.load_checkpoint, map_location='cpu',weights_only=False)

    model.load_state_dict(checkpoint['model'])

    # Ensure sparsity parameters are correctly set after loading
    if hasattr(model, 'enable_sparsity') and hasattr(args, 'enable_sparsity'):
        model.enable_sparsity = args.enable_sparsity
        
        # If model has an encoder with sparsity parameters
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'enable_sparsity'):
            model.encoder.enable_sparsity = args.enable_sparsity
            
            # Update sparsity parameters if they exist
            if hasattr(model.encoder, 'sparsity_weight') and hasattr(args, 'sparsity_weight'):
                model.encoder.sparsity_weight = args.sparsity_weight
                
            if hasattr(model.encoder, 'sparsity_type') and hasattr(args, 'sparsity_type'):
                model.encoder.sparsity_type = args.sparsity_type
                
            if hasattr(model.encoder, 'k_ratio') and hasattr(args, 'k_ratio'):
                model.encoder.k_ratio = args.k_ratio
                
            if hasattr(model.encoder, 'apply_k_sparse_to') and hasattr(args, 'apply_k_sparse_to'):
                model.encoder.apply_k_sparse_to = args.apply_k_sparse_to

    # Ensure NMF parameters are correctly set after loading
    if hasattr(model, 'enable_nmf') and hasattr(args, 'enable_nmf'):
        model.enable_nmf = args.enable_nmf
        
        # If model has an encoder with NMF parameters
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'enable_nmf'):
            model.encoder.enable_nmf = args.enable_nmf
            
            # Update NMF parameters if they exist
            if hasattr(model.encoder, 'rank_ratio') and hasattr(args, 'rank_ratio'):
                model.encoder.rank_ratio = args.rank_ratio
                
            if hasattr(model.encoder, 'iters_train') and hasattr(args, 'iters_train'):
                model.encoder.iters_train = args.iters_train
                
            if hasattr(model.encoder, 'iters_eval') and hasattr(args, 'iters_eval'):
                model.encoder.iters_eval = args.iters_eval
                
            if hasattr(model.encoder, 'space') and hasattr(args, 'space'):
                model.encoder.space = args.space
                
            if hasattr(model.encoder, 'nmf_apply_to') and hasattr(args, 'nmf_apply_to'):
                model.encoder.nmf_apply_to = args.nmf_apply_to
                
            # Update NMF space if model supports it
            if hasattr(model.encoder, 'set_nmf_space') and hasattr(args, 'space'):
                model.encoder.set_nmf_space(args.space)

    if 'optimizer' in checkpoint:
        if checkpoint['args'].optimizer == args.optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = args.lr

            if (lr_scheduler is not None) and ('lr_scheduler' in checkpoint):
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            print("Warning: Could not load optimizer and lr-scheduler state_dict. Different optimizer in configuration ({}) and checkpoint ({}).".format(args.optimizer, checkpoint['args'].optimizer))

    epoch = 0
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch'] + 1

    return model, optimizer, lr_scheduler, epoch



def load_model_checkpoint(model, checkpoint_path):
    """ Loads a checkpoint from file-system. """
    checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])

    return model

def select_model(img_dim, num_classes, args):
    """ Selects and sets up an available model and returns it. """
    
    enc_args = {
        'img_dim': img_dim,
        'embed_dim': args.embedding_dim,
        'num_classes': num_classes,
        'bias': args.encoder_manifold == "lorentz"
    }
    print(args.encoder_manifold)
    if args.encoder_manifold == "lorentz":
        enc_args['learn_k'] = getattr(args, 'learn_k', False)
        enc_args['k'] = getattr(args, 'encoder_k', 1.0)
    elif args.encoder_manifold == "hybrid":
        enc_args['learn_k'] = getattr(args, 'learn_k', False)
        enc_args['k'] = getattr(args, 'encoder_k', 1.0)
    
    # Add sparsity parameters to encoder args
    if getattr(args, 'enable_sparsity', False):
        enc_args.update({
            'enable_sparsity': args.enable_sparsity,
            'sparsity_weight': getattr(args, 'sparsity_weight', 0.01),
            'sparsity_type': getattr(args, 'sparsity_type', 'l1'),
            'sparsity_schedule': getattr(args, 'sparsity_schedule', 'constant'),
            'k_ratio': getattr(args, 'k_ratio', 0.1),
            'apply_k_sparse_to': getattr(args, 'apply_k_sparse_to', 'final_layer')
        })

    # Add NMF parameters to encoder args
    if getattr(args, 'enable_nmf', False):
        enc_args.update({
            'enable_nmf': args.enable_nmf,
            'rank_ratio': getattr(args, 'rank_ratio', 0.35),
            'iters_train': getattr(args, 'iters_train', 5),
            'iters_eval': getattr(args, 'iters_eval', 5),
            'per_channel_gate': getattr(args, 'per_channel_gate', True),
            'gate_init': getattr(args, 'gate_init', -2.0),
            'space': getattr(args, 'space', 'hyperbolic'),
            'nmf_apply_to': getattr(args, 'nmf_apply_to', 'all_blocks')
        })

    dec_args = {
        'embed_dim': args.embedding_dim,
        'num_classes': num_classes,
        'k': getattr(args, 'decoder_k', 1.0),
        'learn_k': getattr(args, 'learn_k', False),
        'type': 'mlr',
        'clip_r': getattr(args, 'clip_features', None)
    }

    model = ResNetClassifier(
        num_layers=args.num_layers,
        enc_type=args.encoder_manifold,
        dec_type=args.decoder_manifold,
        enc_kwargs=enc_args,
        dec_kwargs=dec_args,
        enable_sparsity=getattr(args, 'enable_sparsity', False),
        sparsity_weight=getattr(args, 'sparsity_weight', 0.01),
        sparsity_type=getattr(args, 'sparsity_type', 'l1'),
        sparsity_schedule=getattr(args, 'sparsity_schedule', 'constant'),
        k_ratio=getattr(args, 'k_ratio', 0.1),
        apply_k_sparse_to=getattr(args, 'apply_k_sparse_to', 'final_layer'),
        # NMF parameters passed to ResNetClassifier
        enable_nmf=getattr(args, 'enable_nmf', False),
        rank_ratio=getattr(args, 'rank_ratio', 0.35),
        iters_train=getattr(args, 'iters_train', 5),
        iters_eval=getattr(args, 'iters_eval', 5),
        per_channel_gate=getattr(args, 'per_channel_gate', True),
        gate_init=getattr(args, 'gate_init', -2.0),
        space=getattr(args, 'space', 'hyperbolic'),
        nmf_apply_to=getattr(args, 'nmf_apply_to', 'all_blocks')
    )

    return model


def select_optimizer(model, args):
    """ Selects and sets up an available optimizer and returns it. """

    model_parameters = get_param_groups(model, args.lr, args.weight_decay)

    if args.optimizer == "RiemannianAdam":
        optimizer = RiemannianAdam(model_parameters, lr=args.lr, weight_decay=args.weight_decay, stabilize=1)
    elif args.optimizer == "RiemannianSGD":
        optimizer = RiemannianSGD(model_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9,  stabilize=1)#nesterov = True
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model_parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":  # Modified to match mmm.py
        optimizer = torch.optim.SGD(model_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise "Optimizer not found. Wrong optimizer in configuration... -> " + args.model

    lr_scheduler = None
    if args.use_lr_scheduler:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)  # Modified to use args.num_epochs

    return optimizer, lr_scheduler




def get_param_groups(model, lr_manifold, weight_decay_manifold):
    """Create parameter groups for different types of parameters including NMF parameters"""
    no_decay = ["scale"]
    k_params = ["manifold.k"]
    nmf_params = ["nmf_modules"]  # NMF-specific parameters

    # Standard Euclidean parameters
    standard_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and not any(nd in n for nd in no_decay)
        and not isinstance(p, ManifoldParameter)
        and not any(nd in n for nd in k_params)
        and not any(nd in n for nd in nmf_params)
    ]

    # Manifold parameters (including NMF manifold parameters)
    manifold_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and isinstance(p, ManifoldParameter)
        and not any(nd in n for nd in nmf_params)
    ]

    # Curvature parameters
    k_parameters = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and any(nd in n for nd in k_params)
    ]

    # NMF-specific parameters (may need special handling)
    nmf_parameters = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and any(nd in n for nd in nmf_params)
        and not isinstance(p, ManifoldParameter)
    ]

    # NMF manifold parameters (for Lorentz NMF)
    nmf_manifold_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and isinstance(p, ManifoldParameter)
        and any(nd in n for nd in nmf_params)
    ]

    parameters = [
        {
            "params": standard_params,
        },
        {
            "params": manifold_params,
            'lr': lr_manifold,
            "weight_decay": weight_decay_manifold
        },
        {  # k parameters
            "params": k_parameters,
            "weight_decay": 0,
            "lr": 1e-4
        }
    ]

    # Add NMF parameter groups if they exist
    if nmf_parameters:
        parameters.append({
            "params": nmf_parameters,
            "lr": lr_manifold * 0.1,  # Slightly lower LR for NMF parameters
            "weight_decay": weight_decay_manifold * 0.1  # Reduced weight decay for NMF
        })

    if nmf_manifold_params:
        parameters.append({
            "params": nmf_manifold_params,
            "lr": lr_manifold * 0.1,  # Slightly lower LR for NMF manifold parameters
            "weight_decay": 0  # No weight decay for manifold parameters
        })

    return parameters

import os
import tarfile
from six.moves import urllib
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader

class Cub2011(VisionDataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Cub2011, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        self._load_metadata()

    def _load_metadata(self):
        images = []
        with open(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), 'r') as f:
            for line in f:
                img_id, path = line.strip().split()
                images.append(path)

        labels = []
        with open(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'), 'r') as f:
            for line in f:
                img_id, label = line.strip().split()
                labels.append(int(label) - 1)  # 0-index

        train_test = []
        with open(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'), 'r') as f:
            for line in f:
                img_id, is_train = line.strip().split()
                train_test.append(int(is_train))

        self.data, self.labels = [], []
        for i, is_train_img in enumerate(train_test):
            if (self.train and is_train_img) or (not self.train and not is_train_img):
                self.data.append(os.path.join(self.root, self.base_folder, images[i]))
                self.labels.append(labels[i])

    def __getitem__(self, index):
        path, target = self.data[index], self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        md5 = self.tgz_md5
        fpath = os.path.join(root, self.filename)
        os.makedirs(root, exist_ok=True)

        urllib.request.urlretrieve(self.url, fpath)

        with tarfile.open(fpath) as file:
            file.extractall(root)

# Updated select_dataset function with CUB support
def select_dataset(args, validation_split=False):
    """ Selects an available dataset and returns PyTorch dataloaders for training, validation and testing. """

    if args.dataset == 'MNIST':
        
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32), antialias=None)
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32), antialias=None)
        ])

        train_set = datasets.MNIST('data', train=True, download=True, transform=train_transform)
        if validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.MNIST('data', train=False, download=True, transform=test_transform)

        img_dim = [1, 32, 32]
        num_classes = 10

    elif args.dataset == 'CIFAR-10':
        train_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # Updated to match mmm.py
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # Updated to match mmm.py
        ])

        train_set = datasets.CIFAR10('data', train=True, download=True, transform=train_transform)
        if validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.CIFAR10('data', train=False, download=True, transform=test_transform)

        img_dim = [3, 32, 32]
        num_classes = 10

    elif args.dataset == 'CIFAR-100':
        train_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

        train_set = datasets.CIFAR100('data', train=True, download=True, transform=train_transform)
        if validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.CIFAR100('data', train=False, download=True, transform=test_transform)

        img_dim = [3, 32, 32]
        num_classes = 100

    elif args.dataset == 'Tiny-ImageNet':
        root_dir = "data/tiny-imagenet-200/"
        train_dir = root_dir + "train/images"
        val_dir = root_dir + "val/images"
        test_dir = root_dir + "val/images" # TODO: No labels for test were given, so treat validation as test

        train_transform=transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_set = datasets.ImageFolder(train_dir, train_transform)
        val_set = datasets.ImageFolder(val_dir, test_transform)
        test_set = datasets.ImageFolder(test_dir, test_transform)

        img_dim = [3, 64, 64]
        num_classes = 200

    elif args.dataset == 'CUB':
        train_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        root = 'data'  # Adjust if your data path is different
        train_set = Cub2011(root, train=True, transform=train_transform, download=True)
        test_set = Cub2011(root, train=False, transform=test_transform, download=True)

        if validation_split:
            total_train = len(train_set)
            val_size = total_train // 5  # Approximately 20% for validation
            train_size = total_train - val_size
            train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size], generator=torch.Generator().manual_seed(1))
        else:
            val_set = test_set

        img_dim = [3, 224, 224]
        num_classes = 200

    else:
        raise "Selected dataset '{}' not available.".format(args.dataset)
    
    # Dataloader
    train_loader = DataLoader(train_set, 
        batch_size=args.batch_size, 
        num_workers=8, 
        pin_memory=True, 
        shuffle=True
    )
    test_loader = DataLoader(test_set, 
        batch_size=args.batch_size_test, 
        num_workers=8, 
        pin_memory=True, 
        shuffle=False
    ) 
    
    if validation_split:
        val_loader = DataLoader(val_set, 
            batch_size=args.batch_size_test, 
            num_workers=8, 
            pin_memory=True, 
            shuffle=False
        )
    else:
        val_loader = test_loader
        
    return train_loader, val_loader, test_loader, img_dim, num_classes