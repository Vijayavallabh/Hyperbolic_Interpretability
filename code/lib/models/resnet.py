import torch.nn as nn


from lib.lorentz.manifold import CustomLorentz

import torch
from lib.Euclidean.blocks.resnet_blocks import BasicBlock,Bottleneck

from lib.lorentz.blocks.resnet_blocks import (
    LorentzBasicBlock,
    LorentzInputBlock,LorentzBottleneck
)

from lib.lorentz.layers import LorentzMLR, LorentzGlobalAvgPool2d
from lib.lorentz.manifold import CustomLorentz

from lib.utils.activation_logger import register_all_hooks

class KSparseLayer(nn.Module):
    """K-sparse layer for Lorentz tensors"""
    
    def __init__(self, manifold,k_ratio=0.1, temperature=1.0, straight_through=True, tensor_format='lorentz'):
        super(KSparseLayer, self).__init__()
        self.k_ratio = k_ratio  
        self.manifold = manifold
        self.temperature = temperature
        self.straight_through = straight_through
        self.tensor_format = tensor_format
        
    def forward(self, x):
        """Apply k-sparse constraint to tensor"""
        if not self.training:
            return x
        
        orig_shape = x.shape
        device = x.device
        
        if len(orig_shape) == 4:  # [B, H, W, C] format for Lorentz or [B, C, H, W] for Euclidean
            if self.tensor_format == 'lorentz':
                batch_size, height, width, num_channels = orig_shape
                
                if num_channels > 1:
                    spatial_components = x[..., 1:]  # [B, H, W, C-1]
                    work_channels = num_channels - 1
                else:
                    spatial_components = x
                    work_channels = num_channels
                
                spatial_dim = height * width
                k = max(1, int(spatial_dim * self.k_ratio))
                
                spatial_reshaped = spatial_components.reshape(batch_size * work_channels, spatial_dim)
                
            else:  
                batch_size, num_channels, height, width = orig_shape
                spatial_dim = height * width
                k = max(1, int(spatial_dim * self.k_ratio))
                
                spatial_reshaped = x.reshape(batch_size * num_channels, spatial_dim)

                work_channels = num_channels
            
            abs_values = torch.abs(spatial_reshaped)
            
            if self.straight_through:
                _, top_k_indices = torch.topk(abs_values, k, dim=1, largest=True)
                
                mask = torch.zeros_like(spatial_reshaped)
                batch_indices = torch.arange(batch_size * work_channels, device=device).unsqueeze(1)
                mask[batch_indices, top_k_indices] = 1.0
                
                spatial_sparse = spatial_reshaped * mask
            else:
                logits = abs_values / self.temperature
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
                _, soft_indices = torch.topk(logits + gumbel_noise, k, dim=1, largest=True)
                
                mask = torch.zeros_like(spatial_reshaped)
                batch_indices = torch.arange(batch_size * work_channels, device=device).unsqueeze(1)
                mask[batch_indices, soft_indices] = 1.0
                
                spatial_sparse = spatial_reshaped * mask
            
            if self.tensor_format == 'lorentz':
                spatial_sparse = spatial_sparse.view(batch_size, height, width, work_channels)
                
                time_component =  self.manifold.calc_time(spatial_sparse)
                return torch.cat([time_component, spatial_sparse], dim=-1)

            else: 
                spatial_sparse = spatial_sparse.view(batch_size, num_channels, height, width)
                return spatial_sparse
                
        else:  
            batch_size = orig_shape[0]
            feature_dim = orig_shape[1]
            
            if self.tensor_format == 'lorentz' and feature_dim > 1:
                spatial_components = x[..., 1:]  # [B, features-1]
                work_dim = feature_dim - 1
            else:
                spatial_components = x
                work_dim = feature_dim
            
            k = max(1, int(work_dim * self.k_ratio))
            
            abs_values = torch.abs(spatial_components)
            
            if self.straight_through:
                _, top_k_indices = torch.topk(abs_values, k, dim=1, largest=True)
                
                mask = torch.zeros_like(spatial_components)
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
                mask[batch_indices, top_k_indices] = 1.0
                
                spatial_sparse = spatial_components * mask
            else:
                logits = abs_values / self.temperature
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
                _, soft_indices = torch.topk(logits + gumbel_noise, k, dim=1, largest=True)
                
                mask = torch.zeros_like(spatial_components)
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
                mask[batch_indices, soft_indices] = 1.0
                
                spatial_sparse = spatial_components * mask
            
            if self.tensor_format == 'lorentz':
                time_component =  self.manifold.calc_time(spatial_sparse)
                return torch.cat([time_component, spatial_sparse], dim=-1)

            else: 
                return spatial_sparse
                



class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        manifold: CustomLorentz=None,
        img_dim=[3,32,32],
        embed_dim=512,
        num_classes=100,
        bias=True,
        remove_linear=False,
        enable_sparsity=False,
        sparsity_weight=0.01,
        sparsity_type=None, 
        sparsity_schedule='constant',
        k_ratio=0.1,
        apply_k_sparse_to='all_conv_layers',
        enable_nmf=False,
        rank_ratio=0.35,
        # Add all the missing NMF parameters
        iters_train=5,
        iters_eval=5,
        per_channel_gate=True,
        gate_init=-2.0,
        space="hyperbolic",
        nmf_apply_to="all_blocks",
        **kwargs  # Catch any additional parameters
    ):
        super(ResNet, self).__init__()

        self.img_dim = img_dim[0]
        self.in_channels = 64
        self.conv3_dim = 128
        self.conv4_dim = 256
        self.embed_dim = embed_dim
        self.activation_logger = None
        self.activation_hooks = []
        
        self.bias = bias
        self.block = block
        self.manifold = manifold
        self.tensor_format = 'lorentz' if isinstance(manifold, CustomLorentz) else 'euclidean'

        # Sparsity parameters
        self.enable_sparsity = enable_sparsity
        self.sparsity_weight = sparsity_weight
        self.sparsity_type = sparsity_type
        self.sparsity_schedule = sparsity_schedule
        self.k_ratio = k_ratio
        self.apply_k_sparse_to = apply_k_sparse_to
        self.training_step = 0
        self.block_outputs = {}
        self.block_sparsity_losses = {}
        self.final_k_sparse = None
        
        # NMF parameters
        self.enable_nmf = enable_nmf
        self.rank_ratio = rank_ratio
        self.iters_train = iters_train
        self.iters_eval = iters_eval
        self.per_channel_gate = per_channel_gate
        self.gate_init = gate_init
        self.space = space
        self.nmf_apply_to = nmf_apply_to
        self.nmf_losses = {}
        
        self.output_dir = None
        
        if (self.enable_sparsity and 
              self.sparsity_type == 'k_sparse' and 
              self.apply_k_sparse_to == 'final_layer'):
            self.final_k_sparse = KSparseLayer(self.manifold,
                k_ratio=self.k_ratio,
                straight_through=True,
                tensor_format=self.tensor_format
            )

        self.conv1 = self._get_inConv()

        self.in_channels = 64
        self.conv2_x = self._make_layer(block, out_channels=self.in_channels, num_blocks=num_blocks[0], stride=1,enable_nmf=self.enable_nmf,rank_ratio=self.rank_ratio)
        self.conv3_x = self._make_layer(block, out_channels=self.conv3_dim, num_blocks=num_blocks[1], stride=2,enable_nmf=self.enable_nmf,rank_ratio=self.rank_ratio)
        self.conv4_x = self._make_layer(block, out_channels=self.conv4_dim, num_blocks=num_blocks[2], stride=2,enable_nmf=self.enable_nmf,rank_ratio=self.rank_ratio)
        self.conv5_x = self._make_layer(block, out_channels=self.embed_dim, num_blocks=num_blocks[3], stride=2,enable_nmf=self.enable_nmf,rank_ratio=self.rank_ratio)
        
        self.avg_pool = self._get_GlobalAveragePooling()
        
        if not remove_linear:
            self.predictor = self._get_predictor(self.embed_dim*block.expansion, num_classes)

        else:
            self.predictor = None


    def get_total_nmf_loss(self):
        """Get total NMF loss from all blocks that have NMF enabled"""
        if not self.enable_nmf:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Collect from all child modules that have NMF
        for name, module in self.named_modules():
            if hasattr(module, 'nmf_module') and hasattr(module.nmf_module, 'loss_dict'):
                # Sum all NMF losses (rec_loss, sparse_loss, orth_loss)
                for loss_name, loss_value in module.nmf_module.loss_dict.items():
                    if torch.is_tensor(loss_value):
                        total_loss = total_loss + loss_value
        
        return total_loss

    def get_nmf_losses(self):
        """Get detailed NMF losses from all blocks"""
        if not self.enable_nmf:
            return {}
        
        nmf_losses = {}
        
        # Collect detailed losses from all modules with NMF
        for name, module in self.named_modules():
            if hasattr(module, 'nmf_module') and hasattr(module.nmf_module, 'loss_dict'):
                for loss_name, loss_value in module.nmf_module.loss_dict.items():
                    if torch.is_tensor(loss_value):
                        nmf_losses[f"{name}_{loss_name}"] = loss_value
        
        return nmf_losses
    def get_nmf_stats(self):
        """Get NMF statistics"""
        if not self.enable_nmf:
            return {'nmf_enabled': False}
        
        stats = {
            'nmf_enabled': True,
            'rank_ratio': self.rank_ratio,
            'iters_train': self.iters_train,
            'iters_eval': self.iters_eval,
            'per_channel_gate': self.per_channel_gate,
            'total_nmf_loss': 0.0
        }
        
        # Count NMF modules and debug
        nmf_module_count = 0
        for name, module in self.named_modules():
            if hasattr(module, 'nmf_module') and module.nmf_module is not None:
                nmf_module_count += 1
                print(f"Found NMF module: {name}")
        
        print(f"Total NMF modules found: {nmf_module_count}")
        stats['nmf_module_count'] = nmf_module_count
        
        if nmf_module_count == 0:
            print("WARNING: NMF enabled but no NMF modules found!")
            print(f"enable_nmf: {self.enable_nmf}")
            print(f"manifold type: {type(self.manifold)}")
        
        try:
            total_loss = self.get_total_nmf_loss()
            stats['total_nmf_loss'] = total_loss.item()
        except:
            stats['total_nmf_loss'] = float('nan')
        
        return stats
    def get_sparsity_stats(self):
        """Get sparsity statistics"""
        if not self.enable_sparsity:
            return {'sparsity_enabled': False}
        
        return {
            'sparsity_enabled': True,
            'sparsity_weight': self._get_current_sparsity_weight(),
            'sparsity_type': self.sparsity_type,
            'training_step': self.training_step,
            'k_ratio': self.k_ratio,
            'apply_k_sparse_to': self.apply_k_sparse_to,
            'block_stats': self.get_block_sparsity_stats()
        }      
    def set_activation_logger(self, logger):
        """Set activation logger and register hooks"""
        self.activation_logger = logger
        
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
        
        if logger is not None:
            self.activation_hooks = register_all_hooks(self, logger)

   
    def _get_current_sparsity_weight(self):
        """Get sparsity weight based on schedule"""
        if self.sparsity_schedule == 'constant':
            return self.sparsity_weight
        elif self.sparsity_schedule == 'annealing':
            decay_factor = 0.95 ** (self.training_step // 1000)
            return self.sparsity_weight * decay_factor
        elif self.sparsity_schedule == 'warmup':
            if self.training_step < 5000:
                return self.sparsity_weight * (self.training_step / 5000)
            else:
                return self.sparsity_weight
        else:
            return self.sparsity_weight

    def update_training_step(self):
        """Update training step for sparsity scheduling"""
        if self.enable_sparsity:
            self.training_step += 1

    def _compute_block_sparsity_loss(self, output, block_name):
        """Compute sparsity loss for a block output"""
        if not self.enable_sparsity:
            return torch.tensor(0.0, device=output.device)
        
        if self.sparsity_type == 'l1':
            if self.tensor_format == 'lorentz':
                if len(output.shape) == 4:  # [B, H, W, C+1]
                    spatial_components = output[..., 1:] 
                    return torch.mean(torch.abs(spatial_components))
                else:  # [B, C+1]
                    spatial_components = output[..., 1:]
                    return torch.mean(torch.abs(spatial_components))
            else:
                return torch.mean(torch.abs(output))
        
        elif self.sparsity_type == 'k_sparse':
            return torch.tensor(0.0, device=output.device)
        
        else:
            return torch.tensor(0.0, device=output.device)

    def get_sparsity_loss(self):
        """Get total sparsity loss - required by training loop"""
        return self.get_total_sparsity_loss()

    def get_total_sparsity_loss(self):
        """Calculate total sparsity loss from all blocks"""
        if not self.enable_sparsity or not self.block_sparsity_losses:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for block_name, loss in self.block_sparsity_losses.items():
            if loss is not None:
                total_loss += loss
        
        return total_loss * self._get_current_sparsity_weight()

    def get_block_sparsity_stats(self):
        """Get sparsity statistics for each block"""
        stats = {}
        for block_name, output in self.block_outputs.items():
            if output is not None:
                if self.tensor_format == 'lorentz' and len(output.shape) == 4:
                    spatial_output = output[..., 1:]  # Exclude time component
                    total_elements = spatial_output.numel()
                    zero_elements = (torch.abs(spatial_output) < 1e-6).sum().item()
                else:
                    total_elements = output.numel()
                    zero_elements = (torch.abs(output) < 1e-6).sum().item()
                
                sparsity_ratio = zero_elements / total_elements if total_elements > 0 else 0
                stats[block_name] = {
                    'sparsity_ratio': sparsity_ratio,
                    'total_elements': total_elements,
                    'zero_elements': zero_elements
                }
        return stats

    def forward(self, x):
        """Forward pass with sparsity tracking"""
        self.block_outputs = {}
        self.block_sparsity_losses = {}
        
        out = self.conv1(x)
        out = self.conv2_x(out)
        self.block_outputs['conv2_x'] = out
        if self.training and self.enable_sparsity:
            self.block_sparsity_losses['conv2_x'] = self._compute_block_sparsity_loss(out, 'conv2_x')
        
        out = self.conv3_x(out)
        self.block_outputs['conv3_x'] = out
        if self.training and self.enable_sparsity:
            self.block_sparsity_losses['conv3_x'] = self._compute_block_sparsity_loss(out, 'conv3_x')
        
        out = self.conv4_x(out)
        self.block_outputs['conv4_x'] = out
        if self.training and self.enable_sparsity:
            self.block_sparsity_losses['conv4_x'] = self._compute_block_sparsity_loss(out, 'conv4_x')
        
        out = self.conv5_x(out)
        self.block_outputs['conv5_x'] = out
        if self.training and self.enable_sparsity:
            self.block_sparsity_losses['conv5_x'] = self._compute_block_sparsity_loss(out, 'conv5_x')
 

        if self.final_k_sparse is not None and self.training:
            out = self.final_k_sparse(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        if self.predictor is not None:
            out = self.predictor(out)
        
        return out
    
    def _make_layer(self, block, out_channels, num_blocks, stride, enable_nmf=False, rank_ratio=0.35):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            if self.manifold is None:
                # Euclidean blocks
                kwargs = {
                    'nmf': enable_nmf,
                    'rank_ratio': rank_ratio
                }
                # Add all NMF parameters for Euclidean blocks too
                if enable_nmf:
                    kwargs.update({
                        'iters_train': self.iters_train,
                        'iters_eval': self.iters_eval,
                        'per_channel_gate': self.per_channel_gate,
                        'gate_init': self.gate_init
                    })
                
                layers.append(block(
                    self.in_channels, 
                    out_channels, 
                    stride, 
                    self.bias,
                    **kwargs
                ))
            elif type(self.manifold) is CustomLorentz:
                # Lorentz blocks - pass all NMF parameters
                kwargs = {
                    'nmf': enable_nmf,
                    'rank_ratio': rank_ratio
                }
                if enable_nmf:
                    kwargs.update({
                        'iters_train': self.iters_train,
                        'iters_eval': self.iters_eval,
                        'per_channel_gate': self.per_channel_gate,
                        'gate_init': self.gate_init
                    })
                
                layers.append(
                    block(
                        self.manifold,
                        self.in_channels,
                        out_channels,
                        stride,
                        self.bias,
                        **kwargs
                    )
                )
            else:
                raise RuntimeError(
                    f"Manifold {type(self.manifold)} not supported in ResNet."
                )

            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _get_inConv(self):
        if self.manifold is None:

            return nn.Sequential(
                nn.Conv2d(
                    self.img_dim,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    bias=self.bias
                ),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(inplace=True),
            )

        elif type(self.manifold) is CustomLorentz:
            return LorentzInputBlock(
                self.manifold, 
                self.img_dim, 
                self.in_channels, 
                self.bias,
            )

        else:
            raise RuntimeError(
                f"Manifold {type(self.manifold)} not supported in ResNet."
            )

    def _get_predictor(self, in_features, num_classes):
        if self.manifold is None:
        
            return nn.Linear(in_features, num_classes, bias=self.bias)

        elif type(self.manifold) is CustomLorentz:
            return LorentzMLR(self.manifold, in_features+1, num_classes)

        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ResNet.")

    def _get_GlobalAveragePooling(self):
        if self.manifold is None:
            return nn.AdaptiveAvgPool2d((1, 1))

        elif type(self.manifold) is CustomLorentz:
            return LorentzGlobalAvgPool2d(self.manifold, keep_dim=True)

        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ResNet.")

class HybridResNet(nn.Module):
    def __init__(
        self,
        num_blocks,
        lorentz_manifold=None,
        img_dim=[3,32,32],
        embed_dim=512,
        num_classes=100,
        bias=True,
        remove_linear=True,
        enable_sparsity=False,
        sparsity_weight=0.01,
        sparsity_type=None,
        sparsity_schedule='constant',
        k_ratio=0.1,
        apply_k_sparse_to='final_layer',
        enable_nmf=False,
        rank_ratio=0.35,
        # Add all the missing NMF parameters
        iters_train=5,
        iters_eval=5,
        per_channel_gate=True,
        gate_init=-2.0,
        space="hyperbolic",
        nmf_apply_to="all_blocks",
        **kwargs  # Catch any additional parameters
    ):
        super(HybridResNet, self).__init__()
        self.img_dim = img_dim[0]
        self.in_channels = 64
        self.conv3_dim = 128
        self.conv4_dim = 256
        self.embed_dim = embed_dim

        self.bias = bias
        self.manifold = lorentz_manifold
        
        # Sparsity parameters
        self.enable_sparsity = enable_sparsity and (lorentz_manifold is not None)
        self.sparsity_weight = sparsity_weight
        self.sparsity_type = sparsity_type
        self.sparsity_schedule = sparsity_schedule
        self.k_ratio = k_ratio
        self.apply_k_sparse_to = apply_k_sparse_to
        self.training_step = 0
        self.block_outputs = {}
        self.block_sparsity_losses = {}
        self.final_k_sparse = None
        
        # NMF parameters
        self.enable_nmf = enable_nmf
        self.rank_ratio = rank_ratio
        self.iters_train = iters_train
        self.iters_eval = iters_eval
        self.per_channel_gate = per_channel_gate
        self.gate_init = gate_init
        self.space = space
        self.nmf_apply_to = nmf_apply_to
        self.nmf_losses = {}
        
        if (self.enable_sparsity and 
            self.sparsity_type == 'k_sparse' and 
            self.apply_k_sparse_to == 'final_layer'):
            self.final_k_sparse = KSparseLayer(self.manifold,tensor_format='euclidean',
                k_ratio=self.k_ratio,
                straight_through=(self.sparsity_type == 'k_sparse')
            )
        self.activation_logger = None
        self.activation_hooks = []

        self.conv1 = self._get_euclidean_inConv()

        self.in_channels = 64
        
        
        self.conv2_x = self._make_lorentz_layer(LorentzBasicBlock, out_channels=64, 
                                            num_blocks=num_blocks[0], stride=1)

        self.in_channels = 64  
        self.conv3_x = self._make_euclidean_layer(BasicBlock, out_channels=self.conv3_dim, 
                                                num_blocks=num_blocks[1], stride=2)
        
        self.in_channels = self.conv3_dim  # Should be 128
        self.conv4_x = self._make_lorentz_layer(LorentzBasicBlock, out_channels=self.conv4_dim, 
                                            num_blocks=num_blocks[2], stride=2)
        
        self.in_channels = self.conv4_dim  # Should be 256
        self.conv5_x = self._make_euclidean_layer(BasicBlock, out_channels=self.embed_dim, 
                                                num_blocks=num_blocks[3], stride=2)

        self.avg_pool = LorentzGlobalAvgPool2d(self.manifold, keep_dim=True)
        
        if not remove_linear:
            self.predictor = nn.Linear(self.embed_dim, num_classes, bias=self.bias)
        else:
            self.predictor = None

    def _get_current_sparsity_weight(self):
        """Get current sparsity weight based on schedule"""
        if not self.enable_sparsity:
            return 0.0
            
        if self.sparsity_schedule == 'constant':
            return self.sparsity_weight
        elif self.sparsity_schedule == 'annealing':
            return self.sparsity_weight * min(1.0, self.training_step / 1000.0)
        elif self.sparsity_schedule == 'warmup':
            if self.training_step < 500:
                return self.sparsity_weight * 0.1
            else:
                return self.sparsity_weight
        else:
            return self.sparsity_weight
        
    def set_activation_logger(self, logger):
        """Set activation logger and register hooks"""
        self.activation_logger = logger
        
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
        
        if logger is not None:
            from lib.utils.activation_logger import register_lorentz_hooks
            self.activation_hooks = register_lorentz_hooks(self, logger)




    def get_sparsity_stats(self):
        """Get comprehensive sparsity statistics"""
        if not self.enable_sparsity:
            return {'sparsity_enabled': False}
        
        return {
            'sparsity_enabled': True,
            'sparsity_weight': self._get_current_sparsity_weight(),
            'sparsity_type': self.sparsity_type,
            'training_step': self.training_step,
            'k_ratio': self.k_ratio,
            'apply_k_sparse_to': self.apply_k_sparse_to,
            'block_stats': self.get_block_sparsity_stats()
        }
    @property
    def block(self):
        """
        Property to make HybridResNet compatible with ResNetClassifier.
        Returns the block type used in the final layer.
        """
        return BasicBlock  
    
    def update_training_step(self):
        """Update training step for sparsity scheduling"""
        if self.enable_sparsity:
            self.training_step += 1
    
    def _compute_block_sparsity_loss(self, output, block_name, tensor_format='lorentz'):
        """Compute sparsity loss for a block output with specified format"""
        
        if not self.enable_sparsity or self.sparsity_type is None:
            return torch.tensor(0.0, device=output.device, requires_grad=False)
        
        elif self.sparsity_type == 'k_sparse':
            return torch.tensor(0.0, device=output.device)
        
        current_weight = self._get_current_sparsity_weight()
        if current_weight == 0.0:
            return torch.tensor(0.0, device=output.device, requires_grad=False)
        
        if tensor_format == 'euclidean':
            if len(output.shape) == 4:
                batch_size, channels, height, width = output.shape
                output_reshaped = output.permute(0, 2, 3, 1).contiguous().view(-1, channels)
                spatial_components = output_reshaped
            else:
                spatial_components = output.view(-1, output.shape[-1])
        else:
            if len(output.shape) == 4:
                batch_size, height, width, channels = output.shape
                output_reshaped = output.view(-1, channels)
                spatial_components = output_reshaped[..., 1:] if channels > 1 else output_reshaped
            else:
                spatial_components = output[..., 1:] if output.shape[-1] > 1 else output
                spatial_components = spatial_components.reshape(-1, spatial_components.shape[-1])
            
        if self.sparsity_type == 'l1':
            sparsity_loss = torch.mean(torch.abs(spatial_components))

        return current_weight * sparsity_loss
    
    def get_block_sparsity_stats(self):
        """Get detailed sparsity statistics for all blocks"""
        if not self.enable_sparsity:
            return {'sparsity_enabled': False}
                
        stats = {
            'sparsity_enabled': True,
            'sparsity_weight': self._get_current_sparsity_weight(),
            'sparsity_type': 'l1',  
            'training_step': self.training_step,
            'k_ratio': self.k_ratio,
            'total_sparsity_loss': self.get_total_sparsity_loss().item() if self.block_sparsity_losses else 0.0
        }
        
        for block_name, output in self.block_outputs.items():
            if output is not None:
                if hasattr(output, 'shape') and len(output.shape) >= 3:
                    if block_name.startswith('conv2') or block_name.startswith('conv4'):
                        spatial_components = output[:, 1:] if output.shape[1] > 1 else output
                    else:
                        spatial_components = output
                        
                    if len(spatial_components.shape) == 4:
                        spatial_flat = spatial_components.reshape(-1)
                    else:
                        spatial_flat = spatial_components
                        
                    sparsity_ratio = (torch.abs(spatial_flat) < 1e-3).float().mean().item()
                    avg_magnitude = torch.abs(spatial_flat).mean().item()
                    max_magnitude = torch.abs(spatial_flat).max().item()
                    

                    stats[f'{block_name}_sparsity_ratio'] = sparsity_ratio
                    stats[f'{block_name}_avg_magnitude'] = avg_magnitude
                    stats[f'{block_name}_max_magnitude'] = max_magnitude
                    
                    if block_name in self.block_sparsity_losses:
                        stats[f'{block_name}_loss'] = self.block_sparsity_losses[block_name].item()
        
        return stats
    def get_total_sparsity_loss(self):
        """Compute total sparsity loss from Lorentz blocks"""
        if not self.enable_sparsity or not self.block_sparsity_losses:
            return torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=False)
            
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
        
        for block_name, sparsity_loss in self.block_sparsity_losses.items():
            if sparsity_loss is not None and torch.is_tensor(sparsity_loss):
                total_loss = total_loss + sparsity_loss
            
        return total_loss

    def _get_euclidean_inConv(self):
        """Create Euclidean input conv block"""
        return nn.Sequential(
            nn.Conv2d(
                self.img_dim,
                self.in_channels,
                kernel_size=3,
                padding=1,
                bias=self.bias
            ),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

   

    def _make_euclidean_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for block_idx, stride in enumerate(strides):
            # Pass NMF parameters to Euclidean blocks
            kwargs = {}
            if self.enable_nmf:
                kwargs.update({
                    'nmf': True,
                    'rank_ratio': self.rank_ratio
                })
            
            layers.append(block(
                self.in_channels, 
                out_channels, 
                stride, 
                self.bias, 
                **kwargs
            ))
            
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _make_lorentz_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for block_idx, stride in enumerate(strides):
            # Pass all NMF parameters to Lorentz blocks
            kwargs = {}
            if self.enable_nmf:
                kwargs.update({
                    'nmf': True,
                    'rank_ratio': self.rank_ratio,
                    'iters_train': self.iters_train,
                    'iters_eval': self.iters_eval,
                    'per_channel_gate': self.per_channel_gate,
                    'gate_init': self.gate_init
                })
            
            layers.append(
                block(
                    self.manifold,
                    self.in_channels,
                    out_channels,
                    stride,
                    self.bias,
                    **kwargs
                )
            )
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
        
    def _lorentz_to_euclidean(self, x):
        """Map from Lorentz manifold to Euclidean space via tangent space"""
        # Input: x is in Lorentz format [B, H, W, C+1] where C+1 includes time component
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D tensor for Lorentz to Euclidean conversion, got {x.shape}")
        
        batch_size, height, width, channels = x.shape
        
        # Extract spatial components (exclude time component which is first channel)
        if channels > 1:
            spatial_components = x[..., 1:]  # [B, H, W, C]
            # Convert to NCHW format expected by Euclidean layers
            spatial_euclidean = spatial_components.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            return spatial_euclidean
        else:
            # Edge case: if only time component, create a minimal spatial representation
            return torch.zeros(batch_size, 1, height, width, device=x.device, dtype=x.dtype)

    def _euclidean_to_lorentz(self, x):
        """Map from Euclidean space to Lorentz manifold via exponential map"""
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D tensor for Euclidean to Lorentz conversion, got {x.shape}")
        
        batch_size, channels, height, width = x.shape
        batch_size, channels, height, width = x.shape
        
        # Convert from NCHW to NHWC format for Lorentz processing
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        
        # Flatten spatial dimensions for manifold operations
        x_flat = x_nhwc.view(-1, channels)  # [B*H*W, C]
        
        # Calculate time component using manifold
        time_component = self.manifold.calc_time(x_flat)  # [B*H*W, 1]
        
        # Combine time and spatial components
        lorentz_flat = torch.cat([time_component, x_flat], dim=-1)  # [B*H*W, C+1]
        
        # Reshape back to spatial format
        lorentz_tensor = lorentz_flat.view(batch_size, height, width, channels + 1).contiguous()  # [B, H, W, C+1]
        
        return lorentz_tensor
        
    def get_sparsity_loss(self):
        """Get total sparsity loss - required by training loop"""
        return self.get_total_sparsity_loss()
    def forward(self, x):
        # Reset tracking dictionaries
        self.block_outputs = {}
        self.block_sparsity_losses = {}
        
        # Initial conv (Euclidean): [B, 3, H, W] -> [B, 64, H, W]
        out = self.conv1(x)
        # Convert to Lorentz format: [B, 64, H, W] -> [B, H, W, 65]
        out_lorentz1 = self._euclidean_to_lorentz(out)
        
        # Block 1 (Lorentz): [B, H, W, 65] -> [B, H, W, 65] (64->64 channels + time)
        out_2 = self.conv2_x(out_lorentz1)
        self.block_outputs['conv2_x'] = out_2
        if self.training and self.enable_sparsity:
            self.block_sparsity_losses['conv2_x'] = self._compute_block_sparsity_loss(out_2, 'conv2_x', 'lorentz')

        # Convert back to Euclidean: [B, H, W, 65] -> [B, 64, H, W]
        out_euclidean2 = self._lorentz_to_euclidean(out_2)
        
        # Block 2 (Euclidean): [B, 64, H, W] -> [B, 128, H/2, W/2]
        out_3 = self.conv3_x(out_euclidean2)
        self.block_outputs['conv3_x'] = out_3
        if self.training and self.enable_sparsity:
            self.block_sparsity_losses['conv3_x'] = self._compute_block_sparsity_loss(out_3, 'conv3_x', 'euclidean')
        
        # Convert to Lorentz: [B, 128, H/2, W/2] -> [B, H/2, W/2, 129]
        out_lorentz3 = self._euclidean_to_lorentz(out_3)
        
        # Block 3 (Lorentz): [B, H/2, W/2, 129] -> [B, H/2, W/2, 257]
        out_4 = self.conv4_x(out_lorentz3)
        self.block_outputs['conv4_x'] = out_4
        if self.training and self.enable_sparsity:
            self.block_sparsity_losses['conv4_x'] = self._compute_block_sparsity_loss(out_4, 'conv4_x', 'lorentz')

        # Convert back to Euclidean: [B, H/2, W/2, 257] -> [B, 256, H/2, W/2]
        out_euclidean4 = self._lorentz_to_euclidean(out_4)
        
        # Block 4 (Euclidean): [B, 256, H/2, W/2] -> [B, 512, H/4, W/4]
        out_5 = self.conv5_x(out_euclidean4)
        self.block_outputs['conv5_x'] = out_5
        if self.training and self.enable_sparsity:
            self.block_sparsity_losses['conv5_x'] = self._compute_block_sparsity_loss(out_5, 'conv5_x', 'euclidean')
        
        # Apply k-sparse constraint if enabled
        if self.final_k_sparse is not None and self.training:
            out_5 = self.final_k_sparse(out_5)
        
        # Convert final output to Lorentz for pooling: [B, 512, H/4, W/4] -> [B, H/4, W/4, 513]
        out_lorentz_final = self._euclidean_to_lorentz(out_5)
        self.block_outputs['final_lorentz'] = out_lorentz_final
        
        # Global pooling: [B, H/4, W/4, 513] -> [B, 513]
        out = self.avg_pool(out_lorentz_final)
        out = out.view(out.size(0), -1)
        
        if self.predictor is not None:
            out = self.predictor(out)
        
        return out
    def get_total_nmf_loss(self):
        """Get total NMF loss from all blocks that have NMF enabled"""
        if not self.enable_nmf:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Collect from all child modules that have NMF
        for name, module in self.named_modules():
            if hasattr(module, 'nmf_module') and hasattr(module.nmf_module, 'loss_dict'):
                # Sum all NMF losses (rec_loss, sparse_loss, orth_loss)
                for loss_name, loss_value in module.nmf_module.loss_dict.items():
                    if torch.is_tensor(loss_value):
                        total_loss = total_loss + loss_value
        
        return total_loss

    def get_nmf_losses(self):
        """Get detailed NMF losses from all blocks"""
        if not self.enable_nmf:
            return {}
        
        nmf_losses = {}
        
        # Collect detailed losses from all modules with NMF
        for name, module in self.named_modules():
            if hasattr(module, 'nmf_module') and hasattr(module.nmf_module, 'loss_dict'):
                for loss_name, loss_value in module.nmf_module.loss_dict.items():
                    if torch.is_tensor(loss_value):
                        nmf_losses[f"{name}_{loss_name}"] = loss_value
        
        return nmf_losses

    def get_nmf_stats(self):
        """Get NMF statistics"""
        if not self.enable_nmf:
            return {'nmf_enabled': False}
        
        stats = {
            'nmf_enabled': True,
            'rank_ratio': self.rank_ratio,
            'iters_train': self.iters_train,
            'iters_eval': self.iters_eval,
            'per_channel_gate': self.per_channel_gate,
            'total_nmf_loss': self.get_total_nmf_loss().item()
        }
        
        # Count NMF modules
        nmf_module_count = 0
        for name, module in self.named_modules():
            if hasattr(module, 'nmf_module'):
                nmf_module_count += 1
        
        stats['nmf_module_count'] = nmf_module_count
        
        return stats
# Update function signatures to include NMF parameters
def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model
def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def Lorentz_resnet18(k=1, learn_k=False, manifold=None, **kwargs):
    """Constructs a Lorentz ResNet-18 model."""
    if not manifold:
        manifold = CustomLorentz(k=k, learnable=learn_k)
    model = ResNet(LorentzBasicBlock, [2, 2, 2, 2], manifold, **kwargs)
    return model

def Lorentz_resnet50(k=1, learn_k=False, manifold=None, **kwargs):
    """Constructs a Lorentz ResNet-50 model."""
    if not manifold:
        manifold = CustomLorentz(k=k, learnable=learn_k)
    model = ResNet(LorentzBottleneck, [3, 4, 6, 3], manifold, **kwargs)
    return model
def Lorentz_resnet34(k=1, learn_k=False, manifold=None, **kwargs):
    """Constructs a ResNet-34 model."""
    if not manifold:
        manifold = CustomLorentz(k=k, learnable=learn_k)
    model = ResNet(LorentzBasicBlock, [3, 4, 6, 3], manifold, **kwargs)
    return model
def hybrid_resnet18(k=1.0, learn_k=False, **kwargs):
    """Constructs a Hybrid ResNet-18 model with mixed Euclidean and Lorentz blocks."""
    lorentz_manifold = CustomLorentz(k=k, learnable=learn_k)
    model = HybridResNet([2, 2, 2, 2], lorentz_manifold=lorentz_manifold, **kwargs)
    return model


