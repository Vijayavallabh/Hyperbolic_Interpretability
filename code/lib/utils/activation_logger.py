import torch
import os
import pickle
from collections import defaultdict
import numpy as np
from threading import Thread
import queue
import time

class ActivationLogger:
    """Efficient activation logger that saves data to CPU immediately"""
    
    def __init__(self, output_dir, save_every_n_epochs=10, max_samples_per_epoch=100):
        self.output_dir = output_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.max_samples_per_epoch = max_samples_per_epoch
        self.current_epoch = 0
        self.should_log = False
        self.sample_count = 0
        
        self.activations = defaultdict(list)
        self.optimizer = None
        os.makedirs(output_dir, exist_ok=True)
        
        self.save_queue = queue.Queue()
        self.save_thread = Thread(target=self._background_saver, daemon=True)
        self.save_thread.start()
        
    def set_optimizer(self, optimizer):
        """Set optimizer reference for state logging"""
        self.optimizer = optimizer
        
    def set_epoch(self, epoch):
        """Set current epoch and determine if we should log this epoch"""
        self.current_epoch = epoch
        self.should_log = (epoch % self.save_every_n_epochs == 0)
        self.sample_count = 0
        

            
    def _get_optimizer_state_summary(self):
        """Extract relevant optimizer state information"""
        if self.optimizer is None:
            return None
            
        optimizer_state = {}
        
        optimizer_state['optimizer_type'] = type(self.optimizer).__name__
        
        optimizer_state['param_groups'] = []
        for i, group in enumerate(self.optimizer.param_groups):
            group_info = {
                'lr': group.get('lr', None),
                'weight_decay': group.get('weight_decay', None),
                'momentum': group.get('momentum', None),
                'dampening': group.get('dampening', None),
                'nesterov': group.get('nesterov', None),
                'betas': group.get('betas', None),
                'eps': group.get('eps', None),
                'amsgrad': group.get('amsgrad', None),
            }
            group_info = {k: v for k, v in group_info.items() if v is not None}
            optimizer_state['param_groups'].append(group_info)
        
        optimizer_state['state_summary'] = {}
        param_count = 0

        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    param_state = {}
                    
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            try:
                                if value.numel() <= 1:
                                    std_val = 0.0
                                else:
                                    std_val = float(value.std(unbiased=False).cpu().detach())
                                
                                param_state[key] = {
                                    'shape': list(value.shape),
                                    'numel': value.numel(),
                                    'mean': float(value.mean().cpu().detach()),
                                    'std': std_val,
                                    'min': float(value.min().cpu().detach()),
                                    'max': float(value.max().cpu().detach()),
                                }
                            except Exception as e:
                                # Skip problematic tensors
                                param_state[key] = f"Error extracting stats: {str(e)}"
                        else:
                            param_state[key] = value
                    
                    optimizer_state['state_summary'][f'param_{param_count}'] = param_state
                param_count += 1

        return optimizer_state
    
    def log_activation(self, layer_name, input_tensor, output_tensor, layer_type='conv'):
        """Log activation immediately moving to CPU"""
        if not self.should_log or self.sample_count >= self.max_samples_per_epoch:
            return
            
        input_cpu = input_tensor.detach().cpu().clone()
        output_cpu = output_tensor.detach().cpu().clone()
        optimizer_state = self._get_optimizer_state_summary()
        activation_data = {
            'epoch': self.current_epoch,
            'layer_name': layer_name,
            'layer_type': layer_type,
            'input_shape': tuple(input_cpu.shape),
            'output_shape': tuple(output_cpu.shape),
            'input': input_cpu,
            'output': output_cpu,
            'sample_idx': self.sample_count,
            'optimizer_state': optimizer_state,
            'timestamp': time.time()
        }
        
        self.save_queue.put(activation_data)
        self.sample_count += 1
    
    def _background_saver(self):
        """Background thread to save activations to disk"""
        while True:
            try:
                activation_data = self.save_queue.get(timeout=1)
                if activation_data is None:
                    break
                    
                epoch = activation_data['epoch']
                layer_name = activation_data['layer_name']
                sample_idx = activation_data['sample_idx']
                
                filename = f"epoch_{epoch}_{layer_name}_sample_{sample_idx}.pkl"
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    pickle.dump(activation_data, f)
                    
                self.save_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error saving activation: {e}")
    
    def save_epoch_summary(self):
        """Save epoch summary if logging was enabled"""
        if not self.should_log:
            return
        
        # Get optimizer state for epoch summary
        optimizer_state = self._get_optimizer_state_summary()
            
        summary = {
            'epoch': self.current_epoch,
            'total_samples': self.sample_count,
            'timestamp': time.time(),
            'optimizer_state': optimizer_state
        }
        
        summary_path = os.path.join(self.output_dir, f"epoch_{self.current_epoch}_summary.pkl")
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)
            
        #print(f"Saved {self.sample_count} activation samples for epoch {self.current_epoch}")
    
    def shutdown(self):
        """Shutdown the background saver thread"""
        self.save_queue.put(None)  
        self.save_thread.join(timeout=5)


class LorentzLayerHook:
    """Hook class to capture activations from Lorentz layers"""
    
    def __init__(self, logger, layer_name, layer_type='conv'):
        self.logger = logger
        self.layer_name = layer_name
        self.layer_type = layer_type
        self.input_tensor = None
    
    def forward_pre_hook(self, module, input):
        """Capture input tensor"""
        if len(input) > 0:
            self.input_tensor = input[0]
    
    def forward_hook(self, module, input, output):
        """Capture output tensor and log both input and output"""
        if self.input_tensor is not None:
            self.logger.log_activation(
                self.layer_name, 
                self.input_tensor, 
                output, 
                self.layer_type
            )


def register_lorentz_hooks(model, logger):
    """Register hooks for all LorentzConv2d and LorentzGlobalAvgPool2d layers"""
    hooks = []
    layer_count = defaultdict(int)
    
    def register_hooks_recursive(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if hasattr(child, '__class__'):
                class_name = child.__class__.__name__
                
                if class_name == 'LorentzConv2d':
                    layer_count['LorentzConv2d'] += 1
                    layer_name = f"LorentzConv2d_{layer_count['LorentzConv2d']}_{full_name}"
                    hook_obj = LorentzLayerHook(logger, layer_name, 'conv')
                    
                    pre_hook = child.register_forward_pre_hook(hook_obj.forward_pre_hook)
                    post_hook = child.register_forward_hook(hook_obj.forward_hook)
                    hooks.extend([pre_hook, post_hook])
                    
                    print(f"Registered hooks for {layer_name}")
                    
                elif class_name == 'LorentzGlobalAvgPool2d':
                    layer_count['LorentzGlobalAvgPool2d'] += 1
                    layer_name = f"LorentzGlobalAvgPool2d_{layer_count['LorentzGlobalAvgPool2d']}_{full_name}"
                    hook_obj = LorentzLayerHook(logger, layer_name, 'pool')
                    
                    pre_hook = child.register_forward_pre_hook(hook_obj.forward_pre_hook)
                    post_hook = child.register_forward_hook(hook_obj.forward_hook)
                    hooks.extend([pre_hook, post_hook])
                    
                    print(f"Registered hooks for {layer_name}")
            
            register_hooks_recursive(child, full_name)
    
    register_hooks_recursive(model)
    
    print(f"Total hooks registered: {len(hooks)}")
    print(f"LorentzConv2d layers: {layer_count['LorentzConv2d']}")
    print(f"LorentzGlobalAvgPool2d layers: {layer_count['LorentzGlobalAvgPool2d']}")
    
    return hooks


def load_activations(activation_dir, epoch, layer_name=None):
    """Load saved activations for analysis"""
    activations = []
    
    if layer_name:
        pattern = f"epoch_{epoch}_{layer_name}_sample_*.pkl"
    else:
        pattern = f"epoch_{epoch}_*.pkl"
    
    import glob
    files = glob.glob(os.path.join(activation_dir, pattern))
    
    for file in sorted(files):
        with open(file, 'rb') as f:
            activation_data = pickle.load(f)
            activations.append(activation_data)
    
    return activations
class EuclideanLayerHook:
    """Hook class to capture activations from Euclidean layers"""
    
    def __init__(self, logger, layer_name, layer_type='conv'):
        self.logger = logger
        self.layer_name = layer_name
        self.layer_type = layer_type
        self.input_tensor = None
    
    def forward_pre_hook(self, module, input):
        """Capture input tensor"""
        if len(input) > 0:
            self.input_tensor = input[0]
    
    def forward_hook(self, module, input, output):
        """Capture output tensor and log both input and output"""
        if self.input_tensor is not None:
            self.logger.log_activation(
                self.layer_name, 
                self.input_tensor, 
                output, 
                self.layer_type
            )


def register_all_hooks(model, logger):
    """Register hooks for both Lorentz and Euclidean layers"""
    hooks = []
    layer_count = defaultdict(int)
    
    def register_hooks_recursive(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if hasattr(child, '__class__'):
                class_name = child.__class__.__name__
                
                # Lorentz layers
                if class_name == 'LorentzConv2d':
                    layer_count['LorentzConv2d'] += 1
                    layer_name = f"LorentzConv2d_{layer_count['LorentzConv2d']}_{full_name}"
                    hook_obj = LorentzLayerHook(logger, layer_name, 'conv')
                    
                    pre_hook = child.register_forward_pre_hook(hook_obj.forward_pre_hook)
                    post_hook = child.register_forward_hook(hook_obj.forward_hook)
                    hooks.extend([pre_hook, post_hook])
                    
                    print(f"Registered hooks for {layer_name}")
                    
                elif class_name == 'LorentzGlobalAvgPool2d':
                    layer_count['LorentzGlobalAvgPool2d'] += 1
                    layer_name = f"LorentzGlobalAvgPool2d_{layer_count['LorentzGlobalAvgPool2d']}_{full_name}"
                    hook_obj = LorentzLayerHook(logger, layer_name, 'pool')
                    
                    pre_hook = child.register_forward_pre_hook(hook_obj.forward_pre_hook)
                    post_hook = child.register_forward_hook(hook_obj.forward_hook)
                    hooks.extend([pre_hook, post_hook])
                    
                    print(f"Registered hooks for {layer_name}")
                
                # Euclidean layers
                elif class_name == 'Conv2d':
                    layer_count['Conv2d'] += 1
                    layer_name = f"Conv2d_{layer_count['Conv2d']}_{full_name}"
                    hook_obj = EuclideanLayerHook(logger, layer_name, 'conv')
                    
                    pre_hook = child.register_forward_pre_hook(hook_obj.forward_pre_hook)
                    post_hook = child.register_forward_hook(hook_obj.forward_hook)
                    hooks.extend([pre_hook, post_hook])
                    
                    print(f"Registered hooks for {layer_name}")
                
                elif class_name == 'AdaptiveAvgPool2d':
                    layer_count['AdaptiveAvgPool2d'] += 1
                    layer_name = f"AdaptiveAvgPool2d_{layer_count['AdaptiveAvgPool2d']}_{full_name}"
                    hook_obj = EuclideanLayerHook(logger, layer_name, 'pool')
                    
                    pre_hook = child.register_forward_pre_hook(hook_obj.forward_pre_hook)
                    post_hook = child.register_forward_hook(hook_obj.forward_hook)
                    hooks.extend([pre_hook, post_hook])
                    
                    print(f"Registered hooks for {layer_name}")

            register_hooks_recursive(child, full_name)
    
    register_hooks_recursive(model)
    
    print(f"\nHook Registration Summary:")
    print(f"Total hooks registered: {len(hooks)}")
    for layer_type, count in layer_count.items():
        print(f"{layer_type} layers: {count}")
    
    return hooks


def register_euclidean_hooks(model, logger):
    """Register hooks specifically for Euclidean layers only"""
    hooks = []
    layer_count = defaultdict(int)
    
    def register_hooks_recursive(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if hasattr(child, '__class__'):
                class_name = child.__class__.__name__
                
                if class_name == 'Conv2d':
                    layer_count['Conv2d'] += 1
                    layer_name = f"Conv2d_{layer_count['Conv2d']}_{full_name}"
                    hook_obj = EuclideanLayerHook(logger, layer_name, 'conv')
                    
                    pre_hook = child.register_forward_pre_hook(hook_obj.forward_pre_hook)
                    post_hook = child.register_forward_hook(hook_obj.forward_hook)
                    hooks.extend([pre_hook, post_hook])
                    
                    print(f"Registered hooks for {layer_name}")
                
                elif class_name == 'AdaptiveAvgPool2d':
                    layer_count['AdaptiveAvgPool2d'] += 1
                    layer_name = f"AdaptiveAvgPool2d_{layer_count['AdaptiveAvgPool2d']}_{full_name}"
                    hook_obj = EuclideanLayerHook(logger, layer_name, 'pool')
                    
                    pre_hook = child.register_forward_pre_hook(hook_obj.forward_pre_hook)
                    post_hook = child.register_forward_hook(hook_obj.forward_hook)
                    hooks.extend([pre_hook, post_hook])
                    
                    print(f"Registered hooks for {layer_name}")
            
            register_hooks_recursive(child, full_name)
    
    register_hooks_recursive(model)
    
    print(f"\nEuclidean Hook Registration Summary:")
    print(f"Total hooks registered: {len(hooks)}")
    for layer_type, count in layer_count.items():
        print(f"{layer_type} layers: {count}")
    
    return hooks
