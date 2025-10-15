import torch
class HierarchicalGradCAMBase:
            def __init__(self, model, target_layers, device):
                self.model = model
                self.target_layers = target_layers
                self.device = device
                self.layer_activations = {}
                self.layer_gradients = {}
                self.hooks = []
                
            def _create_activation_hook(self, layer_name, hierarchy_level):
                """Create activation hook with validation"""
                def hook(module, input, output):
                    if output is not None:
                        activation = output.detach()
                        
                        # Validate activation
                        if torch.isnan(activation).any() or torch.isinf(activation).any():
                            print(f"Warning: Invalid activation values in {layer_name}")
                            return
                        
                        # Check for meaningful activation
                        act_std = torch.std(activation)
                        if act_std < 1e-8:
                            print(f"Warning: {layer_name} activation has no variation (std={act_std:.2e})")
                        
                        self.layer_activations[layer_name] = {
                            'activation': activation,
                            'hierarchy_level': hierarchy_level,
                            'shape': activation.shape
                        }
                return hook
            def _create_gradient_hook(self, layer_name, hierarchy_level):
                """Create gradient hook with validation"""
                def hook(module, grad_input, grad_output):
                    if grad_output[0] is not None:
                        grad = grad_output[0].detach()
                        
                        # Validate gradient
                        if torch.isnan(grad).any() or torch.isinf(grad).any():
                            print(f"Warning: Invalid gradient values in {layer_name}")
                            return
                        
                        # Check for meaningful gradient
                        grad_std = torch.std(grad)
                        if grad_std < 1e-8:
                            print(f"Warning: {layer_name} gradient has no variation (std={grad_std:.2e})")
                        
                        self.layer_gradients[layer_name] = {
                            'gradient': grad,
                            'hierarchy_level': hierarchy_level,
                            'shape': grad.shape
                        }
                return hook
            
            def register_hooks(self):
                """Register hooks for all target layers"""
                for layer_info in self.target_layers:
                    module = layer_info['module']
                    name = layer_info['name']
                    level = layer_info['hierarchy_level']
                    
                    h1 = module.register_forward_hook(self._create_activation_hook(name, level))
                    h2 = module.register_full_backward_hook(self._create_gradient_hook(name, level))
                    self.hooks.extend([h1, h2])
                    print(f"Registered hierarchical hooks for {level} layer: {name}")
            
            def remove_hooks(self):
                """Remove all hooks"""
                for hook in self.hooks:
                    hook.remove()
                self.hooks = []
            
            def generate_cams(self, input_tensor, target_class=None):
                """Generate CAMs for all hierarchical layers"""
                self.layer_activations = {}
                self.layer_gradients = {}
                
                # Forward pass
                output = self.model(input_tensor)
                
                if target_class is None:
                    target_class = output.argmax(dim=1).item()
                
                # Backward pass
                self.model.zero_grad()
                target_score = output[0, target_class]
                target_score.backward(retain_graph=True)
                
                # Generate CAMs for each hierarchy level
                cams = {}
                
                for layer_info in self.target_layers:
                    name = layer_info['name']
                    level = layer_info['hierarchy_level']
                    
                    if name in self.layer_activations and name in self.layer_gradients:
                        activation_data = self.layer_activations[name]
                        gradient_data = self.layer_gradients[name]
                        
                        cam = self._compute_cam(
                            gradient_data['gradient'], 
                            activation_data['activation'],
                             name
                        )
                        
                        if cam is not None:
                            cams[name] = {
                                'cam': cam,
                                'hierarchy_level': level,
                                'shape': activation_data['shape'],
                                'activation': activation_data['activation']
                            }
                                
                return cams, output
            
            def _compute_cam(self, gradients, activations, layer_name):
                if len(gradients.shape) != 4 or len(activations.shape) != 4:
                    return None
                
                batch_size, height, width, channels = gradients.shape

                is_sparse = False
                sparsity_ratio = 0.0

                if channels >= 2:
                    zeros = (torch.abs(activations) < 1e-6).float().mean().item()
                    is_sparse = zeros > 0.5  # If more than 50% of values are ~0
                    sparsity_ratio = zeros
                else:
                    zeros = (torch.abs(activations) < 1e-6).float().mean().item()
                    is_sparse = zeros > 0.5
                    sparsity_ratio = zeros
                
                if is_sparse:
                    print(f"Detected sparse activations in {layer_name}: {sparsity_ratio:.2f} sparsity ratio")
                
                # Initialize cam
                cam = None
                
                try:
                    # Process based on channel structure and sparsity
                    if channels < 2:
                        # Standard case for non-Lorentz layers
                        if is_sparse:
                            mask = (torch.abs(activations) > 1e-6).float()
                            importance = torch.abs(gradients) * mask
                        else:
                            importance = torch.abs(gradients)
                        
                        cam = torch.sum(importance * torch.abs(activations), dim=-1)
                    else:
                        
                        grad_norm = torch.sqrt(torch.sum(gradients**2, dim=-1) + 1e-8)
                        act_norm = torch.sqrt(torch.sum(activations**2, dim=-1) + 1e-8)
                        
                        if is_sparse:
                            # Apply sparsity mask for zero components
                            mask = (torch.sum(torch.abs(activations), dim=-1) > 1e-6).float()
                            
                            # Compute base spatial importance
                            base_space_importance = grad_norm * act_norm * mask
                            
                            # Apply the enhancement formula: ξsparse_spatial = ξspatial · (1 + 0.2(1 − ρ))
                            sparsity_enhancement = 1.0 + 0.2 * (1.0 - sparsity_ratio)
                            cam = base_space_importance * sparsity_enhancement
                            
                            print(f"Applied sparsity enhancement: ρ={sparsity_ratio:.3f}, enhancement={sparsity_enhancement:.3f}")
                            
                        else:
                            cam = grad_norm * act_norm


                    # Apply ReLU to focus on positive contributions
                    if cam is not None:
                        cam = torch.relu(cam)
                        
                        # Normalize CAM for each sample in batch
                        for b in range(batch_size):
                            if cam[b].max() > cam[b].min():
                                cam_b = cam[b]
                                
                                if is_sparse:
                                    # Robust normalization for sparse features
                                    nonzero_values = cam_b[cam_b > 1e-6]
                                    if len(nonzero_values) > 0:
                                        cam_min = torch.min(nonzero_values)
                                        cam_max = torch.max(nonzero_values)
                                        cam[b] = torch.clamp((cam_b - cam_min) / (cam_max - cam_min + 1e-8), 0, 1)
                                else:
                                    try:
                                        cam_min = torch.quantile(cam_b.flatten(), 0.05)
                                        cam_max = torch.quantile(cam_b.flatten(), 0.95)
                                        cam[b] = torch.clamp((cam_b - cam_min) / (cam_max - cam_min + 1e-8), 0, 1)
                                    except:
                                        # Fallback to standard min-max if quantile fails
                                        cam[b] = (cam_b - cam_b.min()) / (cam_b.max() - cam_b.min() + 1e-8)
                
                except Exception as e:
                    print(f"Error computing CAM for {layer_name}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Final safety check
                if cam is None:
                    print(f"Warning: CAM computation failed for {layer_name}. Using zero tensor.")
                    cam = torch.zeros((batch_size, height, width), device=gradients.device)
                
                return cam