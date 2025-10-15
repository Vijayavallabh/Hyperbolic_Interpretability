import torch
class HierarchicalGradCAM:
    def __init__(self, model, target_layers, lorentz_manifolds, device,filters_per_layer = 8):
        self.model = model
        self.target_layers = target_layers
        self.lorentz_manifolds = lorentz_manifolds
        self.device = device
        self.layer_activations = {}
        self.layer_gradients = {}
        self.filters_per_layer = filters_per_layer
        self.hooks = []
        self._cache_model_sparsity_config()
        
    def _cache_model_sparsity_config(self):
        """Cache model sparsity configuration for efficient lookup"""
        self.model_sparsity_config = {
            'sparsity_enabled': False,
            'k_ratio': 0.1,
            'apply_k_sparse_to': 'none',
            'sparsity_type': None
        }
        
        try:
            # Check direct model attributes (works for all model types)
            if hasattr(self.model, 'enable_sparsity'):
                self.model_sparsity_config['sparsity_enabled'] = self.model.enable_sparsity
                
            if hasattr(self.model, 'k_ratio'):
                self.model_sparsity_config['k_ratio'] = self.model.k_ratio
                
            if hasattr(self.model, 'apply_k_sparse_to'):
                self.model_sparsity_config['apply_k_sparse_to'] = self.model.apply_k_sparse_to
                
            if hasattr(self.model, 'sparsity_type'):
                self.model_sparsity_config['sparsity_type'] = self.model.sparsity_type
                
            # Check encoder settings (for ResNetClassifier wrapper)
            if hasattr(self.model, 'encoder'):
                encoder = self.model.encoder
                if hasattr(encoder, 'enable_sparsity'):
                    self.model_sparsity_config['sparsity_enabled'] = encoder.enable_sparsity
                if hasattr(encoder, 'k_ratio'):
                    self.model_sparsity_config['k_ratio'] = encoder.k_ratio
                if hasattr(encoder, 'apply_k_sparse_to'):
                    self.model_sparsity_config['apply_k_sparse_to'] = encoder.apply_k_sparse_to
                if hasattr(encoder, 'sparsity_type'):
                    self.model_sparsity_config['sparsity_type'] = encoder.sparsity_type
                    
            # Detect model type for better understanding
            self.model_type = self._detect_model_architecture()
            
            print(f"Detected model type: {self.model_type}")
            print(f"Cached model sparsity config: {self.model_sparsity_config}")
            
        except Exception as e:
            print(f"Warning: Error caching model sparsity config: {e}")

    def _detect_model_architecture(self):
        """Detect the model architecture type"""
        try:
            # Check if it's a classifier wrapper
            if hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
                # It's a ResNetClassifier wrapper, check the encoder type
                encoder = self.model.encoder
                if hasattr(encoder, 'manifold') and encoder.manifold is not None:
                    # Check if it's hybrid (has both Euclidean and Lorentz components)
                    if isinstance(encoder, type(self.model.encoder)) and 'Hybrid' in str(type(encoder)):
                        return 'hybrid'
                    else:
                        return 'lorentz'
                else:
                    return 'euclidean'
            else:
                # Direct model, check its attributes
                if hasattr(self.model, 'manifold'):
                    if self.model.manifold is not None:
                        # Check if it's hybrid
                        if 'Hybrid' in str(type(self.model)):
                            return 'hybrid'
                        else:
                            return 'lorentz'
                    else:
                        return 'euclidean'
                else:
                    return 'euclidean'
                    
        except Exception as e:
            print(f"Warning: Could not detect model type: {e}")
            return 'unknown'

    def _determine_sparsity_from_model(self, layer_name):
        """Determine if layer should use sparse processing based on model configuration"""
        try:
            # First check cached config
            if self.model_sparsity_config['sparsity_enabled']:
                apply_to = self.model_sparsity_config['apply_k_sparse_to']
                
                if apply_to == 'all_conv_layers':
                    return True
                elif apply_to == 'final_layer':
                    # Check if this is the final layer
                    return 'conv5' in layer_name or 'decoder' in layer_name
                elif apply_to == 'lorentz_layers':
                    # For hybrid models, only Lorentz layers get sparsity
                    if self.model_type == 'hybrid':
                        return self._detect_manifold_type(layer_name) == 'lorentz'
                    # For pure Lorentz models, all layers are Lorentz
                    elif self.model_type == 'lorentz':
                        return True
                    # For Euclidean models, no Lorentz layers
                    else:
                        return False
                
                return True
                
            return False
            
        except Exception as e:
            print(f"Warning: Could not determine sparsity from model config for {layer_name}: {e}")
            return False

    def _get_expected_sparsity_ratio(self, layer_name):
        """Get expected sparsity ratio from model configuration"""
        try:
            k_ratio = self.model_sparsity_config['k_ratio']

            expected_sparsity = 1.0 - k_ratio
  
            return expected_sparsity
            
        except Exception as e:
            print(f"Warning: Could not get expected sparsity ratio for {layer_name}: {e}")
            return 0.3  # Conservative default

    def _detect_tensor_format(self, layer_name, tensor_shape=None):
        """Detect if layer uses channel-first or channel-last format"""
        # For hybrid models, check layer type
        if self.model_type == 'hybrid':
            if any(pattern in layer_name for pattern in ['conv2_x', 'conv4_x', 'decoder']):
                return 'channel_last'
            else:
                return 'channel_first'
        elif self.model_type == 'euclidean':
            return 'channel_first'
        else:  # lorentz
            return 'channel_last'

    def _detect_manifold_type(self, layer_name):
        """Detect manifold type for the layer"""
        if self.model_type == 'hybrid':
            # Hybrid: conv2_x, conv4_x, decoder are lorentz
            if any(pattern in layer_name for pattern in ['conv2_x', 'conv4_x', 'decoder']):
                return 'lorentz'
            else:
                return 'euclidean'
        elif self.model_type == 'euclidean':
            return 'euclidean'
        else:  # lorentz
            return 'lorentz'
         
    def _create_activation_hook(self, layer_name, hierarchy_level):
        """Create activation hook for specific layer"""
        def hook(module, input, output):
            activation = output.detach()
            
            self.layer_activations[layer_name] = {
                'activation': activation,
                'hierarchy_level': hierarchy_level,
                'shape': activation.shape
            }
        return hook
        
    def _create_gradient_hook(self, layer_name, hierarchy_level):
        """Create gradient hook for specific layer"""
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                grad = grad_output[0].detach()
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
  
    def generate_hierarchical_cams(self, input_tensor, target_class=None):
        """Generate CAMs for all hierarchical layers"""
        self.layer_activations = {}
        self.layer_gradients = {}

        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)
        
        hierarchical_cams = {}
        
        for layer_info in self.target_layers:
            name = layer_info['name']
            level = layer_info['hierarchy_level']
            k_val = layer_info['k_value']

            if name in self.layer_activations and name in self.layer_gradients:
                activation_data = self.layer_activations[name]
                gradient_data = self.layer_gradients[name]

                cam = self._compute_hierarchical_cam(
                    gradient_data['gradient'], 
                    activation_data['activation'],
                    level, k_val, name
                )

                if cam is not None:
                    hierarchical_cams[name] = {
                        'cam': cam,
                        'hierarchy_level': level,
                        'k_value': k_val,
                        'shape': activation_data['shape'],
                        'activation': activation_data['activation']  # <-- Add this line
                    }
                        
        return hierarchical_cams, output
    
    def _get_hierarchy_params(self, hierarchy_level, manifold_type):
        """Get hierarchy-specific parameters based on manifold type"""
        if manifold_type == 'euclidean':
            params = {
                'shallow': (1, 1.0),
                'mid': (1, 1.0), 
                'deep': (1, 1.0)
            }
        else:  # lorentz
            params = {
                'shallow': (0.05, 1.0),
                'mid': (0.1, 1.0),
                'deep': (0.15, 0.9)
            }
        
        return params.get(hierarchy_level, (0.1, 1.0))
        
    def _compute_hierarchical_cam(self, gradients, activations, hierarchy_level, k_value, layer_name):
        """Compute CAM adapted for hierarchy level and hyperbolic geometry"""
        if len(gradients.shape) != 4 or len(activations.shape) != 4:
            return None
        
        actual_tensor_format = self._detect_tensor_format(layer_name)
        actual_manifold_type = self._detect_manifold_type(layer_name)
        
        if actual_tensor_format == 'channel_first':
            gradients = gradients.permute(0, 2, 3, 1)
            activations = activations.permute(0, 2, 3, 1)
            
        batch_size, height, width, channels = gradients.shape
        
        alpha, beta = self._get_hierarchy_params(hierarchy_level, actual_manifold_type)
        
        # MODIFIED: Determine sparsity based on model configuration (works for all model types)
        is_sparse = self._determine_sparsity_from_model(layer_name)
        sparsity_ratio = self._get_expected_sparsity_ratio(layer_name)
        
        # Fallback: also check actual sparsity if model config indicates no sparsity
        if not is_sparse:
            if actual_manifold_type == 'lorentz':  
                space_acts = activations[:, :, :, 1:]
                sparsity_ratio = (torch.abs(space_acts) < 1e-6).float().mean().item()
            else:
                sparsity_ratio = (torch.abs(activations) < 1e-6).float().mean().item()

        if is_sparse:
            
            print(f"Using sparse processing for {layer_name} ({self.model_type} model, {actual_manifold_type} manifold): expected sparsity ratio {sparsity_ratio:.2f}")
        
        cam = None
        
        try:
            if actual_manifold_type == 'euclidean' or channels < 2:
              
                if is_sparse:
                    mask = (torch.abs(activations) > 1e-6).float()
                    importance = torch.abs(gradients) * mask
                    
                    # Enhanced processing for sparse Euclidean features
                    sparsity_enhancement = 1.0 + (0.2 * sparsity_ratio)  # Moderate enhancement for Euclidean
                    importance = importance * sparsity_enhancement
                    
                    print(f"Applied Euclidean sparse enhancement: {sparsity_enhancement:.3f}")
                else:
                    importance = torch.abs(gradients)
                
                cam = torch.sum(importance * torch.abs(activations), dim=-1)
                
            elif actual_manifold_type == 'lorentz' and channels >= 2:
                time_grads = gradients[:, :, :, 0]
                space_grads = gradients[:, :, :, 1:]
                
                time_acts = activations[:, :, :, 0]
                space_acts = activations[:, :, :, 1:]
                
                # Compute time component importance
                time_importance = torch.abs(time_grads * time_acts)
                
                # Compute space component importance
                space_grad_norm = torch.sqrt(torch.sum(space_grads**2, dim=-1) + 1e-8)
                space_act_norm = torch.sqrt(torch.sum(space_acts**2, dim=-1) + 1e-8)
                
                if is_sparse:

                    space_mask = (torch.sum(torch.abs(space_acts), dim=-1) > 1e-6).float()
                    
                    # Compute base spatial importance
                    base_space_importance = space_grad_norm * space_act_norm * space_mask
                    
                    # Adjust enhancement based on expected sparsity
                    sparsity_enhancement = 1.0 + (0.2 * sparsity_ratio)  # Scale by expected sparsity
                    space_importance = base_space_importance * sparsity_enhancement
                    
                    print(f"Applied Lorentz sparse enhancement: ρ={sparsity_ratio:.3f}, enhancement={sparsity_enhancement:.3f}")
                    
                else:
                    space_importance = space_grad_norm * space_act_norm
            
                cam = alpha * time_importance + beta * space_importance
        
            # Apply ReLU to focus on positive contributions
            if cam is not None:
                cam = torch.relu(cam)
                
                # Normalize CAM for each sample in batch
                for b in range(batch_size):
                    if cam[b].max() > cam[b].min():
                        cam_b = cam[b]
                        
                        if is_sparse:
                            # Robust normalization considering expected sparsity
                            nonzero_values = cam_b[cam_b > 1e-6]
                            if len(nonzero_values) > 0:
                                # Use percentile-based normalization for sparse features
                                cam_min = torch.quantile(nonzero_values, 0.1)
                                cam_max = torch.quantile(nonzero_values, 0.9)
                                cam[b] = torch.clamp((cam_b - cam_min) / (cam_max - cam_min + 1e-8), 0, 1)
                            else:
                                cam[b] = torch.zeros_like(cam_b)
                        else:
                            # Standard normalization for non-sparse features
                            if hierarchy_level == 'shallow':
                                cam[b] = (cam_b - cam_b.min()) / (cam_b.max() - cam_b.min() + 1e-8)
                            else:
                                try:
                                    cam_min = torch.quantile(cam_b.flatten(), 0.05)
                                    cam_max = torch.quantile(cam_b.flatten(), 0.95)
                                    cam[b] = torch.clamp((cam_b - cam_min) / (cam_max - cam_min + 1e-8), 0, 1)
                                except:
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
        
    def generate_filter_cams(self, input_tensor, target_class=None):
        """Generate CAMs for individual filters AND whole layer CAMs across hierarchy levels"""
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
        
        # Generate both filter-level and whole layer CAMs
        filter_cams = {}
        
        for layer_info in self.target_layers:
            name = layer_info['name']
            level = layer_info['hierarchy_level']
            k_val = layer_info['k_value']
            
            if name in self.layer_activations and name in self.layer_gradients:
                activation_data = self.layer_activations[name]
                gradient_data = self.layer_gradients[name]
                
                # Generate CAMs for individual filters
                layer_filter_cams = self._compute_filter_level_cams(
                    gradient_data['gradient'],
                    activation_data['activation'],
                    level, name
                )
                
                # Generate whole layer CAM using existing hierarchical method
                whole_layer_cam = self._compute_hierarchical_cam(
                    gradient_data['gradient'],
                    activation_data['activation'],
                    level, k_val, name
                )
                
                if layer_filter_cams or whole_layer_cam is not None:
                    filter_cams[name] = {
                        'filter_cams': layer_filter_cams if layer_filter_cams else [],
                        'whole_layer_cam': whole_layer_cam,
                        'hierarchy_level': level,
                        'k_value': k_val,
                        'shape': activation_data['shape']
                    }
        
        return filter_cams, output

    def _compute_filter_level_cams(self, gradients, activations, hierarchy_level, layer_name):
        """Compute CAMs for individual filters to reveal part-whole relationships"""
        if len(gradients.shape) != 4 or len(activations.shape) != 4:
            return None
        
        # Detect tensor format and convert to consistent format
        actual_tensor_format = self._detect_tensor_format(layer_name)
        actual_manifold_type = self._detect_manifold_type(layer_name)
        
        if actual_tensor_format == 'channel_first':
            # Convert BCHW to BHWC for consistent processing
            gradients = gradients.permute(0, 2, 3, 1)
            activations = activations.permute(0, 2, 3, 1)
        
        batch_size, height, width, num_filters = gradients.shape
        
        # Set hierarchy-specific parameters
        if hierarchy_level == 'shallow':
            alpha = 0.05
            beta = 1.0
            top_k_ratio = 0.3  # Top 30% of filters
        elif hierarchy_level == 'mid':
            # Balance between parts and composition
            alpha = 0.1
            beta = 1.0
            top_k_ratio = 0.4  # Top 40% of filters
        elif hierarchy_level == 'deep':
            # Focus on object-level composition
            alpha = 0.15
            beta = 0.9
            top_k_ratio = 0.5  # Top 50% of filters
        
        filter_cams = []
        filter_importance_scores = []
        
        # Compute CAM for each filter individually
        for filter_idx in range(num_filters):
            try:
                # Extract filter-specific gradients and activations
                filter_grad = gradients[:, :, :, filter_idx]  # [batch, height, width]
                filter_act = activations[:, :, :, filter_idx]  # [batch, height, width]
                
                # Compute filter importance score
                importance = torch.abs(filter_grad * filter_act).mean().item()
                filter_importance_scores.append((filter_idx, importance))
                
                # Process based on manifold type
                if actual_manifold_type == 'lorentz' and num_filters >= 2:
                    # Handle Lorentz-structured filters
                    if filter_idx == 0:  # Time component
                        cam = alpha * torch.abs(filter_grad * filter_act)
                    else:  # Spatial components
                        cam = beta * torch.abs(filter_grad * filter_act)
                else:
                    # Standard Euclidean filter processing
                    cam = torch.abs(filter_grad * filter_act)
                
                # Apply ReLU and normalize
                cam = torch.relu(cam)
                
                # Normalize per batch
                for b in range(batch_size):
                    if cam[b].max() > cam[b].min():
                        cam[b] = (cam[b] - cam[b].min()) / (cam[b].max() - cam[b].min() + 1e-8)
                
                filter_cams.append({
                    'filter_id': filter_idx,
                    'cam': cam,
                    'importance': importance,
                    'hierarchy_level': hierarchy_level
                })
                
            except Exception as e:
                print(f"Error computing CAM for filter {filter_idx} in {layer_name}: {e}")
                continue
        
        # Sort filters by importance and select top-k for part-whole analysis
        filter_importance_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = max(1, int(len(filter_importance_scores) * top_k_ratio))
        top_k = min(top_k, self.filters_per_layer)  # Limit display
        
        # Return top-k most important filters for part-whole relationship analysis
        selected_filters = []
        for i, (filter_idx, importance) in enumerate(filter_importance_scores[:top_k]):
            # Find the corresponding CAM
            for filter_cam in filter_cams:
                if filter_cam['filter_id'] == filter_idx:
                    filter_cam['rank'] = i + 1
                    selected_filters.append(filter_cam)
                    break
        
        print(f"Selected {len(selected_filters)} top filters from {layer_name} ({hierarchy_level})")
        return selected_filters


