import torch
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import torch.nn as nn
from sklearn.metrics import auc
from pytorch_grad_cam.utils.image import show_cam_on_image

# Change working directory to parent directory
working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)

from utils.initialize import select_dataset, select_model
from gradcam_evaluator import GradCAMEvaluatorBase
from hierarcheal_gradcam import HierarchicalGradCAMBase


class EuclideanInterpreterBase:
    """
    Multi-layer interpretability for standard Euclidean CNN models
    Captures hierarchical representations across shallow to deep layers
    """

    def __init__(self, model_path, device='cuda:0'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.args = None
        self.dataset_info = None
        self.manifold_layers = {}
        self.hierarchical_layers = {}
        self._load_model()
        self.evaluator = GradCAMEvaluatorBase(self.model, device)
        self._analyze_hierarchical_architecture()

    def _load_model(self):
        print(f"Loading Euclidean model from {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.args = checkpoint['args']
        print(f"Model: {self.args.dataset}")
        _, _, _, img_dim, num_classes = select_dataset(self.args)
        self.dataset_info = {
            'img_dim': img_dim,
            'num_classes': num_classes,
            'dataset_name': self.args.dataset
        }
        
        self.model = select_model(img_dim, num_classes, self.args)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Euclidean model loaded: {num_classes} classes, input shape: {img_dim}")

    def _analyze_hierarchical_architecture(self):
            """Analyze the hierarchical Lorentz architecture for compositional understanding"""
            print("\nAnalyzing Hierarchical Lorentz Architecture...")
            
            # Categorize layers by depth and type for hierarchical analysis
            shallow_layers = []
            mid_layers = []
            deep_layers = []
            classifier_layers = []
            sparsity_layers = []  # NEW: Track sparsity layers
            
            layer_count = 0
            
            # Check if model has sparsity configuration
            self.has_sparsity = False
            self.sparsity_config = {
                'enabled': False,
                'type': None,
                'weight': 0,
                'k_ratio': None,
                'apply_to': None
            }
            
            # Extract sparsity configuration from model
            if hasattr(self.model, 'enable_sparsity'):
                self.has_sparsity = self.model.enable_sparsity
                
                if self.has_sparsity and hasattr(self.model, 'sparsity_type'):
                    self.sparsity_config = {
                        'enabled': True,
                        'type': getattr(self.model, 'sparsity_type', 'l1'),
                        'weight': getattr(self.model, 'sparsity_weight', 0.01),
                        'k_ratio': getattr(self.model, 'k_ratio', 0.1),
                        'apply_to': getattr(self.model, 'apply_k_sparse_to', 'final_layer')
                    }
                    print(f"\nDetected sparsity configuration:")
                    print(f"- Type: {self.sparsity_config['type']}")
                    print(f"- Weight: {self.sparsity_config['weight']}")
                    if self.sparsity_config['type'] == 'k_sparse':
                        print(f"- K-ratio: {self.sparsity_config['k_ratio']}")
                        print(f"- Apply to: {self.sparsity_config['apply_to']}")
            
            # Check encoder if model has encoder-decoder structure
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'enable_sparsity'):
                self.has_sparsity = self.model.encoder.enable_sparsity
                
                if self.has_sparsity:
                    self.sparsity_config = {
                        'enabled': True,
                        'type': getattr(self.model.encoder, 'sparsity_type', 'l1'),
                        'weight': getattr(self.model.encoder, 'sparsity_weight', 0.01),
                        'k_ratio': getattr(self.model.encoder, 'k_ratio', 0.1),
                        'apply_to': getattr(self.model.encoder, 'apply_k_sparse_to', 'final_layer')
                    }
                    print(f"\nDetected encoder sparsity configuration:")
                    print(f"- Type: {self.sparsity_config['type']}")
                    print(f"- Weight: {self.sparsity_config['weight']}")
                    if self.sparsity_config['type'] == 'k_sparse':
                        print(f"- K-ratio: {self.sparsity_config['k_ratio']}")
                        print(f"- Apply to: {self.sparsity_config['apply_to']}")
            
            # Original layer categorization logic
            for name, module in self.model.named_modules():
                module_type = type(module).__name__
                
                # Identify sparsity layers
                if 'Sparsity' in module_type:
                    self.manifold_layers[name] = {
                        'module': module,
                        'type': module_type,
                        'hierarchy_level': 'sparsity'
                    }
                    sparsity_layers.append((name, module))
                    print(f"Found sparsity layer: {name} ({module_type})")
                    

                elif 'Euclidean' in module_type:
                    self.manifold_layers[name] = {
                        'module': module,
                        'type': module_type,
                        'depth': layer_count,
                        'hierarchy_level': 'unknown'
                    }
                    
                    # Store manifold reference

                        
                        # Categorize by depth for hierarchical analysis
                    if layer_count < 23:  # Early layers - basic features
                        self.manifold_layers[name]['hierarchy_level'] = 'shallow'
                        shallow_layers.append((name, module))
                    elif layer_count < 68:  # Mid layers - part compositions
                        self.manifold_layers[name]['hierarchy_level'] = 'mid'
                        mid_layers.append((name, module))
                    else:  # Deep layers - object compositions
                        self.manifold_layers[name]['hierarchy_level'] = 'deep'
                        deep_layers.append((name, module))
                    
                    print(f"Found {self.manifold_layers[name]['hierarchy_level']} Lorentz layer: {name} ({module_type}), depth={layer_count}")
                   
                    
                    layer_count += 1
                
                # Special handling for LorentzMLR classifier
                elif isinstance(module, nn.Linear):
                    self.manifold_layers[name] = {
                        'module': module,
                        'type': 'Linear',
                        'depth': layer_count,
                        'hierarchy_level': 'classifier'
                    }
                    classifier_layers.append((name, module))
                    print(f"Found Linear classifier: {name}")
            
            # Store hierarchical organization
            self.hierarchical_layers = {
                'shallow': shallow_layers,
                'mid': mid_layers, 
                'deep': deep_layers,
                'classifier': classifier_layers,
                'sparsity': sparsity_layers  # Add sparsity layers
            }
            
            print(f"\nHierarchical Architecture Summary:")
            print(f"- Shallow layers (basic features): {len(shallow_layers)}")
            print(f"- Mid layers (part compositions): {len(mid_layers)}")
            print(f"- Deep layers (object compositions): {len(deep_layers)}")
            print(f"- Classifier layers: {len(classifier_layers)}")
            print(f"- Sparsity layers: {len(sparsity_layers)}")
            
            if self.has_sparsity:
                print(f"\nSparsity is enabled: {self.sparsity_config['type']} (weight={self.sparsity_config['weight']})")

    def get_target_layers(self):
            """Get exactly the 18 specified ResNet layers"""
            print("\nSelecting exactly 18 specified ResNet layers:")
            
            # Define the exact 18 layer names to extract
            specified_layers = [
                'encoder.conv1.0',
                'encoder.conv2_x.0.conv.0', 
                'encoder.conv2_x.0.conv.3', 
                'encoder.conv2_x.1.conv.0', 
                'encoder.conv2_x.1.conv.3', 
                'encoder.conv3_x.0.conv.0', 
                'encoder.conv3_x.0.conv.3', 
                'encoder.conv3_x.1.conv.0', 
                'encoder.conv3_x.1.conv.3', 
                'encoder.conv4_x.0.conv.0', 
                'encoder.conv4_x.0.conv.3', 
                'encoder.conv4_x.1.conv.0', 
                'encoder.conv4_x.1.conv.3', 
                'encoder.conv5_x.0.conv.0',
                'encoder.conv5_x.0.conv.3',
                'encoder.conv5_x.1.conv.0',
                'encoder.conv5_x.1.conv.3',
                'decoder'
            ]
            
            # Map layers to their hierarchy levels
            hierarchy_mapping = {
                'encoder.conv1': 'shallow',
                'encoder.conv2_x': 'shallow',
                'encoder.conv3_x': 'mid',
                'encoder.conv4_x': 'mid',
                'encoder.conv5_x': 'deep',
                'decoder': 'deep'
            }
            
            target_layers = []
            found_layers = set()
            not_found_layers = []
            
            # Look for each specified layer in the model
            for specified_layer in specified_layers:
                found = False
                
                # Determine hierarchy level for this layer
                hierarchy_level = None
                for prefix, level in hierarchy_mapping.items():
                    if specified_layer.startswith(prefix):
                        hierarchy_level = level
                        break
                
                if hierarchy_level is None:
                    hierarchy_level = 'deep'  # Default to deep for unrecognized patterns
                
                # Search through all modules
                for name, module in self.model.named_modules():
                    if name == specified_layer:

                        target_layers.append({
                            'name': name,
                            'module': module,
                            'hierarchy_level': hierarchy_level,
                       
                        })
                        found_layers.add(name)
                        found = True
                        print(f"  ✓ Found layer: {name} ({hierarchy_level}")
                        break
                
                if not found:
                    not_found_layers.append(specified_layer)
            
            # Report on missing layers
            if not_found_layers:
                print(f"\nWarning: Could not find {len(not_found_layers)} specified layers:")
                for layer in not_found_layers:
                    print(f"  ✗ Missing: {layer}")
                
                # Try to find alternative layers to maintain the count of 18
                available_layers = []
                for level in ['shallow', 'mid', 'deep']:
                    for name, module in self.hierarchical_layers[level]:
                        if name not in found_layers:
                            available_layers.append((name, module, level))
                
                if available_layers and len(target_layers) < 18:
                    print("\nAdding alternative layers to maintain 18 total:")
                    needed = 18 - len(target_layers)
                    
                    # Try to match the distribution of missing layers across hierarchy levels
                    missing_by_level = {'shallow': 0, 'mid': 0, 'deep': 0}
                    for layer in not_found_layers:
                        for prefix, level in hierarchy_mapping.items():
                            if layer.startswith(prefix):
                                missing_by_level[level] += 1
                                break
                    
                    added_by_level = {'shallow': 0, 'mid': 0, 'deep': 0}
                    
                    # Add alternatives with preference to matching hierarchy levels
                    for i in range(min(needed, len(available_layers))):
                        # Determine which level needs more layers
                        target_level = max(missing_by_level.items(), key=lambda x: x[1] - added_by_level.get(x[0], 0))[0]
                        
                        # Find an alternative in that level
                        alt_found = False
                        for name, module, level in available_layers:
                            if level == target_level and name not in found_layers:
                                target_layers.append({
                                    'name': name,
                                    'module': module,
                                    'hierarchy_level': level,
                                
                                })
                                found_layers.add(name)
                                added_by_level[level] = added_by_level.get(level, 0) + 1
                                print(f"  + Added alternative {level} layer: {name} ")
                                available_layers.remove((name, module, level))
                                alt_found = True
                                break
                        
                        # If no alternative in target level, add any available layer
                        if not alt_found and available_layers:
                            name, module, level = available_layers[0]
                            target_layers.append({
                                'name': name,
                                'module': module,
                                'hierarchy_level': level,
                            
                            })
                            found_layers.add(name)
                            added_by_level[level] = added_by_level.get(level, 0) + 1
                            print(f"  + Added alternative {level} layer: {name}")
                            available_layers.remove((name, module, level))
            
            # Final layer count summary
            layer_counts = {'shallow': 0, 'mid': 0, 'deep': 0}
            for layer in target_layers:
                layer_counts[layer['hierarchy_level']] += 1
            
            print(f"\nSelected {len(target_layers)} layers for visualization:")
            print(f"- Shallow layers: {layer_counts['shallow']}")
            print(f"- Mid layers: {layer_counts['mid']}")
            print(f"- Deep layers: {layer_counts['deep']}")
            
            return target_layers
    def run_comprehensive_analysis_with_evaluation(self, test_loader, num_samples=5, 
                                                  output_dir='comprehensive_analysis', 
                                                  evaluate_gradcam=True):
        """Run comprehensive analysis including GradCAM evaluation metrics"""
        print("=" * 100)
        print("COMPREHENSIVE EUCLIDEAN CNN INTERPRETABILITY ANALYSIS WITH EVALUATION")
        print("=" * 100)
        
        base_output_dir = output_dir
        hierarchical_dir = os.path.join(base_output_dir, 'hierarchical')
        evaluation_dir = os.path.join(base_output_dir, 'evaluation')
        
        os.makedirs(hierarchical_dir, exist_ok=True)
        os.makedirs(evaluation_dir, exist_ok=True)
        
        print(f"Dataset: {self.args.dataset}")
        print(f"Total Conv Layers: {len([l for l in self.manifold_layers.values() if 'Conv' in l['type']])}")
        print(f"Analysis Samples: {num_samples}")
        
        # Run hierarchical analysis
        print("\n" + "="*50)
        print("PHASE 1: HIERARCHICAL COMPOSITION ANALYSIS")
        print("="*50)
        self.generate_hierarchical_analysis(test_loader, num_samples, hierarchical_dir)
        
        # Run GradCAM evaluation
        if evaluate_gradcam:
            print("\n" + "="*50)
            print("PHASE 2: GRADCAM EVALUATION METRICS")
            print("="*50)
            self._run_gradcam_evaluation(test_loader, evaluation_dir, num_eval_samples=min(20, num_samples))
        
        print("=" * 100)
        print(f"COMPREHENSIVE ANALYSIS WITH EVALUATION COMPLETE")
        print(f"Results saved to:")
        print(f"- Hierarchical: {hierarchical_dir}/")
        
        if evaluate_gradcam:
            print(f"- Evaluation: {evaluation_dir}/")
        print("=" * 100)

    def _run_gradcam_evaluation(self, test_loader, output_dir, num_eval_samples=20):
        """Run GradCAM evaluation on sample data"""
        target_layers = self.get_target_layers()
        
        if not target_layers:
            print("No suitable layers found for evaluation")
            return
        
        # Initialize GradCAM generator for evaluation
        class EvaluationGradCAM:
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
                for layer_info in self.target_layers:
                    module = layer_info['module']
                    name = layer_info['name']
                    level = layer_info['hierarchy_level']
                    
                    h1 = module.register_forward_hook(self._create_activation_hook(name, level))
                    h2 = module.register_full_backward_hook(self._create_gradient_hook(name, level))
                    self.hooks.extend([h1, h2])
            
            def remove_hooks(self):
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
                            
                            
                            # Apply the enhancement formula: ξsparse_spatial = ξspatial · (1 + 0.2(1 − ρ))
                            sparsity_enhancement = 1.0 + 0.2 * (1.0 - sparsity_ratio)
                            cam = (grad_norm * act_norm * mask) * sparsity_enhancement
                            
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
        # Run evaluation on samples
        gradcam = EvaluationGradCAM(self.model, target_layers, self.device)
        gradcam.register_hooks()
        
        sample_count = 0
        all_evaluations = []
        
        for batch_idx, (data, targets) in enumerate(test_loader):
            if sample_count >= num_eval_samples:
                break
            
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            for i in range(data.size(0)):
                if sample_count >= num_eval_samples:
                    break
                single_input = data[i:i+1]
                single_target = targets[i:i+1]
                target_class = single_target.item()
                
                print(f"\nEvaluating sample {sample_count + 1}/{num_eval_samples} (Class {target_class})")
                
                try:
                    # Run comprehensive evaluation
                    sample_eval_dir = os.path.join(output_dir, f'sample_{sample_count + 1:03d}')
                    evaluation_results = self.evaluator.generate_comprehensive_evaluation_report(
                        single_input, gradcam, target_class, output_dir=sample_eval_dir
                    )
                    all_evaluations.append(evaluation_results)
                except Exception as e:
                    print(f"Error evaluating sample {sample_count + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                
                sample_count += 1
        
        gradcam.remove_hooks()
        
        # Generate aggregate evaluation report
        if all_evaluations:
            self._generate_aggregate_evaluation(all_evaluations, output_dir)
        
        print(f"\nGradCAM evaluation complete! Results saved to {output_dir}/")

    def _generate_aggregate_evaluation(self, all_evaluations, output_dir):
        """Generate aggregate statistics across all evaluated samples"""
        aggregate_path = os.path.join(output_dir, 'aggregate_evaluation.txt')
        
        with open(aggregate_path, 'w') as f:
            f.write("AGGREGATE GRADCAM EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            # Collect all layer names
            all_layers = set()
            for eval_result in all_evaluations:
                for metric_type, metric_data in eval_result.items():
                    if isinstance(metric_data, dict):
                        all_layers.update(metric_data.keys())
            
            # Aggregate metrics by layer
            for layer_name in sorted(all_layers):
                f.write(f"LAYER: {layer_name}\n")
                f.write("-" * 40 + "\n")
                
                # Robustness
                robustness_scores = []
                for eval_result in all_evaluations:
                    if (layer_name in eval_result.get('robustness', {}) and 
                        'robustness_score' in eval_result['robustness'][layer_name]):
                        robustness_scores.append(eval_result['robustness'][layer_name]['robustness_score'])
                
                if robustness_scores:
                    f.write(f"Robustness: {np.mean(robustness_scores):.3f} ± {np.std(robustness_scores):.3f}\n")
                
                # Faithfulness
                faithfulness_scores = []
                for eval_result in all_evaluations:
                    if (layer_name in eval_result.get('faithfulness', {}) and 
                        'faithfulness_score' in eval_result['faithfulness'][layer_name]):
                        faithfulness_scores.append(eval_result['faithfulness'][layer_name]['faithfulness_score'])
                
                if faithfulness_scores:
                    f.write(f"Faithfulness: {np.mean(faithfulness_scores):.3f} ± {np.std(faithfulness_scores):.3f}\n")
                
                # Localisation
                localisation_scores = []
                for eval_result in all_evaluations:
                    if (layer_name in eval_result.get('localisation', {}) and 
                        'localisation_score' in eval_result['localisation'][layer_name]):
                        localisation_scores.append(eval_result['localisation'][layer_name]['localisation_score'])
                
                if localisation_scores:
                    f.write(f"Localisation: {np.mean(localisation_scores):.3f} ± {np.std(localisation_scores):.3f}\n")
                
                # Complexity
                complexity_scores = []
                interpretability_scores = []
                for eval_result in all_evaluations:
                    if layer_name in eval_result.get('complexity', {}):
                        complexity_data = eval_result['complexity'][layer_name]
                        if 'complexity_score' in complexity_data:
                            complexity_scores.append(complexity_data['complexity_score'])
                        if 'interpretability_score' in complexity_data:
                            interpretability_scores.append(complexity_data['interpretability_score'])
                
                if complexity_scores:
                    f.write(f"Complexity: {np.mean(complexity_scores):.3f} ± {np.std(complexity_scores):.3f}\n")
                if interpretability_scores:
                    f.write(f"Interpretability: {np.mean(interpretability_scores):.3f} ± {np.std(interpretability_scores):.3f}\n")
                
                f.write("\n")

    def generate_hierarchical_analysis(self, test_loader, num_samples=100, output_dir='hierarchical_analysis'):
        """Generate hierarchical compositional analysis across multiple CNN layers"""
        print("Generating Hierarchical Compositional Analysis...")
        
        os.makedirs(output_dir, exist_ok=True)
        target_layers = self.get_target_layers()
        
        if not target_layers:
            print("No suitable hierarchical layers found for analysis")
            return
        
        # Initialize hierarchical GradCAM
        gradcam = HierarchicalGradCAMBase(self.model, target_layers, self.device)
        gradcam.register_hooks()
        
        # Process samples
        samples_processed = 0
        saved_visualizations = 0
        processed_classes = set() 
        
        for batch_idx, (data, targets) in enumerate(test_loader):
            if samples_processed >= num_samples:
                break
            
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            try:
                for i in range(data.size(0)):
                    single_input = data[i:i+1]
                    single_target = targets[i:i+1]
                    target_class = single_target.item()
                    
                    # Skip if we've already processed this class (for diversity)
                    if len(processed_classes) < 10 and target_class in processed_classes:
                        continue
                    
                    print(f"\nProcessing hierarchical sample {samples_processed + 1} (Class {target_class})")
                    
                    # Generate hierarchical CAMs
                    cams, output = gradcam.generate_cams(single_input, target_class)
                    
                    if cams:
                        # Create comprehensive hierarchical visualization
                        self._create_hierarchical_visualization(
                            single_input, cams, output, single_target, 
                            samples_processed + 1, output_dir
                        )
                        saved_visualizations += 1
                        
                        # Track that we've processed this class
                        processed_classes.add(target_class)
                    
                    samples_processed += 1
                    
                    if samples_processed >= num_samples:
                        break
                
                if samples_processed % 5 == 0:
                    print(f"Hierarchical progress: {samples_processed}/{num_samples}")
                    
            except Exception as e:
                print(f"Error in hierarchical batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        gradcam.remove_hooks()
        print(f"\nHierarchical analysis complete! Saved {saved_visualizations} visualizations from {len(processed_classes)} different classes to {output_dir}/")

    def _create_hierarchical_visualization(self, input_tensor, cams, output, target, sample_num, output_dir):
        """Create hierarchical visualization showing progression from shallow to deep"""
        # Process input image
        img_tensor = input_tensor.squeeze(0).detach()
        
        if img_tensor.shape[0] <= 4:  # Channel-first
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = img_tensor.cpu().numpy()
        
        # Normalize image
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Ensure RGB format
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np, img_np, img_np], axis=2)
        elif img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        elif img_np.shape[2] > 3:
            img_np = img_np[:, :, :3]
        
        # Get predictions
        predicted_class = output.argmax(dim=1)[0].item()
        true_class = target.item()
        class_names = self._get_class_names()
        confidence = torch.softmax(output, dim=1)[0, predicted_class].item()
        
        # Organize CAMs by hierarchy level
        hierarchy_cams = {'shallow': [], 'mid': [], 'deep': []}
        for layer_name, cam_data in cams.items():
            level = cam_data['hierarchy_level']
            if level in hierarchy_cams:
                cam_np = cam_data['cam'][0].detach().cpu().numpy()
                if cam_np.shape != img_np.shape[:2]:
                    cam_np = cv2.resize(cam_np, (img_np.shape[1], img_np.shape[0]))
                hierarchy_cams[level].append({
                    'name': layer_name,
                    'cam': cam_np,
                })
        
        # Calculate number of layers to show
        total_layers = sum(len(layers) for layers in hierarchy_cams.values())
        layers_per_row = 6
        num_layer_rows = (total_layers + layers_per_row - 1) // layers_per_row
        total_rows = num_layer_rows + 1  # +1 for header row
        
        fig = plt.figure(figsize=(20, 4 * total_rows))
        
        # Create grid layout
        gs = fig.add_gridspec(total_rows, layers_per_row, height_ratios=[1] + [1] * num_layer_rows)
        
        # Original image in header row
        ax_orig = fig.add_subplot(gs[0, 0:2])
        ax_orig.imshow(img_np)
        ax_orig.set_title(f'Original Image\nTrue: {class_names[true_class]}\nPred: {class_names[predicted_class]} ({confidence:.3f})', 
                        fontsize=12, fontweight='bold')
        ax_orig.axis('off')
        
        # Model info in header row
        ax_info = fig.add_subplot(gs[0, 2:6])
        model_info = f"""Hierarchical CNN Model Analysis
        
Dataset: {self.args.dataset}

Layer Distribution:
• Shallow: {len(hierarchy_cams['shallow'])} layers - Basic features (edges, textures)
• Mid: {len(hierarchy_cams['mid'])} layers - Part composition (shapes, parts)  
• Deep: {len(hierarchy_cams['deep'])} layers - Object composition (semantics)

Total Layers: {total_layers}
"""
        
        ax_info.text(0.05, 0.95, model_info, transform=ax_info.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
        ax_info.set_title('CNN Layer Hierarchy', fontsize=12, fontweight='bold')
        ax_info.axis('off')
        
        # Plot all layers by hierarchy level
        current_row = 1
        current_col = 0
        
        # Function to add a layer to the visualization
        def add_layer_to_grid(level, layer_data):
            nonlocal current_row, current_col
            
            # Create new row if needed
            if current_col >= layers_per_row:
                current_row += 1
                current_col = 0
            
            # Add layer visualization
            ax = fig.add_subplot(gs[current_row, current_col])
            
            # Create overlay
            overlay = show_cam_on_image(img_np, layer_data['cam'], use_rgb=True)
            ax.imshow(overlay)
            
            # Extract layer info
            display_name = layer_data['name'].split('.')[-1] if '.' in layer_data['name'] else layer_data['name']
            
            # Format title with hierarchy level
            ax.set_title(f'{level.upper()}: {display_name}', fontsize=10)
            ax.axis('off')
            
            current_col += 1
        
        # Add layers in order: shallow, mid, deep
        for level in ['shallow', 'mid', 'deep']:
            for layer_data in hierarchy_cams[level]:
                add_layer_to_grid(level, layer_data)
        
        plt.suptitle(f'Hierarchical CNN Layer Visualization (Sample {sample_num} - Class {true_class})', 
                    fontsize=16, fontweight='bold')
        
        # Save visualization
        save_path = os.path.join(output_dir, f'hierarchical_cnn_{sample_num:03d}_class{true_class}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Saved hierarchical visualization: {save_path}")

    def _get_class_names(self):
        """Get class names for the dataset"""
        if self.args.dataset.lower() == 'cifar10':
            return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.args.dataset.lower() == 'cifar100':
            return [f'class_{i}' for i in range(100)]
        else:
            return [f'class_{i}' for i in range(self.dataset_info['num_classes'])]