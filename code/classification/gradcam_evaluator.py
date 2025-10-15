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

# Import required modules
from utils.initialize import select_dataset, select_model

class GradCAMEvaluatorBase:
    """
    Comprehensive evaluation metrics for GradCAM interpretability in Euclidean models
    Implements: Robustness, Faithfulness, Localisation, Complexity, Randomisation
    """
    
    def __init__(self, model, device='cuda:0'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_robustness(self, input_tensor, cam_generator, target_class=None, 
                        noise_levels=[0.01, 0.05, 0.1, 0.15, 0.2], num_trials=5):
        """
        Evaluate robustness by measuring CAM stability under input perturbations
        Higher stability (lower variance) indicates better robustness
        """
        print("Evaluating GradCAM Robustness...")
        
        # Generate baseline CAM
        baseline_cams, _ = cam_generator.generate_cams(input_tensor, target_class)
        
        robustness_scores = {}
        
        for layer_name, baseline_cam_data in baseline_cams.items():
            baseline_cam = baseline_cam_data['cam'][0].detach().cpu().numpy()
            similarities = []
            
            # Check if baseline CAM has valid variation
            baseline_std = np.std(baseline_cam.flatten())
            if baseline_std < 1e-8:
                print(f"Warning: Baseline CAM for {layer_name} has no variation (std={baseline_std:.2e})")
                robustness_scores[layer_name] = {
                    'mean_similarity': 0.0,
                    'std_similarity': 0.0,
                    'robustness_score': 0.0,
                    'num_valid_trials': 0,
                    'warning': 'baseline_cam_no_variation'
                }
                continue
            
            for noise_level in noise_levels:
                trial_similarities = []
                
                for trial in range(num_trials):
                    # Add Gaussian noise
                    noise = torch.randn_like(input_tensor) * noise_level
                    perturbed_input = input_tensor + noise
                    
                    # Generate CAM for perturbed input
                    try:
                        perturbed_cams, _ = cam_generator.generate_cams(perturbed_input, target_class)
                        
                        if layer_name in perturbed_cams:
                            perturbed_cam = perturbed_cams[layer_name]['cam'][0].detach().cpu().numpy()
                            
                            # Check if perturbed CAM has valid variation
                            perturbed_std = np.std(perturbed_cam.flatten())
                            if perturbed_std < 1e-8:
                                continue
                            
                            # Compute similarity (correlation coefficient) with safety checks
                            baseline_flat = baseline_cam.flatten()
                            perturbed_flat = perturbed_cam.flatten()
                            
                            # Check for sufficient variation in both arrays
                            if np.std(baseline_flat) > 1e-8 and np.std(perturbed_flat) > 1e-8:
                                # Use Pearson correlation with error handling
                                try:
                                    correlation_matrix = np.corrcoef(baseline_flat, perturbed_flat)
                                    correlation = correlation_matrix[0, 1]
                                    
                                    # Check for valid correlation value
                                    if not (np.isnan(correlation) or np.isinf(correlation)):
                                        trial_similarities.append(abs(correlation))
                                except Exception as corr_e:
                                    print(f"Correlation computation error for {layer_name}: {corr_e}")
                                    continue
                            
                    except Exception as e:
                        print(f"Error in robustness trial for {layer_name}: {e}")
                        continue
                
                if trial_similarities:
                    similarities.extend(trial_similarities)
            
            # Compute robustness metrics
            if similarities and len(similarities) >= 2:  # Need at least 2 values for meaningful stats
                mean_similarity = np.mean(similarities)
                std_similarity = np.std(similarities)
                robustness_score = mean_similarity - std_similarity  # Higher is better
                
                robustness_scores[layer_name] = {
                    'mean_similarity': float(mean_similarity),
                    'std_similarity': float(std_similarity),
                    'robustness_score': float(robustness_score),
                    'num_valid_trials': len(similarities)
                }
            else:
                print(f"Warning: Insufficient valid trials for {layer_name} ({len(similarities)} trials)")
                robustness_scores[layer_name] = {
                    'mean_similarity': 0.0,
                    'std_similarity': 0.0,
                    'robustness_score': 0.0,
                    'num_valid_trials': len(similarities),
                    'warning': 'insufficient_valid_trials'
                }
        
        return robustness_scores
    
    def evaluate_faithfulness(self, input_tensor, cam_generator, target_class=None, 
                            percentiles=[10, 20, 30, 40, 50]):
        """
        Evaluate faithfulness using deletion/insertion metrics
        Measures how much prediction changes when important/unimportant regions are removed
        """
        print("Evaluating GradCAM Faithfulness...")
        
        # Generate CAMs
        cams, original_output = cam_generator.generate_cams(input_tensor, target_class)
        
        if target_class is None:
            target_class = original_output.argmax(dim=1).item()
        
        original_confidence = torch.softmax(original_output, dim=1)[0, target_class].item()
        
        faithfulness_scores = {}
        
        for layer_name, cam_data in cams.items():
            cam = cam_data['cam'][0].detach().cpu().numpy()
            
            # Resize CAM to input dimensions
            input_size = input_tensor.shape[-2:]
            if cam.shape != input_size:
                cam_resized = cv2.resize(cam, (input_size[1], input_size[0]))
            else:
                cam_resized = cam
            
            deletion_scores = []
            insertion_scores = []
            
            for percentile in percentiles:
                # Deletion test: remove most important regions
                threshold = np.percentile(cam_resized, 100 - percentile)
                deletion_mask = (cam_resized >= threshold).astype(np.float32)
                
                # Create masked input (set important regions to mean)
                masked_input = input_tensor.clone()
                input_mean = input_tensor.mean()
                
                for c in range(input_tensor.shape[1]):  # Apply to all channels
                    masked_input[0, c] = masked_input[0, c] * (1 - torch.tensor(deletion_mask).to(self.device)) + \
                                       input_mean * torch.tensor(deletion_mask).to(self.device)
                
                # Get prediction for masked input
                with torch.no_grad():
                    masked_output = self.model(masked_input)
                    masked_confidence = torch.softmax(masked_output, dim=1)[0, target_class].item()
                
                # Deletion score: confidence drop (higher is better for faithfulness)
                deletion_score = original_confidence - masked_confidence
                deletion_scores.append(deletion_score)
                
                # Insertion test: start with mean image, add important regions
                insertion_mask = deletion_mask
                inserted_input = torch.full_like(input_tensor, input_mean.item())
                
                for c in range(input_tensor.shape[1]):
                    inserted_input[0, c] = inserted_input[0, c] * (1 - torch.tensor(insertion_mask).to(self.device)) + \
                                         input_tensor[0, c] * torch.tensor(insertion_mask).to(self.device)
                
                with torch.no_grad():
                    inserted_output = self.model(inserted_input)
                    inserted_confidence = torch.softmax(inserted_output, dim=1)[0, target_class].item()
                
                # Insertion score: confidence gain
                insertion_score = inserted_confidence
                insertion_scores.append(insertion_score)
            
            # Compute AUC for deletion and insertion curves
            x_axis = np.array(percentiles) / 100.0
            deletion_auc = auc(x_axis, deletion_scores) if len(deletion_scores) > 1 else deletion_scores[0]
            insertion_auc = auc(x_axis, insertion_scores) if len(insertion_scores) > 1 else insertion_scores[0]
            
            faithfulness_scores[layer_name] = {
                'deletion_auc': deletion_auc,
                'insertion_auc': insertion_auc,
                'deletion_scores': deletion_scores,
                'insertion_scores': insertion_scores,
                'faithfulness_score': (deletion_auc + insertion_auc) / 2  # Combined metric
            }
        
        return faithfulness_scores
    def evaluate_localisation(self, input_tensor, cam_generator, target_class=None,
                        bbox=None, segmentation_mask=None):
        """
        Evaluate localisation quality using spatial metrics
        If ground truth bounding box or segmentation is available, compute overlap
        Otherwise, use concentration and peak detection metrics
        """
        print("Evaluating GradCAM Localisation...")
        
        cams, _ = cam_generator.generate_cams(input_tensor, target_class)
        
        localisation_scores = {}
        
        for layer_name, cam_data in cams.items():
            cam = cam_data['cam'][0].detach().cpu().numpy()
            
            # Check if CAM has any meaningful values
            cam_sum = np.sum(cam)
            cam_max = np.max(cam)
            
            if cam_sum <= 1e-8 or cam_max <= 1e-8:
                print(f"Warning: CAM for {layer_name} has no meaningful values (sum={cam_sum:.2e}, max={cam_max:.2e})")
                localisation_scores[layer_name] = {
                    'concentration_ratio': 0.0,
                    'center_deviation': 1.0,
                    'effective_area': 0.0,
                    'normalized_entropy': 1.0,
                    'localisation_score': 0.0,
                    'warning': 'cam_no_meaningful_values'
                }
                continue
            
            # Basic spatial concentration metrics
            # 1. Peak concentration (how focused the attention is)
            cam_flat = cam.flatten()
            cam_sorted = np.sort(cam_flat)[::-1]
            top_10_percent = int(0.1 * len(cam_flat))
            
            # Safe division with check for zero denominator
            total_cam_sum = np.sum(cam_sorted)
            if total_cam_sum > 1e-8:
                concentration_ratio = np.sum(cam_sorted[:top_10_percent]) / total_cam_sum
            else:
                concentration_ratio = 0.0
            
            # 2. Center of mass deviation from image center
            center_y, center_x = np.array(cam.shape) // 2
            y_coords, x_coords = np.mgrid[0:cam.shape[0], 0:cam.shape[1]]
            
            total_mass = np.sum(cam)
            if total_mass > 1e-8:
                com_y = np.sum(y_coords * cam) / total_mass
                com_x = np.sum(x_coords * cam) / total_mass
                center_deviation = np.sqrt((com_y - center_y)**2 + (com_x - center_x)**2)
                
                # Avoid division by zero in normalization
                max_deviation = np.sqrt(center_y**2 + center_x**2)
                if max_deviation > 1e-8:
                    normalized_deviation = center_deviation / max_deviation
                else:
                    normalized_deviation = 0.0
            else:
                normalized_deviation = 1.0
            
            # 3. Effective area (percentage of image with significant attention)
            if cam_max > 1e-8:
                threshold = 0.2 * cam_max
                effective_area = np.sum(cam > threshold) / cam.size
            else:
                effective_area = 0.0
            
            # 4. Entropy-based dispersion with safe division
            if cam_sum > 1e-8:
                cam_prob = cam / cam_sum
                # Add small epsilon to avoid log(0)
                cam_prob_safe = cam_prob + 1e-12
                entropy = -np.sum(cam_prob * np.log(cam_prob_safe))
                max_entropy = np.log(cam.size)
                
                if max_entropy > 1e-8:
                    normalized_entropy = entropy / max_entropy
                else:
                    normalized_entropy = 0.0
            else:
                normalized_entropy = 1.0
            
            # Combined localisation score with safe computation
            if concentration_ratio > 0 and normalized_entropy < 1.0:
                localisation_score = concentration_ratio * (1 - normalized_entropy)
            else:
                localisation_score = 0.0
            
            localisation_result = {
                'concentration_ratio': float(concentration_ratio),
                'center_deviation': float(normalized_deviation),
                'effective_area': float(effective_area),
                'normalized_entropy': float(normalized_entropy),
                'localisation_score': float(localisation_score)
            }
            
            # If ground truth is available, compute overlap metrics
            if bbox is not None:
                # Bounding box IoU
                y1, x1, y2, x2 = bbox
                bbox_mask = np.zeros_like(cam)
                bbox_mask[y1:y2, x1:x2] = 1
                
                # Threshold CAM to create binary mask
                if cam_max > 1e-8:
                    cam_binary = (cam > 0.5 * cam_max).astype(np.float32)
                    
                    intersection = np.sum(cam_binary * bbox_mask)
                    union = np.sum(np.maximum(cam_binary, bbox_mask))
                    iou = intersection / (union + 1e-8)
                    
                    localisation_result['bbox_iou'] = float(iou)
                else:
                    localisation_result['bbox_iou'] = 0.0
            
            if segmentation_mask is not None:
                # Segmentation overlap
                if cam_max > 1e-8:
                    cam_binary = (cam > 0.5 * cam_max).astype(np.float32)
                    intersection = np.sum(cam_binary * segmentation_mask)
                    union = np.sum(np.maximum(cam_binary, segmentation_mask))
                    seg_iou = intersection / (union + 1e-8)
                    
                    localisation_result['segmentation_iou'] = float(seg_iou)
                else:
                    localisation_result['segmentation_iou'] = 0.0
            
            localisation_scores[layer_name] = localisation_result
        
        return localisation_scores

    def evaluate_complexity(self, input_tensor, cam_generator, target_class=None):
        """
        Evaluate complexity of the generated explanations
        Measures visual complexity, smoothness, and interpretability
        """
        print("Evaluating GradCAM Complexity...")
        
        cams, _ = cam_generator.generate_cams(input_tensor, target_class)
        
        complexity_scores = {}
        
        for layer_name, cam_data in cams.items():
            cam = cam_data['cam'][0].detach().cpu().numpy()
            
            # 1. Edge density (measure of visual complexity)
            # Apply Sobel filter to detect edges
            from scipy.ndimage import sobel
            edges_x = sobel(cam, axis=1)
            edges_y = sobel(cam, axis=0)
            edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
            edge_density = np.mean(edge_magnitude)
            
            # 2. Smoothness (inverse of total variation)
            tv_x = np.mean(np.abs(np.diff(cam, axis=1)))
            tv_y = np.mean(np.abs(np.diff(cam, axis=0)))
            total_variation = tv_x + tv_y
            smoothness = 1.0 / (1.0 + total_variation)
            
            # 3. Number of connected components (fragmentation)
            from scipy.ndimage import label
            binary_cam = (cam > 0.3 * cam.max()).astype(int)
            labeled_array, num_components = label(binary_cam)
            fragmentation = num_components / (cam.size / 100)  # Normalized by size
            
            # 4. Spatial coherence (neighboring pixel similarity)
            coherence_sum = 0
            count = 0
            for i in range(cam.shape[0] - 1):
                for j in range(cam.shape[1] - 1):
                    # Compare pixel with its neighbors
                    neighbors = [cam[i+1, j], cam[i, j+1], cam[i+1, j+1]]
                    coherence_sum += np.mean([abs(cam[i, j] - n) for n in neighbors])
                    count += 1
            spatial_coherence = 1.0 / (1.0 + coherence_sum / count)
            
            # 5. Entropy-based complexity
            cam_hist, _ = np.histogram(cam.flatten(), bins=50, density=True)
            cam_hist = cam_hist[cam_hist > 0]
            entropy = -np.sum(cam_hist * np.log(cam_hist))
            
            # Combined complexity score (lower is better for interpretability)
            complexity_score = (edge_density + total_variation + fragmentation + entropy) / 4
            interpretability_score = (smoothness + spatial_coherence) / 2
            
            complexity_scores[layer_name] = {
                'edge_density': edge_density,
                'total_variation': total_variation,
                'smoothness': smoothness,
                'fragmentation': fragmentation,
                'spatial_coherence': spatial_coherence,
                'entropy': entropy,
                'complexity_score': complexity_score,
                'interpretability_score': interpretability_score
            }
        
        return complexity_scores
    
    def evaluate_randomisation(self, input_tensor, cam_generator, target_class=None,
                            randomisation_tests=['model_randomisation', 'data_randomisation']):
        """
        Evaluate if explanations are better than random
        Tests model parameter randomisation and data label randomisation
        """
        print("Evaluating GradCAM Randomisation...")
        
        # Generate baseline CAMs
        baseline_cams, baseline_output = cam_generator.generate_cams(input_tensor, target_class)
        
        randomisation_scores = {}
        
        for layer_name, baseline_cam_data in baseline_cams.items():
            baseline_cam = baseline_cam_data['cam'][0].detach().cpu().numpy()
            
            # Check baseline CAM validity
            baseline_std = np.std(baseline_cam.flatten())
            if baseline_std < 1e-8:
                print(f"Warning: Baseline CAM for {layer_name} has no variation, skipping randomisation tests")
                randomisation_scores[layer_name] = {
                    'warning': 'baseline_cam_no_variation'
                }
                continue
            
            test_results = {}
            
            if 'model_randomisation' in randomisation_tests:
                # Test 1: Model parameter randomisation
                print(f"Running model randomisation test for {layer_name}...")
                
                # Store original parameters
                original_params = {}
                for name, param in self.model.named_parameters():
                    if 'weight' in name:  # Only randomize weights, not biases
                        original_params[name] = param.data.clone()
                
                random_similarities = []
                num_random_tests = 5
                
                for test_idx in range(num_random_tests):
                    # Randomize model parameters
                    for name, param in self.model.named_parameters():
                        if name in original_params:
                            param.data = torch.randn_like(param.data) * param.data.std()
                    
                    try:
                        # Generate CAM with randomized model
                        random_cams, _ = cam_generator.generate_cams(input_tensor, target_class)
                        
                        if layer_name in random_cams:
                            random_cam = random_cams[layer_name]['cam'][0].detach().cpu().numpy()
                            
                            # Check random CAM validity
                            random_std = np.std(random_cam.flatten())
                            if random_std < 1e-8:
                                continue
                            
                            # Compute similarity with baseline using safe correlation
                            baseline_flat = baseline_cam.flatten()
                            random_flat = random_cam.flatten()
                            
                            if np.std(baseline_flat) > 1e-8 and np.std(random_flat) > 1e-8:
                                try:
                                    correlation_matrix = np.corrcoef(baseline_flat, random_flat)
                                    correlation = correlation_matrix[0, 1]
                                    
                                    if not (np.isnan(correlation) or np.isinf(correlation)):
                                        random_similarities.append(abs(correlation))
                                except Exception as corr_e:
                                    print(f"Correlation error in randomisation test: {corr_e}")
                                    continue
                            
                    except Exception as e:
                        print(f"Error in randomisation test: {e}")
                        continue
                
                # Restore original parameters
                for name, param in self.model.named_parameters():
                    if name in original_params:
                        param.data = original_params[name]
                
                if random_similarities:
                    mean_random_similarity = np.mean(random_similarities)
                    # Sanity check: baseline should be different from random
                    sanity_score = 1.0 - mean_random_similarity
                    test_results['model_randomisation'] = {
                        'mean_random_similarity': float(mean_random_similarity),
                        'sanity_score': float(sanity_score),
                        'num_valid_tests': len(random_similarities)
                    }
                else:
                    test_results['model_randomisation'] = {
                        'warning': 'no_valid_random_similarities'
                    }
            
            if 'data_randomisation' in randomisation_tests:
                # Test 2: Data label randomisation (cascade to incorrect class)
                print(f"Running data randomisation test for {layer_name}...")
                
                # Get all possible classes
                num_classes = baseline_output.shape[1]
                incorrect_classes = [c for c in range(num_classes) if c != target_class]
                
                label_similarities = []
                
                for incorrect_class in incorrect_classes[:5]:  # Test top 5 incorrect classes
                    try:
                        # Generate CAM for incorrect class
                        incorrect_cams, _ = cam_generator.generate_cams(input_tensor, incorrect_class)
                        
                        if layer_name in incorrect_cams:
                            incorrect_cam = incorrect_cams[layer_name]['cam'][0].detach().cpu().numpy()
                            
                            # Check incorrect CAM validity
                            incorrect_std = np.std(incorrect_cam.flatten())
                            if incorrect_std < 1e-8:
                                continue
                            
                            # Compute similarity with correct class CAM
                            baseline_flat = baseline_cam.flatten()
                            incorrect_flat = incorrect_cam.flatten()
                            
                            if np.std(baseline_flat) > 1e-8 and np.std(incorrect_flat) > 1e-8:
                                try:
                                    correlation_matrix = np.corrcoef(baseline_flat, incorrect_flat)
                                    correlation = correlation_matrix[0, 1]
                                    
                                    if not (np.isnan(correlation) or np.isinf(correlation)):
                                        label_similarities.append(abs(correlation))
                                except Exception as corr_e:
                                    continue
                            
                    except Exception as e:
                        continue
                
                if label_similarities:
                    mean_label_similarity = np.mean(label_similarities)
                    # Class selectivity: should be different for different classes
                    selectivity_score = 1.0 - mean_label_similarity
                    test_results['data_randomisation'] = {
                        'mean_label_similarity': float(mean_label_similarity),
                        'selectivity_score': float(selectivity_score),
                        'num_valid_tests': len(label_similarities)
                    }
                else:
                    test_results['data_randomisation'] = {
                        'warning': 'no_valid_label_similarities'
                    }
            
            randomisation_scores[layer_name] = test_results
        
        return randomisation_scores
    
    def generate_comprehensive_evaluation_report(self, input_tensor, cam_generator, target_class=None,
                                                bbox=None, output_dir='evaluation_results'):
        """
        Generate comprehensive evaluation report with all metrics
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE GRADCAM EVALUATION")
        print("="*80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Run all evaluations
        print("\n1. Evaluating Robustness...")
        robustness_scores = self.evaluate_robustness(input_tensor, cam_generator, target_class)
        
        print("\n2. Evaluating Faithfulness...")
        faithfulness_scores = self.evaluate_faithfulness(input_tensor, cam_generator, target_class)
        
        print("\n3. Evaluating Localisation...")
        localisation_scores = self.evaluate_localisation(input_tensor, cam_generator, target_class, bbox)
        
        print("\n4. Evaluating Complexity...")
        complexity_scores = self.evaluate_complexity(input_tensor, cam_generator, target_class)
        
        print("\n5. Evaluating Randomisation...")
        randomisation_scores = self.evaluate_randomisation(input_tensor, cam_generator, target_class)
        
        # Compile comprehensive report
        evaluation_report = {
            'robustness': robustness_scores,
            'faithfulness': faithfulness_scores,
            'localisation': localisation_scores,
            'complexity': complexity_scores,
            'randomisation': randomisation_scores
        }
        
        # Generate summary statistics
        self._generate_evaluation_summary(evaluation_report, output_dir)
        
        # Save detailed results
        import json
        report_path = os.path.join(output_dir, 'gradcam_evaluation_report.json')
        with open(report_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_report = self._convert_numpy_types(evaluation_report)
            json.dump(json_report, f, indent=2)
        
        print(f"\nEvaluation complete! Results saved to {output_dir}/")
        return evaluation_report
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    def _generate_evaluation_summary(self, evaluation_report, output_dir):
        """Generate human-readable evaluation summary"""
        summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("GRADCAM EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Robustness Summary
            f.write("ROBUSTNESS (Higher is Better):\n")
            f.write("-" * 30 + "\n")
            if 'robustness' in evaluation_report:
                for layer, scores in evaluation_report['robustness'].items():
                    f.write(f"{layer}:\n")
                    f.write(f"  Score: {scores['robustness_score']:.3f}\n")
                    f.write(f"  Stability: {scores['mean_similarity']:.3f} ± {scores['std_similarity']:.3f}\n\n")
            
            # Faithfulness Summary
            f.write("FAITHFULNESS (Higher is Better):\n")
            f.write("-" * 30 + "\n")
            if 'faithfulness' in evaluation_report:
                for layer, scores in evaluation_report['faithfulness'].items():
                    f.write(f"{layer}:\n")
                    f.write(f"  Combined Score: {scores['faithfulness_score']:.3f}\n")
                    f.write(f"  Deletion AUC: {scores['deletion_auc']:.3f}\n")
                    f.write(f"  Insertion AUC: {scores['insertion_auc']:.3f}\n\n")
            
            # Localisation Summary
            f.write("LOCALISATION (Higher is Better):\n")
            f.write("-" * 30 + "\n")
            if 'localisation' in evaluation_report:
                for layer, scores in evaluation_report['localisation'].items():
                    f.write(f"{layer}:\n")
                    f.write(f"  Score: {scores['localisation_score']:.3f}\n")
                    f.write(f"  Concentration: {scores['concentration_ratio']:.3f}\n")
                    f.write(f"  Effective Area: {scores['effective_area']:.3f}\n\n")
            
            # Complexity Summary
            f.write("COMPLEXITY (Lower is Better):\n")
            f.write("-" * 30 + "\n")
            if 'complexity' in evaluation_report:
                for layer, scores in evaluation_report['complexity'].items():
                    f.write(f"{layer}:\n")
                    f.write(f"  Complexity Score: {scores['complexity_score']:.3f}\n")
                    f.write(f"  Interpretability: {scores['interpretability_score']:.3f}\n")
                    f.write(f"  Smoothness: {scores['smoothness']:.3f}\n\n")
            
            # Randomisation Summary
            f.write("RANDOMISATION (Higher is Better):\n")
            f.write("-" * 30 + "\n")
            if 'randomisation' in evaluation_report:
                for layer, scores in evaluation_report['randomisation'].items():
                    f.write(f"{layer}:\n")
                    if 'model_randomisation' in scores:
                        f.write(f"  Model Sanity: {scores['model_randomisation']['sanity_score']:.3f}\n")
                    if 'data_randomisation' in scores:
                        f.write(f"  Class Selectivity: {scores['data_randomisation']['selectivity_score']:.3f}\n")
                    f.write("\n")
