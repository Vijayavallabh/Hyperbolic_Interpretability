import os
import sys
import torch
import numpy as np
import cv2
from sklearn.metrics import auc
# Change working directory to parent directory
working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)


class GradCAMEvaluator:
    """
    Comprehensive evaluation metrics for GradCAM interpretability
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
        

        baseline_cams, _ = cam_generator.generate_hierarchical_cams(input_tensor, target_class)
        
        robustness_scores = {}
        
        for layer_name, baseline_cam_data in baseline_cams.items():
            baseline_cam = baseline_cam_data['cam'][0].detach().cpu().numpy()
            similarities = []
            
            for noise_level in noise_levels:
                trial_similarities = []
                
                for trial in range(num_trials):
                    noise = torch.randn_like(input_tensor) * noise_level
                    perturbed_input = input_tensor + noise
                    
                    try:
                        perturbed_cams, _ = cam_generator.generate_hierarchical_cams(perturbed_input, target_class)
                        
                        if layer_name in perturbed_cams:
                            perturbed_cam = perturbed_cams[layer_name]['cam'][0].detach().cpu().numpy()
                   
                            correlation = np.corrcoef(baseline_cam.flatten(), perturbed_cam.flatten())[0, 1]
                            if not np.isnan(correlation):
                                trial_similarities.append(correlation)
                    except Exception as e:
                        print(f"Error in robustness trial: {e}")
                        continue
                
                if trial_similarities:
                    similarities.extend(trial_similarities)
            
            if similarities:
                mean_similarity = np.mean(similarities)
                std_similarity = np.std(similarities)
                robustness_score = mean_similarity - std_similarity
                
                robustness_scores[layer_name] = {
                    'mean_similarity': mean_similarity,
                    'std_similarity': std_similarity,
                    'robustness_score': robustness_score,
                    'num_valid_trials': len(similarities)
                }
        
        return robustness_scores
    
    def evaluate_faithfulness(self, input_tensor, cam_generator, target_class=None, 
                            percentiles=[10, 20, 30, 40, 50]):
        """
        Evaluate faithfulness using deletion/insertion metrics
        Measures how much prediction changes when important/unimportant regions are removed
        """
        print("Evaluating GradCAM Faithfulness...")
        
  
        cams, original_output = cam_generator.generate_hierarchical_cams(input_tensor, target_class)
        
        if target_class is None:
            target_class = original_output.argmax(dim=1).item()
        
        original_confidence = torch.softmax(original_output, dim=1)[0, target_class].item()
        
        faithfulness_scores = {}
        
        for layer_name, cam_data in cams.items():
            cam = cam_data['cam'][0].detach().cpu().numpy()
            
       
            input_size = input_tensor.shape[-2:]
            if cam.shape != input_size:
                cam_resized = cv2.resize(cam, (input_size[1], input_size[0]))
            else:
                cam_resized = cam
            
            deletion_scores = []
            insertion_scores = []
            
            for percentile in percentiles:
                threshold = np.percentile(cam_resized, 100 - percentile)
                deletion_mask = (cam_resized >= threshold).astype(np.float32)
                
                masked_input = input_tensor.clone()
                input_mean = input_tensor.mean()
                
                for c in range(input_tensor.shape[1]):  
                    masked_input[0, c] = masked_input[0, c] * (1 - torch.tensor(deletion_mask).to(self.device)) + \
                                       input_mean * torch.tensor(deletion_mask).to(self.device)
                
                with torch.no_grad():
                    masked_output = self.model(masked_input)
                    masked_confidence = torch.softmax(masked_output, dim=1)[0, target_class].item()
                
          
                deletion_score = original_confidence - masked_confidence
                deletion_scores.append(deletion_score)

                insertion_mask = deletion_mask
                inserted_input = torch.full_like(input_tensor, input_mean.item())
                
                for c in range(input_tensor.shape[1]):
                    inserted_input[0, c] = inserted_input[0, c] * (1 - torch.tensor(insertion_mask).to(self.device)) + \
                                         input_tensor[0, c] * torch.tensor(insertion_mask).to(self.device)
                
                with torch.no_grad():
                    inserted_output = self.model(inserted_input)
                    inserted_confidence = torch.softmax(inserted_output, dim=1)[0, target_class].item()
                
          
                insertion_score = inserted_confidence
                insertion_scores.append(insertion_score)
            
          
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
        
        cams, _ = cam_generator.generate_hierarchical_cams(input_tensor, target_class)
        
        localisation_scores = {}
        
        for layer_name, cam_data in cams.items():
            cam = cam_data['cam'][0].detach().cpu().numpy()
            
            # Skip localisation for non-spatial CAMs (e.g., decoder)
            if len(cam.shape) != 2:
                localisation_scores[layer_name] = {
                    'concentration_ratio': 0.0,
                    'center_deviation': 1.0,
                    'effective_area': 0.0,
                    'normalized_entropy': 1.0,
                    'localisation_score': 0.0
                }
                continue
            
            cam_flat = cam.flatten()
            cam_sorted = np.sort(cam_flat)[::-1]
            top_10_percent = int(0.1 * len(cam_flat))
            concentration_ratio = np.sum(cam_sorted[:top_10_percent]) / np.sum(cam_sorted)
            
            center_y, center_x = np.array(cam.shape) // 2
            y_coords, x_coords = np.mgrid[0:cam.shape[0], 0:cam.shape[1]]
            
            total_mass = np.sum(cam)
            if total_mass > 0:
                com_y = np.sum(y_coords * cam) / total_mass
                com_x = np.sum(x_coords * cam) / total_mass
                center_deviation = np.sqrt((com_y - center_y)**2 + (com_x - center_x)**2)
                normalized_deviation = center_deviation / np.sqrt(center_y**2 + center_x**2)
            else:
                normalized_deviation = 1.0
            
            threshold = 0.2 * cam.max()
            effective_area = np.sum(cam > threshold) / cam.size
            
        
            cam_prob = cam / (cam.sum() + 1e-8)
            entropy = -np.sum(cam_prob * np.log(cam_prob + 1e-8))
            max_entropy = np.log(cam.size)
            normalized_entropy = entropy / max_entropy
            
            localisation_score = {
                'concentration_ratio': concentration_ratio,
                'center_deviation': normalized_deviation,
                'effective_area': effective_area,
                'normalized_entropy': normalized_entropy,
                'localisation_score': concentration_ratio * (1 - normalized_entropy)
            }
            
            if bbox is not None:
                # Bounding box IoU
                y1, x1, y2, x2 = bbox
                bbox_mask = np.zeros_like(cam)
                bbox_mask[y1:y2, x1:x2] = 1
                
                cam_binary = (cam > 0.5 * cam.max()).astype(np.float32)
                
                intersection = np.sum(cam_binary * bbox_mask)
                union = np.sum(np.maximum(cam_binary, bbox_mask))
                iou = intersection / (union + 1e-8)
                
                localisation_score['bbox_iou'] = iou
            
            if segmentation_mask is not None:
                # Segmentation overlap
                cam_binary = (cam > 0.5 * cam.max()).astype(np.float32)
                intersection = np.sum(cam_binary * segmentation_mask)
                union = np.sum(np.maximum(cam_binary, segmentation_mask))
                seg_iou = intersection / (union + 1e-8)
                
                localisation_score['segmentation_iou'] = seg_iou
            
            localisation_scores[layer_name] = localisation_score
        
        return localisation_scores

    def evaluate_complexity(self, input_tensor, cam_generator, target_class=None):
        """
        Evaluate complexity of the generated explanations
        Measures visual complexity, smoothness, and interpretability
        """
        print("Evaluating GradCAM Complexity...")
        
        cams, _ = cam_generator.generate_hierarchical_cams(input_tensor, target_class)
        
        complexity_scores = {}
        
        for layer_name, cam_data in cams.items():
            cam = cam_data['cam'][0].detach().cpu().numpy()
            
         
            from scipy.ndimage import sobel
            edges_x = sobel(cam, axis=1)
            edges_y = sobel(cam, axis=0)
            edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
            edge_density = np.mean(edge_magnitude)
            
            tv_x = np.mean(np.abs(np.diff(cam, axis=1)))
            tv_y = np.mean(np.abs(np.diff(cam, axis=0)))
            total_variation = tv_x + tv_y
            smoothness = 1.0 / (1.0 + total_variation)
            
            from scipy.ndimage import label
            binary_cam = (cam > 0.3 * cam.max()).astype(int)
            labeled_array, num_components = label(binary_cam)
            fragmentation = num_components / (cam.size / 100)  # Normalized by size
            
            coherence_sum = 0
            count = 0
            for i in range(cam.shape[0] - 1):
                for j in range(cam.shape[1] - 1):
                    # Compare pixel with its neighbors
                    neighbors = [cam[i+1, j], cam[i, j+1], cam[i+1, j+1]]
                    coherence_sum += np.mean([abs(cam[i, j] - n) for n in neighbors])
                    count += 1
            spatial_coherence = 1.0 / (1.0 + coherence_sum / count)
            
            
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
        
        baseline_cams, baseline_output = cam_generator.generate_hierarchical_cams(input_tensor, target_class)
        
        randomisation_scores = {}
        
        for layer_name, baseline_cam_data in baseline_cams.items():
            baseline_cam = baseline_cam_data['cam'][0].detach().cpu().numpy()
            
            # Check if baseline CAM has variation
            baseline_flat = baseline_cam.flatten()
            baseline_std = np.std(baseline_flat)
            
            if baseline_std < 1e-8:
                print(f"Warning: Baseline CAM for {layer_name} has no variation, skipping randomisation tests")
                randomisation_scores[layer_name] = {
                    'model_randomisation': {'sanity_score': 0.0, 'mean_random_similarity': 1.0},
                    'data_randomisation': {'selectivity_score': 0.0, 'mean_label_similarity': 1.0}
                }
                continue
            
            test_results = {}
            
            if 'model_randomisation' in randomisation_tests:
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
                        random_cams, _ = cam_generator.generate_hierarchical_cams(input_tensor, target_class)
                        
                        if layer_name in random_cams:
                            random_cam = random_cams[layer_name]['cam'][0].detach().cpu().numpy()
                            random_flat = random_cam.flatten()
                            random_std = np.std(random_flat)
                            
                            # Check if random CAM has variation
                            if random_std < 1e-8:
                                print(f"Warning: Random CAM for {layer_name} has no variation")
                                continue
                            
                            # Use safer correlation calculation
                            try:
                                correlation = self._safe_correlation(baseline_flat, random_flat)
                                if not np.isnan(correlation):
                                    random_similarities.append(abs(correlation))
                            except Exception as corr_e:
                                print(f"Error computing correlation: {corr_e}")
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
                    sanity_score = 1.0 - mean_random_similarity
                    test_results['model_randomisation'] = {
                        'mean_random_similarity': mean_random_similarity,
                        'sanity_score': sanity_score
                    }
                else:
                    test_results['model_randomisation'] = {
                        'mean_random_similarity': 1.0,
                        'sanity_score': 0.0
                    }
            
            if 'data_randomisation' in randomisation_tests:
                print(f"Running data randomisation test for {layer_name}...")
                
                # Get all possible classes
                num_classes = baseline_output.shape[1]
                incorrect_classes = [c for c in range(num_classes) if c != target_class]
                
                label_similarities = []
                
                for incorrect_class in incorrect_classes[:5]:  # Test top 5 incorrect classes
                    try:
                        # Generate CAM for incorrect class
                        incorrect_cams, _ = cam_generator.generate_hierarchical_cams(input_tensor, incorrect_class)
                        
                        if layer_name in incorrect_cams:
                            incorrect_cam = incorrect_cams[layer_name]['cam'][0].detach().cpu().numpy()
                            incorrect_flat = incorrect_cam.flatten()
                            incorrect_std = np.std(incorrect_flat)
                            
                            # Check if incorrect CAM has variation
                            if incorrect_std < 1e-8:
                                continue
                            
                            # Use safer correlation calculation
                            try:
                                correlation = self._safe_correlation(baseline_flat, incorrect_flat)
                                if not np.isnan(correlation):
                                    label_similarities.append(abs(correlation))
                            except Exception as corr_e:
                                print(f"Error computing correlation: {corr_e}")
                                continue
                                
                    except Exception as e:
                        continue
                
                if label_similarities:
                    mean_label_similarity = np.mean(label_similarities)
                    selectivity_score = 1.0 - mean_label_similarity
                    test_results['data_randomisation'] = {
                        'mean_label_similarity': mean_label_similarity,
                        'selectivity_score': selectivity_score
                    }
                else:
                    test_results['data_randomisation'] = {
                        'mean_label_similarity': 1.0,
                        'selectivity_score': 0.0
                    }
            
            randomisation_scores[layer_name] = test_results
        
        return randomisation_scores
    
    def _safe_correlation(self, x, y):
        """
        Compute correlation safely, handling edge cases
        """
        # Remove NaN values
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        
        if len(x) != len(y):
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
        
        if len(x) < 2:
            return 0.0
        
        # Check for zero variance
        x_std = np.std(x)
        y_std = np.std(y)
        
        if x_std < 1e-8 or y_std < 1e-8:
            # If one or both arrays have no variance, return appropriate correlation
            if x_std < 1e-8 and y_std < 1e-8:
                # Both constant - perfect correlation if same value, else 0
                return 1.0 if abs(np.mean(x) - np.mean(y)) < 1e-8 else 0.0
            else:
                # One constant, one varying - no correlation
                return 0.0
        
        # Compute Pearson correlation manually to avoid numpy warnings
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        
        numerator = np.sum(x_centered * y_centered)
        denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
        
        if denominator < 1e-8:
            return 0.0
        
        correlation = numerator / denominator
        return np.clip(correlation, -1.0, 1.0)
    
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

        evaluation_report = {'sample_class': target_class,
            'robustness': robustness_scores,
            'faithfulness': faithfulness_scores,
            'localisation': localisation_scores,
            'complexity': complexity_scores,
            'randomisation': randomisation_scores
        }
        

        self._generate_evaluation_summary(evaluation_report, output_dir)
        
        # Save detailed results
        import json
        report_path = os.path.join(output_dir, 'gradcam_evaluation_report.json')
        with open(report_path, 'w') as f:
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
            if 'sample_class' in evaluation_report:
                f.write(f"Sample Class: {evaluation_report['sample_class']}\n\n")
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
