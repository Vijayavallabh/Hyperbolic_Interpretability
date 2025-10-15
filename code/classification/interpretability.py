import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

# Change working directory to parent HyperbolicCV/code
working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)
from eval import GradCAMEvaluator
from gcam import HierarchicalGradCAM

# Import required modules
from utils.initialize import select_dataset, select_model
from pytorch_grad_cam.utils.image import show_cam_on_image


class HierarchicalHyperbolicInterpreter:
    """
    Multi-layer hierarchical interpretability for Hyperbolic CV models
    Captures compositional representations across shallow to deep Lorentz layers
    """
    
    def __init__(self, model_path, device='cuda:0'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.args = None
        self.dataset_info = None
        self.lorentz_manifolds = {}
        self.hierarchical_layers = {}
        self.filter_activations = {}
        self.filter_gradients = {}
        self._load_model()
        self.evaluator = GradCAMEvaluator(self.model, device)

    def _create_part_whole_visualization(self, input_tensor, filter_cams, output, target, 
                                    sample_num, output_dir):
        """Create separate visualization for each layer showing part-whole relationships"""
        # Process input image
        img_tensor = input_tensor.squeeze(0).detach()
        if img_tensor.shape[0] <= 4:
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
        
        # Organize filters from ALL layers
        layer_count = 0
        for layer_name, layer_data in filter_cams.items():
            layer_count += 1
            level = layer_data['hierarchy_level']
            
            # Create separate part-whole visualization for each layer
            self._create_single_layer_part_whole(
                img_np, layer_name, layer_data, level, sample_num,
                true_class, predicted_class, class_names, confidence,
                layer_count, len(filter_cams), output_dir
            )
        
        print(f"Created separate part-whole visualizations for all {layer_count} layers!")
        
        
    def _create_single_layer_part_whole(self, img_np, layer_name, layer_data, level, 
                                    sample_num, true_class, predicted_class, 
                                    class_names, confidence, layer_count, 
                                    total_layers, output_dir):
        """Create part-whole visualization for a single layer"""
        
        # MODIFIED: Show 4-5 filters instead of 3
        max_filters_to_show = 4
        level_filters = layer_data['filter_cams'][:max_filters_to_show]
        whole_layer_cam = layer_data['whole_layer_cam']
        k_value = layer_data['k_value']
        
        if not level_filters and whole_layer_cam is None:
            return
        
        # Calculate layout: original + whole layer + individual filters
        total_items = 1 + (1 if whole_layer_cam is not None else 0) + len(level_filters)
        
        # Use optimal layout
        if total_items <= 3:
            rows, cols = 1, total_items
        elif total_items <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 2, 4
        
        # MODIFIED: Larger figure for better text display
        fig_width = cols * 6
        fig_height = rows * 7
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        # Handle subplot configuration
        if total_items == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
            if not isinstance(axes, list):
                axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        item_idx = 0
        
        # 1. Original Image
        if item_idx < len(axes):
            axes[item_idx].imshow(img_np)
            
            # MODIFIED: Better text alignment and formatting
            title_text = f"ORIGINAL IMAGE\n\n"
            title_text += f"True: {class_names[true_class]}\n\n"
            title_text += f"Predicted:\n{class_names[predicted_class]}\n\n"
            title_text += f"Conf: {confidence:.3f}"
            
            axes[item_idx].set_title(title_text, fontsize=12, fontweight='bold',
                                    pad=20, ha='center', va='top')
            axes[item_idx].axis('off')
            
            # Blue border for original
            for spine in axes[item_idx].spines.values():
                spine.set_edgecolor('blue')
                spine.set_linewidth(3)
                spine.set_visible(True)
            
            item_idx += 1
        
        # 2. Whole Layer CAM (if available)
        if whole_layer_cam is not None and item_idx < len(axes):
            whole_cam_np = whole_layer_cam[0].detach().cpu().numpy()
            if whole_cam_np.shape != img_np.shape[:2]:
                whole_cam_np = cv2.resize(whole_cam_np, (img_np.shape[1], img_np.shape[0]))
            
            overlay = show_cam_on_image(img_np, whole_cam_np, use_rgb=True)
            axes[item_idx].imshow(overlay)
            
            # MODIFIED: Display full layer name with better alignment
            layer_display = layer_name.split('.')[-1] if '.' in layer_name else layer_name
            
            title_text = f"WHOLE OBJECT\n\n"
            title_text += f"Layer: {layer_display}\n\n"
            title_text += f"Full: {layer_name}\n\n"
            title_text += f"Level: {level.upper()}\n\n"
            title_text += f"K: {k_value:.4f}"
            
            axes[item_idx].set_title(title_text, fontsize=11, fontweight='bold',
                                    pad=20, ha='center', va='top')
            axes[item_idx].axis('off')
            
            # Red border for whole layer
            for spine in axes[item_idx].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
                spine.set_visible(True)
            
            item_idx += 1
        
        # 3. Individual Filter Parts
        for i, filter_data in enumerate(level_filters):
            if item_idx < len(axes):
                cam_np = filter_data['cam'][0].detach().cpu().numpy()
                
                if cam_np.shape != img_np.shape[:2]:
                    cam_np = cv2.resize(cam_np, (img_np.shape[1], img_np.shape[0]))
                
                overlay = show_cam_on_image(img_np, cam_np, use_rgb=True)
                axes[item_idx].imshow(overlay)
                
                # MODIFIED: Enhanced part context with full layer info
                part_context = self._get_part_whole_context(level, filter_data['rank'])
                
                title_text = f"PART #{i+1}\n\n"
                title_text += f"Filter: {filter_data['filter_id']}\n\n"
                title_text += f"Layer: {layer_name}\n\n"
                title_text += f"Rank: {filter_data['rank']}\n\n"
                title_text += f"Imp: {filter_data['importance']:.3f}\n\n"
                title_text += f"{part_context}"
                
                axes[item_idx].set_title(title_text, fontsize=10, fontweight='bold',
                                        pad=20, ha='center', va='top')
                axes[item_idx].axis('off')
                
                # Color-coded borders for different parts
                part_colors = ['orange', 'green', 'purple', 'brown', 'pink']
                color = part_colors[i] if i < len(part_colors) else 'gray'
                
                for spine in axes[item_idx].spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
                    spine.set_visible(True)
                
                item_idx += 1
        
        # Hide unused subplots
        for i in range(item_idx, len(axes)):
            axes[i].axis('off')
        
        # Level descriptions
        level_descriptions = {
            'shallow': 'Basic Parts → Simple features and edges',
            'mid': 'Complex Parts → Object components and relationships',
            'deep': 'Whole Objects → Complete objects and global context'
        }
        
        # MODIFIED: Enhanced title with full layer information
        layer_short = layer_name.split('.')[-1] if '.' in layer_name else layer_name
        main_title = f"PART-WHOLE ANALYSIS: {layer_short}"
        subtitle = f"Full Layer Path: {layer_name}"
        context_line = f"{level_descriptions[level]} | Sample {sample_num} | Layer {layer_count}/{total_layers}"
        
        fig.suptitle(f'{main_title}\n{subtitle}\n{context_line}', 
                    fontsize=13, fontweight='bold', y=0.95, ha='center')
        
        # MODIFIED: Better spacing for text visibility
        plt.tight_layout()
        plt.subplots_adjust(top=0.82, bottom=0.05, hspace=0.25, wspace=0.15)
        
        # MODIFIED: Create descriptive filename
        layer_safe = layer_name.replace('.', '_').replace('/', '_')
        save_path = os.path.join(output_dir, 
                            f'partwhole_{layer_count:02d}_{level}_{layer_safe}_sample{sample_num:03d}.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Saved part-whole analysis {layer_count}/{total_layers}: {save_path}")
    def _create_detailed_filter_analysis(self, input_tensor, filter_cams, output, target,
                                    sample_num, output_dir):
        """Create detailed analysis for ALL layers with separate images per layer"""
        # Process input image
        img_tensor = input_tensor.squeeze(0).detach()
        if img_tensor.shape[0] <= 4:
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = img_tensor.cpu().numpy()
        
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np, img_np, img_np], axis=2)
        elif img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        elif img_np.shape[2] > 3:
            img_np = img_np[:, :, :3]
        
        true_class = target.item()
        predicted_class = output.argmax(dim=1)[0].item()
        class_names = self._get_class_names()
        confidence = torch.softmax(output, dim=1)[0, predicted_class].item()
        
        # Organize all layers by hierarchy level
        level_layers = {'shallow': [], 'mid': [], 'deep': []}
        
        for layer_name, layer_data in filter_cams.items():
            level = layer_data['hierarchy_level']
            if level in level_layers:
                level_layers[level].append({
                    'name': layer_name,
                    'filters': layer_data['filter_cams'],
                    'whole_layer_cam': layer_data['whole_layer_cam'],
                    'k_value': layer_data['k_value']
                })
        
        print(f"\nProcessing detailed filter analysis with SEPARATE images for each layer:")
        total_layers = sum(len(layers) for layers in level_layers.values())
        print(f"Total layers to process: {total_layers}")
        
        # Create separate image for each individual layer
        layer_count = 0
        for level, layers_list in level_layers.items():
            if not layers_list:
                continue
            
            print(f"\nProcessing {level.upper()} level with {len(layers_list)} layers...")
            
            for layer_data in layers_list:
                layer_count += 1
                self._create_single_layer_analysis(
                    img_np, layer_data, level, sample_num, true_class, 
                    predicted_class, class_names, confidence, 
                    layer_count, total_layers, output_dir
                )
        
        print(f"\nCompleted detailed analysis for all {total_layers} layers!")
        
    def _create_single_layer_analysis(self, img_np, layer_data, level, sample_num, 
                                    true_class, predicted_class, class_names, confidence,
                                    layer_count, total_layers, output_dir):
        """Create a separate detailed analysis image for a single layer"""
        
        layer_name = layer_data['name']
        filters = layer_data['filters']
        whole_layer_cam = layer_data['whole_layer_cam']
        k_value = layer_data['k_value']
        
        # MODIFIED: Show 4-5 filters instead of 3
        max_filters_per_layer = 5
        selected_filters = filters[:max_filters_per_layer] if filters else []
        
        # Calculate layout: 1 whole layer + up to 5 individual filters + 1 original image
        total_items = 1 + len(selected_filters) + 1  # original + whole + filters
        
        # Use 3 rows layout for better organization
        if total_items <= 3:
            rows, cols = 1, total_items
        elif total_items <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        # MODIFIED: Larger figure size for better text visibility
        fig_width = cols * 6  # Increased width for better text spacing
        fig_height = rows * 6  # Increased height for better text spacing
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        # Handle different subplot configurations
        if total_items == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
            if not isinstance(axes, list):
                axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # MODIFIED: Enhanced layout with better text alignment
        item_idx = 0
        
        # 1. Original Image (top-left)
        if item_idx < len(axes):
            axes[item_idx].imshow(img_np)
            
            # MODIFIED: Better aligned text with full class names
            title_text = f"ORIGINAL IMAGE\n\n"
            title_text += f"True Class:\n{class_names[true_class]}\n\n"
            title_text += f"Predicted:\n{class_names[predicted_class]}\n\n"
            title_text += f"Confidence: {confidence:.3f}\n\n"
            title_text += f"Sample #{sample_num}"
            
            axes[item_idx].set_title(title_text, fontsize=12, fontweight='bold', 
                                    pad=20, ha='center', va='top')
            axes[item_idx].axis('off')
            
            # Add distinctive border
            for spine in axes[item_idx].spines.values():
                spine.set_edgecolor('blue')
                spine.set_linewidth(3)
                spine.set_visible(True)
            
            item_idx += 1
        
        # 2. Whole Layer CAM
        if whole_layer_cam is not None and item_idx < len(axes):
            whole_cam_np = whole_layer_cam[0].detach().cpu().numpy()
            
            if whole_cam_np.shape != img_np.shape[:2]:
                whole_cam_np = cv2.resize(whole_cam_np, (img_np.shape[1], img_np.shape[0]))
            
            overlay = show_cam_on_image(img_np, whole_cam_np, use_rgb=True)
            axes[item_idx].imshow(overlay)
            
            # MODIFIED: Display full layer name with better formatting
            title_text = f"WHOLE LAYER CAM\n\n"
            title_text += f"Layer Name:\n{layer_name}\n\n"
            title_text += f"Hierarchy: {level.upper()}\n\n"
            title_text += f"K-value: {k_value:.4f}\n\n"
            title_text += f"Layer {layer_count}/{total_layers}"
            
            axes[item_idx].set_title(title_text, fontsize=12, fontweight='bold',
                                    pad=20, ha='center', va='top')
            axes[item_idx].axis('off')
            
            # Add distinctive border for whole layer
            for spine in axes[item_idx].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
                spine.set_visible(True)
            
            item_idx += 1
        
        # 3. Individual Filter CAMs (up to 5)
        for i, filter_data in enumerate(selected_filters):
            if item_idx < len(axes):
                cam_np = filter_data['cam'][0].detach().cpu().numpy()
                
                if cam_np.shape != img_np.shape[:2]:
                    cam_np = cv2.resize(cam_np, (img_np.shape[1], img_np.shape[0]))
                
                overlay = show_cam_on_image(img_np, cam_np, use_rgb=True)
                axes[item_idx].imshow(overlay)
                
                # MODIFIED: Better aligned text with full layer name
                part_context = self._get_part_whole_context(level, filter_data['rank'])
                
                title_text = f"FILTER #{filter_data['filter_id']}\n\n"
                title_text += f"Layer: {layer_name}\n\n"
                title_text += f"Rank: {filter_data['rank']}\n\n"
                title_text += f"Importance:\n{filter_data['importance']:.4f}\n\n"
                title_text += f"Context: {part_context}"
                
                axes[item_idx].set_title(title_text, fontsize=11, fontweight='bold',
                                        pad=20, ha='center', va='top')
                axes[item_idx].axis('off')
                
                # Add colored border based on rank
                border_colors = ['gold', 'silver', '#CD7F32', 'green', 'purple']  # Top 5 colors
                color = border_colors[i] if i < len(border_colors) else 'gray'
                
                for spine in axes[item_idx].spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
                    spine.set_visible(True)
                
                item_idx += 1
        
        # Hide unused subplots
        for i in range(item_idx, len(axes)):
            axes[i].axis('off')
        
        # Level descriptions for better context
        level_descriptions = {
            'shallow': 'Basic Feature Detection (edges, textures, corners)',
            'mid': 'Part Composition (shapes, object parts, relationships)', 
            'deep': 'Object Understanding (complete objects, global context)'
        }
        
        # MODIFIED: Enhanced main title with full layer information
        layer_short = layer_name.split('.')[-1] if '.' in layer_name else layer_name
        main_title = f"{level.upper()} LAYER ANALYSIS: {layer_short}"
        subtitle = f"Full Layer: {layer_name} | {level_descriptions[level]}"
        context_line = f"Sample {sample_num} | Layer {layer_count}/{total_layers} | Showing {len(selected_filters)} top filters"
        
        fig.suptitle(f'{main_title}\n{subtitle}\n{context_line}', 
                    fontsize=14, fontweight='bold', y=0.98, ha='center')
        
        # MODIFIED: Better layout adjustment for improved text visibility
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.05, hspace=0.3, wspace=0.2)
        
        # MODIFIED: Create safe filename with layer information
        layer_safe = layer_name.replace('.', '_').replace('/', '_')
        save_path = os.path.join(output_dir, 
                            f'layer_{layer_count:02d}_{level}_{layer_safe}_sample{sample_num:03d}.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Saved layer analysis {layer_count}/{total_layers}: {save_path}") 
        
    def _get_part_whole_context(self, level, rank):
        """Get contextual description for part-whole relationship based on hierarchy level"""
        contexts = {
            'shallow': [
                'Edge Detection', 'Texture Pattern', 'Color Gradient', 
                'Corner Detection', 'Line Segment', 'Local Contrast'
            ],
            'mid': [
                'Shape Formation', 'Part Assembly', 'Spatial Relation',
                'Object Part', 'Structural Element', 'Pattern Combination'
            ],
            'deep': [
                'Object Identity', 'Global Structure', 'Semantic Whole',
                'Complete Object', 'Contextual Scene', 'High-level Concept'
            ]
        }
        
        level_contexts = contexts.get(level, ['Feature Detection'])
        context_idx = min(rank - 1, len(level_contexts) - 1)
        return level_contexts[context_idx]
    
   
    
    def generate_filter_level_heatmaps(self, test_loader, num_samples=5, 
                                     filters_per_layer=8, output_dir='filter_analysis'):
        """
        Generate filter-level heatmaps to reveal part-whole relationships
        """
        print("="*80)
        print("GENERATING FILTER-LEVEL HIERARCHICAL HEATMAPS")
        print("="*80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get target layers for analysis
        target_layers = self.get_hierarchical_target_layers()
        
        if not target_layers:
            print("No suitable layers found for filter analysis")
            return
        
        # Initialize filter-level GradCAM
        filter_gradcam = HierarchicalGradCAM(
            self.model, target_layers, self.lorentz_manifolds, 
            self.device, filters_per_layer
        )
        filter_gradcam.register_hooks()
        
        samples_processed = 0
        processed_classes = set()
        
        for batch_idx, (data, targets) in enumerate(test_loader):
            if samples_processed >= num_samples:
                break
                
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            for i in range(data.size(0)):
                if samples_processed >= num_samples:
                    break
                    
                single_input = data[i:i+1]
                single_target = targets[i:i+1]
                target_class = single_target.item()
                
                # Skip if already processed this class
                if target_class in processed_classes:
                    continue
                
                print(f"\nProcessing filter-level analysis for sample {samples_processed + 1} (Class {target_class})")
                
                # Generate filter-level CAMs
                filter_cams, output = filter_gradcam.generate_filter_cams(single_input, target_class)
                
                if filter_cams:
                    # Create part-whole relationship visualization
                    self._create_part_whole_visualization(
                        single_input, filter_cams, output, single_target,
                        samples_processed + 1, output_dir
                    )
                    
                    # Create detailed filter analysis
                    self._create_detailed_filter_analysis(
                        single_input, filter_cams, output, single_target,
                        samples_processed + 1, output_dir
                    )
                    
                    processed_classes.add(target_class)
                    samples_processed += 1
                
        filter_gradcam.remove_hooks()
        print(f"\nFilter-level analysis complete! Results saved to {output_dir}/")
        
    def run_comprehensive_analysis_with_evaluation(self, test_loader, num_samples=5, 
                                                  output_dir='comprehensive_analysis', 
                                                  ):
        """Run comprehensive analysis including GradCAM evaluation metrics"""
        print("=" * 100)
        print("COMPREHENSIVE HYPERBOLIC INTERPRETABILITY ANALYSIS WITH EVALUATION")
        print("=" * 100)
        
        evaluation_dir = output_dir
        
        print(f"Dataset: {self.args.dataset}")
        print(f"Manifolds: {self.args.encoder_manifold} → {self.args.decoder_manifold}")

        print(f"Analysis Samples: {num_samples}")
        


        print("\n" + "="*50)
        print("GRADCAM EVALUATION METRICS")
        print("="*50)
        self._run_gradcam_evaluation(test_loader, evaluation_dir, num_eval_samples=num_samples)
        
        print("=" * 100)
        print(f"COMPREHENSIVE ANALYSIS WITH EVALUATION COMPLETE")
        print(f"Results saved to:")

        print(f"- Evaluation: {evaluation_dir}/")
        print("=" * 100)
    
    def _run_gradcam_evaluation(self, test_loader, output_dir, num_eval_samples=10):
        """Run GradCAM evaluation on sample data"""
        target_layers = self.get_hierarchical_target_layers()
        
        if not target_layers:
            print("No suitable layers found for evaluation")
            return
        
        gradcam = HierarchicalGradCAM(self.model, target_layers, self.lorentz_manifolds, self.device)
        gradcam.register_hooks()
        
        sample_count = 0
        all_evaluations = []
        processed_classes = set()
        
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
                
                if target_class in processed_classes:
                    continue

                print(f"\nEvaluating sample {sample_count + 1}/{num_eval_samples} (Class {target_class})")

                
                try:
                    # Run comprehensive evaluation
                    sample_eval_dir = os.path.join(output_dir, f'sample_{sample_count + 1:03d}')
                    evaluation_results = self.evaluator.generate_comprehensive_evaluation_report(
                        single_input, gradcam, target_class, output_dir=sample_eval_dir
                    )
                    all_evaluations.append(evaluation_results)
                    processed_classes.add(target_class)
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
               
    def _detect_model_type(self):
        """Detect if model is Euclidean, Lorentz, or Hybrid"""
        encoder_manifold = getattr(self.args, 'encoder_manifold', 'euclidean')
        decoder_manifold = getattr(self.args, 'decoder_manifold', 'euclidean')
        
        # Check for hybrid model indicators
        if hasattr(self.args, 'hybrid_layers') or encoder_manifold == 'hybrid':
            return 'hybrid'
        elif encoder_manifold == 'euclidean' and decoder_manifold == 'euclidean':
            return 'euclidean'
        elif encoder_manifold == 'lorentz' and decoder_manifold == 'lorentz':
            return 'lorentz'
        else:
            # Mixed manifolds could indicate hybrid
            return 'hybrid'
        
    def _load_model(self):
        """Load the trained model from checkpoint"""
        print(f"Loading Hierarchical Hyperbolic model from {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        print('checkpoint keys:', checkpoint.keys(), flush=True)
        if isinstance(checkpoint, dict) and any(k.startswith('module.') for k in checkpoint.keys()):
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            print("Stripped 'module.' prefix from checkpoint keys")
        
        # Check if 'args' is in checkpoint; if not, infer from model path or set defaults
        if 'args' in checkpoint:
            self.args = checkpoint['args']
        else:
            print("Warning: 'args' not found in checkpoint. Inferring defaults from model path...")
            self.args = self._infer_args_from_path()
        
        self.model_type = self._detect_model_type()
        print(f"Detected model type: {self.model_type}", flush=True)
 
        print(f"Model: {self.args.dataset}, {self.args.encoder_manifold}-{self.args.decoder_manifold}")

        _, _, _, img_dim, num_classes = select_dataset(self.args)
        self.dataset_info = {
            'img_dim': img_dim,
            'num_classes': num_classes,
            'dataset_name': self.args.dataset
        }
        
        self.model = select_model(img_dim, num_classes, self.args)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Hierarchical model loaded: {num_classes} classes, input shape: {img_dim}")

    def _infer_args_from_path(self):
        """Infer args from model path when checkpoint lacks 'args'"""
        import argparse
        
        # Create a default args object
        args = argparse.Namespace()
        
        # Infer dataset from path (e.g., "tiny" -> Tiny-ImageNet)
        if 'tiny' in self.model_path.lower():
            args.dataset = 'Tiny-ImageNet'
            args.img_dim = [3, 64, 64]
            args.num_classes = 200
        elif 'cifar10' in self.model_path.lower():
            args.dataset = 'CIFAR-10'
            args.img_dim = [3, 32, 32]
            args.num_classes = 10
        elif 'cifar100' in self.model_path.lower():
            args.dataset = 'CIFAR-100'
            args.img_dim = [3, 32, 32]
            args.num_classes = 100
        else:
            # Default fallback
            args.dataset = 'CIFAR-10'
            args.img_dim = [3, 32, 32]
            args.num_classes = 10
        
        # Infer manifolds from path (e.g., "euclid" -> euclidean)
        if 'euclid' in self.model_path.lower():
            args.encoder_manifold = 'euclidean'
            args.decoder_manifold = 'euclidean'
        elif 'lorentz' in self.model_path.lower():
            args.encoder_manifold = 'lorentz'
            args.decoder_manifold = 'lorentz'
        elif 'hybrid' in self.model_path.lower():
            args.encoder_manifold = 'hybrid'
            args.decoder_manifold = 'lorentz'
        else:
            args.encoder_manifold = 'euclidean'
            args.decoder_manifold = 'euclidean'
        
        # Set other common defaults (adjust as needed based on your training)
        args.num_layers = 18  # Assuming ResNet-18
        args.embedding_dim = 512
        args.batch_size = 64
        args.batch_size_test = 64
        args.lr = 0.1
        args.weight_decay = 5e-4
        args.optimizer = 'SGD'
        args.use_lr_scheduler = True
        args.num_epochs = 200
        args.device = self.device
        
        # Add manifold-specific params if needed
        if args.encoder_manifold == 'lorentz':
            args.learn_k = False
            args.encoder_k = 1.0
        if args.decoder_manifold == 'lorentz':
            args.decoder_k = 1.0
        
        # Sparsity and NMF (from path: "without_nmf" -> disable NMF)
        args.enable_sparsity = False
        args.enable_nmf = 'nmf' in self.model_path.lower() and 'without_nmf' not in self.model_path.lower()
        
        print(f"Inferred args: dataset={args.dataset}, manifolds={args.encoder_manifold}-{args.decoder_manifold}, nmf={args.enable_nmf}")
        return args

    
    def _create_complete_hierarchical_visualization(self, input_tensor, hierarchical_cams, output, target, sample_num, output_dir):
        """Create hierarchical visualization with improved text placement and layout"""
        # Process input image (same as before)
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
        for layer_name, cam_data in hierarchical_cams.items():
            level = cam_data['hierarchy_level']
            if level in hierarchy_cams:
                cam_np = cam_data['cam'][0].detach().cpu().numpy()
                if cam_np.shape != img_np.shape[:2]:
                    cam_np = cv2.resize(cam_np, (img_np.shape[1], img_np.shape[0]))
                hierarchy_cams[level].append({
                    'name': layer_name,
                    'cam': cam_np,
                    'k_value': cam_data['k_value']
                })
        
        
        max_layers_per_level = 8  # Increased to show all 18 layers (shallow:5, mid:8, deep:5)
        for level in hierarchy_cams:
            if len(hierarchy_cams[level]) > max_layers_per_level:
                # Keep first, middle, and last layers, plus more to reach max
                layers = hierarchy_cams[level]
                selected = [
                    layers[0],  # First layer
                    layers[len(layers)//4],  # Early
                    layers[len(layers)//2],  # Middle
                    layers[3*len(layers)//4],  # Late-middle
                    layers[-1]  # Last layer
                ]
                # Add more if needed to reach max_layers_per_level
                if len(selected) < max_layers_per_level:
                    additional = [layers[i] for i in range(len(layers)) if layers[i] not in selected][:max_layers_per_level - len(selected)]
                    selected.extend(additional)
                hierarchy_cams[level] = selected
        
        # Calculate total layers for visualization
        total_layers = sum(len(layers) for layers in hierarchy_cams.values())
        print(f"\nCreating improved visualization with {total_layers} layers:")
        for level, layers in hierarchy_cams.items():
            print(f"- {level.upper()}: {len(layers)} layers")
        
        # Create improved layout
        fig_width = 22  # Wider for better spacing
        fig_height = 12  # Fixed height for consistency
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Main grid: header row + layer display
        gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.2)
        
        # Header section with better organization
        header_gs = fig.add_subplot(gs_main[0, 0])
        header_gs.axis('off')
        
        header_fig = fig.add_subfigure(gs_main[0, 0])
        header_axes = header_fig.subplots(1, 4, gridspec_kw={'width_ratios': [2, 2, 3, 2]})
        
        # Original image
        header_axes[0].imshow(img_np)
        header_axes[0].set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
        header_axes[0].axis('off')
        
        # Prediction info
        header_axes[1].text(0.1, 0.8, f'True Class: {class_names[true_class]}', 
                           transform=header_axes[1].transAxes, fontsize=12, fontweight='bold')
        header_axes[1].text(0.1, 0.6, f'Predicted: {class_names[predicted_class]}', 
                           transform=header_axes[1].transAxes, fontsize=12)
        header_axes[1].text(0.1, 0.4, f'Confidence: {confidence:.3f}', 
                           transform=header_axes[1].transAxes, fontsize=12)
        header_axes[1].text(0.1, 0.2, f'Sample: {sample_num}', 
                           transform=header_axes[1].transAxes, fontsize=12)
        header_axes[1].set_title('Classification Results', fontsize=14, fontweight='bold', pad=10)
        header_axes[1].axis('off')
        
        # Model architecture info
        model_info_text = f"""Model Architecture: {self.args.encoder_manifold} → {self.args.decoder_manifold}
        
Hierarchical Structure:
• Shallow Layers: Basic feature detection (edges, textures)
• Mid Layers: Part composition (shapes, object parts)  
• Deep Layers: Complete object understanding

Total Layers Visualized: {total_layers}
Distribution: {len(hierarchy_cams['shallow'])} + {len(hierarchy_cams['mid'])} + {len(hierarchy_cams['deep'])}"""
        
        header_axes[2].text(0.05, 0.95, model_info_text, transform=header_axes[2].transAxes,
                           fontsize=11, verticalalignment='top', fontfamily='monospace')
        header_axes[2].set_title('Architecture Overview', fontsize=14, fontweight='bold', pad=10)
        header_axes[2].axis('off')
        
        # Interpretation guide
        guide_text = """Interpretation Guide:

• Brighter regions = Higher importance
• Color intensity = Attention strength
• Progression shows feature hierarchy
• Left → Right: Simple → Complex"""
        
        header_axes[3].text(0.05, 0.95, guide_text, transform=header_axes[3].transAxes,
                           fontsize=11, verticalalignment='top')
        header_axes[3].set_title('Reading the Heatmaps', fontsize=14, fontweight='bold', pad=10)
        header_axes[3].axis('off')
        
        # Layer visualization section
        layers_gs = fig.add_subplot(gs_main[1, 0])
        layers_gs.axis('off')
        
        layers_fig = fig.add_subfigure(gs_main[1, 0])
        
        # Calculate grid for layers
        layers_per_row = 6  # Optimal for readability
        num_rows = (total_layers + layers_per_row - 1) // layers_per_row
        
        layer_axes = layers_fig.subplots(num_rows, layers_per_row)
                                       #figsize=(layers_per_row * 3, num_rows * 3))
        
        # Handle single row case
        if num_rows == 1 and layers_per_row == 1:
                # Single subplot case
                layer_axes = [layer_axes]
        elif num_rows == 1:
            # Single row case - layer_axes is already a 1D array/tuple
            if isinstance(layer_axes, tuple):
                layer_axes = list(layer_axes)
            elif hasattr(layer_axes, 'flatten'):
                layer_axes = layer_axes.flatten().tolist()
            else:
                # layer_axes is already a list/array
                pass
        elif layers_per_row == 1:
            # Single column case
            if hasattr(layer_axes, 'flatten'):
                layer_axes = layer_axes.flatten().tolist()
            else:
                layer_axes = list(layer_axes)
        else:
            # Multiple rows and columns
            if hasattr(layer_axes, 'flatten'):
                layer_axes = layer_axes.flatten().tolist()
            else:
                # Convert nested structure to flat list
                flat_axes = []
                for row in layer_axes:
                    if hasattr(row, '__iter__'):
                        flat_axes.extend(row)
                    else:
                        flat_axes.append(row)
                layer_axes = flat_axes
        
        # Plot layers by hierarchy level with better spacing
        current_idx = 0
        level_colors = {'shallow': 'blue', 'mid': 'green', 'deep': 'red'}
        
        for level in ['shallow', 'mid', 'deep']:
            for layer_data in hierarchy_cams[level]:
                if current_idx < len(layer_axes):
                    # Create overlay
                    overlay = show_cam_on_image(img_np, layer_data['cam'], use_rgb=True)
                    layer_axes[current_idx].imshow(overlay)
                    
                    # Better layer naming
                    layer_parts = layer_data['name'].split('.')
                    if len(layer_parts) >= 2:
                        display_name = f"{layer_parts[-2]}.{layer_parts[-1]}"
                    else:
                        display_name = layer_parts[-1]
                    
                    # Cleaner title with level indicator
                    layer_axes[current_idx].set_title(f'{level.upper()}\n{display_name}\nk={layer_data["k_value"]:.3f}', 
                                                     fontsize=10, color=level_colors[level], fontweight='bold', pad=8)
                    layer_axes[current_idx].axis('off')
                    
                    current_idx += 1
        
        # Hide unused axes
        for i in range(current_idx, len(layer_axes)):
            layer_axes[i].axis('off')
        
        # Main title
        plt.suptitle(f'Complete Hierarchical Layer Analysis: Shallow → Mid → Deep Progression\nSample {sample_num}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.94, hspace=0.15, wspace=0.1)
        
        # Save visualization
        save_path = os.path.join(output_dir, f'complete_hierarchy_improved_{sample_num:03d}_class{true_class}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved improved complete hierarchical visualization: {save_path}")

    def get_hierarchical_target_layers(self, use_all_layers=False):
        """Get exactly the 18 specified ResNet layers"""
        # Define the exact 18 layer names to extract
        if self.model_type == 'euclidean':
            specified_layers = ['encoder.conv1.0',
            'encoder.conv2_x.0.conv1', 
            'encoder.conv2_x.0.conv2', 
            'encoder.conv2_x.1.conv1', 
            'encoder.conv2_x.1.conv2', 
            'encoder.conv3_x.0.conv1', 
            'encoder.conv3_x.0.conv2', 
            'encoder.conv3_x.1.conv1', 
            'encoder.conv3_x.1.conv2', 
            'encoder.conv4_x.0.conv1', 
            'encoder.conv4_x.0.conv2', 
            'encoder.conv4_x.1.conv1', 
            'encoder.conv4_x.1.conv2', 
            'encoder.conv5_x.0.conv1',
            'encoder.conv5_x.0.conv2',
            'encoder.conv5_x.1.conv1',
            'encoder.conv5_x.1.conv2',
            'decoder']
        elif self.model_type == 'hybrid':
            specified_layers = [
            'encoder.conv1.0',
            'encoder.conv2_x.0.conv1', 
            'encoder.conv2_x.0.conv2', 
            'encoder.conv2_x.1.conv1', 
            'encoder.conv2_x.1.conv2', 
            'encoder.conv3_x.0.conv1', 
            'encoder.conv3_x.0.conv2', 
            'encoder.conv3_x.1.conv1', 
            'encoder.conv3_x.1.conv2', 
            'encoder.conv4_x.0.conv1', 
            'encoder.conv4_x.0.conv2', 
            'encoder.conv4_x.1.conv1', 
            'encoder.conv4_x.1.conv2', 
            'encoder.conv5_x.0.conv1',
            'encoder.conv5_x.0.conv2',
            'encoder.conv5_x.1.conv1',
            'encoder.conv5_x.1.conv2',
            'decoder'
        ]
        else:
            specified_layers = [
            'encoder.conv1.conv.0',
            'encoder.conv2_x.0.conv1', 
            'encoder.conv2_x.0.conv2', 
            'encoder.conv2_x.1.conv1', 
            'encoder.conv2_x.1.conv2', 
            'encoder.conv3_x.0.conv1', 
            'encoder.conv3_x.0.conv2', 
            'encoder.conv3_x.1.conv1', 
            'encoder.conv3_x.1.conv2', 
            'encoder.conv4_x.0.conv1', 
            'encoder.conv4_x.0.conv2', 
            'encoder.conv4_x.1.conv1', 
            'encoder.conv4_x.1.conv2', 
            'encoder.conv5_x.0.conv1',
            'encoder.conv5_x.0.conv2',
            'encoder.conv5_x.1.conv1',
            'encoder.conv5_x.1.conv2',
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
        
        if self.model_type == 'hybrid':
            manifold_mapping = {
            'encoder.conv1': 'euclidean',
            'encoder.conv2_x': 'lorentz',
            'encoder.conv3_x': 'euclidean',
            'encoder.conv4_x': 'lorentz',
            'encoder.conv5_x': 'euclidean',
            'decoder': 'lorentz'
        }
        elif self.model_type == 'euclidean':
            manifold_mapping = {key: 'euclidean' for key in hierarchy_mapping.keys()}
        else:  # lorentz
            manifold_mapping = {key: 'lorentz' for key in hierarchy_mapping.keys()}
            
        target_layers = []
        found_layers = set()
        not_found_layers = []
        
        for specified_layer in specified_layers:
            found = False
            
            # Determine hierarchy level and manifold type for this layer
            hierarchy_level = None
            layer_manifold_type = 'euclidean'  # default
            
            for prefix, level in hierarchy_mapping.items():
                if specified_layer.startswith(prefix):
                    hierarchy_level = level
                    layer_manifold_type = manifold_mapping.get(prefix, 'euclidean')
                    break
            
            if hierarchy_level is None:
                hierarchy_level = 'deep'  # Default to deep for unrecognized patterns
            
            # Search through all modules
            for name, module in self.model.named_modules():

                if name == specified_layer:
                    # Get k value if available
                    k_val = 1.0
                    if hasattr(module, 'manifold') and hasattr(module.manifold, 'k'):
                        k_val = module.manifold.k.item()
                    elif specified_layer in self.lorentz_manifolds:
                        k_val = self.lorentz_manifolds[specified_layer].k.item()
                    
                    target_layers.append({
                        'name': name,
                        'module': module,
                        'hierarchy_level': hierarchy_level,
                        'k_value': k_val,
                        'manifold_type': layer_manifold_type  # Add manifold type info
                    })
                    found_layers.add(name)
                    found = True
                    print(f"  ✓ Found layer: {name} ({hierarchy_level}, {layer_manifold_type}, k={k_val:.3f})")
                    break
            
            if not found:
                not_found_layers.append(specified_layer)

        layer_counts = {'shallow': 0, 'mid': 0, 'deep': 0}
        manifold_counts = {'euclidean': 0, 'lorentz': 0}
        
        for layer in target_layers:
            layer_counts[layer['hierarchy_level']] += 1
            manifold_counts[layer.get('manifold_type', 'euclidean')] += 1
        
        print(f"\nSelected {len(target_layers)} layers for visualization:")
        print(f"- Shallow layers: {layer_counts['shallow']}")
        print(f"- Mid layers: {layer_counts['mid']}")
        print(f"- Deep layers: {layer_counts['deep']}")
        print(f"- Euclidean layers: {manifold_counts['euclidean']}")
        print(f"- Lorentz layers: {manifold_counts['lorentz']}")
        
        if not_found_layers:
            print(f"\nWarning: {len(not_found_layers)} layers not found:")
            for layer in not_found_layers[:5]:  # Show first 5
                print(f"  - {layer}")
        
        return target_layers


    def _get_class_names(self):
        """Get class names for the dataset"""
        if self.args.dataset.lower() == 'cifar10':
            return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.args.dataset.lower() == 'cifar100':
            return [f'class_{i}' for i in range(100)]
        else:
            return [f'class_{i}' for i in range(self.dataset_info['num_classes'])]

    def generate_hierarchical_analysis(self, test_loader, num_samples=10, output_dir='hierarchical_analysis'):
        """Generate hierarchical compositional analysis across multiple Lorentz layers"""
        print("Generating Hierarchical Compositional Analysis...")
        
        os.makedirs(output_dir, exist_ok=True)
        # Modified to select exactly 18 layers for ResNet-18, not all layers
        target_layers = self.get_hierarchical_target_layers(use_all_layers=False)
        
        if not target_layers:
            print("No suitable hierarchical layers found for analysis")
            return

        gradcam = HierarchicalGradCAM(self.model, target_layers, self.lorentz_manifolds, self.device)
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
                    
                    # Skip if we've already processed this class
                    if target_class in processed_classes:
                        continue
                    
                    print(f"\nProcessing hierarchical sample {samples_processed + 1} (Class {target_class})")
                    
                    # Generate hierarchical CAMs
                    hierarchical_cams, output = gradcam.generate_hierarchical_cams(single_input, target_class)
                    
                    if hierarchical_cams:
                        # Create comprehensive hierarchical visualization with all layers
                        self._create_complete_hierarchical_visualization(
                            single_input, hierarchical_cams, output, single_target, 
                            samples_processed + 1, output_dir
                        )
                        saved_visualizations += 1
                        
                        # Track that we've processed this class
                        processed_classes.add(target_class)
                        

                    samples_processed += 1
                    print(f"Processed classes so far: {sorted(processed_classes)}")
                    
                    if samples_processed >= num_samples:
                        break
                
                if samples_processed % 3 == 0:
                    print(f"Hierarchical progress: {samples_processed}/{num_samples} (unique classes)")
                    
            except Exception as e:
                print(f"Error in hierarchical batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        gradcam.remove_hooks()
        print(f"\nHierarchical analysis complete! Saved {saved_visualizations} visualizations from {len(processed_classes)} different classes to {output_dir}/")    


def main():
    parser = argparse.ArgumentParser(description='Complete Hyperbolic Interpretability Analysis')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=7)
    parser.add_argument('--output_dir', type=str, default='complete_hierarchical_analysis_3')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--analysis_type', type=str, default='evaluation',
                       choices=['hierarchical', 'filter', 'evaluation'],
                       help='Type of analysis: hierarchical (image-level), filter (filter-level), complete (both)')
    
    args = parser.parse_args()
    analyzer = HierarchicalHyperbolicInterpreter(args.model_path, args.device)
    _, test_loader, _, _, _ = select_dataset(analyzer.args)
    if args.analysis_type == 'filter':
        filter_dir = os.path.join(args.output_dir, 'filter_level')
        analyzer.generate_filter_level_heatmaps(test_loader, args.num_samples, output_dir=filter_dir)
    
    elif args.analysis_type == 'hierarchical':
        hierarchical_dir = os.path.join(args.output_dir, 'hierarchical')
        analyzer.generate_hierarchical_analysis(test_loader, args.num_samples, hierarchical_dir)
    else:
        hierarchical_dir = os.path.join(args.output_dir, 'hierarchical')
        analyzer.generate_hierarchical_analysis(test_loader, args.num_samples, hierarchical_dir)
        evaluation_dir = os.path.join(args.output_dir, 'evaluation')
        analyzer.run_comprehensive_analysis_with_evaluation(test_loader, args.num_samples,evaluation_dir)
        filter_dir = os.path.join(args.output_dir, 'filter_level')
        analyzer.generate_filter_level_heatmaps(test_loader, args.num_samples, output_dir=filter_dir)
if __name__ == "__main__":
    main()
