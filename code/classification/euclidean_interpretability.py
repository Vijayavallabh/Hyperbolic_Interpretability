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
from gradcam_evaluator import GradCAMEvaluatorBase
from hierarcheal_gradcam import HierarchicalGradCAMBase
from euclidean_interpreter_base import EuclideanInterpreterBase
from filterlevel_gradcam import FilterLevelGradCAMBase

class FilterLevelHierarchicalAnalyzer(EuclideanInterpreterBase):
    """
    Extended analyzer for filter-level heatmap generation to reveal part-whole relationships
    """
    
    def __init__(self, model_path, device='cuda:0'):
        super().__init__(model_path, device)
        self.filter_activations = {}
        self.filter_gradients = {}
        
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
        target_layers = self.get_target_layers()
        
        if not target_layers:
            print("No suitable layers found for filter analysis")
            return
        
        # Initialize filter-level GradCAM
        filter_gradcam = FilterLevelGradCAMBase(
            self.model, target_layers, 
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
    def _create_part_whole_visualization(self, input_tensor, filter_cams, output, target, 
                                    sample_num, output_dir):
        """Create visualization showing part-whole relationships across hierarchy levels with improved text placement"""
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
        
        # Organize filters and get ONE whole layer CAM per hierarchy level
        hierarchy_filters = {'shallow': [], 'mid': [], 'deep': []}
        hierarchy_whole_cams = {'shallow': None, 'mid': None, 'deep': None}
        
        for layer_name, layer_data in filter_cams.items():
            level = layer_data['hierarchy_level']
            if level in hierarchy_filters:
                # Add individual filters
                hierarchy_filters[level].extend(layer_data['filter_cams'])
                
                # Take the first whole layer CAM for this level (one per level)
                if hierarchy_whole_cams[level] is None and layer_data['whole_layer_cam'] is not None:
                    whole_cam_np = layer_data['whole_layer_cam'][0].detach().cpu().numpy()
                    if whole_cam_np.shape != img_np.shape[:2]:
                        whole_cam_np = cv2.resize(whole_cam_np, (img_np.shape[1], img_np.shape[0]))
                    
                    hierarchy_whole_cams[level] = {
                        'name': layer_name,
                        'cam': whole_cam_np
                    }
        
        # Create improved visualization with better spacing
        max_filters_per_level = 4  # Reduced for better layout
        
        # Count levels with data
        active_levels = [level for level in ['shallow', 'mid', 'deep'] 
                        if hierarchy_filters[level] or hierarchy_whole_cams[level]]
        
        if not active_levels:
            print("No data available for visualization")
            return
        
        # Create figure with better proportions
        fig_width = 20
        fig_height = len(active_levels) * 8 + 3  # More space per level
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create grid: header + (whole_layer + filters) for each level
        total_rows = 1 + len(active_levels) * 2  # header + 2 rows per level
        gs_main = fig.add_gridspec(total_rows, 1, height_ratios=[1] + [1.2, 2] * len(active_levels))
        
        # Header with original image and basic info
        header_gs = fig.add_subplot(gs_main[0, 0])
        header_gs.axis('off')
        
        header_fig = fig.add_subfigure(gs_main[0, 0])
        header_axes = header_fig.subplots(1, 3, gridspec_kw={'width_ratios': [2, 3, 2]})
        
        # Original image (center)
        header_axes[1].imshow(img_np)
        header_axes[1].set_title(f'Original Image\nTrue: {class_names[true_class]} | Predicted: {class_names[predicted_class]}\nConfidence: {confidence:.3f}', 
                                fontsize=14, fontweight='bold', pad=10)
        header_axes[1].axis('off')
        
        # Model info (left)
        header_axes[0].text(0.05, 0.8, f'Model: {self.args.encoder_manifold}-{self.args.decoder_manifold}', 
                           transform=header_axes[0].transAxes, fontsize=12, fontweight='bold')
        header_axes[0].text(0.05, 0.6, f'Sample: {sample_num}', 
                           transform=header_axes[0].transAxes, fontsize=11)
        header_axes[0].text(0.05, 0.4, f'Class: {true_class}', 
                           transform=header_axes[0].transAxes, fontsize=11)
        header_axes[0].set_title('Analysis Info', fontsize=12, fontweight='bold')
        header_axes[0].axis('off')
        
        # Legend (right)
        header_axes[2].text(0.05, 0.8, 'Part-Whole Analysis:', 
                           transform=header_axes[2].transAxes, fontsize=12, fontweight='bold')
        header_axes[2].text(0.05, 0.65, '• Whole Layer = Complete level view', 
                           transform=header_axes[2].transAxes, fontsize=10)
        header_axes[2].text(0.05, 0.5, '• Individual Filters = Part details', 
                           transform=header_axes[2].transAxes, fontsize=10)
        header_axes[2].text(0.05, 0.35, '• Hierarchy: Shallow → Mid → Deep', 
                           transform=header_axes[2].transAxes, fontsize=10)
        header_axes[2].set_title('Interpretation Guide', fontsize=12, fontweight='bold')
        header_axes[2].axis('off')
        
        # Process each hierarchy level
        level_descriptions = {
            'shallow': 'Basic Parts (edges, textures, local features)',
            'mid': 'Part Composition (shapes, object parts, relationships)',
            'deep': 'Whole Objects (complete objects, global structure)'
        }
        
        row_idx = 1
        for level in active_levels:
            level_filters = hierarchy_filters[level][:max_filters_per_level]
            whole_cam_data = hierarchy_whole_cams[level]
            
            # Whole layer CAM row
            whole_layer_gs = fig.add_subplot(gs_main[row_idx, 0])
            whole_layer_gs.axis('off')
            
            whole_layer_fig = fig.add_subfigure(gs_main[row_idx, 0])
            
            if whole_cam_data:
                # Single whole layer visualization
                whole_ax = whole_layer_fig.subplots(1, 1)
                overlay = show_cam_on_image(img_np, whole_cam_data['cam'], use_rgb=True)
                whole_ax.imshow(overlay)
                
                layer_short = whole_cam_data['name'].split('.')[-1] if '.' in whole_cam_data['name'] else whole_cam_data['name']
                whole_ax.set_title(f'WHOLE LAYER: {layer_short} ', 
                                  fontsize=14, fontweight='bold', pad=15)
                whole_ax.axis('off')
                
                # Add colored border for whole layer
                for spine in whole_ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(4)
                    spine.set_visible(True)
                
                whole_layer_fig.suptitle(f'{level.upper()} LEVEL - COMPLETE LAYER VIEW', 
                                        fontsize=16, fontweight='bold', y=0.95)
            
            row_idx += 1
            
            # Individual filters row
            if level_filters:
                filters_gs = fig.add_subplot(gs_main[row_idx, 0])
                filters_gs.axis('off')
                
                filters_fig = fig.add_subfigure(gs_main[row_idx, 0])
                
                num_filters = len(level_filters)
                filter_axes = filters_fig.subplots(1, num_filters)
                
                if num_filters == 1:
                    filter_axes = [filter_axes]
                
                # Plot individual filters
                for i, filter_data in enumerate(level_filters):
                    cam_np = filter_data['cam'][0].detach().cpu().numpy()
                    
                    # Resize CAM to match image
                    if cam_np.shape != img_np.shape[:2]:
                        cam_np = cv2.resize(cam_np, (img_np.shape[1], img_np.shape[0]))
                    
                    # Create overlay
                    overlay = show_cam_on_image(img_np, cam_np, use_rgb=True)
                    filter_axes[i].imshow(overlay)
                    
                    # Cleaner title with better spacing
                    part_context = self._get_part_whole_context(level, filter_data['rank'])
                    filter_axes[i].set_title(f'Filter #{filter_data["filter_id"]}\n{part_context}\nImp: {filter_data["importance"]:.3f}', 
                                           fontsize=12, pad=10)
                    filter_axes[i].axis('off')
                
                filters_fig.suptitle(f'{level.upper()} LEVEL - INDIVIDUAL PARTS: {level_descriptions[level]}', 
                                   fontsize=16, fontweight='bold', y=0.95)
            
            row_idx += 1
        
        # Main title with better spacing
        plt.suptitle(f'Hierarchical Part-Whole Analysis (Sample {sample_num})', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # FIXED: Replace tight_layout with manual layout adjustment for subfigures
        # tight_layout() doesn't work well with complex subfigure layouts
        # Use manual spacing instead
        plt.subplots_adjust(top=0.94, bottom=0.02, hspace=0.25, wspace=0.1)
        
        # Save visualization
        save_path = os.path.join(output_dir, f'part_whole_hierarchy_{sample_num:03d}_class{true_class}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved improved part-whole visualization: {save_path}")

    def _create_detailed_filter_analysis(self, input_tensor, filter_cams, output, target,
                                       sample_num, output_dir):
        """Create detailed analysis with improved layout and single whole layer CAM per level"""
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
        
        # Organize by hierarchy level and select ONE representative layer per level
        level_data = {'shallow': None, 'mid': None, 'deep': None}
        
        # Select the most representative layer for each level (first one found)
        for layer_name, layer_data in filter_cams.items():
            level = layer_data['hierarchy_level']
            if level in level_data and level_data[level] is None:
                level_data[level] = {
                    'name': layer_name,
                    'filters': layer_data['filter_cams'],
                    'whole_layer_cam': layer_data['whole_layer_cam'],
          
                }
        
        # Create detailed analysis for each level
        for level, data in level_data.items():
            if data is None:
                continue
                
            filters = data['filters']
            whole_layer_cam = data['whole_layer_cam']
            
            if not filters and whole_layer_cam is None:
                continue
            
            # Calculate layout: whole layer + top filters
            max_filters = 5  # Limit for better layout
            selected_filters = filters[:max_filters] if filters else []
            total_items = len(selected_filters) + (1 if whole_layer_cam is not None else 0)
            
            if total_items == 0:
                continue
            
            # Create optimal grid layout with extra height for titles
            cols = min(3, total_items)  # Max 3 columns for readability
            rows = (total_items + cols - 1) // cols
            
            # FIXED: Increase figure height to accommodate all text
            fig_height = max(rows * 6, 8)  # More height for proper text spacing
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, fig_height))
            
            # Handle different subplot configurations
            if total_items == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
                if not isinstance(axes, list):
                    axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            item_idx = 0
            
            # Plot whole layer CAM first (if available)
            if whole_layer_cam is not None and item_idx < len(axes):
                whole_cam_np = whole_layer_cam[0].detach().cpu().numpy()
                
                if whole_cam_np.shape != img_np.shape[:2]:
                    whole_cam_np = cv2.resize(whole_cam_np, (img_np.shape[1], img_np.shape[0]))
                
                overlay = show_cam_on_image(img_np, whole_cam_np, use_rgb=True)
                axes[item_idx].imshow(overlay)
                
                # FIXED: Better title formatting with reduced text
                layer_short = data['name'].split('.')[-1] if '.' in data['name'] else data['name']
                axes[item_idx].set_title(f'WHOLE LAYER\n{layer_short}', 
                                        fontsize=12, fontweight='bold', pad=20)  # Increased pad
                axes[item_idx].axis('off')
                
                # Distinctive border for whole layer
                for spine in axes[item_idx].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
                
                item_idx += 1
            
            # Plot top individual filters
            for filter_data in selected_filters:
                if item_idx < len(axes):
                    cam_np = filter_data['cam'][0].detach().cpu().numpy()
                    
                    if cam_np.shape != img_np.shape[:2]:
                        cam_np = cv2.resize(cam_np, (img_np.shape[1], img_np.shape[0]))
                    
                    overlay = show_cam_on_image(img_np, cam_np, use_rgb=True)
                    axes[item_idx].imshow(overlay)
                    
                    # FIXED: Simplified and better positioned titles
                    part_context = self._get_part_whole_context(level, filter_data['rank'])
                    detection_analysis = self._analyze_filter_detection(cam_np, level, filter_data['rank'])
                    
                    # Shorter, cleaner title
                    axes[item_idx].set_title(f'Filter #{filter_data["filter_id"]}\n{part_context}\n{detection_analysis}', 
                                            fontsize=11, pad=20)  # Increased pad
                    axes[item_idx].axis('off')
                    
                    item_idx += 1
            
            # Hide unused subplots
            for i in range(item_idx, len(axes)):
                axes[i].axis('off')
            
            # Level descriptions
            level_descriptions = {
                'shallow': 'Basic Feature Detection',
                'mid': 'Part Composition', 
                'deep': 'Object-level Understanding'
            }
            
            # FIXED: Much shorter main title and proper positioning
            layer_short = data['name'].split('.')[-1] if '.' in data['name'] else data['name']
            
            # Create a more compact title
            main_title = f'{level.upper()} LEVEL: {layer_short} ({level_descriptions[level]})'
            subtitle = f'Sample {sample_num} | True: {class_names[true_class]} | Pred: {class_names[predicted_class]} ({confidence:.3f})'
            
            # FIXED: Position main title higher to avoid overlap
            fig.suptitle(f'{main_title}\n{subtitle}', 
                        fontsize=14, fontweight='bold', y=0.98)  # Higher y position
            
            # FIXED: Better layout adjustment with more top space
            plt.tight_layout()
            plt.subplots_adjust(top=0.85, bottom=0.05, hspace=0.3, wspace=0.2)  # More top space
            
            # Save detailed analysis
            detail_save_path = os.path.join(output_dir, f'detailed_{level}_analysis_{sample_num:03d}.png')
            plt.savefig(detail_save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Saved detailed {level} analysis: {detail_save_path}")

    
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
    
    def _analyze_filter_detection(self, cam_np, level, rank):
        """Analyze what specific part or feature this filter is detecting"""
        # Calculate activation statistics
        max_activation = cam_np.max()
        mean_activation = cam_np.mean()
        std_activation = cam_np.std()
        
        # Determine activation pattern
        if max_activation > 0.8:
            intensity = "Strong"
        elif max_activation > 0.5:
            intensity = "Moderate"
        else:
            intensity = "Weak"
        
        # Spatial analysis
        activation_area = (cam_np > 0.3 * max_activation).sum() / cam_np.size
        
        if activation_area > 0.5:
            spatial = "Global"
        elif activation_area > 0.2:
            spatial = "Regional" 
        else:
            spatial = "Local"
        
        return f"{intensity} {spatial}"

# Usage example
def run_filter_level_analysis():
    """
    Main function to run filter-level hierarchical analysis
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter-Level Part-Whole Relationship Analysis')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained hyperbolic model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to analyze')
    parser.add_argument('--filters_per_layer', type=int, default=8,
                       help='Number of top filters to visualize per layer')
    parser.add_argument('--output_dir', type=str, default='filter_level_analysis',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run analysis on')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FilterLevelHierarchicalAnalyzer(args.model_path, args.device)
    
    # Load test data
    from utils.initialize import select_dataset
    _, test_loader, _, _, _ = select_dataset(analyzer.args)
    
    # Run filter-level analysis
    analyzer.generate_filter_level_heatmaps(
        test_loader, 
        num_samples=args.num_samples,
        filters_per_layer=args.filters_per_layer,
        output_dir=args.output_dir
    )
    
    print("Filter-level part-whole relationship analysis complete!")

def main():
    parser = argparse.ArgumentParser(description='Complete Hyperbolic Interpretability Analysis')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='complete_hierarchical_analysis_eu')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--analysis_type', type=str, default='filter',
                       choices=['hierarchical', 'filter', 'complete'],
                       help='Type of analysis: hierarchical (image-level), filter (filter-level), complete (both)')
    
    args = parser.parse_args()
    
    if args.analysis_type == 'filter':
        # Run filter-level analysis
        analyzer = FilterLevelHierarchicalAnalyzer(args.model_path, args.device)
        _, test_loader, _, _, _ = select_dataset(analyzer.args)
        analyzer.generate_filter_level_heatmaps(test_loader, args.num_samples, output_dir=args.output_dir)
    
    elif args.analysis_type == 'hierarchical':
        # Run hierarchical analysis
        analyzer = EuclideanInterpreterBase(args.model_path, args.device)
        _, test_loader, _, _, _ = select_dataset(analyzer.args)
        analyzer.generate_hierarchical_analysis(test_loader, args.num_samples, args.output_dir)
    
    elif args.analysis_type == 'complete':
        # Run both analyses
        print("="*100)
        print("RUNNING COMPLETE HIERARCHICAL INTERPRETABILITY ANALYSIS")
        print("="*100)
        
        # Phase 1: Hierarchical Analysis
        print("\nPHASE 1: HIERARCHICAL IMAGE-LEVEL ANALYSIS")
        print("-" * 50)
        analyzer = EuclideanInterpreterBase(args.model_path, args.device)
        _, test_loader, _, _, _ = select_dataset(analyzer.args)
        hierarchical_dir = os.path.join(args.output_dir, 'hierarchical')
        analyzer.generate_hierarchical_analysis(test_loader, args.num_samples, hierarchical_dir)
        
        # Phase 2: Filter-Level Analysis
        print("\nPHASE 2: FILTER-LEVEL PART-WHOLE ANALYSIS")
        print("-" * 50)
        filter_analyzer = FilterLevelHierarchicalAnalyzer(args.model_path, args.device)
        filter_dir = os.path.join(args.output_dir, 'filter_level')
        filter_analyzer.generate_filter_level_heatmaps(test_loader, args.num_samples, output_dir=filter_dir)
        
        print("="*100)
        print(f"COMPLETE ANALYSIS FINISHED! Results saved to:")
        print(f"- Hierarchical: {hierarchical_dir}/")
        print(f"- Filter-level: {filter_dir}/")
        print("="*100)

if __name__ == "__main__":
    main()

