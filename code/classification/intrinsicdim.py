from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

from sklearn import linear_model
from scipy.stats import pearsonr
import sys
from classification.utils.initialize import select_dataset
from lib.lorentz.manifold import CustomLorentz
from classification.models.classifier import ResNetClassifier
from lib.geoopt.manifolds.euclidean import Euclidean


def compute_intrinsic_dimension(features, nres=20, fraction=0.9, method='euclidean', manifold=None, layer_name='avg_pool'):
    """
    Compute intrinsic dimension of feature representations using multiple resamplings.
    
    Args:
        features: numpy array of shape (n_samples, n_features)
        nres: number of resamplings for error estimation
        fraction: fraction of data used for estimation
        method: distance metric ('euclidean' or 'lorentz')
        manifold: CustomLorentz manifold object (required when method='lorentz')
    
    Returns:
        mean_id: mean intrinsic dimension
        error_id: standard deviation of intrinsic dimension estimates
    """
    print(f"\nProcessing layer: {layer_name}")
    print(f"Input features shape: {features.shape}")
    
    features_clean = features
    
    if len(features_clean) < features.shape[0]:
        print(f"Removed {features.shape - len(features_clean)} samples with NaN/Inf. Using {len(features_clean)} samples.")
    
    # Ensure minimum number of samples
    if len(features_clean) < 10:
        print(f"Insufficient samples after filtering: {len(features_clean)}")
        return np.nan, np.nan
    # Project features to hyperbolic space if using Lorentz method
    if method == 'lorentz':
        if manifold is None:
            raise ValueError("Manifold is required for lorentz method")
        
      
        try:
            features_tensor = torch.tensor(features_clean, dtype=torch.float32,device = 'cuda:0')

            features_clean = manifold.projx(features_tensor).detach().cpu().numpy()
        except Exception as e:
            print(f"Error projecting to manifold for {layer_name}: {e}")
            return np.nan, np.nan
            
    elif method == 'tangent':
        if manifold is None:
            raise ValueError("Manifold is required for tangent method")
        

        try:
            features_tensor = torch.tensor(features_clean, dtype=torch.float32, device='cuda:0')
            projected = manifold.logmap0(manifold.projx(features_tensor))
            
            # Check for NaN/Inf in projection
            if torch.any(torch.isnan(projected)) or torch.any(torch.isinf(projected)):
                print(f"Tangent projection introduced NaN/Inf for {layer_name}")
                return np.nan, np.nan
                
            features_clean = projected.detach().cpu().numpy()
            print(f"Projected features to tangent space for {layer_name}.")
        except Exception as e:
            print(f"Error projecting to tangent space for {layer_name}: {e}")
            return np.nan, np.nan

    # Final check for NaN/Inf after projection
    if np.any(~np.isfinite(features_clean)):
        print(f"Features contain NaN/Inf after projection for {layer_name}")
        return np.nan, np.nan
    # Compute distance matrix
    if method == 'euclidean':

        dist = squareform(pdist(features_clean, 'euclidean'))

    
    elif method == 'tangent':
        aaa = Euclidean()
        def euclidean_distance_func(u, v):
            # Convert numpy arrays to tensors
            u_tensor = torch.tensor(u, dtype=torch.float32,device = 'cuda:0')
            v_tensor = torch.tensor(v, dtype=torch.float32,device = 'cuda:0')
            
            # Compute distance using manifold with error handling
            dist_val = aaa.dist(u_tensor, v_tensor)
            
            # Handle tensor with multiple values - take scalar value
            if dist_val.numel() > 1:
                dist_val = dist_val.sum()
            
            # Convert to scalar before checking
            dist_scalar = dist_val.item()
                
            # Check for NaN/Inf
            if np.isnan(dist_scalar) or np.isinf(dist_scalar):
                return 1e-10  # Return small positive value instead of NaN
            
            return max(dist_scalar, 1e-10)  # Ensure positive distance
            
        
        dist = np.zeros((features_clean.shape[0], features_clean.shape[0]))
        for i in range(features_clean.shape[0]):
            for j in range(i+1, features_clean.shape[0]):
                dist[i,j] = euclidean_distance_func(features_clean[i], features_clean[j])
                dist[j,i] = dist[i,j]

    elif method == 'lorentz':
        def lorentz_distance_func(u, v):
            # Convert numpy arrays to tensors
            u_tensor = torch.tensor(u, dtype=torch.float32,device = 'cuda:0')
            v_tensor = torch.tensor(v, dtype=torch.float32,device = 'cuda:0')
            
            # Compute distance using manifold with error handling
            dist_val = manifold.dist(u_tensor, v_tensor)
            
            # Handle tensor with multiple values - take scalar value
            if dist_val.numel() > 1:
                dist_val = dist_val.sum()
            
            # Convert to scalar before checking
            dist_scalar = dist_val.item()
                
            # Check for NaN/Inf
            if np.isnan(dist_scalar) or np.isinf(dist_scalar):
                return 1e-10  # Return small positive value instead of NaN
            
            return max(dist_scalar, 1e-10)  # Ensure positive distance
                
        
        try:
            # Write this in a nested for loop
            dist = np.zeros((features_clean.shape[0], features_clean.shape[0]))
            for i in range(features_clean.shape[0]):
                for j in range(i+1, features_clean.shape[0]):
                    dist[i,j] = lorentz_distance_func(features_clean[i], features_clean[j])
                    dist[j,i] = dist[i,j]
            # dist = squareform(pdist(features_clean, metric=lorentz_distance_func))
            
            # Check distance matrix for issues
            if np.any(~np.isfinite(dist)):
                print(f"Distance matrix contains NaN/Inf for {layer_name}")
                return np.nan, np.nan
                
        except Exception as e:
            print(f"Error computing lorentz distances for {layer_name}: {e}")
            return np.nan, np.nan
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Distance matrix shape: {dist.shape}, min: {np.min(dist):.6f}, max: {np.max(dist):.6f}")
    
    # Multiple resamplings for robust estimation
    ID = []
    n_samples = dist.shape[0]  # Use the actual distance matrix size
    n = max(10, int(np.round(n_samples * fraction)))  # Ensure minimum samples
    n = min(n, n_samples)  # Don't exceed available samples (no -1 needed here)
    
    print(f"Using {n} samples per resampling out of {n_samples} total")
    
    for i in range(nres):
    # Random subsample - FIX: Ensure indices are within bounds
        if n >= n_samples:
            # If we want all samples, just use all indices
            perm = np.arange(n_samples)
        else:
            # Otherwise, randomly select n samples
            perm = np.random.choice(n_samples, size=n, replace=False)
    
    # Extract submatrix
        dist_s = dist[np.ix_(perm, perm)]  # More robust way to extract submatrix
        
        # Estimate intrinsic dimension with enhanced error handling
        try:
            _, _, reg, r, pval = estimate(dist_s, fraction=fraction, verbose=False)
            
            # More stringent validation
            if (not np.isnan(reg) and not np.isinf(reg) and 
                reg > 0 and reg < 1000 and  # Reasonable upper bound
                not np.isnan(r) and abs(r) > 0.1):  # Require decent correlation
                ID.append(reg)
            else:
                if i < 3:  # Only print first few rejections to avoid spam
                    print(f"Rejected estimate {i}: reg={reg}, r={r}, pval={pval}")
                
        except Exception as e:
            if i < 3:  # Only print first few errors to avoid spam
                print(f"Error in estimate for resampling {i}: {e}")
            continue
        
        print(f"Valid estimates: {len(ID)} out of {nres}")
    
    if len(ID) == 0:
        print(f"No valid estimates for {layer_name}")
        return np.nan, np.nan
    elif len(ID) < nres // 3:  # If less than 1/3 of estimates are valid
        print(f"Warning: Only {len(ID)} valid estimates out of {nres} for {layer_name}")
        
    mean_id = np.mean(ID)
    error_id = np.std(ID) if len(ID) > 1 else 0.0
    
    print(f"Final result for {layer_name}: ID = {mean_id:.3f} ± {error_id:.3f}")
    
    return mean_id, error_id


def estimate(X,fraction=0.9,verbose=False):    
    '''
        Estimates the intrinsic dimension of a system of points from
        the matrix of their distances X
        
        Args:
        X : 2-D Matrix X (n,n) where n is the number of points
        fraction : fraction of the data considered for the dimensionality
        estimation (default : fraction = 0.9)

        Returns:            
        x : log(mu)    (*)
        y : -(1-F(mu)) (*)
        reg : linear regression y ~ x structure obtained with scipy.stats.linregress
        (reg.slope is the intrinsic dimension estimate)
        r : determination coefficient of y ~ x
        pval : p-value of y ~ x
            
        (*) See cited paper for description
        
        Usage:
            
        _,_,reg,r,pval = estimate(X,fraction=0.85)
            
        The technique is described in : 
            
        "Estimating the intrinsic dimension of datasets by a 
        minimal neighborhood information"       
        Authors : Elena Facco, Maria d’Errico, Alex Rodriguez & Alessandro Laio        
        Scientific Reports 7, Article number: 12140 (2017)
        doi:10.1038/s41598-017-11873-y
    
    '''             
     
    # sort distance matrix
    Y = np.sort(X,axis=1,kind='quicksort')

    # clean data
    k1 = Y[:,1]
    k2 = Y[:,2]

    zeros = np.where(k1 == 0)[0]
    if verbose:
        print('Found n. {} elements for which r1 = 0'.format(zeros.shape[0]))
        print(zeros)

    degeneracies = np.where(k1 == k2)[0]
    if verbose:
        print('Found n. {} elements for which r1 = r2'.format(degeneracies.shape[0]))
        print(degeneracies)

    good = np.setdiff1d(np.arange(Y.shape[0]), np.array(zeros) )
    good = np.setdiff1d(good,np.array(degeneracies))
    
    if verbose:
        print('Fraction good points: {}'.format(good.shape[0]/Y.shape[0]))
    
    k1 = k1[good]
    k2 = k2[good]    
    
    # n.of points to consider for the linear regression
    npoints = int(np.floor(good.shape[0]*fraction))

    # define mu and Femp
    N = good.shape[0]
    mu = np.sort(np.divide(k2, k1), axis=None,kind='quicksort')
    Femp = (np.arange(1,N+1,dtype=np.float64) )/N
    
    # take logs (leave out the last element because 1-Femp is zero there)
    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])

    # regression
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(x[0:npoints,np.newaxis],y[0:npoints,np.newaxis]) 
    r,pval = pearsonr(x[0:npoints], y[0:npoints])  
    return x,y,regr.coef_[0][0],r,pval


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path=None, num_classes=200):
        super(ResNet18FeatureExtractor, self).__init__()
        
        # Use ResNetClassifier for Euclidean ResNet18
        self.model = ResNetClassifier(
            num_layers=18,
            enc_type="euclidean",
            dec_type="euclidean",
            enc_kwargs={
                'img_dim': [3, 64, 64],
                'embed_dim': 512,
                'num_classes': num_classes,
                'bias': False
            },
            dec_kwargs={
                'embed_dim': 512,
                'num_classes': num_classes,
                'k': 1.0,
                'learn_k': False,
                'type': 'mlr',
                'clip_r': None
            }
        )
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading pretrained Euclidean ResNet18 from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded pretrained Euclidean ResNet18")
        else:
            print("No checkpoint provided for Euclidean ResNet18, using random initialization")
        
        # Store intermediate features
        self.features = {}
        self.layer_names = []
        
        # Register hooks
        self.register_hooks()
    
    def register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output.detach()
            return hook
        
        def get_layers(module, layer_types):
            layers = []
            for name, child in module.named_modules():
                if any(isinstance(child, t) for t in layer_types):
                    layers.append((name, child))
            return layers
        
        # Get all Conv2d layers
        conv_layers = get_layers(self.model, [nn.Conv2d])
        pool_layers = get_layers(self.model, [nn.AvgPool2d, nn.AdaptiveAvgPool2d])
        linear_layers = get_layers(self.model, [nn.Linear])
        
        all_layers = conv_layers + pool_layers + linear_layers
        for i, (name, layer) in enumerate(all_layers):
            layer.register_forward_hook(get_activation(f'layer_{i:02d}_{name}'))
            self.layer_names.append(f'layer_{i:02d}_{name}')
        
        print(f"Total points registered: {len(self.layer_names)}")
    
    def forward(self, x):
        self.features = {}  # Clear previous features
        output = self.model(x)
        return output, self.features


class LorentzResNet18FeatureExtractor(nn.Module):
    def __init__(self, num_classes=200, k=1.0, learn_k=False, checkpoint_path=None):
        super(LorentzResNet18FeatureExtractor, self).__init__()
        self.manifold = CustomLorentz(k=k, learnable=learn_k)
        # Create Lorentz manifold
        self.model = ResNetClassifier(
            num_layers=18,
            enc_type="lorentz",
            dec_type="lorentz",
            enc_kwargs={
                'img_dim': [3, 64, 64],
                'embed_dim': 512,
                'num_classes': num_classes,
                'bias': True
            },
            dec_kwargs={
                'embed_dim': 512,
                'num_classes': num_classes,
                'k': k,
                'learn_k': learn_k,
                'type': 'mlr',
                'clip_r': None
            }
        )
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading pretrained Lorentz ResNet18 from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded pretrained Lorentz ResNet18")
        else:
            print("No checkpoint provided for Lorentz ResNet18, using random initialization")
        
        # Store intermediate features
        self.features = {}
        self.layer_names = []
        
        # Register hooks
        self.register_hooks()
    
    def register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    self.features[name] = output.detach()
                else:
                    self.features[name] = output
            return hook
        
        def get_layers(module, layer_types):
            layers = []
            for name, child in module.named_modules():
                if any(isinstance(child, t) for t in layer_types):
                    layers.append((name, child))
            return layers
        
        # Import Lorentz layers
        from lib.lorentz.layers import LorentzConv2d, LorentzMLR,LorentzGlobalAvgPool2d
        
        # Get all LorentzConv2d, AvgPool2d, and LorentzMLR layers
        conv_layers = get_layers(self.model, [LorentzConv2d])
        pool_layers = get_layers(self.model, [LorentzGlobalAvgPool2d])
        linear_layers = get_layers(self.model, [LorentzMLR])
        
        all_layers = conv_layers + pool_layers + linear_layers
        
        for i, (name, layer) in enumerate(all_layers):
            layer.register_forward_hook(get_activation(f'layer_{i:02d}_{name}'))
            self.layer_names.append(f'layer_{i:02d}_{name}')
        
        print(f"Total Lorentz points registered: {len(self.layer_names)}")
    
    def forward(self, x):
        self.features = {}  # Clear previous features
        output = self.model(x)
        return output, self.features

def main():
    print(os.getcwd())
    _, val_loader, _, img_dim, num_classes = select_dataset(            type('Args', (), {
                'dataset': 'Tiny-ImageNet',
                'batch_size': 128,
                'batch_size_test': 256,
                'validation_split': False
            })()
            )
    dataset = val_loader.dataset
    subset_size = 300
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    subset_dataset = torch.utils.data.Subset(dataset, indices)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Processing {subset_size} samples...")

    # ===== EUCLIDEAN ResNet18 =====
    print("\n" + "="*50)
    print("PROCESSING EUCLIDEAN ResNet18")
    print("="*50)
    
    dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=20, shuffle=False, num_workers=2)
    
    model_euclidean = ResNet18FeatureExtractor('/home/prachh/source_code/iccv/markup/Jithamanyu_dont_open/Vijayavallabh/Hyperbolic_Interpretability/code/classification/output_tiny_euclid_without_nmf/final_E-ResNet18.pth',num_classes=num_classes)
    model_euclidean.to(device)
    model_euclidean.eval()

    print(f"Euclidean model has {len(model_euclidean.layer_names)} points")
    print("Euclidean layer names:")
    for i, name in enumerate(model_euclidean.layer_names):
        print(f"  {i:2d}: {name}")
    
    # Extract features
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(dataloader, desc="Extracting Euclidean features")):
            inputs = inputs.to(device)
            _, features = model_euclidean(inputs)
            
            if i == 0:  # Initialize feature storage
                all_features_euclidean = {name: [] for name in features.keys()}
            
            for layer_name, feature_tensor in features.items():
                flat_features = feature_tensor.view(feature_tensor.size(0), -1)
                all_features_euclidean[layer_name].append(flat_features.cpu())
    
    # Concatenate features
    for layer_name in all_features_euclidean.keys():
        all_features_euclidean[layer_name] = torch.cat(all_features_euclidean[layer_name], dim=0).numpy()
        print(f"Euclidean {layer_name} shape: {all_features_euclidean[layer_name].shape}")
    
    
    # ===== LORENTZ ResNet18 =====
    print("\n" + "="*50)
    print("PROCESSING LORENTZ ResNet18")
    print("="*50)
    
    checkpoint_path = '/home/prachh/source_code/iccv/markup/Jithamanyu_dont_open/Vijayavallabh/Hyperbolic_Interpretability/code/classification/output_tiny_lorentz_without_nmf/best_L-ResNet18.pth'
    model_lorentz = LorentzResNet18FeatureExtractor(
        num_classes=num_classes, 
        k=1.0, 
        learn_k=True,
        checkpoint_path=checkpoint_path
    )
    model_lorentz.to(device)
    model_lorentz.eval()
    print(f"Lorentz model has {len(model_lorentz.layer_names)} points")
    print("Lorentz layer names:")
    for i, name in enumerate(model_lorentz.layer_names):
        print(f"  {i:2d}: {name}")
    
    # Extract features
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(dataloader, desc="Extracting Lorentz features")):
            inputs = inputs.to(device)
            _, features = model_lorentz(inputs)
      
            if i == 0:  # Initialize feature storage
                all_features_lorentz = {name: [] for name in features.keys()}

            for layer_name, feature_tensor in features.items():
                if isinstance(feature_tensor, torch.Tensor):
                    if len(feature_tensor.shape) > 2:
                        flat_features = feature_tensor
                    else:
                        flat_features = feature_tensor
                    all_features_lorentz[layer_name].append(flat_features.cpu())
    
    # Concatenate features
    for layer_name in all_features_lorentz.keys():
        if all_features_lorentz[layer_name]:
            all_features_lorentz[layer_name] = torch.cat(all_features_lorentz[layer_name], dim=0).numpy()
            print(f"Lorentz {layer_name} shape: {all_features_lorentz[layer_name].shape}")
    
    # ===== COMPUTE INTRINSIC DIMENSIONS =====
    print("\n" + "="*50)
    print("COMPUTING INTRINSIC DIMENSIONS")
    print("="*50)
    
    # Euclidean
    euclidean_intrinsic_dims = []
    euclidean_errors = []
    euclidean_valid_layers = []
    euclidean_layer_indices = []
    
    print("\nComputing Euclidean intrinsic dimensions...")
    for idx, layer_name in enumerate(tqdm(model_euclidean.layer_names)):
        if layer_name in all_features_euclidean:
            features = all_features_euclidean[layer_name]
            mean_id, error_id = compute_intrinsic_dimension(
                features,
                method='euclidean',
                layer_name=layer_name
            )
            if not np.isnan(mean_id):
                euclidean_intrinsic_dims.append(mean_id)
                euclidean_errors.append(error_id)
                euclidean_valid_layers.append(layer_name)
                euclidean_layer_indices.append(idx)
                print(f"✓ Euclidean {layer_name}: ID = {mean_id:.2f} ± {error_id:.2f}")
            else:
                print(f"✗ Euclidean {layer_name}: Failed to compute ID")
                
    # Lorentz Tangent
    lorentz_tangent_intrinsic_dims = []
    lorentz_tangent_errors = []
    lorentz_tangent_valid_layers = []
    lorentz_tangent_layer_indices = []
    
    print("\nComputing Lorentz Tangent intrinsic dimensions...")
    for idx, layer_name in enumerate(tqdm(model_lorentz.layer_names)):
        if layer_name in all_features_lorentz:
            features = all_features_lorentz[layer_name]
            mean_id, error_id = compute_intrinsic_dimension(
                features,
                method='tangent',
                manifold=model_lorentz.manifold,
                layer_name=layer_name
            )
            if not np.isnan(mean_id):
                lorentz_tangent_intrinsic_dims.append(mean_id)
                lorentz_tangent_errors.append(error_id)
                lorentz_tangent_valid_layers.append(layer_name)
                lorentz_tangent_layer_indices.append(idx)
                print(f"✓ Lorentz Tangent {layer_name}: ID = {mean_id:.2f} ± {error_id:.2f}")
            else:
                print(f"✗ Lorentz Tangent {layer_name}: Failed to compute ID")
                
    # Lorentz Lorentz
    lorentz_lorentz_intrinsic_dims = []
    lorentz_lorentz_errors = []
    lorentz_lorentz_valid_layers = []
    lorentz_lorentz_layer_indices = []
    
    print("\nComputing Lorentz Lorentz intrinsic dimensions...")
    for idx, layer_name in enumerate(tqdm(model_lorentz.layer_names)):
        if layer_name in all_features_lorentz:
            features = all_features_lorentz[layer_name]
            mean_id, error_id = compute_intrinsic_dimension(
                features,
                method='lorentz',
                manifold=model_lorentz.manifold,
                layer_name=layer_name
            )
            if not np.isnan(mean_id):
                lorentz_lorentz_intrinsic_dims.append(mean_id)
                lorentz_lorentz_errors.append(error_id)
                lorentz_lorentz_valid_layers.append(layer_name)
                lorentz_lorentz_layer_indices.append(idx)
                print(f"✓ Lorentz Lorentz {layer_name}: ID = {mean_id:.2f} ± {error_id:.2f}")
            else:
                print(f"✗ Lorentz Lorentz {layer_name}: Failed to compute ID")
    
    # ===== PLOTTING =====
    print("\n" + "="*50)
    print("CREATING PLOTS")
    print("="*50)
    
    # Plot parameters
    lw = 2.5
    alpha = 1
    ms = 5
    fs = 20
    
    fig = plt.figure(figsize=(18, 10))
    
    # Compute relative depths
    def compute_relative_depths(layer_indices):
        if len(layer_indices) <= 1:
            return [0.0] * len(layer_indices)
        max_idx = max(layer_indices)
        return [idx / max_idx for idx in layer_indices]
    
    euclidean_relative_depths = compute_relative_depths(euclidean_layer_indices)
    lorentz_tangent_relative_depths = compute_relative_depths(lorentz_tangent_layer_indices)
    lorentz_lorentz_relative_depths = compute_relative_depths(lorentz_lorentz_layer_indices)
    
    # Plot Euclidean
    if euclidean_intrinsic_dims:
        plt.plot(euclidean_relative_depths, euclidean_intrinsic_dims, '-ob', 
                 markersize=ms, alpha=alpha, linewidth=lw, label=f'Euclidean ResNet18 ({len(euclidean_valid_layers)} points)')
        plt.errorbar(euclidean_relative_depths, euclidean_intrinsic_dims, yerr=euclidean_errors, 
                     fmt='-b', alpha=alpha, linewidth=lw)
    
    # Plot Lorentz Tangent
    if lorentz_tangent_intrinsic_dims:
        plt.plot(lorentz_tangent_relative_depths, lorentz_tangent_intrinsic_dims, '-sr', 
                 markersize=ms, alpha=alpha, linewidth=lw, label=f'Lorentz Tangent ResNet18 ({len(lorentz_tangent_valid_layers)} points)')
        plt.errorbar(lorentz_tangent_relative_depths, lorentz_tangent_intrinsic_dims, yerr=lorentz_tangent_errors, 
                     fmt='-r', alpha=alpha, linewidth=lw)
    
    # Plot Lorentz Lorentz
    if lorentz_lorentz_intrinsic_dims:
        plt.plot(lorentz_lorentz_relative_depths, lorentz_lorentz_intrinsic_dims, '-^g', 
                 markersize=ms, alpha=alpha, linewidth=lw, label=f'Lorentz Lorentz ResNet18 ({len(lorentz_lorentz_valid_layers)} points)')
        plt.errorbar(lorentz_lorentz_relative_depths, lorentz_lorentz_intrinsic_dims, yerr=lorentz_lorentz_errors, 
                     fmt='-g', alpha=alpha, linewidth=lw)
    
    plt.xlabel('Relative Layer Depth', fontsize=fs)
    plt.ylabel('Intrinsic Dimensionality (ID)', fontsize=fs)
    plt.ylim(bottom=0)
    plt.legend(fontsize=fs)
    plt.xticks(fontsize=fs-2)
    plt.yticks(fontsize=fs)
    plt.grid(True, alpha=0.3)
    
    plt.title(f'Intrinsic Dimensionality vs Relative Layer Depth\nEuclidean vs Lorentz Tangent vs Lorentz Lorentz on Tiny-ImageNet', 
              fontsize=fs+2, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('euclidean_vs_lorentz_tangent_vs_lorentz_1.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plot saved to 'euclidean_vs_lorentz_tangent_vs_lorentz.png'")
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"Euclidean: {len(euclidean_valid_layers)}/{len(model_euclidean.layer_names)} layers computed successfully")
    print(f"Lorentz Tangent: {len(lorentz_tangent_valid_layers)}/{len(model_lorentz.layer_names)} layers computed successfully")
    print(f"Lorentz Lorentz: {len(lorentz_lorentz_valid_layers)}/{len(model_lorentz.layer_names)} layers computed successfully")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()
