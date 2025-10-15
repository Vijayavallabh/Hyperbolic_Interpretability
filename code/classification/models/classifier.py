import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzMLR
# Remove duplicate import
# from lib.lorentz.manifold import CustomLorentz

from lib.models.resnet import hybrid_resnet18

from lib.models.resnet import (
    resnet18,resnet50,resnet34,
    Lorentz_resnet18,Lorentz_resnet50,Lorentz_resnet34
    
)

EUCLIDEAN_RESNET_MODEL = {
    18: resnet18,50:resnet50,34:resnet34
}


LORENTZ_RESNET_MODEL = {
    18: Lorentz_resnet18,
    50:Lorentz_resnet50,34:Lorentz_resnet34
}


EUCLIDEAN_DECODER = {
    'mlr' : nn.Linear
}

LORENTZ_DECODER = {
    'mlr' : LorentzMLR
}


HYBRID_RESNET_MODEL = {
    18: hybrid_resnet18
}

RESNET_MODEL = {
    "euclidean": EUCLIDEAN_RESNET_MODEL,
    "lorentz": LORENTZ_RESNET_MODEL,
    "hybrid": HYBRID_RESNET_MODEL
}

class ResNetClassifier(nn.Module):
    def __init__(self, 
            num_layers:int, 
            enc_type:str="lorentz", 
            dec_type:str="lorentz",
            enc_kwargs={},
            dec_kwargs={},
            enable_sparsity=True,
            sparsity_weight=0.01,
            sparsity_type='l1',
            sparsity_schedule='constant',
            k_ratio=0.1,
            apply_k_sparse_to='final_layer',
            # NMF parameters
            enable_nmf=False,
            rank_ratio=0.35,
            iters_train=5,
            iters_eval=5,
            per_channel_gate=True,
            gate_init=-2.0,
            space="hyperbolic",
            nmf_apply_to="all_blocks",
        ):
        super(ResNetClassifier, self).__init__()
        self.enc_type = enc_type
        self.dec_type = dec_type
        self.enable_sparsity = enable_sparsity
        self.enable_nmf = enable_nmf

        # Fix: Add safety check for clip_r
        self.clip_r = dec_kwargs.get('clip_r', 1.0)

        # Fix: Use dictionary copy to avoid mutable default argument issues
        enc_kwargs = enc_kwargs.copy()
        dec_kwargs = dec_kwargs.copy()

        # Add sparsity parameters to encoder kwargs for all models that support it
        if enable_sparsity:
            enc_kwargs.update({
                'enable_sparsity': enable_sparsity,
                'sparsity_weight': sparsity_weight,
                'sparsity_type': sparsity_type,
                'sparsity_schedule': sparsity_schedule,
                'k_ratio': k_ratio,
                'apply_k_sparse_to': apply_k_sparse_to,
            })

        # Add NMF parameters to encoder kwargs - THIS IS THE CRITICAL FIX
        if enable_nmf:
            enc_kwargs.update({
                'enable_nmf': enable_nmf,
                'rank_ratio': rank_ratio,
                'iters_train': iters_train,
                'iters_eval': iters_eval,
                'per_channel_gate': per_channel_gate,
                'gate_init': gate_init,
                'space': space,
                'nmf_apply_to': nmf_apply_to,
            })


        self.encoder = RESNET_MODEL[enc_type][num_layers](remove_linear=True, **enc_kwargs)
        self.enc_manifold = self.encoder.manifold if hasattr(self.encoder, 'manifold') else None

        self.dec_manifold = None
        # Fix: Add safety check for expansion attribute
        expansion = getattr(self.encoder, 'expansion', 1)
        if hasattr(self.encoder, 'block') and hasattr(self.encoder.block, 'expansion'):
            expansion = self.encoder.block.expansion
        dec_kwargs['embed_dim'] *= expansion
        
        if dec_type == "euclidean":
            self.decoder = EUCLIDEAN_DECODER[dec_kwargs['type']](dec_kwargs['embed_dim'], dec_kwargs['num_classes'])
                
        elif dec_type == "lorentz":
            self.dec_manifold = CustomLorentz(k=dec_kwargs["k"], learnable=dec_kwargs['learn_k'])

            self.decoder = LORENTZ_DECODER[dec_kwargs['type']](
                self.dec_manifold, 
                dec_kwargs['embed_dim']+1, 
                dec_kwargs['num_classes'],
            )
        else:
            raise RuntimeError(f"Decoder manifold {dec_type} not available...")

    def set_activation_logger(self, logger):
        """Set activation logger for encoder"""
        if hasattr(self.encoder, 'set_activation_logger'):
            self.encoder.set_activation_logger(logger)
        else:
            print("Warning: Encoder does not support activation logging")

    def check_manifold(self, x):
        if self.enc_type=="euclidean" and self.dec_type=="euclidean":
            pass
        elif self.enc_type=="euclidean" and self.dec_type=="lorentz":
            x_norm = torch.norm(x,dim=-1, keepdim=True)
            x = torch.minimum(torch.ones_like(x_norm), self.clip_r/x_norm)*x # Clipped HNNs
            x = self.dec_manifold.expmap0(F.pad(x, pad=(1,0), value=0))
        elif self.enc_type=="euclidean" and self.dec_type=="poincare":
            x_norm = torch.norm(x,dim=-1, keepdim=True)
            x = torch.minimum(torch.ones_like(x_norm), self.clip_r/x_norm)*x # Clipped HNNs
            x = self.dec_manifold.expmap0(x)
        elif self.enc_type=="lorentz" and self.dec_type=="euclidean":
            x = self.enc_manifold.logmap0(x)[..., 1:]
        # Fix: Add safety check for manifold comparison
        elif (self.enc_manifold is not None and self.dec_manifold is not None and 
              hasattr(self.enc_manifold, 'k') and hasattr(self.dec_manifold, 'k') and
              self.enc_manifold.k != self.dec_manifold.k):
            x =  self.dec_manifold.expmap0(self.enc_manifold.logmap0(x))
        
        return x
    
    def forward(self, x):
        x = self.check_manifold(self.encoder(x))
        x = self.decoder(x)
        return x
        
    def get_sparsity_loss(self):
        """Get total sparsity loss from encoder"""
        if self.enable_sparsity and hasattr(self.encoder, 'get_total_sparsity_loss'):
            return self.encoder.get_total_sparsity_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def get_sparsity_stats(self):
        """Get sparsity statistics from encoder"""
        if self.enable_sparsity and hasattr(self.encoder, 'get_block_sparsity_stats'):
            return self.encoder.get_block_sparsity_stats()
        return {'sparsity_enabled': False}
    
    def update_training_step(self):
        """Update training step for sparsity scheduling"""
        if self.enable_sparsity and hasattr(self.encoder, 'update_training_step'):
            self.encoder.update_training_step()

    # NMF-related methods
    def get_nmf_loss(self):
        """Get total NMF loss from encoder"""
        if self.enable_nmf and hasattr(self.encoder, 'get_total_nmf_loss'):
            return self.encoder.get_total_nmf_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def get_nmf_losses(self):
        """Get detailed NMF losses from encoder"""
        if self.enable_nmf and hasattr(self.encoder, 'get_nmf_losses'):
            return self.encoder.get_nmf_losses()
        return {}

    def set_nmf_space(self, space):
        """Set NMF computation space for encoder"""
        if self.enable_nmf and hasattr(self.encoder, 'set_nmf_space'):
            self.encoder.set_nmf_space(space)

    def get_nmf_stats(self):
        """Get NMF statistics for logging"""
        if not self.enable_nmf:
            return {'nmf_enabled': False}
        
        nmf_losses = self.get_nmf_losses()

        stats = {
            'nmf_enabled': True,
            'nmf_total_loss': self.get_nmf_loss().item(),
        }
        
        if nmf_losses:
            rec_losses = [loss.item() for key, loss in nmf_losses.items() if 'rec' in key]
            sparse_losses = [loss.item() for key, loss in nmf_losses.items() if 'sparse' in key]
            orth_losses = [loss.item() for key, loss in nmf_losses.items() if 'orth' in key]
            
            stats.update({
                'nmf_rec_loss': sum(rec_losses),
                'nmf_sparse_loss': sum(sparse_losses),
                'nmf_orth_loss': sum(orth_losses),
            })

        return stats
            
    def load_state_dict_with_compatibility(self, state_dict, strict=True):
        incompatible_keys = [
            'encoder.tangent_sparsity_layer.manifold.k',
            'encoder.final_k_sparse.temperature',
            'encoder.final_k_sparse.straight_through'
        ]
        
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if not any(incompatible_key in key for incompatible_key in incompatible_keys):
                filtered_state_dict[key] = value
            else:
                print(f"Skipping incompatible key: {key}")
        
        try:
            return super().load_state_dict(filtered_state_dict, strict=strict)
        except RuntimeError as e:
            if "Missing key(s)" in str(e) and strict:
                print("Attempting to load with strict=False due to architecture changes...")
                return super().load_state_dict(filtered_state_dict, strict=False)
            else:
                raise e
  
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **override_kwargs):
        """Load model from checkpoint with compatibility handling"""
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only = False)
        
        # Extract model arguments and state dict
        if 'model_args' in checkpoint:
            model_args = checkpoint['model_args']
            model_state_dict = checkpoint['model_state_dict']
        else:
            # Fallback for older checkpoint format
            model_state_dict = checkpoint
            model_args = {}
        
        # Override arguments if provided
        model_args.update(override_kwargs)
        
        # Create model instance
        model = cls(**model_args)
        
        # Load state dict with compatibility
        model.load_state_dict_with_compatibility(model_state_dict, strict=False)
        
        return model