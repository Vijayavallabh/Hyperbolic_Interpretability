import torch.nn as nn
import torch.nn.functional as F
import torch

class NMFModule(nn.Module):
    def __init__(
        self,
        channels: int,
        rank: int,
        iters_train: int = 10,
        iters_eval: int = 10,
        eps: float = 1e-6,
        normalize_w: bool = True,
        per_channel_gate: bool = False,
        gate_init: float = -2.0,  # bias toward identity early
    ):
        super().__init__()
        self.channels = channels
        self.rank = rank
        self.iters_train = iters_train
        self.iters_eval = iters_eval
        self.eps = eps
        self.normalize_w = normalize_w

        self.W_raw = nn.Parameter(torch.randn(channels, rank) * 0.02)

        if per_channel_gate:
            self.gate = nn.Parameter(torch.full((channels, 1, 1), gate_init))
            self.per_channel_gate = True
        else:
            self.gate = nn.Parameter(torch.tensor(gate_init))
            self.per_channel_gate = False
        self.loss_dict = {}
    def _pos(self, x):
        return F.softplus(x) + self.eps

    def _normalize_cols(self, W):
        col_sum = W.sum(dim=0, keepdim=True) + self.eps
        return W / col_sum

    def forward(self, x):
        n, c, h, w = x.shape
        s = h * w

        W = self._pos(self.W_raw)
        if self.normalize_w:
            W = self._normalize_cols(W)
        Wt = W.transpose(0, 1)

        G = Wt @ W  

        # Fix: Reshape x to (c, n*s) for correct NMF matrix multiplication
        V = x.permute(1, 0, 2, 3).reshape(c, n * s)
        numerator = torch.matmul(Wt, V)  
        H = numerator.clamp_min(self.eps)

        iters = self.iters_train if self.training else self.iters_eval
        for _ in range(max(iters, 0)):
            # Fix: Remove unsqueeze(0) to match standard NMF update
            denom = torch.matmul(G, H).clamp_min(self.eps)
            H = (H * (numerator / denom)).clamp_min(self.eps)

        V_hat = torch.matmul(W, H).reshape(c, n, h, w).permute(1, 0, 2, 3)
        alpha = torch.sigmoid(self.gate)
        y = alpha * V_hat + (1.0 - alpha) * x

        rec_loss = F.mse_loss(V_hat, x)
        sparse_loss = H.abs().mean()

        G = W.t() @ W 
        off_diag = G - torch.diag(torch.diag(G))
        orth_loss = (off_diag.pow(2)).mean()

        self.loss_dict['rec_loss'] = rec_loss
        self.loss_dict['sparse_loss'] = sparse_loss
        self.loss_dict['orth_loss'] = orth_loss
        return y


    
    

# Also need to add BasicBlock for Euclidean ResNet with k-sparse support
class BasicBlock(nn.Module):
    """Basic Block for Euclidean ResNet-10, -18 and -34"""
    
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, bias=True, nmf=False, 
                 constraint_type='abs', k_sparse_layers=None, stage_idx=0, block_idx=0, 
                 rank_ratio=0.35, iters_train=5, iters_eval=5, 
                 per_channel_gate=True, gate_init=-2.0, **kwargs):
        super(BasicBlock, self).__init__()
        
        self.nmf = nmf


        self.constraint_type = constraint_type
        self.k_sparse_layers = k_sparse_layers
        self.stage_idx = stage_idx
        self.block_idx = block_idx

        if nmf:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
            self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second conv layer
        if nmf:
            self.conv2 = nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=bias)
            self.bn2 = nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=bias)
            self.bn2 = nn.BatchNorm2d(out_channels * BasicBlock.expansion)

        self.shortcut = nn.Sequential()
        self.has_shortcut_conv = False

        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            if nmf:
                self.shortcut_conv = nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=bias)
                self.shortcut_bn = nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            else:
                self.shortcut_conv = nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=bias)
                self.shortcut_bn = nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            self.has_shortcut_conv = True

        # ...existing initialization...

        if nmf:
            rank = max(1, int(out_channels * BasicBlock.expansion * rank_ratio))
            self.nmf_module = NMFModule(
                channels=out_channels * BasicBlock.expansion, 
                rank=rank,
                iters_train=iters_train,
                iters_eval=iters_eval,
                per_channel_gate=per_channel_gate,
                gate_init=gate_init
            )



    def forward(self, x):
        identity = x

        # First conv + bn + relu
        out = self.conv1(x)
        
        # Apply k-sparse after first conv if enabled
        if self.k_sparse_layers is not None:
            conv1_name = f'conv{self.stage_idx+2}_block{self.block_idx}_conv1'
            if conv1_name in self.k_sparse_layers and self.training:
                out = self.k_sparse_layers[conv1_name](out)
        
        out = self.bn1(out)
        out = self.relu1(out)

        # Second conv + bn
        out = self.conv2(out)
        
        # Apply k-sparse after second conv if enabled
        if self.k_sparse_layers is not None:
            conv2_name = f'conv{self.stage_idx+2}_block{self.block_idx}_conv2'
            if conv2_name in self.k_sparse_layers and self.training:
                out = self.k_sparse_layers[conv2_name](out)
        
        out = self.bn2(out)

        # Shortcut connection
        if self.has_shortcut_conv:
            shortcut_out = self.shortcut_conv(identity)
            
            # Apply k-sparse after shortcut conv if enabled
            if self.k_sparse_layers is not None:
                shortcut_name = f'conv{self.stage_idx+2}_block{self.block_idx}_shortcut'
                if shortcut_name in self.k_sparse_layers and self.training:
                    shortcut_out = self.k_sparse_layers[shortcut_name](shortcut_out)
            
            shortcut_out = self.shortcut_bn(shortcut_out)
        else:
            shortcut_out = identity

        # Residual connection
        out += shortcut_out
        out = F.relu(out)

        if self.nmf:
            out = self.nmf_module(out)
        return out



class Bottleneck(nn.Module):
    """ Residual block for ResNet with > 50 layers """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, bias=False, nmf=False, 
                 rank_ratio=0.35, iters_train=5, iters_eval=5, 
                 per_channel_gate=True, gate_init=-2.0, **kwargs):
        super(Bottleneck, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.nmf = nmf
        self.rank_ratio = rank_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels * Bottleneck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )
        
        if nmf:
            rank = max(1, int(out_channels * Bottleneck.expansion * rank_ratio))
            self.nmf_module = NMFModule(
                channels=out_channels * Bottleneck.expansion, 
                rank=rank,
                iters_train=iters_train,
                iters_eval=iters_eval,
                per_channel_gate=per_channel_gate,
                gate_init=gate_init
            )



    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        out = out + res

        out = self.activation(out)

        if self.nmf:
            out = self.nmf_module(out)

        return out
