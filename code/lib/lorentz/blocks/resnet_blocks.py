import torch.nn as nn
import torch.nn.functional as F

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import (
    LorentzConv2d,
    LorentzBatchNorm2d,
    LorentzReLU,
)
import torch
from lib.lorentz.manifold import CustomLorentz

class NMFModule(nn.Module):

    def __init__(
        self,
        channels: int,
        rank: int,
        iters_train: int = 5,
        iters_eval: int = 5,
        eps: float = 1e-6,
        normalize_w: bool = True,
        per_channel_gate: bool = False,
        gate_init: float = -2.0,
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

        self.manifold = CustomLorentz() 

        self.loss_dict = {}

    def _pos(self, x):
        return F.softplus(x) + self.eps

    def _normalize_cols(self, W):
        col_sum = W.sum(dim=0, keepdim=True) + self.eps
        return W / col_sum

    def forward(self, x_lorentz):
        sp_in =x_lorentz[:,:,:,1:]
        n, h, w ,c = sp_in.shape
        s = h * w

        W = self._pos(self.W_raw)                      
        if self.normalize_w:
            W = self._normalize_cols(W)
        Wt = W.transpose(0, 1)                         
        G = Wt @ W                                    

        V = sp_in.permute(3, 0, 1, 2).reshape(c,n*s)             
        numerator = torch.matmul(Wt, V)               
        H = numerator.clamp_min(self.eps)              

        iters = self.iters_train if self.training else self.iters_eval
        for _ in range(max(iters, 0)):
            denom = torch.matmul(G.unsqueeze(0), H).clamp_min(self.eps)
            H = (H * (numerator / denom)).clamp_min(self.eps)

        V_hat = torch.matmul(W, H).reshape(n,  h, w,c) 


        alpha = torch.sigmoid(self.gate).view(1,1,1,-1)  #[c,1,1]

        y = alpha * V_hat + (1.0 - alpha) * sp_in

        V_hat_t = self.manifold.add_time(V_hat)
        rec_loss = (self.manifold.dist(V_hat_t, x_lorentz)**2).mean()
        sparse_loss = H.abs().mean()

        G = W.t() @ W
        off_diag = G - torch.diag(torch.diag(G))
        orth_loss = (off_diag.pow(2)).mean()

        self.loss_dict['rec_loss'] = rec_loss
        self.loss_dict['sparse_loss'] = sparse_loss
        self.loss_dict['orth_loss'] = orth_loss
        y = self.manifold.add_time(y)
        return y



def get_Conv2d(manifold, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, LFC_normalize=False):
    return LorentzConv2d(
        manifold=manifold, 
        in_channels=in_channels+1, 
        out_channels=out_channels+1, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        bias=bias, 
        LFC_normalize=LFC_normalize,
    )

def get_BatchNorm2d(manifold, num_channels):
    return LorentzBatchNorm2d(manifold=manifold, num_channels=num_channels+1)

def get_Activation(manifold):
    return LorentzReLU(manifold)
    
class LorentzInputBlock(nn.Module):
    """ Input Block of ResNet model """

    def __init__(self, manifold: CustomLorentz, img_dim, in_channels, bias=True):
        super(LorentzInputBlock, self).__init__()

        self.manifold = manifold

        self.conv = nn.Sequential(
            get_Conv2d(
                self.manifold,
                img_dim,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
            get_BatchNorm2d(self.manifold, in_channels),
            get_Activation(self.manifold),
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # Make channel last (bs x H x W x C)
        

        x = self.manifold.projx(F.pad(x, pad=(1, 0)))
        #print("in input block after projection ",self.manifold.inner(None,x,x))
        return self.conv(x)



class LorentzBasicBlock(nn.Module):
    """ Basic Block for Lorentz ResNet-10, -18 and -34 """

    expansion = 1

    def __init__(self, manifold: CustomLorentz, in_channels, out_channels, stride=1, bias=True, 
                 nmf=False, rank_ratio=0.35, iters_train=5, iters_eval=5, 
                 per_channel_gate=True, gate_init=-2.0, **kwargs):
        super(LorentzBasicBlock, self).__init__()

        self.manifold = manifold
        self.nmf = nmf
        self.rank_ratio = rank_ratio
   
        self.activation = get_Activation(self.manifold)

        # First conv layer
        self.conv1 = get_Conv2d(
            self.manifold,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )
        self.bn1 = get_BatchNorm2d(self.manifold, out_channels)
        self.relu1 = get_Activation(self.manifold, )
        
        # Second conv layer
        self.conv2 = get_Conv2d(
            self.manifold,
            out_channels,
            out_channels * LorentzBasicBlock.expansion,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.bn2 = get_BatchNorm2d(self.manifold, out_channels * LorentzBasicBlock.expansion,)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzBasicBlock.expansion:
            self.shortcut_conv = get_Conv2d(
                self.manifold,
                in_channels,
                out_channels * LorentzBasicBlock.expansion,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=bias,
            )
            self.shortcut_bn = get_BatchNorm2d(
                self.manifold, out_channels * LorentzBasicBlock.expansion, 
            )
            self.has_shortcut_conv = True
        else:
            self.has_shortcut_conv = False
        
        if nmf:
            rank = max(1, int(out_channels * LorentzBasicBlock.expansion * rank_ratio))
            #print(f"Creating NMF module for LorentzBasicBlock: channels={out_channels * LorentzBasicBlock.expansion}, rank={rank}")
            self.nmf_module = NMFModule(
                channels=out_channels * LorentzBasicBlock.expansion,
                rank=rank,
                iters_train=iters_train,
                iters_eval=iters_eval,
                per_channel_gate=per_channel_gate,
                gate_init=gate_init
            )
        else:
            self.nmf_module = None


    def forward(self, x):

        identity = x

        # First conv + bn + relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Second conv + bn
        out = self.conv2(out)

        out = self.bn2(out)

        # Shortcut connection
        if self.has_shortcut_conv:
            shortcut_out = self.shortcut_conv(identity)
          
            shortcut_out = self.shortcut_bn(shortcut_out)
        else:
            shortcut_out = identity

        out_space = out.narrow(-1, 1, shortcut_out.shape[-1]-1) + shortcut_out.narrow(-1, 1, shortcut_out.shape[-1]-1)
        
        out = self.manifold.add_time(out_space)
        out = self.activation(out)
        
        if self.nmf:
            out = self.nmf_module(out)


        return out




class LorentzBottleneck(nn.Module):
    """ Residual block for Lorentz ResNet with > 50 layers """

    expansion = 4
    def __init__(self, manifold: CustomLorentz, in_channels, out_channels, stride=1, bias=False, 
                 nmf=False, rank_ratio=0.35, iters_train=5, iters_eval=5, 
                 per_channel_gate=True, gate_init=-2.0, **kwargs):
        super(LorentzBottleneck, self).__init__()

        self.manifold = manifold
        self.nmf = nmf
        self.rank_ratio = rank_ratio

        self.activation = get_Activation(self.manifold)

        # First 1x1 conv
        self.conv1 = get_Conv2d(
            self.manifold,
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=bias,
        )
        self.bn1 = get_BatchNorm2d(self.manifold, out_channels)
        self.relu1 = get_Activation(self.manifold, )
        
        # 3x3 conv
        self.conv2 = get_Conv2d(
            self.manifold,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )
        self.bn2 = get_BatchNorm2d(self.manifold, out_channels)
        self.relu2 = get_Activation(self.manifold)

        # Final 1x1 conv
        self.conv3 = get_Conv2d(
            self.manifold,
            out_channels,
            out_channels * LorentzBottleneck.expansion,
            kernel_size=1,
            padding=0,
            bias=bias,
        )
        self.bn3 = get_BatchNorm2d(self.manifold, out_channels * LorentzBottleneck.expansion)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzBottleneck.expansion:
            self.shortcut_conv = get_Conv2d(
                self.manifold,
                in_channels,
                out_channels * LorentzBottleneck.expansion,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=bias,
            )
            self.shortcut_bn = get_BatchNorm2d(
                self.manifold, out_channels * LorentzBottleneck.expansion,
            )
            self.has_shortcut_conv = True
        else:
            self.has_shortcut_conv = False
        
        
        if self.nmf:
            rank = max(1, int(out_channels * LorentzBottleneck.expansion * self.rank_ratio))
            self.nmf_module = NMFModule(
                channels=out_channels * LorentzBottleneck.expansion, 
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
 
        out = self.bn1(out)
        out = self.relu1(out)

        # Second conv + bn + relu
        out = self.conv2(out)
        
        out = self.bn2(out)
        out = self.relu2(out)

        # Third conv + bn
        out = self.conv3(out)
        
        
        out = self.bn3(out)

        # Shortcut connection
        if self.has_shortcut_conv:
            shortcut_out = self.shortcut_conv(identity)
            

            shortcut_out = self.shortcut_bn(shortcut_out)
        else:
            shortcut_out = identity
        # out = self.manifold.pt_addition(out, shortcut_out) 
        out_space = out.narrow(-1, 1, shortcut_out.shape[-1]-1) + shortcut_out.narrow(-1, 1, shortcut_out.shape[-1]-1)
        out = self.manifold.add_time(out_space)
        out = self.activation(out)

        if self.nmf:
            out = self.nmf_module(out)

        return out
