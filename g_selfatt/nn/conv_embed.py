
import torch.nn as nn
from einops import rearrange
from g_selfatt.utils import _ntuple
from e2cnn import gspaces, nn as enn

to_2tuple = _ntuple(2)

from e2cnn import gspaces, nn as enn

class GroupEquivariantPatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, patch_size=7, stride=4, rotations=4):
        """
        Group Equivariant Patch Embedding using Group Equivariant Convolution (G-Conv2D).
        
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            embed_dim (int): Output embedding dimension.
            patch_size (int): Kernel size for G-Conv.
            stride (int): Stride for downsampling.
            rotations (int): Number of rotation symmetries (e.g., 8 for C8 group equivariance).
        """
        super().__init__()
        
        # Define the rotational symmetry group (C_N for N-fold rotation equivariance)
        self.r2_act = gspaces.Rot2dOnR2(N=rotations)
        
        # Input field: Standard image (trivial representation)
        in_type = enn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])
        
        # Output field: G-steerable feature maps
        out_type = enn.FieldType(self.r2_act, embed_dim * [self.r2_act.regular_repr])
        
        # Group Equivariant Convolutional Layer (Patch Extraction)
        self.proj = enn.R2Conv(in_type, out_type, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        
        # Activation function (ReLU applied to the group representation)
        self.non_linearity = enn.ReLU(out_type)
        
    def forward(self, x):
        # Convert input tensor to an enn.GeometricTensor
        x = enn.GeometricTensor(x, self.proj.in_type)
        
        # Apply equivariant convolution
        x = self.proj(x)
        x = self.non_linearity(x)
        
        # Keep the spatial grid format instead of flattening
        return x.tensor  # (B, embed_dim, H', W') where H' and W' depend on patch size & stride

class ConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x