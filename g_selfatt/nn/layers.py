import torch
import torch.nn as nn
from escnn import gspaces, nn as enn
from escnn.nn import FieldType, GeometricTensor

class E2Linear(nn.Module):
    def __init__(self, in_features, out_features, N=8, bias=False):
        super(E2Linear, self).__init__()

        # Define the group space for 2D rotations
        # Correctly access Rot2dOnR2 from its submodule
        self.gspace = gspaces.rot2dOnR2(N=8) # N=-1 for continuous rotations

        # Define the input and output field types
        # We'll use trivial representations for a standard 'linear' mapping between channels
        self.in_type = FieldType(self.gspace, [self.gspace.trivial_repr] * in_features)
        self.out_type = FieldType(self.gspace, [self.gspace.trivial_repr] * out_features)

        # Initialize the R2Conv layer with a 1x1 kernel
        # This acts like a channel-wise linear transformation at each spatial location
        self.conv_layer = enn.R2Conv(
            self.in_type,
            self.out_type,
            kernel_size=1,
            padding=0, # 1x1 kernel needs 0 padding
            bias=bias  # Standard linear layers have bias
        )

    def forward(self, x):
        # x is expected to be a standard PyTorch Tensor (batch, channels, H, W)

        # Wrap the input tensor into a GeometricTensor, specifying its field type
        x_geometric = GeometricTensor(x, self.in_type)

        # Apply the R2Conv layer
        y_geometric = self.conv_layer(x_geometric)

        # Extract the tensor from the output GeometricTensor
        y = y_geometric.tensor

        return y

def Conv2d1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = False,
) -> torch.nn.Module:
    """
    Implements a point-wise convolution for 2d images, i.e., kernel_size=1x1.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def Conv3d1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = False,
) -> torch.nn.Module:
    """
    Implements a point-wise convolution for signals in the group, i.e., kernel_size=1x1x1.
    """
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


class LayerNorm(nn.Module):
    def __init__(
        self,
        num_channels,
        eps=1e-12,
    ):
        """Uses GroupNorm implementation with group=1 for speed reason."""
        super(LayerNorm, self).__init__()
        # we use GroupNorm to implement this efficiently and fast.
        self.layer_norm = torch.nn.GroupNorm(1, num_channels=num_channels, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)
