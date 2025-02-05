import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn.gspaces import rot2dOnR2
import escnn.nn as enn

import torch
import torch.nn as nn
from escnn.nn import FieldType, GeometricTensor, R2Conv, SequentialModule


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Define a GSpace for rotations and translations in 2D
        self.r2_act = rot2dOnR2(N=4)  # Example with N=4 for 4-fold rotation symmetry

        # Define the input type for the R2Conv layer
        self.input_type = FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])

        # Define the output type for the R2Conv layer
        self.output_type = FieldType(self.r2_act, embed_dim * [self.r2_act.regular_repr])

        # Define the equivariant convolutional layer
        self.conv = R2Conv(self.input_type, self.output_type, kernel_size=patch_size, stride=patch_size)

        # Flatten the patches
        self.flatten = nn.Flatten(2)

    def forward(self, x):
        # Check input dimensions
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Input image size must match the specified image size"

        # Convert to GeometricTensor
        x = GeometricTensor(x, self.input_type)
        print(x.shape)
        # Apply the equivariant convolution
        x = self.conv(x)
        print(x.shape)
        # Convert back to torch.Tensor and flatten the patches
        x = x.tensor
        print(x.shape)
        x = x.view(B, self.embed_dim, -1)
        x = self.flatten(x)
        return x

# Example usage
img_size = 224
patch_size = 16
in_channels = 3
embed_dim = 768

patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1
output = patch_embedding(dummy_input)

print("Output shape:", output.shape)  # Should be (1, embed_dim, num_patches)
