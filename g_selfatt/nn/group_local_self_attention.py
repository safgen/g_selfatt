import math

import einops
import torch
import torch.nn as nn

import g_selfatt.nn
from g_selfatt.groups import Group
from g_selfatt.nn.activations import Swish
from g_selfatt.utils import normalize_tensor_one_minone


class GroupLocalSelfAttention(nn.Module):
    def __init__(
        self,
        group: Group,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_heads: int,
        patch_size: int,
        attention_dropout_rate: float,
    ):
        """
        Creates a group self-attention layer with local receptive fields.

        The dimension of the positional encoding is given by mid_channels // 3, where mid_channels // 3 are used to represent
        the row embedding, mid_channels // 3 channels are used to represent the col embedding, and (mid_channels - 2 * mid_channels // 3)
        are used to represent the embedding of the group element (e.g., rotation) .

        Args:
            group: The group to be used, e.g., rotations.
            in_channels:  Number of channels in the input signal
            mid_channels: Number of channels of the hidden representation on which attention is performed.
            out_channels: Number of channels in the output signal
            num_heads: Number of heads in the operation
            patch_size: The maximum size of the positional embedding to use.
            attention_dropout_rate: Dropout applied to the resulting attention coefficients.
        """
        super().__init__()

        # Define self parameters
        self.group = group
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        # Define embeddings
        self.pos_encoding_dim = mid_channels // 3
        self.group_encoding_dim = mid_channels - 2 * self.pos_encoding_dim

        self.row_embedding = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1),
            g_selfatt.nn.LayerNorm(num_channels=16),
            Swish(),
            torch.nn.Conv2d(in_channels=16, out_channels=self.pos_encoding_dim, kernel_size=1),
        )
        self.col_embedding = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1),
            g_selfatt.nn.LayerNorm(num_channels=16),
            Swish(),
            torch.nn.Conv2d(in_channels=16, out_channels=self.pos_encoding_dim, kernel_size=1),
        )
        self.group_embedding = nn.Embedding(self.group.num_elements, self.group_encoding_dim)

        # Create the relative position indices for each element of the group.
        self.initialize_indices(patch_size)

        # Define linears using convolution 1x1.
        self.query = nn.Conv3d(in_channels, mid_channels * num_heads, kernel_size=1)
        self.key = nn.Conv3d(in_channels, mid_channels * num_heads, kernel_size=1)
        self.value = nn.Conv3d(in_channels, mid_channels * num_heads, kernel_size=1)
        self.wout = nn.Conv3d(mid_channels * num_heads, out_channels, kernel_size=1)

        # Patch extractor.
        self.unfold = nn.Unfold(
            kernel_size=(self.patch_size, self.patch_size),
            padding=(self.patch_size // 2, self.patch_size // 2),
        )

        # Define dropout
        self.attention_dropout = nn.Dropout(attention_dropout_rate)

        # Define dummy variable to handle values outside of image
        self.inf_outside = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        b, c, g, h, w = x.shape

        # Compute attention scores
        att_scores = self.compute_attention_scores(x)

        # Normalize to obtain probabilities.
        shape = att_scores.shape
        att_probs = self.attention_dropout(
            nn.Softmax(dim=-1)(att_scores.view(*shape[:-3], -1)).view(shape)
        )

        # Compute value projections.
        v = self.unfold(self.value(x).view(b, -1, h, w)).view(
            b, self.mid_channels, self.num_heads, g, self.patch_size, self.patch_size, h, w
        )

        # Re-weight values via attention and map to output dimension.
        v = torch.einsum("bhvgijmkl,bchmklij->bchvij", att_probs, v)
        out = self.wout(
            v.contiguous().view(
                b, self.mid_channels * self.num_heads, self.group.num_elements, h, w
            )
        )

        return out

    def compute_attention_scores(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, dim_input, g, height, width = x.shape
        sqrt_normalizer = math.sqrt(dim_input)

        # Compute query and key data
        q = self.query(x).view(batch_size, self.mid_channels, self.num_heads, g, height, width)

        # -- batch_size, mid_channels, num_heads, g, patch_size, patch_size, height, width
        k_local = self.unfold(self.key(x).view(batch_size, -1, height, width)).view(
            batch_size,
            self.mid_channels,
            self.num_heads,
            g,
            self.patch_size,
            self.patch_size,
            height,
            width,
        )

        # Compute attention scores based on data.
        attention_content_scores = torch.einsum("bchgij,bchmklij->bhgijmkl", q, k_local).unsqueeze(
            2
        )

        # Compute attention scores based on position
        # # -- batch, width, height, num_attention_heads, dim // 3
        q_row = q[:, : self.pos_encoding_dim, :, :, :, :]
        q_col = q[:, self.pos_encoding_dim : 2 * self.pos_encoding_dim, :, :, :, :]
        q_group = q[:, 2 * self.pos_encoding_dim :, :, :, :, :]

        row_scores = torch.einsum(
            "bchgij,vklc->bhvgijkl",
            q_row,
            self.row_embedding(self.row_indices.view(-1, 1, 1, 1)).view(
                self.row_indices.shape + (-1,)
            ),
        )
        col_scores = torch.einsum(
            "bchgij,vklc->bhvgijkl",
            q_col,
            self.col_embedding(self.col_indices.view(-1, 1, 1, 1)).view(
                self.col_indices.shape + (-1,)
            ),
        )
        g_scores = torch.einsum(
            "bchgij,vgmc->bhvgijm",
            q_group,
            self.group_embedding(self.g_indices.view(-1)).view(self.g_indices.shape + (-1,)),
        )

        attention_scores = (
            row_scores.unsqueeze(-3)
            + col_scores.unsqueeze(-3)
            + g_scores.unsqueeze(-1).unsqueeze(-1)
        )
        # print (g_scores.shape, row_scores.shape, col_scores.shape)
        # attention_scores = g_scores
        
        # Combine attention scores
        attention_scores = (
            attention_scores + attention_content_scores.expand_as(attention_scores)
        ) / sqrt_normalizer
        # print(attention_scores.shape, attention_content_scores.shape)
        # Handle attention scores outside of ima2ge
        self.handle_values_outside_image(attention_scores, height, width)

        # Return attention scores
        return attention_scores

    def handle_values_outside_image(
        self,
        attention_scores: torch.Tensor,
        height: int,
        width: int,
    ):
        """
        Set (un-normalized) attention scores outside of the image to -inf for the softmax to set these probabilities to 0.

        Args:
            attention_scores: the calculated unn-normalized attention coefficients
            height: the height of the image, e.g., 28 for MNIST
            width: the width of the image, e.g., 28 for MNIST
        """
        if self.inf_outside is None:
            zeros_outside = self.unfold(
                torch.ones((1, 1, height, width), device=attention_scores.device)
            ).view(self.patch_size, self.patch_size, height, width)
            ones_outside = 1 - zeros_outside
            inf_outside = ones_outside
            inf_outside[ones_outside == 1.0] = float("Inf")
            inf_outside = einops.rearrange(inf_outside, "ph pw h w -> () () () () h w () ph pw")
            self.inf_outside = inf_outside

        attention_scores -= self.inf_outside

    def initialize_indices(
        self,
        patch_size: int,
    ):
        """
        Creates a set of acted relative positions for each of the elements in the group.

        Args:
            patch_size: the size of the patch for which attention coefficients are calculated.
        """

        #  Create 2D relative indices
        indices_1d = normalize_tensor_one_minone(torch.arange(patch_size))
        # -- height, width, 2
        indices = torch.stack(
            [
                indices_1d.view(-1, 1).repeat(1, patch_size),
                indices_1d.view(1, -1).repeat(patch_size, 1),
            ],
            dim=0,
        )
        # Group indices
        indices_g = self.group.relative_positions

        # Get acted versions of the positional encoding
        row_indices = []
        col_indices = []
        g_indices = []

        for i, h in enumerate(self.group.elements):
            Lh_indices_Rd, Lh_indices_g = self.group.left_action_on_G(h, indices, indices_g)
            assert Lh_indices_Rd.shape == (2, patch_size, patch_size)
            row_indices.append(Lh_indices_Rd[0])
            col_indices.append(Lh_indices_Rd[1])
            g_indices.append(Lh_indices_g)

        self.register_buffer("row_indices", torch.stack(row_indices, dim=0))
        self.register_buffer("col_indices", torch.stack(col_indices, dim=0))
        self.register_buffer("g_indices", torch.stack(g_indices, dim=0))
