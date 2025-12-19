import math

import torch
import torch.nn as nn

import g_selfatt.functional.encoding as encoding_functions
import g_selfatt.nn
from g_selfatt.groups import Group
from g_selfatt.nn.activations import Swish
from g_selfatt.utils import normalize_tensor_one_minone
from collections import OrderedDict

import torch.nn.functional as F
from einops import rearrange

from escnn import gspaces, nn as enn

class ESConvAttention(nn.Module):
    def __init__(self,
                 group,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=(1,1),
                 stride_q=1,
                 padding_kv=(1,1),
                 padding_q=1,
                 with_cls_token=False,
                 max_pos_embedding=0,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        self.group = group
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.group_embedding = nn.Embedding(self.group.num_elements, self.dim//self.num_heads)

        # define GeometricTensor types
        self.r2_act = gspaces.rot2dOnR2(N=self.group.num_elements)
        in_type = enn.FieldType(self.r2_act, dim_in * [self.r2_act.regular_repr])
        out_type = enn.FieldType(self.r2_act, dim_out * [self.r2_act.regular_repr])

        self.conv_proj_q = self._build_projection(
            in_type, out_type, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            in_type, out_type, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            in_type, out_type, kernel_size, padding_kv,
            stride_kv, method
        )

        # define GeometricTensor types
        self.r2_act = gspaces.rot2dOnR2(N=self.group.num_elements)
        in_type = enn.FieldType(self.r2_act, dim_in * [self.r2_act.regular_repr])
        out_type = enn.FieldType(self.r2_act, dim_out * [self.r2_act.regular_repr])

        self.proj_q = enn.R2Conv(in_type, out_type, kernel_size=1)
        self.proj_k = enn.R2Conv(in_type, out_type, kernel_size=1)
        self.proj_v = enn.R2Conv(in_type, out_type, kernel_size=1)
        self.proj_out = enn.R2Conv(out_type, out_type, kernel_size=1)

        # self.proj_q = nn.Conv3d(dim_in, dim_out, kernel_size=1)
        # self.proj_k = nn.Conv3d(dim_in, dim_out, kernel_size=1)
        # self.proj_v = nn.Conv3d(dim_in, dim_out, kernel_size=1)
        # self.proj_out = nn.Conv3d(dim_out, dim_out // num_heads, kernel_size=1)

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim_out, dim_out)
        if proj_drop > 0:
            self.proj_drop = nn.Dropout(proj_drop)
        else:
            self.proj_drop = None

        self.initialize_indices(max_pos_embedding)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            # print("dim_in", dim_in, "dim_out", dim_out)
            proj = nn.Sequential(OrderedDict([
                ('conv', enn.R2Conv(
                    dim_in, dim_in, kernel_size=kernel_size, padding=padding, stride=stride, bias=False
                )),
                # ('bn', enn.NormBatchNorm(dim_out)),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool3d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                #('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h*w], 1)

        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # print("type of x", x.type, x.shape)
        # exit()

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)
        
        return q, k, v

    def forward(self, x, h, w):
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)
            # q = rearrange(q, 'b (h c) g x y -> b c h g x y', h=self.num_heads)
            # k = rearrange(k, 'b (h c) g x y -> b c h g x y', h=self.num_heads)
            # v = rearrange(v, 'b (h c) g x y -> b c h g x y', h=self.num_heads)
        
        # else:
        # print(q.shape, k.shape)
        if False:
            q = rearrange(self.proj_q(q), 'b (h c) g x y -> b c h g x y', h=self.num_heads)
            k = rearrange(self.proj_k(k), 'b (h c) g x y -> b c h g x y', h=self.num_heads)
            v = rearrange(self.proj_v(v), 'b (h c) g x y -> b c h g x y', h=self.num_heads)
        
        else:
            q = self.proj_q(q)
            k = self.proj_k(k)
            v = self.proj_v(v)
            q_t = rearrange(q.tensor, 'b (h c g) x y -> b (h g) c x y', h=self.num_heads, g=self.group.num_elements)
            k_t = rearrange(k.tensor, 'b (h c g) x y -> b (h g) c x y', h=self.num_heads, g=self.group.num_elements)
            v_t = rearrange(v.tensor, 'b (h c g) x y -> b (h g) c x y', h=self.num_heads, g=self.group.num_elements)
        # print(q.shape, k.shape)
        q_group = q
        g_embedding = self.group_embedding(self.g_indices.view(-1)).view(self.g_indices.shape + (-1,))
        # print(q_group.shape, g_embedding.shape)
        # g_scores = torch.einsum(                                #REPLACE THIS WITH tensor product for q_group and then turn into a tensor
        #     "bchgij,vgmc->bhvgijm",
        #     q_group,
        #     g_embedding,
        # )
        g_tp = enn.TensorProductModule(
            q.type,
            q.type
        ).to('cuda:0')
        g_scores = g_tp(q.to('cuda:0'))

        print(g_scores.shape, q_t.shape, k_t.flatten(3).shape, v_t.shape)
        b, chg, i, j = q.shape
        # attn_c_score = (torch.einsum('bchgij,bchmkl->bhgijmkl', q, k)).unsqueeze(2) #.repeat(1,1,self.group.num_elements,1,1,1,1,1,1) * self.scalen  # turn q and k into tensor and REPLACE THIS WITH a dot product of q and k  
        attn_c_score = torch.matmul(q_t.flatten(3), k_t.flatten(3).transpose(-2,-1))
        print("duh", g_scores.shape, attn_c_score.shape)
        exit()
        # attn_score = g_scores.tensor.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1, attn_c_score.shape[-2], attn_c_score.shape[-1])         # adjust according to the shape of attn_c_score
        # print(attn_score.shape, attn_c_score.shape)
        # exit()
        attn_score = (
            attn_score + attn_c_score.expand_as(attn_score)
        ) * self.scale

        shape = attn_score.shape
        attn = F.softmax(attn_score.view(*shape[:-2], -1), dim=-1)
        attn = self.attn_drop(attn.view(shape))

        x = torch.einsum('bhvgijmkl,bchmkl->bchvij', attn, v)                     # replace with the right kond of product for attn and v after converting v to a tensor
        x = rearrange(x, 'b c h g i j -> b (c h) g i j')                         # if needed

        x = self.proj_out(x)
        if self.proj_drop is not None:
            x = self.proj_drop(x)

        return x                                                        # return the output as geometric tensor, maybe
    
    def initialize_indices(
        self,
        max_pos_embedding: int,
    ):
        """
        Creates a set of acted relative positions for each of the elements in the group.
        """

        #  Create 2D relative indices
        indices_1d = normalize_tensor_one_minone(torch.arange(2 * max_pos_embedding - 1))
        indices = torch.stack(
            [
                indices_1d.view(-1, 1).repeat(1, 2 * max_pos_embedding - 1),
                indices_1d.view(1, -1).repeat(2 * max_pos_embedding - 1, 1),
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
            # patches = encoding_functions.extract_patches(Lh_indices_Rd)
            # row_indices_windows = patches[..., 0]
            # col_indices_windows = patches[..., 1]
            # row_indices.append(row_indices_windows)
            # col_indices.append(col_indices_windows)
            g_indices.append(Lh_indices_g)

        # self.register_buffer("row_indices", torch.stack(row_indices, dim=0))
        # self.register_buffer("col_indices", torch.stack(col_indices, dim=0))
        self.register_buffer("g_indices", torch.stack(g_indices, dim=0))



class ESGroupConvAttention(nn.Module):
    def __init__(
        self,
        group: Group,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_heads: int,
        max_pos_embedding: int,
        attention_dropout_rate: float,
    ):
        """
        Creates a group self-attention layer with global receptive fields.

        The dimension of the positional encoding is given by mid_channels // 3, where mid_channels // 3 are used to represent
        the row embedding, mid_channels // 3 channels are used to represent the col embedding, and (mid_channels - 2 * mid_channels // 3)
        are used to represent the embedding of the group element (e.g., rotation) .

        Args:
            group: The group to be used, e.g., rotations.
            in_channels:  Number of channels in the input signal
            mid_channels: Number of channels of the hidden representation on which attention is performed.
            out_channels: Number of channels in the output signal
            num_heads: Number of heads in the operation
            max_pos_embedding: The maximum size of the positional embedding to use.
            attention_dropout_rate: Dropout applied to the resulting attention coefficients.
        """
        super().__init__()

        # Define self parameters
        self.group = group
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.dim_out = out_channels * self.num_heads

        # # Define embeddings.
        # self.dim_pos_encoding = mid_channels // 3
        # self.dim_group_encoding = mid_channels - 2 * self.dim_pos_encoding

        # self.row_embedding = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1),
        #     g_selfatt.nn.LayerNorm(num_channels=16),
        #     Swish(),
        #     torch.nn.Conv2d(in_channels=16, out_channels=self.dim_pos_encoding, kernel_size=1),
        # )
        # self.col_embedding = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1),
        #     g_selfatt.nn.LayerNorm(num_channels=16),
        #     Swish(),
        #     torch.nn.Conv2d(in_channels=16, out_channels=self.dim_pos_encoding, kernel_size=1),
        # )
        # self.group_embedding = nn.Embedding(self.group.num_elements, self.dim_group_encoding)

        # # Create the relative position indices for each element of the group.
        # self.initialize_indices(max_pos_embedding)

        # # Define linears using convolution 1x1.
        # self.query = nn.Conv3d(in_channels, mid_channels * num_heads, kernel_size=1)
        # self.key = nn.Conv3d(in_channels, mid_channels * num_heads, kernel_size=1)
        # self.value = nn.Conv3d(in_channels, mid_channels * num_heads, kernel_size=1)
        # self.wout = nn.Conv3d(mid_channels * num_heads, out_channels, kernel_size=1)

        # # Define dropout
        # self.dropout_attention = nn.Dropout(attention_dropout_rate)
        self.attention = ESConvAttention(group=self.group, dim_in=self.in_channels, dim_out=self.dim_out, num_heads=self.num_heads, 
                                         attn_drop=attention_dropout_rate, max_pos_embedding=max_pos_embedding)


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        b, c, w, h = x.shape
        # print(x.shape)
        out = self.attention(x, h, w)


        # # Compute attention scores
        # att_scores = self.compute_attention_scores(x)

        # # Normalize to obtain probabilities.
        # shape = att_scores.shape
        # att_probs = self.dropout_attention(
        #     nn.Softmax(dim=-1)(att_scores.view(*shape[:-3], -1)).view(shape)
        # )

        # # Compute value projections.
        # v = self.value(x).view(b, self.mid_channels, self.num_heads, g, w, h)

        # # Re-weight values via attention and map to output dimension.
        # v = torch.einsum("bhvgijmkl,bchmkl->bchvij", att_probs, v)
        # out = self.wout(
        #     v.contiguous().view(
        #         b, self.mid_channels * self.num_heads, self.group.num_elements, w, h
        #     )
        # )

        return out

