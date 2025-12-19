# import torch
# import torch.nn as nn
# import torch.nn.functional as F

import torch
import torch.nn as nn

# class PatchEmbed(nn.Module):
#     def __init__(self, patch_size=16, in_chans=3, embed_dim=192):
#         super().__init__()
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         return self.proj(x)  # [B, D, H', W']

# class Attention2D(nn.Module):
#     def __init__(self, dim, num_heads=3):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.qkv = nn.Conv2d(dim, dim * 3, 1)
#         self.proj = nn.Conv2d(dim, dim, 1)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
#         q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B, heads, head_dim, N]
#         q, k, v = [t.permute(0, 1, 3, 2) for t in (q, k, v)]  # [B, heads, N, head_dim]

#         attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
#         attn = attn.softmax(dim=-1)
#         out = attn @ v  # [B, heads, N, head_dim]
#         out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
#         return self.proj(out)

# class MLP2D(nn.Module):
#     def __init__(self, dim, mlp_dim):
#         super().__init__()
#         self.fc1 = nn.Conv2d(dim, mlp_dim, 1)
#         self.fc2 = nn.Conv2d(mlp_dim, dim, 1)
#         self.act = nn.GELU()

#     def forward(self, x):
#         return self.fc2(self.act(self.fc1(x)))

# class Block2D(nn.Module):
#     def __init__(self, dim, num_heads=3, mlp_ratio=4.):
#         super().__init__()
#         self.norm1 = nn.BatchNorm2d(dim)
#         self.attn = Attention2D(dim, num_heads)
#         self.norm2 = nn.BatchNorm2d(dim)
#         self.mlp = MLP2D(dim, int(dim * mlp_ratio))

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x

# class ViTTiny(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, num_classes=1000,
#                  in_chans=3, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.):
#         super().__init__()
#         self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)

#         self.blocks = nn.Sequential(*[
#             Block2D(embed_dim, num_heads, mlp_ratio)
#             for _ in range(depth)
#         ])

#         self.norm = nn.BatchNorm2d(embed_dim)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.head = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
#         x = self.patch_embed(x)        # [B, D, H, W]
#         x = self.blocks(x)             # [B, D, H, W]
#         x = self.norm(x)
#         x = self.pool(x).squeeze(-1).squeeze(-1)  # [B, D]
#         return self.head(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, D]
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, in_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=True, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2,0,3,1,4)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_dropout=attn_drop, proj_dropout=drop)
        self.drop_path = nn.Identity()  # you can replace with DropPath for stochastic depth
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_dim, dropout=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ViTTiny(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., drop_rate=0.2, attn_drop=0.2):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))  # add num_patches+1 if using class tokenizer
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        # x = x.mean(dim=1)  # Global average pooling over patches, when not using cls token
        # return self.head(x)
        return self.head(x[:, 0]) # for tokenized class 
