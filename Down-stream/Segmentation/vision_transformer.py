# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from typing import Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

import timm.models.vision_transformer
from utils.pos_embed import get_3d_sincos_pos_embed
import ipdb


class PatchEmbed3D(nn.Module):
    """ 
    3D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_chans=1, embed_dim=768, norm_layer=None, flatten=True, in_chan_last=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = np.prod(self.grid_size)
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.in_chans = in_chans
        self.in_chan_last = in_chan_last
        
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def forward(self, x):
        B, L, Dim = x.shape
        assert Dim == np.prod(self.patch_size) * self.in_chans, \
            f"Input image total size {Dim} doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})"        
        
        # input image is pathified 3D image
        if self.in_chan_last:
            x = x.reshape(B * L, *self.img_size, self.in_chans).permute(0, 4, 1, 2, 3) # When patchification follows HWDC
        else:
            x = x.reshape(B * L, self.in_chans, *self.img_size)
        
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)

        x = x.reshape(B, L, self.embed_dim)
        return x


class ViT(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(
        self,
        in_chans: int = 1,
        img_size: Tuple =(224,224,112),
        patch_size: Tuple =(16,16,8),
        embed_dim: int =768,
        **kwargs
    ):
        super(ViT, self).__init__(**kwargs)

        self.patch_embed = PatchEmbed3D(img_size=patch_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = np.prod(img_size) // np.prod(patch_size)
        # self.pos_embed = nn.Parameter(torch.zeros(1, int(self.num_patches) + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, int(self.num_patches), embed_dim), requires_grad=False)  # fixed sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], np.round(self.num_patches**(1/3)), num_tokens=0, cls_token=False)
        self.pos_embed.data.copy_(pos_embed.float())

        del self.norm
        del self.head
        norm_layer = kwargs['norm_layer']
        self.fc_norm = norm_layer(embed_dim)


    
    def patchify3D(self, imgs):
        """
        imgs: (N, 1, H, W, D)
        x: (N, L,  prod(patch_size) * 1)
        """
        p = self.patch_embed.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p[0] == 0

        h = w = d = imgs.shape[2] // p[0]
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p[0], w, p[1], d, p[2]))
        x = torch.einsum('nchpwqdr->nhwdpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p[0] * p[1] * p[2] * 1))
        return x


    def forward(self, x):
        x = self.patchify3D(x)
        x = self.patch_embed(x)

        # cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)

        x = self.fc_norm(x)

        return x, hidden_states_out



def vit_base_patch16(**kwargs):
    model = ViT(
        patch_size=(16, 16, 8), in_channels=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# def vit_large_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_huge_patch14(**kwargs):
#     model = VisionTransformer(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model