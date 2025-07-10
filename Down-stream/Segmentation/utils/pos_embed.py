import numpy as np
import torch
import torch.nn as nn
from timm.models.layers.helpers import to_3tuple


def create_sin_cos_position_embedding(height, width, depth, embed_dim):
    pe = np.zeros((height, width, depth, embed_dim))
    div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        
    # Using sin-cos embedding for every dimension individually
    for h in range(height):
        for w in range(width):
            for d in range(depth):
                pe[h, w, d, 0::2] = np.sin(h * div_term)
                pe[h, w, d, 1::2] = np.cos(w * div_term)
        
    return torch.tensor(pe, dtype=torch.float32)


def get_3d_sincos_pos_embed(embed_dim, grid_size, num_tokens=1, temperature=10000., cls_token=False):
    grid_size = to_3tuple(int(grid_size))
    h, w, d = grid_size

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    
    pos_embed = create_sin_cos_position_embedding(h, w, d, embed_dim)
    pos_embed = pos_embed.flatten(0, 2).unsqueeze(0)

    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_embed], dim=1))
    else:
        pos_embed = nn.Parameter(pos_embed)
    
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, 1, embed_dim], dtype=torch.float32), pos_embed], dim=1)
    
    return pos_embed