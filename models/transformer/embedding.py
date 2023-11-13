from torch import nn
import torch

from models.sgat.gcn import GCN
from models.transformer.spatial_embedding import SpatialPositionalEncoding


class Embedding(nn.Module):
    def __init__(self, input_dim, embed_dim, time_steps, num_nodes, batch_size, adj, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(time_steps, num_nodes, embed_dim))
        )
        self.spatial_emb = None
        if adj is not None:
            self.spatial_emb = SpatialPositionalEncoding(embed_dim, num_nodes, 0.2, GCN(adj, embed_dim, embed_dim))
        self.batch_size = batch_size

    def forward(self, x):
        x = self.token_embed(x)
        adp_emb = self.adaptive_embedding.expand(size=(self.batch_size, *self.adaptive_embedding.shape))
        if self.spatial_emb is not None:
            x = x + self.spatial_emb(x).repeat(1, x.size()[1], 1, 1)
        x = x + adp_emb
        x = self.norm(x)
        return x