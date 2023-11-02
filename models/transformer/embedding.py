from torch import nn
import torch


class Embedding(nn.Module):
    def __init__(self, input_dim, embed_dim, time_steps, num_nodes, batch_size, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim * 2) if norm_layer is not None else nn.Identity()
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(time_steps, num_nodes, embed_dim))
        )
        self.batch_size = batch_size

    def forward(self, x):
        x = self.token_embed(x)
        adp_emb = self.adaptive_embedding.expand(size=(self.batch_size, *self.adaptive_embedding.shape))
        x = torch.concat((x, adp_emb), dim=-1)
        # x = self.norm(x)
        return x