import torch
from torch import nn

from models.sgcn.gcn import GCN


class SGCNEmbedding(nn.Module):
    def __init__(self, sgat_configs):
        super(SGCNEmbedding, self).__init__()
        self.gcns = GCN(sgat_configs)

    def forward(self, x):
        out = self.gcns(x)
        return out