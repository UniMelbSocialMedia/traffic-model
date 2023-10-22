from torch import nn
import torch
from torch.nn import MultiheadAttention


class TimeStepAttention(nn.Module):
    def __init__(self, emb_dim):
        super(TimeStepAttention, self).__init__()
        self.lin_cross_1 = nn.Linear(emb_dim, emb_dim)
        self.lin_cross_2 = nn.Linear(emb_dim, emb_dim)
        self.lin_cross = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, cross_x):
        cross_x_1 = self.lin_cross_1(cross_x[0]).exp()
        cross_x_2 = self.lin_cross_2(cross_x[1]).exp()

        cross_x_1 = cross_x_1 / (cross_x_1 + cross_x_2 + 1e-16)
        cross_x_2 = cross_x_2 / (cross_x_1 + cross_x_2 + 1e-16)

        cross_x = torch.concat((cross_x_1, cross_x_2), dim=-1)
        cross_x = self.lin_cross(cross_x)

        return cross_x
