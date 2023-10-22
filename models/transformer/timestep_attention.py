from torch import nn
# import torch.nn.functional as F
import torch


class TimeStepAttention(nn.Module):
    def __init__(self, emb_dim):
        super(TimeStepAttention, self).__init__()
        self.lin_cross_1 = nn.Linear(emb_dim, emb_dim)
        self.lin_cross_2 = nn.Linear(emb_dim, emb_dim)
        self.lin_cross = nn.Linear(emb_dim * 2, emb_dim)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, cross_x, device):
        cross_x_exp = torch.zeros((cross_x[0].shape[0], cross_x[0].shape[1], 2, cross_x[0].shape[2])).to(device)
        cross_x_1 = self.lin_cross_1(cross_x[0])
        cross_x_2 = self.lin_cross_2(cross_x[1])

        cross_x_exp[:, :, 0] = cross_x_1
        cross_x_exp[:, :, 1] = cross_x_2

        cross_x_softmax = self.softmax(cross_x_exp)
        cross_x_softmax_reshaped = cross_x_softmax.view(cross_x_softmax.shape[0], cross_x_softmax.shape[1], -1)

        cross_x = self.lin_cross(cross_x_softmax_reshaped)

        return cross_x
