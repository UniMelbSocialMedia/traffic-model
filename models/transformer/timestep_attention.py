from torch import nn
import torch.nn.functional as F
import torch


class TimeStepAttention(nn.Module):
    def __init__(self, emb_dim):
        super(TimeStepAttention, self).__init__()
        self.lin_enc_1 = nn.Linear(emb_dim, emb_dim)
        self.lin_enc_2 = nn.Linear(emb_dim, emb_dim)
        self.lin_enc = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, enc_x, device):
        enc_x_exp = torch.zeros((enc_x[0].shape[0], enc_x[0].shape[1], 2, enc_x[0].shape[2])).to(device)
        enc_x_1 = self.lin_enc_1(enc_x[0])
        enc_x_2 = self.lin_enc_2(enc_x[1])

        enc_x_exp[:, :, 0] = enc_x_1
        enc_x_exp[:, :, 1] = enc_x_2

        enc_x_softmax = F.softmax(enc_x_exp, dim=2)

        return [enc_x_softmax[:, :, 0] * enc_x_1, enc_x_softmax[:, :, 1] * enc_x_2]
