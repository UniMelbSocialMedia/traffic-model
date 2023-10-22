from torch import nn
import torch
from torch.nn import MultiheadAttention


class TimeStepAttention(nn.Module):
    def __init__(self, emb_dim):
        super(TimeStepAttention, self).__init__()
        self.lin_enc = nn.Linear(emb_dim, emb_dim)

    def forward(self, enc_x):
        enc_x_1 = self.lin_enc(enc_x[0]).exp()
        enc_x_2 = self.lin_enc(enc_x[1]).exp()

        enc_x_1 = enc_x_1 / (enc_x_1 + enc_x_2 + 1e-16)
        enc_x_2 = enc_x_2 / (enc_x_1 + enc_x_2 + 1e-16)

        return [enc_x_1, enc_x_2]
