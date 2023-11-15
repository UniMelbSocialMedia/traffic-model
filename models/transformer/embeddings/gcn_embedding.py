from torch import nn
from torch.nn import functional as F
import torch


class GCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(GCN, self).__init__()
        self.sym_norm_Adj_matrix = torch.Tensor(sym_norm_Adj_matrix).to('cuda')  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, F_in)
        :return: (batch_size, N, F_out)
        '''
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)))  # (N,N)(b,N,in)->(b,N,in)->(b,N,out)