from torch import nn

from models.sgcn.gcn_conv_v1 import GCNConv


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, seq_len, emb_dim, num_nodes):
        super(GCNLayer, self).__init__()
        self.in_features = in_features  #
        self.out_features = out_features  #
        self.conv = GCNConv(in_features, out_features, seq_len=seq_len, emb_dim=emb_dim, num_nodes=num_nodes)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight=None)
        return x