from torch import nn
from torch_geometric.nn import GCNConv


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.in_features = in_features  #
        self.out_features = out_features  #
        self.conv = GCNConv(in_features, out_features)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight=None)
        return x