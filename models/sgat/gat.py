import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear

from models.sgat.gat_layer import GATLayer


class GAT(nn.Module):
    def __init__(self, configs):
        super(GAT, self).__init__()

        self.n_layers = configs['n_layers']
        self.dropout = configs['dropout']
        self.dropout_g = configs['dropout_g']
        self.seq_len = configs['seq_len']
        self.out_f_sizes = configs['out_f_sizes']

        n_heads = configs['n_heads']
        first_in_f_size = configs['first_in_f_size']
        alpha = configs['alpha']
        edge_dim = configs['edge_dim']
        seq_len = configs['seq_len']

        self.layer_stack = nn.ModuleList()
        for l in range(self.n_layers):
            in_f_size = self.out_f_sizes[l - 1] * n_heads[l - 1] if l else first_in_f_size
            concat = True if l < (self.n_layers - 1) else False
            gat_layer = GATLayer(in_f_size, self.out_f_sizes[l], n_heads=n_heads[l], dropout=self.dropout_g, alpha=alpha,
                                 concat=concat, edge_dim=edge_dim, seq_len=seq_len)
            self.layer_stack.append(gat_layer)

    def forward(self, batch_data):
        out = ()
        for data in batch_data:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

            for l, gat_layer in enumerate(self.layer_stack):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = gat_layer(x, edge_attr=edge_attr, edge_index=edge_index)
                if l < (self.n_layers - 1):
                    x = F.elu(x)

            out = (*out, x)

        return torch.stack(out)
