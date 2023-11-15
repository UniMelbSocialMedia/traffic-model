import torch
from torch import nn

from models.sgcn.gcn_layer import GCNLayer


class GCN(nn.Module):
    def __init__(self, configs):
        super(GCN, self).__init__()

        self.n_layers = configs['n_layers']
        dropout_g = configs['dropout_g']
        self.seq_len = configs['seq_len']
        self.emb_dim = configs['emb_dim']
        num_nodes = configs['num_nodes']

        self.out_f_sizes = configs['out_f_sizes']
        n_heads = configs['n_heads']
        self.first_in_f_size = configs['first_in_f_size']

        self.lin = nn.Linear(self.first_in_f_size, self.out_f_sizes[-1])
        self.dropout = nn.Dropout(dropout_g)

        self.layer_stack = nn.ModuleList()
        for l in range(self.n_layers):
            in_f_size = self.out_f_sizes[l - 1] * n_heads[l - 1] if l else self.first_in_f_size
            gcn_layer = GCNLayer(in_f_size, self.out_f_sizes[l], seq_len=self.seq_len, emb_dim=self.emb_dim, num_nodes=num_nodes)
            self.layer_stack.append(gcn_layer)

    def forward(self, batch_data):
        # ToDo: implement using pytorch geometric batch data
        out = ()
        for data in batch_data:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

            for l, gcn_layer in enumerate(self.layer_stack):
                x = gcn_layer(x,  edge_weight=edge_attr, edge_index=edge_index)
                x = self.dropout(x)
                # x = self.lin(x)

            out = (*out, x)

        return torch.stack(out)
