import torch
from torch import nn

from models.sgcn.gcn_layer import GCNLayer


class GCN(nn.Module):
    def __init__(self, configs):
        super(GCN, self).__init__()

        self.n_layers = configs['n_layers']
        dropout_g = configs['dropout_g']
        self.seq_len = configs['seq_len']

        self.out_f_sizes = configs['out_f_sizes']
        n_heads = configs['n_heads']
        self.first_in_f_size = configs['first_in_f_size']

        self.lin = nn.Linear(self.first_in_f_size, self.out_f_sizes[-1])
        self.dropout = nn.Dropout(dropout_g)

        self.layer_stack = nn.ModuleList()
        for l in range(self.n_layers):
            in_f_size = self.out_f_sizes[l - 1] * n_heads[l - 1] if l else self.first_in_f_size
            gcn_layer = GCNLayer(in_f_size, self.out_f_sizes[l])
            self.layer_stack.append(gcn_layer)

    def forward(self, batch_data):
        # ToDo: implement using pytorch geometric batch data
        out = ()
        for data in batch_data:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

            for l, gcn_layer in enumerate(self.layer_stack):
                x = gcn_layer(x,  edge_weight=edge_attr, edge_index=edge_index)
                x = self.dropout(x)
                x = x.reshape(x.size()[0], self.seq_len, -1)  # 307, 36, 288
                # x = self.lin(x)
                x = x.permute(1, 0, 2)  # 36, 307, 288

            out = (*out, x)

        return torch.stack(out)
