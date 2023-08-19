import torch
from torch import nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import edge_softmax, GATConv


class SGATEmbedding(nn.Module):
    def __init__(self, configs):
        super(SGATEmbedding, self).__init__()

        self.g = configs['graph_data']

        self.n_layers = configs['n_layers']
        self.dropout = configs['dropout']

        out_f_sizes = configs['out_f_sizes']
        n_heads = configs['n_heads']
        first_in_f_size = configs['first_in_f_size']
        alpha = configs['alpha']

        self.layer_stack = nn.ModuleList()
        for l in range(self.n_layers):
            in_f_size = out_f_sizes[l - 1] * n_heads[l - 1] if l else first_in_f_size
            gat_layer = GATConv(in_feats=in_f_size,
                                out_feats=out_f_sizes[l],
                                num_heads=n_heads[l],
                                feat_drop=self.dropout,
                                attn_drop=self.dropout,
                                negative_slope=alpha,
                                residual=False,
                                activation=F.elu,
                                allow_zero_in_degree=True)

            self.layer_stack.append(gat_layer)

    def forward(self, x):
        [batch_size, step_size, num_of_vertices, fea_size] = x.size()
        batched_g = dgl.batch(batch_size * [self.g])

        h = x.permute(0, 2, 3, 1).reshape(batch_size * num_of_vertices, fea_size * step_size)

        for l, gat_layer in enumerate(self.layer_stack):
            h = gat_layer(batched_g, h).mean(1)

        gc = h.reshape(batch_size, num_of_vertices, fea_size, -1)
        graph_out = gc.permute(0, 3, 1, 2)

        return x + graph_out
