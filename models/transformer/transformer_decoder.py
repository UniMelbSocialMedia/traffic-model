import torch
from torch import nn, Tensor
from torch_geometric.transforms import ToDevice
import torch_geometric.data as data

from models.sgat.sgat_embedding import SGATEmbedding
from models.transformer.positional_embedding import PositionalEmbedding
from models.transformer.decoder_block import DecoderBlock
from models.transformer.embedding import Embedding


class TransformerDecoder(nn.Module):
    def __init__(self, configs):

        super(TransformerDecoder, self).__init__()

        self.emb_dim = configs['emb_dim']
        input_dim = configs['input_dim']
        self.merge_emb = configs['merge_emb']
        emb_expansion_factor = configs['emb_expansion_factor']
        max_lookup_len = configs['max_lookup_len']
        self.lookup_idx = configs['lookup_idx']
        self.device = configs['device']

        n_layers = configs['n_layers']
        out_dim = configs['out_dim']

        self.offset = configs['seq_offset']
        self.seq_len = configs['seq_len']
        self.enc_features = configs['enc_features']
        self.per_enc_feature_len = configs['per_enc_feature_len']
        self.cross_attn_features = configs['decoder_block']['cross_attn_features']

        # graph related
        self.device = configs['device']
        self.edge_index = configs['edge_index']
        self.edge_attr = configs['edge_attr']
        self.edge_details = configs['edge_details']
        self.graph_input = configs['graph_input']
        self.graph_semantic_input = configs['graph_semantic_input']
        configs['sgat'] = configs['sgat_normal']
        configs['sgat']['seq_len'] = self.seq_len
        configs['sgat']['dropout_g'] = configs['sgat']['dropout_g_dis']
        if self.graph_input:
            self.graph_embedding_dis = SGATEmbedding(configs['sgat'])
        configs['sgat']['dropout_g'] = configs['sgat']['dropout_g_sem']
        if self.graph_semantic_input:
            self.graph_embedding_semantic = SGATEmbedding(configs['sgat'])

        # embedding
        num_nodes = configs['num_nodes']
        batch_size = configs['batch_size']
        adj_mx = configs['adj_mx']
        self.embedding = Embedding(input_dim=input_dim, embed_dim=self.emb_dim, time_steps=self.seq_len,
                                   num_nodes=num_nodes, batch_size=batch_size, adj=adj_mx)
        # self.emb_dim = self.emb_dim * 2
        self.position_embedding = PositionalEmbedding(max_lookup_len, self.emb_dim)

        # convolution related
        self.local_trends = configs['local_trends']

        self.conv_q_layer = nn.Conv1d(in_channels=self.emb_dim, out_channels=self.emb_dim, kernel_size=3, stride=1, padding=1)
        self.emb_norm = nn.LayerNorm(self.emb_dim)

        padding_size = 1
        if self.offset == 1:
            padding_size = 2
        self.conv_q_layers = nn.ModuleList([
            nn.Conv2d(in_channels=self.emb_dim, out_channels=self.emb_dim, kernel_size=(1, 3), stride=1,
                      padding=(0, padding_size), bias=False)
            for _ in range(n_layers)
        ])

        self.conv_k_layers = nn.ModuleList([
            nn.ModuleList(
                [nn.Conv1d(in_channels=self.emb_dim, out_channels=self.emb_dim, kernel_size=3, stride=1, padding=1) for i in
                 range(self.cross_attn_features)])
            for j in range(n_layers)
        ])

        configs['decoder_block']['emb_dim'] = self.emb_dim
        self.layers = nn.ModuleList(
            [
                DecoderBlock(configs['decoder_block'])
                for _ in range(n_layers)
            ]
        )

        # by merging embeddings we increase the num embeddings
        if self.merge_emb:
            self.emb_dim = self.emb_dim * emb_expansion_factor

        self.fc_out = nn.Linear(self.emb_dim, out_dim)

        # graph
        self.out_e_lin = nn.Linear(self.emb_dim, self.emb_dim)
        dropout_e_normal = configs['dropout_e_normal']
        self.dropout_e_normal = nn.Dropout(dropout_e_normal)

    def _create_graph(self, x, edge_index, edge_attr):
        graph = data.Data(x=(Tensor(x[0]), Tensor(x[1])),
                          edge_index=torch.LongTensor(edge_index),
                          y=None,
                          edge_attr=Tensor(edge_attr))
        return graph

    def _derive_graphs(self, x_batch, x_time_idx):
        to = ToDevice(self.device)

        x_batch_graphs = []
        x_batch_graphs_semantic = []
        for idx, x_all_t in enumerate(x_batch):
            semantic_edge_index, semantic_edge_attr = self.edge_details
            x_src = x_all_t.permute(1, 0, 2)  # N, T, F

            # x_dst = x_src.unsqueeze(dim=0).repeat(self.seq_len, 1, 1, 1)
            # mask = torch.tril(torch.ones((self.seq_len, self.seq_len))).unsqueeze(dim=1).repeat(1, x_src.shape[0], 1).unsqueeze(dim=-1).repeat(1, 1, 1, self.emb_dim).to(self.device)
            # x_dst = x_dst * mask
            # x_dst = x_dst.transpose(0, 1).reshape(x_src.shape[0], self.seq_len * self.seq_len * self.emb_dim)

            x_src = x_src.reshape(x_src.shape[0], -1)  # N, T*F

            if self.graph_input:
                graph = self._create_graph((x_src, x_src), self.edge_index, self.edge_attr)
                x_batch_graphs.append(to(graph))

            if self.graph_semantic_input:
                graph_semantic = self._create_graph((x_src, x_src), semantic_edge_index, semantic_edge_attr)
                x_batch_graphs_semantic.append(to(graph_semantic))

        return x_batch_graphs, x_batch_graphs_semantic

    def calculate_masked_src(self, x, conv_q, tgt_mask, device='cuda'):
        batch_size = x.shape[0]

        x = x.repeat(1, self.seq_len, 1).view(batch_size, self.seq_len, self.seq_len, self.emb_dim)

        x = tgt_mask.transpose(2, 3) * x
        x = x.permute(0, 3, 1, 2)
        x = conv_q(x)

        out = torch.zeros((batch_size, self.emb_dim, self.seq_len)).to(device)
        for i in range(self.seq_len):
            out[:, :, i] = x[:, :, i, i]

        return out.permute(0, 2, 1)

    def create_conv_mask(self, x, device='cuda'):
        tri = torch.tril(torch.ones((self.seq_len, self.seq_len)))
        tri[:, :self.offset] = 1
        tgt_mask_conv = tri.repeat(1, self.emb_dim).view(-1, self.emb_dim, self.seq_len)
        tgt_mask_conv = tgt_mask_conv.expand(x.shape[0], self.seq_len, self.emb_dim, self.seq_len).to(device)
        return tgt_mask_conv

    def _organize_matrix(self, mat):
        mat = mat.permute(1, 0, 2, 3)  # B, T, N, F -> T, B, N , F (4, 36, 170, 16) -> (36, 4, 170, 16)
        mat_shp = mat.shape
        mat = mat.reshape(mat_shp[0], mat_shp[1] * mat_shp[2], mat_shp[3])  # (36, 4 * 170, 16)
        mat = mat.permute(1, 0, 2)  # (4 * 170, 36, 16)
        return mat

    def _return_mat(self, out):
        out = self.fc_out(out)
        out = out.transpose(1, 2)

        return out

    def forward(self, x, enc_x, tgt_mask, device):
        embed_out = self.embedding(x)
        embed_shp = embed_out.shape
        enc_x_shp = enc_x[0].shape
        embed_out = self._organize_matrix(embed_out)

        out_d = self.position_embedding(embed_out, self.lookup_idx)  # 32x10x512

        tgt_mask_conv = self.create_conv_mask(out_d, device)

        for idx, layer in enumerate(self.layers):
            enc_x_conv = []
            if self.local_trends:
                out_d = out_d.view(-1, embed_shp[1], embed_shp[3])
                out_d = self.calculate_masked_src(out_d, self.conv_q_layers[idx], tgt_mask_conv, device)

                for idx_k, f_layer in enumerate(self.conv_k_layers[idx]):
                    if self.enc_features > 1:
                        enc_x_i = self._organize_matrix(enc_x[idx_k].transpose(1, 2))
                        enc_x_conv.append(f_layer(enc_x_i.transpose(2, 1)).transpose(2, 1))
                    else:
                        start = idx_k * self.per_enc_feature_len
                        enc_x_0 = self._organize_matrix(enc_x[0].transpose(1, 2))
                        enc_x_conv.append(
                            f_layer(enc_x_0[:, start: start + self.per_enc_feature_len].transpose(2, 1)).transpose(2, 1))

            else:
                enc_x_conv = [x for x in enc_x]

            out_d = out_d.reshape(embed_shp[0], embed_shp[2], embed_shp[1], embed_shp[3])
            enc_x_conv = [x.reshape(enc_x_shp[0], enc_x_shp[1], enc_x_shp[2], enc_x_shp[3]) for x in enc_x_conv]
            out_d = layer(out_d, enc_x_conv, tgt_mask)

        graph_x = out_d

        graph_x = graph_x.reshape(x.shape[0], x.shape[2], x.shape[1], graph_x.shape[-1])
        graph_x = graph_x.permute(0, 2, 1, 3)
        graph_x_shp = graph_x.shape
        out_g_dis, out_g_semantic = self._derive_graphs(graph_x, None)

        if self.graph_input:
            batch_size, time_steps, num_nodes, features = graph_x_shp
            out_g_dis = self.graph_embedding_dis(out_g_dis)  # (4, 307, 576)
            # out_g_dis = out_g_dis.reshape(batch_size, num_nodes, time_steps, -1)  # (4, 307, 12, 16)
            # out_g_dis = out_g_dis.permute(0, 2, 1, 3)  # (4, 12, 307, 16)
        if self.graph_semantic_input:
            out_g_semantic = self.graph_embedding_semantic(out_g_semantic)

        if self.graph_input and self.graph_semantic_input:
            out_g = out_g_dis + out_g_semantic
        elif self.graph_input and not self.graph_semantic_input:
            out_g = out_g_dis
        elif not self.graph_input and self.graph_semantic_input:
            out_g = out_g_semantic
        elif not self.graph_input and not self.graph_semantic_input:
            out_e = self.dropout_e_normal(out_d)
            return out_e

        out = self.dropout_e_normal(self.out_e_lin(out_d)) + out_g.transpose(1, 2)
        return self._return_mat(out)
