import torch
from torch import nn

from models.transformer.positional_embedding import PositionalEmbedding
from models.transformer.decoder_block import DecoderBlock
from models.transformer.token_embedding import TokenEmbedding
from utils.transformer_utils import organize_matrix


class TransformerDecoder(nn.Module):
    def __init__(self, configs):

        super(TransformerDecoder, self).__init__()

        self.emb_dim = configs['emb_dim']
        input_dim = configs['input_dim']
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

        # embedding
        self.embedding = TokenEmbedding(input_dim=input_dim, embed_dim=self.emb_dim)
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

        self.fc_out = nn.Linear(self.emb_dim, out_dim)

    def calculate_masked_src(self, x, conv_q, tgt_mask, device='cuda'):
        """
        Calculate output from convolution operation.
        Create a 2D tensor and apply target mask over it before doing the convolution
        Parameters
        ----------
        x: Tensor, input
        conv_q: nn.Module, convolution layer
        tgt_mask: Tensor,
        device: str, cpu or cuda

        Returns
        -------
        out: Tensor, return from the convolution
        """
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
        """
        Create convolution mask to prevent data leakage from future time steps
        Parameters
        ----------
        x: Tensor, decoder input
        device: str, cpu or cuda

        Returns
        -------
        tgt_mask_conv: Tensor, mask
        """
        tri = torch.tril(torch.ones((self.seq_len, self.seq_len)))
        tri[:, :self.offset] = 1
        tgt_mask_conv = tri.repeat(1, self.emb_dim).view(-1, self.emb_dim, self.seq_len)
        tgt_mask_conv = tgt_mask_conv.expand(x.shape[0], self.seq_len, self.emb_dim, self.seq_len).to(device)
        return tgt_mask_conv

    def _return_mat(self, out, shp):
        """
        Reshape the output
        Parameters
        ----------
        out
        shp

        Returns
        -------

        """
        out = self.fc_out(out)

        out = out.permute(1, 0, 2)
        out = out.reshape(shp[1], shp[0], shp[2], out.shape[-1])
        out = out.permute(1, 0, 2, 3)

        return out

    def forward(self, x, enc_x, device):
        embed_out = self.embedding(x)
        embed_shp = embed_out.shape
        enc_x_shp = enc_x[0].shape
        embed_out = organize_matrix(embed_out)
        out_d = self.position_embedding(embed_out, self.lookup_idx)  # 32x36x64

        tgt_mask_conv = self.create_conv_mask(out_d, device)  # to stop data flow from future time steps
        for idx, layer in enumerate(self.layers):
            enc_x_conv = []
            if self.local_trends:
                out_d = out_d.view(-1, embed_shp[1], embed_shp[3])
                out_d = self.calculate_masked_src(out_d, self.conv_q_layers[idx], tgt_mask_conv, device)

                for idx_k, f_layer in enumerate(self.conv_k_layers[idx]):
                    if self.enc_features > 1:
                        enc_x_i = organize_matrix(enc_x[idx_k].transpose(1, 2))
                        enc_x_conv.append(f_layer(enc_x_i.transpose(2, 1)).transpose(2, 1))
                    else:
                        start = idx_k * self.per_enc_feature_len
                        enc_x_0 = organize_matrix(enc_x[0].transpose(1, 2))
                        enc_x_conv.append(
                            f_layer(enc_x_0[:, start: start + self.per_enc_feature_len].transpose(2, 1)).transpose(2, 1))

            else:
                enc_x_conv = [x for x in enc_x]

            out_d = out_d.reshape(embed_shp[0], embed_shp[2], embed_shp[1], embed_shp[3])
            enc_x_conv = [x.reshape(enc_x_shp[0], enc_x_shp[1], enc_x_shp[2], enc_x_shp[3]) for x in enc_x_conv]
            out_d = layer(out_d, enc_x_conv)

        return self.fc_out(out_d).transpose(1, 2)
