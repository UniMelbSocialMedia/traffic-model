import torch
from torch import nn
from torch.nn import MultiheadAttention

from models.transformer.cross_attention import CrossAttentionLayer
from models.transformer.position_wise_feed_forward import PositionWiseFeedForward
from models.transformer.timestep_attention import TimeStepAttention


class DecoderBlock(nn.Module):
    def __init__(self, configs):
        super(DecoderBlock, self).__init__()

        emb_dim = configs['emb_dim']
        n_heads = configs['n_heads']
        expansion_factor = configs['expansion_factor']
        cross_attn_dropout = configs['cross_attn_dropout']
        cross_attn_features = configs['cross_attn_features']
        src_dropout = configs['src_dropout']
        ff_dropout = configs['ff_dropout']

        self.self_attention = MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, batch_first=True, bias=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(src_dropout)

        self.timestep_attention = TimeStepAttention(emb_dim)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(emb_dim, n_heads, dropout=cross_attn_dropout[i]) for i in range(cross_attn_features)
        ])
        self.norm2 = nn.LayerNorm(emb_dim)

        self.feed_forward = PositionWiseFeedForward(emb_dim, expansion_factor * emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)
        self.dropout2 = nn.Dropout(ff_dropout)

    def forward(self, x, enc_x, tgt_mask, device):
        # self attention
        attention = self.self_attention(x, x, x, attn_mask=tgt_mask)  # 32x10x512
        x = self.norm1(x + self.dropout1(attention[0]))

        # cross attention
        # enc_x = self.timestep_attention(enc_x, device)
        # cross_attn = []
        # for idx, layer in enumerate(self.cross_attn_layers):
        #     if idx == 0:
        #         cross_attn.append(layer(x, enc_x[idx], enc_x[idx]))
        #     else:
        #         cross_attn.append(layer(x, enc_x[idx], enc_x[idx]))

        history_with_context = self.cross_attn_layers[0](enc_x[0], enc_x[1], enc_x[1])
        x_with_context = self.cross_attn_layers[1](x, enc_x[1], enc_x[1])
        cross_attn = self.cross_attn_layers[2](x_with_context, history_with_context, history_with_context)

        # enc_x = torch.concat(enc_x, dim=1)
        # cross_attn = self.cross_attn_layers[0](x, enc_x, enc_x)

        x = self.norm2(x + cross_attn)

        # positionwise ffn
        ff_output = self.feed_forward(x)
        out = self.norm3(x + self.dropout2(ff_output))

        return out
