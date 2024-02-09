import torch
from torch import nn
from utils import get_module

class CrossAttnLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_dim, dropout=0.1, kvdim=None, norm_first=True):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, kdim=kvdim, vdim=kvdim,
                                          batch_first=True)  # nn.MultiheadAttention includes QKV projection
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, fc_dim)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)

        self.norm_first = norm_first
        if self.norm_first:
            self.normkv = nn.LayerNorm(kvdim)

    def forward(self, q, kv, kv_padding=None):
        if self.norm_first:
            q = self.norm1(q)
            kv = self.normkv(kv)
            x = self.dropout1(self.attn(query=q, key=kv, value=kv, key_padding_mask=kv_padding, need_weights=False)[0]) + q
            x = self._ff(self.norm2(x))
        else:
            x = self.norm1(self.dropout1(self.attn(query=q, key=kv, value=kv, key_padding_mask=kv_padding, need_weights=False)[0]) + q)
            x = self.norm2(self._ff(x))
        return x

    def _ff(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        return x


class BindingX(nn.Module):
    def __init__(self, emb_dim, n_layers, input_attn_heads, self_attn_heads, kvdim, dim_feedforward_scale=2, dropout=0.1, n_outs=1, norm_first=True):
        super().__init__()
        self.input_attn = CrossAttnLayer(emb_dim, input_attn_heads, emb_dim * dim_feedforward_scale, dropout=dropout, norm_first=norm_first, kvdim=kvdim)
        
        # replace with custom encoder layer
        encoder_layer = nn.TransformerEncoderLayer(emb_dim, self_attn_heads, emb_dim * dim_feedforward_scale, batch_first=True,
                                                   dropout=dropout, activation="gelu", norm_first=norm_first)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False, mask_check=False)

        self.norm_out = nn.LayerNorm(emb_dim) if norm_first else nn.Identity()
        self.fc_out = nn.Linear(emb_dim, n_outs)

    def forward(self, protein_emb, ligand_emb, protein_padding=None, ligand_padding=None):
        x = self.input_attn(protein_emb, ligand_emb, ligand_padding) # protein is q, ligand is kv, this is different from image, also remove proj from image
        x = self.encoder(x)
        if protein_padding is not None:
            mask = protein_padding.unsqueeze(-1).expand_as(x).bool()
            x[mask] = 0
            x = x.sum(1) / (~mask).sum(1)
        else:
            x = x.mean(1)
        return self.fc_out(self.norm_out(x))


def binding_lite(input_dim, hidden_dim, num_hidden_layers, batch_norm, activation, dropout):
    layers = []

    for i in range(num_hidden_layers):
        layers.append(nn.Linear(input_dim, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(get_module(activation))
        if dropout:
            layers.append(nn.Dropout1d(dropout))
        input_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, 1))

    return nn.Sequential(*layers)
