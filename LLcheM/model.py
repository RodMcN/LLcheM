import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import rearrange
import selfies as sf


class LLcheM(nn.Module):
    def __init__(self,
                 vocab,
                 embed_dim, 
                 num_embeddings, 
                 embed_padding_idx, 
                 num_outputs, 
                 num_attn_heads, 
                 num_encoder_layers, 
                 pos_encoding_layer=None, 
                 max_input_len=None,
                 rotary_embeddings=False):

        super().__init__()
        self.vocab = vocab
        
        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.embed_padding_idx = embed_padding_idx
        self.num_outputs = num_outputs
        self.num_attn_heads = num_attn_heads
        self.num_encoder_layers = num_encoder_layers
        self.rotary_embeddings = rotary_embeddings
        self.max_input_len = max_input_len

        assert not (rotary_embeddings and pos_encoding_layer)
        if pos_encoding_layer == "TrainablePosEmbedding":
            pos_encoding_layer = TrainablePosEmbedding(embed_dim, max_input_len)
        elif pos_encoding_layer == "SinusoidalPosEmbedding":
            pos_encoding_layer = SinusoidalPosEmbedding(embed_dim, max_input_len)
        self.pos_encoding_layer = pos_encoding_layer        
                     
        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim, padding_idx=embed_padding_idx)

        encoder_layer = EncoderLayer(d_model=embed_dim, nhead=num_attn_heads, dim_feedforward=embed_dim * 4, use_rotary_embeddings=rotary_embeddings)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.act = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, num_outputs)

        # self.fc_out = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(embed_dim),
        #     nn.Dropout(0.1),
        #     nn.Linear(embed_dim, num_outputs))

        self.init_weights()

    def forward(self, x, padding_mask=None, prediction_mask=None, pre_logits=False):
        x = self.emb(x)
        if self.pos_encoding_layer is not None:
            x = self.pos_encoding_layer(x)
        # rearrange for compatability with multihead attention from torch functional
        # x = rearrange(x, "b t c -> t b c")
        x = x.transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        # x = rearrange(x, "t b c -> b t c")
        x = x.transpose(0, 1)
        x = self.norm(self.act(x))
        if prediction_mask is not None:
            # only need predictions for the masked elements when training
            x = x[prediction_mask]
        
        if pre_logits:
            return x
        x = self.fc_out(x)
        return x


    @torch.no_grad()
    def get_embeddings(self, selfies=None, smiles=None, return_attn=False, encoder_layer: int=None, mean_reduce=True):
        assert bool(selfies) != bool(smiles)
        is_smiles = bool(smiles)

        x = selfies or smiles
        if isinstance(x, str):
            x = [x]
        
        # convert inputs to embedding indices
        def _enc(s):
            if is_smiles:
                s = sf.encoder(s)
            s = list(sf.split_selfies(s))
            s = self.vocab(s)
            return s
        x = [_enc(s) for s in x]
        
        if len(x) > 1:
            # generate padding mask
            max_len = max([len(s) for s in x])
            padding = torch.full((len(x), max_len), False)
            for i, s in enumerate(x):
                padding[i, len(s):] = True
                s.extend([self.embed_padding_idx] * (max_len - len(s)))
        else:
            padding = None

        x = torch.LongTensor(x).to(self.emb.weight.device)

        # forward pass up to encoder_layer
        x = self.emb(x)
        if self.pos_encoding_layer is not None:
            x = self.pos_encoding_layer(x)
        x = rearrange(x, "b t c -> t b c")
        if encoder_layer:
            if abs(encoder_layer) >= len(self.encoder.layers):
                raise ValueError(f"invalid 'encoder_layer' argument for model with {len(self.encoder.layers)} encoder layers")
            # if encoder_layer >= 0:
            #     encoder_layer += 1
        else:
            encoder_layer = len(self.encoder.layers)
        for layer in self.encoder.layers[:encoder_layer]:
            x, attn = layer(x, return_head_weights=True, src_key_padding_mask=padding)
        x = rearrange(x, "t b c -> b t c")
        attn = rearrange(attn, "h b t s -> b h t s")

        def _maybe_reduce(x):
            if mean_reduce:
                return x.mean(0)
            else:
                return x

        if x.shape[0] == 1:
            x = _maybe_reduce(x[0])
        else:
            # remove padding
            out = []
            for e, m in zip(x, padding):
                out.append(_maybe_reduce(e[~m]))
            x = out
        
        if return_attn:
            return x, attn
        else:
            return x


    def step(self, x, y, mask, padding, loss_fn, device):
        x = x.to(device, non_blocking=True)
        y = y.to(device)
        preds = self(x, padding_mask=padding.to(device), prediction_mask=mask.to(device))
        loss = loss_fn(preds, y)
        return loss

    def init_weights(self):
        self.apply(self._init_weights)
        # for c in self.children():
        #     if isinstance(c, MultiheadAttention):
        #         c.reset_parameters()
        # for n, p in self.named_parameters():
        #     if n.endswith('self_attn.out_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.num_encoder_layers))



    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # if isinstance(module, MultiheadAttention):
        #     module.reset_parameters()


    @staticmethod
    def load_model(filename, map_location=None):
        state_dict = torch.load(filename, map_location=map_location)
        vocab = state_dict['vocab']
        model_config = state_dict['model_config']
        model = LLcheM(vocab, **model_config)

        params_dict = state_dict['params_dict']
        model.load_state_dict(params_dict)
        return model
    

    def save_model(self, filename):
        model_config = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        del model_config['training']
        if "pos_encoding_layer" not in model_config or isinstance(model_config["pos_encoding_layer"], nn.Module):
            model_config["pos_encoding_layer"] = self.pos_encoding_layer.__class__.__name__
            model_config["max_input_len"] = self.pos_encoding_layer.pos.shape[0]

        save_dict = {
            "params_dict": self.state_dict(),
            "vocab": self.vocab,
            "model_config": model_config
        }

        torch.save(save_dict, filename)

class TrainablePosEmbedding(nn.Module):
    def __init__(self, embed_dim, num_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.pos = nn.Parameter(torch.randn(num_embeddings, embed_dim) * 0.01)
    
    def encode_pos(self, x):
        # maybe more efficient to swap the order to slice then repeat?
        pos = self.pos.repeat(x.shape[0], 1, 1)
        pos = pos[:, :x.shape[1], :]
        return pos

    def forward(self, x):
        # mask?
        pos = self.encode_pos(x)
        return x + pos


class SinusoidalPosEmbedding(nn.Module):
    def __init__(self, embed_dim, num_embeddings, **kwargs):
        super().__init__(**kwargs)
        
        half_dim = embed_dim // 2
        pos = math.log(10000) / (half_dim - 1)
        pos = torch.exp(torch.arange(half_dim, dtype=torch.float) * -pos)
        pos = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * pos.unsqueeze(0)
        self.register_buffer(name='pos', tensor=torch.cat([torch.sin(pos), torch.cos(pos)], dim=1).view(num_embeddings, -1))

    def encode_pos(self, x):
        pos = self.pos[:x.shape[1], :]

        return pos

    def forward(self, x):
        pos = self.encode_pos(x)
        # return x + pos
        return x + pos[None, ...]


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_embeddings=1000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_embeddings).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)


    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dim=1):
        return self.encode_pos(q), self.encode_pos(k)


    def encode_pos(self, x, seq_dim=1):
        assert seq_dim == 1

        cos = self.cos[:, :x.shape[seq_dim], :] * x

        x1, x2 = x.chunk(2, dim=-1)
        x = torch.cat((-x2, x1), dim=-1)
        sin = self.sin[:, :x.shape[seq_dim], :] * x

        return cos + sin


class MultiheadAttention(nn.Module):
    '''Implementation of MHA which supports rotary embeddings'''
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.batch_first = True

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.rot_emb = None
        if use_rotary_embeddings:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value,
        key_padding_mask=None, attn_mask=None, 
        return_attn_weights=False, before_softmax=False, return_head_weights=False,
    ):
        """
        :param return_attn_weights: return the attention weights averaged over heads
        :param return_head_weights: return the attention weights for each head
        :param before_softmax: return the attention weights and values before the attention softmax
        """
        if return_head_weights:
            return_attn_weights = True

        tgt_len, bsz, embed_dim = query.size()

        if not (self.rot_emb or return_head_weights):
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                key_padding_mask,
                return_attn_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask),
                    ],
                    dim=1,
                )

        if self.rot_emb:
            q, k = self.rot_emb(q, k)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights=None
        if return_attn_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).type_as(attn).transpose(1, 0)
            if not return_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        return attn, attn_weights
    

class EncoderLayer(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        d_model,
        dim_feedforward,
        nhead,
        attn_dropout=0.05,
        fc_dropout=0.15,
        add_bias_kv=False,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.use_rotary_embeddings = use_rotary_embeddings
        
        self.self_attn = MultiheadAttention(
            self.d_model,
            self.nhead,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)

        self.final_layer_norm = nn.LayerNorm(self.d_model)
        self.fc_dropout1 = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.gelu = nn.GELU()
        self.fc_dropout2 = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(self.dim_feedforward, self.d_model)

    def forward(self, x, src_mask=None, src_key_padding_mask=None, return_head_weights=False, is_causal=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=src_key_padding_mask,
            return_attn_weights=True,
            return_head_weights=return_head_weights,
            attn_mask=src_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc_dropout1(x)
        x = self.gelu(self.fc1(x))
        x = self.fc_dropout2(x)
        x = self.fc2(x)
        x = residual + x

        if return_head_weights:
            return x, attn
        else:
            return x
