
import math
import torch
import torch.nn as nn

class PositionalEncoding_forBTC(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B, L, d_model]
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0).to(dtype=x.dtype, device=x.device)

class ChomoTransformer_forBTC(nn.Module):
    """
    Transformer encoder for BTC 4h features → future return.
    """
    def __init__(self, d_in, d_model=128, nhead=4, num_layers=4, d_ff=256,
                 dropout=0.1, out_dim=1, use_last_token=True):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = PositionalEncoding_forBTC(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu',
            norm_first=True,
            layer_norm_eps=1e-5
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.use_last_token = use_last_token
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        # x: [B, L, d_in]
        B, L, _ = x.shape
        h = self.proj(x)
        h = self.pos(h)
        # causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        attn_mask = torch.full((L, L), float('-inf'), device=x.device)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        h = self.encoder(h, mask=attn_mask)          # [B, L, d_model]
        h = h[:, -1] if self.use_last_token else h.mean(dim=1)
        h = self.norm(h)
        return self.head(h)
