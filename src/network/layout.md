Input  [B, L, d_in]  〈— OHLCV + static indicators(ST, VRB ...)
   │
   ▼
Linear projection  
(d_in → d_model)
   │
   ▼
Sin/Cos Positional Encoding  
(add to sequence, length L)
   │
   ▼
TransformerEncoder × num_layers
   • EncoderLayer:
     - (Pre‑LN) LayerNorm
     - Multi‑Head Self‑Attention  (causal mask)
     - Dropout + Residual
     - (Pre‑LN) LayerNorm
     - Feed‑Forward (GELU, dim=d_ff)
     - Dropout + Residual
   │
   ▼
Sequence pooling
   • use_last_token ? take h[:, -1, :] : mean over time
   │
   ▼
LayerNorm
   │
   ▼
Linear head  d_model → out_dim(=1)   〈— log‑return
Output  [B, 1]
