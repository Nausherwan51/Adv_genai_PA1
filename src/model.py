"""Pre-norm Transformer blocks and an untied, bias-free language model."""

from torch import nn

from src.attention import GroupedQuerySelfAttention
from src.layers import Embedding, Linear, RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    """Apply independent pre-normalized attention and SwiGLU residual paths."""

    def __init__(self, d_model, n_q_heads, n_kv_heads, d_ff, context_length,
                 rope_theta, norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.attention_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.attention = GroupedQuerySelfAttention(
            d_model, n_q_heads, n_kv_heads, context_length, rope_theta, device, dtype)
        self.ffn_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        """Keep the residual stream and both sublayer outputs at [B, n, d_model]."""
        x = x + self.attention(self.attention_norm(x), token_positions=token_positions)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    """Map token IDs to next-token logits using independently parameterized blocks."""

    def __init__(self, vocab_size, context_length, d_model, num_layers, n_q_heads,
                 n_kv_heads, d_ff, rope_theta, norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        if min(vocab_size, context_length, d_model, num_layers, d_ff) <= 0:
            raise ValueError("model dimensions and layer count must be positive")
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_q_heads, n_kv_heads, d_ff, context_length,
                             rope_theta, norm_eps, device, dtype)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids, token_positions=None):
        """Return unnormalized [B, n, vocab_size] logits for a nonempty context."""
        if token_ids.ndim != 2 or not 1 <= token_ids.shape[-1] <= self.context_length:
            raise ValueError("token_ids must be [batch, sequence] with sequence within context_length")
        x = self.token_embedding(token_ids)  # [B, n] -> [B, n, d_model]
        for block in self.blocks:
            x = block(x, token_positions=token_positions)
        return self.lm_head(self.final_norm(x))
