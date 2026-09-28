import torch
from torch import nn
 
 
class RotaryPositionalEmbedding(nn.Module): 
    def __init__( 
        self, 
        rope_theta: float, 
        head_dim: int, 
        context_length: int, 
        device: torch.device | None = None, 
    ): 
        super().__init__() 
 
        # TODO: validate head_dim 
        # RoPE uses adjacent coordinate pairs, so head_dim must be even 
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
 
        self.rope_theta = rope_theta 
        self.head_dim = head_dim 
        self.context_length = context_length 
 
        # Number of 2D coordinate pairs 
        num_pairs = head_dim // 2 
 
        # TODO: compute one frequency per pair 
        # Expected shape: [num_pairs] 
        frequencies = []

        for k in range(num_pairs):
            freq = self.rope_theta ** (-(2 * k) / head_dim)
            frequencies.append(freq)

        frequencies = torch.tensor(
            frequencies,
            device=device,
            dtype=torch.float32
        )
 
        # TODO: create all supported token positions 
        # Expected shape: [context_length] 
        positions = []

        for k in range(context_length):
            positions.append(k)

        positions = torch.tensor(
            positions,
            device=device,
            dtype=torch.float32
        )
 
        # TODO: combine positions and frequencies 
        # to get rotation angles 
        # 
        # Desired shape: 
        # [context_length, num_pairs] 
        angles = [] 
 
        for pos in positions: 
            angle = []

            for freq in frequencies: 
                angle.append(pos * freq)
                 
            angles.append(torch.stack(angle))
 
        angles = torch.stack(angles)
 
        # TODO: precompute compact sine/cosine tables 
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)
 
        # Store them as buffers, not Parameters 
        self.register_buffer( 
            "cos_cache", 
            cos_cache, 
            persistent=False 
        ) 
 
        self.register_buffer( 
            "sin_cache", 
            sin_cache, 
            persistent=False 
        ) 
 
    def forward( 
        self, 
        x: torch.Tensor, 
        token_positions: torch.Tensor, 
    ) -> torch.Tensor: 
 
        # x expected shape: 
        # [..., sequence_length, head_dim] 
 
        # token_positions expected shape: 
        # [..., sequence_length] 
        # 
        # examples: 
        # [sequence_length] 
        # [batch_size, sequence_length] 
 
        # TODO: validate final x dimension 
        # x.shape[-1] should equal self.head_dim 
        if x.shape[-1] != self.head_dim:
            raise ValueError("x has the wrong head_dim")
 
        # TODO: validate token_positions integer dtype 
        if token_positions.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("token_positions must contain integers")
 
        # TODO: validate positions are in: 
        # [0, context_length) 
        if torch.any(token_positions < 0):
            raise ValueError("token positions outside [0, context_length)")

        if torch.any(token_positions >= self.context_length):
            raise ValueError("token positions outside [0, context_length)")

        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError("token_positions must match sequence length")
 
        # ------------------------------------------------- 
        # 1. Look up the sin/cos rows for these positions 
        # ------------------------------------------------- 
 
        # Expected final dimension: 
        num_pairs = self.head_dim // 2 
        # Convert small integer dtypes so uint8 is not treated as a Boolean mask.
        positions = token_positions.to(device=self.cos_cache.device, dtype=torch.long)
        cos = self.cos_cache[positions] 
        sin = self.sin_cache[positions] 
 
        # At this point possible shapes are: 
        # 
        # token_positions: [n] 
        # cos/sin:         [n, num_pairs] 
        # 
        # OR 
        # 
        # token_positions: [B, n] 
        # cos/sin:         [B, n, num_pairs] 
 
        # ------------------------------------------------- 
        # 2. Split x into adjacent pairs 
        # ------------------------------------------------- 
 
        # even coordinates: 
        # x0, x2, x4, ... 
        x_even = x[..., 0::2] 
        
        # odd coordinates: 
        # x1, x3, x5, ... 
        x_odd  = x[..., 1::2] 

        # Both should have shape: 
        # [..., sequence_length, num_pairs] 
 
        # ------------------------------------------------- 
        # 3. Make cos/sin broadcast over extra head dims 
        # ------------------------------------------------- 
 
        # TODO: 
        # Insert singleton dimensions where needed 
        # so cos/sin can broadcast against x_even/x_odd 
        # 
        # Example: 
        # 
        # x_even: 
        # [B, H, n, num_pairs] 
        # 
        # cos from positions: 
        # [B, n, num_pairs] 
        # 
        # reshape/unsqueeze to: 
        # [B, 1, n, num_pairs] 
        # 
        # so it broadcasts across H 
         
        while cos.ndim < x_even.ndim: 
            cos = cos.unsqueeze(-3) 
            sin = sin.unsqueeze(-3) 
 
        # ------------------------------------------------- 
        # 4. Apply the 2D rotation pairwise 
        # ------------------------------------------------- 
 
        # For pair (a, b): 
        # 
        # a' = a*cos - b*sin 
        # b' = a*sin + b*cos 
 
        rotated_even = x_even * cos - x_odd * sin 
        rotated_odd = x_even * sin + x_odd * cos 
 
        # ------------------------------------------------- 
        # 5. Interleave the pairs back together 
        # ------------------------------------------------- 
 
        # Create output with same shape as x 
        out = torch.empty_like(x) 
 
        # Put rotated even coordinates back into: 
 
        out[..., 0::2] = rotated_even 
        out[..., 1::2] = rotated_odd 
 
        return out 
    
import torch


def softmax(tensor, dim: int) -> torch.Tensor:
    """
    Numerically stable softmax implementation.
    """
    # Subtract the max for numerical stability
    max_val, _ = torch.max(tensor, dim=dim, keepdim=True)
    exp_tensor = torch.exp(tensor - max_val)
    sum_exp = torch.sum(exp_tensor, dim=dim, keepdim=True)
    return exp_tensor / sum_exp


def scaled_dot_product_attention(q, k, v, mask=None):
    """Attend over keys with broadcastable leading dimensions and an allowed mask."""
    if min(q.ndim, k.ndim, v.ndim) < 2:
        raise ValueError("inputs need sequence and feature dimensions")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] == 0:
        raise ValueError("query and key feature dimensions must match and be positive")
    if k.shape[-2] != v.shape[-2] or k.shape[-2] == 0:
        raise ValueError("key and value sequence lengths must match and be positive")
    # (..., n_q, d_k) @ (..., d_k, n_kv) gives (..., n_q, n_kv).
    d_k = k.shape[-1]
    k_t = k.transpose(-2, -1)
    qk_t = torch.matmul(q, k_t)
    scaled_attention_logits = qk_t / (d_k ** 0.5)

    if mask is not None:

        # mask must be Boolean
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor")

        if mask.ndim < 2 or mask.shape[-2:] != (q.shape[-2], k.shape[-2]):
            raise ValueError("mask final dimensions must match query/key sequence lengths")

        # Every query must have at least one allowed key
        if not torch.all(mask.any(dim=-1)):
            raise ValueError(
                "Every query must have at least one unmasked key"
            )

        # False positions should get zero probability after softmax
        scaled_attention_logits = scaled_attention_logits.masked_fill(
            ~mask,
            float("-inf")
        )

    softmax_scores = softmax(
        scaled_attention_logits,
        dim=-1
    )

    output = torch.matmul(softmax_scores, v)

    return output


class GroupedQuerySelfAttention(nn.Module):
    """Causal attention sharing each key/value head across a group of queries."""

    def __init__(self, d_model, n_q_heads, n_kv_heads, context_length,
                 rope_theta, device=None, dtype=None):
        super().__init__()
        from src.layers import Linear

        if min(d_model, n_q_heads, n_kv_heads, context_length) <= 0:
            raise ValueError("model, head counts, and context length must be positive")
        if d_model % n_q_heads or n_q_heads % n_kv_heads:
            raise ValueError("d_model must divide into query heads, and query heads into KV groups")
        self.d_model = d_model
        self.n_kv_heads = n_kv_heads
        self.group_size = n_q_heads // n_kv_heads
        self.head_dim = d_model // n_q_heads
        self.context_length = context_length
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(rope_theta, self.head_dim, context_length, device)

    def forward(self, x, token_positions=None):
        """Project, rotate Q/K, attend causally, then restore query-head order."""
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape [batch, sequence, d_model]")
        batch, sequence, _ = x.shape
        if not 1 <= sequence <= self.context_length:
            raise ValueError("sequence length outside supported context")
        if token_positions is None:
            token_positions = torch.arange(sequence, device=x.device)

        # x: [B, n, d_model]. Split consecutive query heads into (h_kv, g).
        q = self.q_proj(x).reshape(batch, sequence, self.n_kv_heads, self.group_size, self.head_dim)
        q = q.permute(0, 2, 3, 1, 4)  # [B, h_kv, g, n, d_h]
        k = self.k_proj(x).reshape(batch, sequence, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, sequence, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # k/v: [B, h_kv, n, d_h]; only Q and K encode positions.
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        positions = torch.arange(sequence, device=x.device)
        causal_mask = positions[:, None] >= positions[None, :]
        # Singleton group axes share K/V without repeating heads.
        # Inside attention, scores/probabilities: [B, h_kv, g, n, n].
        output = scaled_dot_product_attention(q, k.unsqueeze(2), v.unsqueeze(2), causal_mask)
        # [B, h_kv, g, n, d_h] -> [B, n, h_kv, g, d_h], preserving (h, r) order.
        output = output.permute(0, 3, 1, 2, 4).reshape(batch, sequence, self.d_model)
        return self.out_proj(output)
