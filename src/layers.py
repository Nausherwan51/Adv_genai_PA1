import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        std = (2 / (in_features + out_features)) ** 0.5

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
    
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        # TODO: create the trainable embedding matrix
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        # TODO: initialize according to the PA spec
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # TODO: look up the rows of self.weight selected by token_ids
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, norm_eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = norm_eps

        # TODO: create the learned scale/gain parameter
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO:
        # 1. compute mean(x^2) over the final dimension
        # 2. divide by sqrt(mean_square + eps)
        # 3. multiply by the learned scale
        in_dtype= x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        x_ms = torch.mean(x ** 2, dim=-1, keepdim=True)
        x = x / torch.sqrt((x_ms + self.eps))
        x = x * self.weight
        return x.to(in_dtype)

def silu(x: torch.Tensor) -> torch.Tensor:
    # TODO: implement SiLU directly from its mathematical definition
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        # TODO: create the three custom Linear sublayers
        self.w_gate = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w_up = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w_down = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO:
        gate = silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        output = self.w_down(hidden)
        return output
