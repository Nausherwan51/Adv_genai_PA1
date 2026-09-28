"""Manual loss, AdamW updates, learning-rate scheduling, and gradient clipping."""

import math
from numbers import Integral

import torch


def cross_entropy(logits, targets):
    """Return mean negative log likelihood without building full log probabilities."""
    if logits.ndim < 1 or targets.shape != logits.shape[:-1] or targets.numel() == 0:
        raise ValueError("targets must match nonempty leading logits dimensions")
    if targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("targets must contain integer class IDs")
    if logits.shape[-1] == 0 or torch.any(targets < 0) or torch.any(targets >= logits.shape[-1]):
        raise ValueError("target outside vocabulary range")
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    # [..., vocab] -> [...]: gather just the correct class at each position.
    target_logits = logits.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
    return (torch.logsumexp(logits, dim=-1) - target_logits).mean()


class AdamW(torch.optim.Optimizer):
    """Adam with per-parameter bias correction and decoupled weight decay."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self._validate_group(defaults)
        super().__init__(params, defaults)
        for group in self.param_groups:
            self._validate_group(group)

    @staticmethod
    def _validate_group(group):
        """Check effective settings, including overrides and later scheduler edits."""
        for name in ("lr", "eps", "weight_decay"):
            if not math.isfinite(group[name]) or group[name] < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if len(group["betas"]) != 2 or any(not 0 <= beta < 1 for beta in group["betas"]):
            raise ValueError("betas must both lie in [0, 1)")

    @torch.no_grad()
    def step(self, closure=None):
        """Update only parameters with dense gradients; evaluate a closure once."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Validate all groups before changing any parameter.
        for group in self.param_groups:
            self._validate_group(group)
            for parameter in group["params"]:
                if parameter.grad is not None and parameter.grad.layout != torch.strided:
                    raise RuntimeError("AdamW does not support sparse gradients")
                
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, decay = group["lr"], group["eps"], group["weight_decay"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                count = state["step"]
                first_moment = state["exp_avg"]
                second_moment = state["exp_avg_sq"]
                first_moment.mul_(beta1).add_(gradient, alpha=1 - beta1)
                second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                first_unbiased = first_moment / (1 - beta1 ** count)
                second_unbiased = second_moment / (1 - beta2 ** count)
                parameter.mul_(1 - lr * decay)
                parameter.addcdiv_(first_unbiased, second_unbiased.sqrt() + eps, value=-lr)
        return loss


def get_lr_cosine_schedule(step, learning_rate_max, learning_rate_min, warmup_steps, cosine_steps):
    """Warm up linearly, decay with cosine, and hold the minimum after the endpoint."""
    if any(isinstance(value, bool) or not isinstance(value, Integral)
           for value in (step, warmup_steps, cosine_steps)):
        raise TypeError("step indices must be integers")
    if step < 0 or not 0 <= warmup_steps < cosine_steps:
        raise ValueError("require step >= 0 and 0 <= warmup_steps < cosine_steps")
    if not 0 <= learning_rate_min <= learning_rate_max or not math.isfinite(learning_rate_max):
        raise ValueError("require finite 0 <= learning_rate_min <= learning_rate_max")
    if step < warmup_steps:
        return float(learning_rate_max * step / warmup_steps)
    if step > cosine_steps:
        return float(learning_rate_min)
    progress = (step - warmup_steps) / (cosine_steps - warmup_steps)
    return float(learning_rate_min + 0.5 * (1 + math.cos(math.pi * progress))
                 * (learning_rate_max - learning_rate_min))


@torch.no_grad()
def gradient_clipping(parameters, max_l2_norm):
    """Scale all gradients by one factor and return their pre-clipping global norm."""
    if not max_l2_norm > 0:
        raise ValueError("max_l2_norm must be positive")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    # Sum in float64 to avoid overflow when gradients use a low precision dtype.
    norm = math.sqrt(sum(gradient.double().square().sum().item() for gradient in gradients))
    if norm > max_l2_norm:
        scale = max_l2_norm / (norm + 1e-6)
        for gradient in gradients:
            gradient.mul_(scale)
    return float(norm)
