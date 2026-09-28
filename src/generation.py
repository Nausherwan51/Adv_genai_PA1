"""Autoregressive decoding with temperature and nucleus sampling."""

import math

import torch

from src.attention import softmax


def generate(model, prompt_ids, max_new_tokens, context_length, *, temperature=1.0,
             top_p=1.0, eot_token_id=None, generator=None):
    """Return the full prompt plus sampled completion, cropping only model input."""
    if prompt_ids.ndim != 1 or prompt_ids.numel() == 0 or prompt_ids.dtype != torch.long:
        raise ValueError("prompt_ids must be a nonempty one-dimensional torch.long tensor")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if context_length <= 0 or context_length != model.context_length:
        raise ValueError("context_length must equal the model's positive context_length")
    if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("require finite temperature > 0 and 0 < top_p <= 1")
    parameter = next(model.parameters(), None)
    if parameter is not None and prompt_ids.device != parameter.device:
        raise ValueError("prompt_ids must be on the model's device")
    if max_new_tokens == 0:
        return prompt_ids
    was_training = model.training
    model.eval()
    sequence = prompt_ids
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                # [n] -> [1, min(n, context_length)]; only the last logits predict a new token.
                model_input = sequence[-context_length:].unsqueeze(0)
                logits = model(model_input)[0, -1]
                if logits.dtype in (torch.float16, torch.bfloat16):
                    logits = logits.float()
                probabilities = softmax(logits / temperature, dim=-1)
                sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
                if top_p < 1:
                    cumulative = sorted_probs.cumsum(dim=-1)
                    # Keep the crossing token too: remove a token only if the prior sum reached p.
                    remove = torch.zeros_like(sorted_probs, dtype=torch.bool)
                    remove[1:] = cumulative[:-1] >= top_p
                    sorted_probs = sorted_probs.masked_fill(remove, 0)
                sorted_probs = sorted_probs / sorted_probs.sum()
                rank = torch.multinomial(sorted_probs, 1, generator=generator)
                next_token = sorted_ids[rank]  # Map sorted rank back to vocabulary ID.
                sequence = torch.cat((sequence, next_token))
                if eot_token_id is not None and next_token.item() == eot_token_id:
                    break
        return sequence
    finally:
        model.train(was_training)
