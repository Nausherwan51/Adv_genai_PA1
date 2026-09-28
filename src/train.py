"""Configurable training entry point: run with ``python -m src.train --help``."""

import argparse
import json

import torch

from src.checkpoint import load_checkpoint, save_checkpoint
from src.data import get_batch, load_token_array
from src.model import TransformerLM
from src.optim import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping


def evaluate_validation(model, dataset, batch_size, sequence_length, device,
                        generator, num_validation_batches):
    """Average exactly the requested batches under inference mode, restoring mode."""
    if num_validation_batches <= 0:
        raise ValueError("num_validation_batches must be positive")
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            total_loss = 0.0
            for _ in range(num_validation_batches):
                x, y = get_batch(dataset, batch_size, sequence_length, device, generator)
                total_loss += cross_entropy(model(x), y).item()
        return total_loss / num_validation_batches
    finally:
        model.train(was_training)


def train(model, optimizer, train_tokens, val_tokens, train_generator, val_generator,
          *, num_steps, batch_size, sequence_length, device, checkpoint_path,
          next_step=0, gradient_accumulation_steps=1, learning_rate_max=3e-4,
          learning_rate_min=3e-5, warmup_steps=200, cosine_steps=9999,
          max_grad_norm=1.0, eval_interval=100, log_interval=10,
          checkpoint_interval=100, num_validation_batches=10):
    
    """Run complete updates, then validate, log, and checkpoint in manual order."""
    positive_values = (batch_size, sequence_length, gradient_accumulation_steps,
                       eval_interval, log_interval, checkpoint_interval, num_validation_batches)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in positive_values):
        raise ValueError("batch sizes, accumulation, and intervals must be positive integers")
    if not 0 <= next_step <= num_steps or sequence_length > model.context_length:
        raise ValueError("invalid step range or sequence length")
    for step in range(next_step, num_steps):
        model.train()
        lr = get_lr_cosine_schedule(step, learning_rate_max, learning_rate_min,
                                    warmup_steps, cosine_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        train_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            x, y = get_batch(train_tokens, batch_size, sequence_length, device, train_generator)
            # logits: [B, n, vocab]; targets: [B, n]. Average across microbatches.
            microbatch_loss = cross_entropy(model(x), y)
            (microbatch_loss / gradient_accumulation_steps).backward()
            train_loss += microbatch_loss.detach().item()
        train_loss /= gradient_accumulation_steps
        grad_norm = gradient_clipping(model.parameters(), max_grad_norm)
        optimizer.step()

        completed_steps = step + 1
        final_step = completed_steps == num_steps
        should_validate = final_step or completed_steps % eval_interval == 0
        should_log = should_validate or completed_steps % log_interval == 0
        should_checkpoint = final_step or completed_steps % checkpoint_interval == 0
        metrics = dict(completed_steps=completed_steps, train_loss=train_loss,
                       learning_rate=lr, preclip_grad_norm=grad_norm)
        if should_validate:
            metrics["validation_loss"] = evaluate_validation(
                model, val_tokens, batch_size, sequence_length, device,
                val_generator, num_validation_batches)
        if should_log:
            print(json.dumps(metrics), flush=True)
        if should_checkpoint:
            # Validation must advance its generator before that state is saved.
            save_checkpoint(model, optimizer, completed_steps, train_generator,
                            val_generator, checkpoint_path)
    return num_steps


def main():
    """Construct configurable local/cloud training objects and optionally resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--resume", help="Checkpoint to restore; use matching model/training settings")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64", "bfloat16", "float16"), default="float32")
    integer_defaults = dict(vocab_size=8192, context_length=256, d_model=512,
                            num_layers=4, n_q_heads=16, n_kv_heads=4, d_ff=1344,
                            num_steps=10000, batch_size=16, sequence_length=256,
                            gradient_accumulation_steps=16, warmup_steps=200,
                            cosine_steps=9999, eval_interval=100, log_interval=10,
                            checkpoint_interval=100, num_validation_batches=10,
                            seed=0, train_seed=1, val_seed=2)
    float_defaults = dict(rope_theta=10000.0, norm_eps=1e-5, learning_rate_max=3e-4,
                          learning_rate_min=3e-5, beta1=0.9, beta2=0.95,
                          adam_eps=1e-8, weight_decay=0.1, max_grad_norm=1.0)
    for name, default in integer_defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in float_defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train_generator = torch.Generator().manual_seed(args.train_seed)
    val_generator = torch.Generator().manual_seed(args.val_seed)
    model = TransformerLM(args.vocab_size, args.context_length, args.d_model,
                          args.num_layers, args.n_q_heads, args.n_kv_heads, args.d_ff,
                          args.rope_theta, args.norm_eps, args.device, getattr(torch, args.dtype))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate_max,
                      betas=(args.beta1, args.beta2), eps=args.adam_eps,
                      weight_decay=args.weight_decay)
    next_step = 0
    if args.resume:
        next_step = load_checkpoint(args.resume, model, optimizer, train_generator, val_generator)
    training_names = ("num_steps", "batch_size", "sequence_length", "device",
                      "gradient_accumulation_steps", "learning_rate_max", "learning_rate_min",
                      "warmup_steps", "cosine_steps", "max_grad_norm", "eval_interval",
                      "log_interval", "checkpoint_interval", "num_validation_batches")
    train(model, optimizer, load_token_array(args.train_path), load_token_array(args.validation_path),
          train_generator, val_generator, checkpoint_path=args.checkpoint_path,
          next_step=next_step, **{name: getattr(args, name) for name in training_names})


if __name__ == "__main__":
    main()
