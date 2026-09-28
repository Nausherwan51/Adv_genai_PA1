"""Checkpoint all state needed to resume at an optimizer-step boundary."""

import torch


def save_checkpoint(model, optimizer, next_step, train_generator, val_generator, out):
    """Save parameters, optimizer moments, next step, and independent sampler states."""
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "next_step": next_step,
        "train_generator": train_generator.get_state(),
        "val_generator": val_generator.get_state(),
    }, out)


def load_checkpoint(src, model, optimizer, train_generator, val_generator):
    """Restore a path or binary stream checkpoint and return its next global step."""
    # CPU loading also keeps generator byte states on the device set_state expects.
    state = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    train_generator.set_state(state["train_generator"])
    val_generator.set_state(state["val_generator"])
    return state["next_step"]
