"""Model checkpointing: save/load on schedule, resume training."""

import os
import torch
import time
import glob

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, step, extra=None):
    tstr = time.strftime("%Y%m%d_%H%M%S")
    fname = f"model_ep{epoch}_step{step}_{tstr}.pt"
    path = os.path.join(CHECKPOINT_DIR, fname)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "extra": extra,
    }
    torch.save(state, path)
    print(f"[Checkpoint] Saved: {path}")
    return path

def load_latest_checkpoint(model, optimizer=None):
    files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "model_ep*.pt")))
    if not files:
        print("[Checkpoint] No checkpoint found.")
        return None
    latest = files[-1]
    print(f"[Checkpoint] Loading: {latest}")
    state = torch.load(latest, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return {
        "epoch": state.get("epoch", 0),
        "step": state.get("step", 0),
        "extra": state.get("extra", None),
    }
