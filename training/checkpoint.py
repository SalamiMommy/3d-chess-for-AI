"""Model checkpointing: save/load on schedule, resume training."""

import os
import torch
import time
import glob
from pathlib import Path

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, scheduler, epoch, step, metrics=None, run_name=None):
    """Save checkpoint with unique naming."""
    if run_name:
        checkpoint_dir = Path(CHECKPOINT_DIR) / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        fname = f"model_ep{epoch}_step{step}.pt"
        path = checkpoint_dir / fname
    else:
        tstr = time.strftime("%Y%m%d_%H%M%S")
        fname = f"model_ep{epoch}_step{step}_{tstr}.pt"
        path = Path(CHECKPOINT_DIR) / fname

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        "timestamp": time.time()
    }
    torch.save(state, path)
    print(f"[Checkpoint] Saved: {path}")
    return path

def load_latest_checkpoint(model, optimizer=None, scheduler=None, run_name=None):
    """Load latest checkpoint, optionally from specific run."""
    if run_name:
        checkpoint_dir = Path(CHECKPOINT_DIR) / run_name
        pattern = str(checkpoint_dir / "model_ep*.pt")
    else:
        pattern = str(Path(CHECKPOINT_DIR) / "model_ep*.pt")

    files = sorted(glob.glob(pattern))
    if not files:
        print("[Checkpoint] No checkpoint found.")
        return None

    latest = files[-1]
    print(f"[Checkpoint] Loading: {latest}")
    state = torch.load(latest, map_location="cpu")

    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return {
        "epoch": state.get("epoch", 0),
        "step": state.get("step", 0),
        "metrics": state.get("metrics", None),
        "path": latest
    }

def find_checkpoints(run_name=None, pattern="model_ep*.pt"):
    """Find all checkpoints for a run or globally."""
    if run_name:
        checkpoint_dir = Path(CHECKPOINT_DIR) / run_name
        search_pattern = str(checkpoint_dir / pattern)
    else:
        search_pattern = str(Path(CHECKPOINT_DIR) / pattern)

    return sorted(glob.glob(search_pattern))
