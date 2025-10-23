"""Optimized training loop for 3D chess model."""
import os
if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import random
import torch.nn.functional as F
from models.resnet3d import OptimizedResNet3D, LightweightResNet3D
from training.self_play import generate_training_data
from training.checkpoint import save_checkpoint, load_latest_checkpoint
from training.types import TrainingExample
from game3d.common.constants import N_CHANNELS, SIZE

BATCH_SIZE = 32
NUM_WORKERS = 0  # Changed to 0 to avoid CUDA multiprocessing issues
PIN_MEMORY = False  # Changed to False to avoid CUDA multiprocessing issues

class ChessDataset(Dataset):
    """Dataset with augmentation and validation."""

    def __init__(self, examples: List[TrainingExample], augment: bool = True):
        self.examples = examples
        self.augment = augment
        self._validate_examples()

    def _validate_examples(self):
        for i, ex in enumerate(self.examples):
            if ex.state_tensor.shape != (N_CHANNELS, SIZE, SIZE, SIZE):
                raise ValueError(f"Example {i}: Invalid state shape")
            if ex.from_target.shape != (729,):
                raise ValueError(f"Example {i}: Invalid from_target shape")
            if ex.to_target.shape != (729,):
                raise ValueError(f"Example {i}: Invalid to_target shape")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        state = ex.state_tensor
        if self.augment:
            state = self._augment_state(state)
        return (
            state,
            ex.from_target,
            ex.to_target,
            torch.tensor(ex.value_target, dtype=torch.float32)
        )

    def _augment_state(self, state: torch.Tensor) -> torch.Tensor:
        """Apply chess-specific augmentations."""
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            state = torch.rot90(state, k, dims=[2, 3])
        if random.random() > 0.5:
            state = torch.flip(state, dims=[3])
        elif random.random() > 0.5:
            state = torch.flip(state, dims=[2])
        return state

class ChessLoss(nn.Module):
    """Combined loss for factorized policy and value."""

    def __init__(self, policy_weight: float = 1.0, value_weight: float = 1.0):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.value_loss = nn.MSELoss()

    def forward(
        self,
        from_logits: torch.Tensor,  # (B, 729)
        to_logits: torch.Tensor,    # (B, 729)
        from_targets: torch.Tensor, # (B, 729)
        to_targets: torch.Tensor,   # (B, 729)
        value_pred: torch.Tensor,   # (B,)
        value_target: torch.Tensor, # (B,)
    ) -> torch.Tensor:
        # Policy losses
        loss_from = -(from_targets * F.log_softmax(from_logits, dim=1)).sum(dim=1).mean()
        loss_to = -(to_targets * F.log_softmax(to_logits, dim=1)).sum(dim=1).mean()
        policy_loss = loss_from + loss_to

        # Value loss
        value_loss = self.value_loss(value_pred.squeeze(), value_target)

        total_loss = self.policy_weight * policy_loss + self.value_weight * value_loss
        return total_loss

@dataclass
class TrainingConfig:
    model_type: str = "optimized"
    blocks: int = 15
    channels: int = 256
    n_moves: int = 1_000_000
    batch_size: int = BATCH_SIZE
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    warmup_epochs: int = 5
    policy_weight: float = 1.0
    value_weight: float = 1.0
    train_split: float = 0.8
    augment_data: bool = True
    max_examples: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    compile_model: bool = True
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    validate_every: int = 1
    gradient_clipping: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999

class ChessTrainer:
    """Optimized trainer for 3D chess neural networks."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        Path(config.log_dir).mkdir(exist_ok=True)
        Path(config.checkpoint_dir).mkdir(exist_ok=True)

        self.model = self._create_model().to(self.device)
        if config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = ChessLoss(config.policy_weight, config.value_weight)
        # Fix: Updated GradScaler to use torch.amp
        self.scaler = torch.amp.GradScaler('cpu') if config.mixed_precision else None

        if config.use_ema:
            self.ema_model = self._create_model()
            state_dict = self.model.state_dict()
            cleaned = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
            self.ema_model.load_state_dict(cleaned)
            self.ema_model.to(self.device)
            self.ema_model.eval()
        else:
            self.ema_model = None

        self.writer = SummaryWriter(config.log_dir)
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _create_model(self) -> nn.Module:
        if self.config.model_type == "optimized":
            return OptimizedResNet3D(self.config.blocks, self.config.n_moves, self.config.channels)
        else:
            return LightweightResNet3D(self.config.blocks, self.config.n_moves, self.config.channels)

    def _create_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs - self.config.warmup_epochs, eta_min=self.config.learning_rate * 0.01
        )
        if self.config.warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, total_iters=self.config.warmup_epochs
            )
            return optim.lr_scheduler.SequentialLR(
                self.optimizer, [warmup_scheduler, main_scheduler], milestones=[self.config.warmup_epochs]
            )
        return main_scheduler

    def update_ema(self) -> None:
        if self.ema_model is None:
            return
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.config.ema_decay).add_(model_param.data, alpha=1 - self.config.ema_decay)

    def prepare_data(self, examples: List[TrainingExample]) -> tuple[ChessDataset, ChessDataset]:
        if self.config.max_examples:
            examples = examples[:self.config.max_examples]
        split_idx = int(len(examples) * self.config.train_split)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        return ChessDataset(train_examples, augment=self.config.augment_data), ChessDataset(val_examples, augment=False)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for states, from_targets, to_targets, value_targets in train_loader:
            states = states.to(self.device)
            from_targets = from_targets.to(self.device)
            to_targets = to_targets.to(self.device)
            value_targets = value_targets.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cpu', enabled=self.config.mixed_precision):
                from_logits, to_logits, value_pred = self.model(states)
                loss = self.criterion(from_logits, to_logits, from_targets, to_targets, value_pred, value_targets)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                self.optimizer.step()

            self.update_ema()

            total_loss += loss.item()
            num_batches += 1

        return {'loss': total_loss / num_batches}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for states, from_targets, to_targets, value_targets in val_loader:
                states = states.to(self.device)
                from_targets = from_targets.to(self.device)
                to_targets = to_targets.to(self.device)
                value_targets = value_targets.to(self.device)

                with torch.amp.autocast('cpu', enabled=self.config.mixed_precision):
                    from_logits, to_logits, value_pred = self.model(states)
                    loss = self.criterion(from_logits, to_logits, from_targets, to_targets, value_pred, value_targets)

                total_loss += loss.item()
                num_batches += 1

        return {'loss': total_loss / num_batches}

    def train(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        train_dataset, val_dataset = self.prepare_data(examples)
        # Fix: Use 0 workers to avoid CUDA multiprocessing issues
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )

        training_history = []
        patience = 10
        no_improvement = 0

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            train_metrics = self.train_epoch(train_loader)
            self.scheduler.step()

            if (epoch + 1) % self.config.validate_every == 0:
                val_metrics = self.validate(val_loader)
                self._log_epoch_metrics(epoch, train_metrics, val_metrics)

                if (epoch + 1) % self.config.save_every == 0:
                    self._save_checkpoint(epoch, val_metrics)

                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self._save_best_model(epoch, val_metrics)
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            training_history.append({'epoch': epoch, 'train': train_metrics, 'val': val_metrics if (epoch + 1) % self.config.validate_every == 0 else None})

        return {'best_val_loss': self.best_val_loss, 'training_history': training_history, 'final_model_path': str(Path(self.config.checkpoint_dir) / "best_model.pt")}

    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        print(f"Epoch {epoch+1}: Train Loss {train_metrics['loss']:.4f}, Val Loss {val_metrics['loss']:.4f}")
        self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
        self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        # choose the model whose weights we want to persist
        to_save = self.ema_model if self.ema_model is not None else self.model

        # unwrap torch.compile wrapper if necessary
        if hasattr(to_save, "_orig_mod"):                 # torch >= 2.0 compiled
            to_save = to_save._orig_mod

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": to_save.state_dict(),   # <- clean keys
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def _save_best_model(self, epoch: int, metrics: Dict[str, float]) -> None:
        to_save = self.ema_model if self.ema_model is not None else self.model

        if hasattr(to_save, "_orig_mod"):                 # compiled wrapper
            to_save = to_save._orig_mod

        best_ckpt = {
            "epoch": epoch,
            "model_state_dict": to_save.state_dict(),   # <- clean keys
            "metrics": metrics,
        }
        path = Path(self.config.checkpoint_dir) / "best_model.pt"
        torch.save(best_ckpt, path)
        print(f"Best model saved: {path}")

    def load_best_model(self) -> None:
        path = Path(self.config.checkpoint_dir) / "best_model.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.ema_model and 'ema_model_state_dict' in checkpoint:
                self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            print(f"Loaded best model from {path}")

def load_or_init_model(config: TrainingConfig) -> tuple[nn.Module, optim.Optimizer, int]:
    """Build model + optim, load weights if present, return (model, optim, epoch)."""
    model = ChessTrainer(config)._create_model().to(config.device)
    optimizer = optim.AdamW(model.parameters(),
                          lr=config.learning_rate,
                          weight_decay=config.weight_decay)

    checkpoint = load_latest_checkpoint(model, optimizer)   # your helper
    if checkpoint:
        # ---------- remove _orig_mod. prefix (compiled wrapper) ----------
        raw = checkpoint["model_state_dict"]
        cleaned = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in raw.items()}
        model.load_state_dict(cleaned)
        # ------------------------------------------------------------------
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, optimizer, checkpoint.get("epoch", 0)

    return model, optimizer, 0

def train_with_self_play(config: TrainingConfig, num_games: int = 10) -> Dict[str, Any]:
    model, _, _ = load_or_init_model(config)
    examples = generate_training_data(model, num_games=num_games, device=config.device)
    trainer = ChessTrainer(config)
    return trainer.train(examples)

if __name__ == "__main__":
    config = TrainingConfig()
    results = train_with_self_play(config, num_games=5)
    print(f"Training completed! Best val loss: {results['best_val_loss']}")
