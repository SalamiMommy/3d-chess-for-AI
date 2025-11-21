"""Optimized training loop for 3D chess model with Graph Transformer."""
import os
if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import random
import torch.nn.functional as F
from tqdm import tqdm

# Import Graph Transformer
from models.graph_transformer import GraphTransformer3D, create_optimized_model, setup_rocm_optimizations
HAS_GRAPH_TRANSFORMER = True

from training.training_types import TrainingExample, TrainingConfig, BatchData, convert_examples_to_tensors
from game3d.common.shared_types import N_CHANNELS, SIZE, MAX_COORD_VALUE, MIN_COORD_VALUE

# Setup ROCm optimizations if available
if HAS_GRAPH_TRANSFORMER:
    setup_rocm_optimizations()

# Batch size optimized for RX 7900 XTX 24GB VRAM
BATCH_SIZE = 48  # Increased from 16 - fully utilizes 24GB VRAM (~18-20GB usage)
NUM_WORKERS = 4  # Parallel data loading - ROCm handles multiprocessing well
PIN_MEMORY = True  # Faster CPU-GPU transfers on ROCm


class ChessDataset(Dataset):
    """Dataset with augmentation and validation - converts numpy arrays to tensors."""

    def __init__(self, examples: List[TrainingExample], augment: bool = True):
        self.examples = examples
        self.augment = augment
        self._validate_examples()

    def _validate_examples(self):
        """Validate all training examples."""
        for i, ex in enumerate(self.examples):
            # Validate using the TrainingExample's built-in validation
            if not ex.validate():
                raise ValueError(f"Example {i} failed validation: {ex}")

            # Additional shape checks for compatibility
            if hasattr(ex.state_tensor, 'shape'):
                expected_shape = (N_CHANNELS, SIZE, SIZE, SIZE)
                if ex.state_tensor.shape != expected_shape:
                    raise ValueError(
                        f"Example {i}: Invalid state shape {ex.state_tensor.shape}, "
                        f"expected {expected_shape}"
                    )
            if hasattr(ex.from_target, 'shape'):
                if ex.from_target.shape != (729,):
                    raise ValueError(f"Example {i}: Invalid from_target shape {ex.from_target.shape}")
            if hasattr(ex.to_target, 'shape'):
                if ex.to_target.shape != (729,):
                    raise ValueError(f"Example {i}: Invalid to_target shape {ex.to_target.shape}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Use the TrainingExample's built-in tensor conversion
        tensor_dict = ex.to_tensor()

        state = tensor_dict['state']
        from_target = tensor_dict['from_target']
        to_target = tensor_dict['to_target']
        value_target = tensor_dict['value_target']

        # Apply augmentation to the state
        if self.augment:
            state = self._augment_state(state)

        return state, from_target, to_target, value_target

    def _augment_state(self, state: torch.Tensor) -> torch.Tensor:
        """Apply chess-specific augmentations."""
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            state = torch.rot90(state, k, dims=[2, 3])  # Rotate around z and y axes
        if random.random() > 0.5:
            state = torch.flip(state, dims=[3])  # Flip along x-axis
        elif random.random() > 0.5:
            state = torch.flip(state, dims=[2])  # Flip along y-axis
        return state

    def get_batch_tensor(self, indices: List[int], device: str = 'cuda') -> BatchData:
        """
        Get a batch of training examples as tensors for efficient GPU processing.

        Args:
            indices: List of indices to include in the batch
            device: Target device for tensors

        Returns:
            BatchData container with all tensors
        """
        batch_examples = [self.examples[i] for i in indices]
        return convert_examples_to_tensors(batch_examples, device)

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



class ChessTrainer:
    """Optimized trainer for 3D chess Graph Transformer."""

    def __init__(self, config: TrainingConfig, model: Optional[nn.Module] = None):
        self.config = config
        self.device = torch.device(config.device)

        Path(config.log_dir).mkdir(exist_ok=True)
        Path(config.checkpoint_dir).mkdir(exist_ok=True)

        # Use provided model or create new one
        if model is not None:
            self.model = model
        else:
            self.model = self._create_model()

        self.model = self.model.to(self.device)

        if config.compile_model and hasattr(torch, 'compile') and not isinstance(self.model, torch._dynamo.eval_frame.OptimizedModule):
            self.model = torch.compile(self.model)
            print("Model compiled with torch.compile")

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = ChessLoss(config.policy_weight, config.value_weight)
        # Fix: Updated GradScaler to use torch.amp
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision and self.device.type == 'cuda' else None
        
        # Gradient accumulation for larger effective batch size
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)

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
        """Create model with fallback to simple implementation."""
        if HAS_GRAPH_TRANSFORMER and self.config.dim is not None and self.config.depth is not None:
            # Use explicit parameters if provided
            return GraphTransformer3D(
                dim=self.config.dim,
                depth=self.config.depth,
                num_heads=self.config.num_heads or 8,
                ff_mult=self.config.ff_mult or 4,
                use_gradient_checkpointing=True,
                use_flash_attention=True
            )
        elif HAS_GRAPH_TRANSFORMER:
            # Use preset model size
            return create_optimized_model(self.config.model_size)
        else:
            # Fallback to simple model
            print("Warning: Using placeholder model - Graph Transformer not available")
            return torch.nn.Identity()

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

    def prepare_data(self, examples: Union[List[TrainingExample], Dataset]) -> tuple[Dataset, Dataset]:
        if isinstance(examples, Dataset):
            # Lazy dataset path (ReplayBuffer)
            total_len = len(examples)
            train_len = int(total_len * self.config.train_split)
            val_len = total_len - train_len
            # Use random_split for datasets
            return torch.utils.data.random_split(examples, [train_len, val_len])

        # Standard list path
        if self.config.max_examples:
            examples = examples[:self.config.max_examples]
        split_idx = int(len(examples) * self.config.train_split)
        # Ensure at least one training example if we have data
        if split_idx == 0 and len(examples) > 0:
            split_idx = 1
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        # Use appropriate datasets
        if self.config.model_type == "transformer" and HAS_GRAPH_TRANSFORMER:
            train_dataset = GraphTransformerDataset(train_examples, self.device)
            val_dataset = GraphTransformerDataset(val_examples, self.device)
        else:
            train_dataset = ChessDataset(train_examples, augment=self.config.augment_data)
            val_dataset = ChessDataset(val_examples, augment=False)

        return train_dataset, val_dataset

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Reset gradients at start
        self.optimizer.zero_grad()

        # Create progress bar for batches
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.epoch+1} [Train]",
            unit="batch",
            dynamic_ncols=True,
            leave=False
        )
        
        for batch_idx, (states, from_targets, to_targets, value_targets) in enumerate(pbar):
            states = states.to(self.device)
            from_targets = from_targets.to(self.device)
            to_targets = to_targets.to(self.device)
            value_targets = value_targets.to(self.device)

            # Mark CUDA graph step boundary before model invocation
            if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                torch.compiler.cudagraph_mark_step_begin()
            
            # Use autocast only for CUDA
            if self.device.type == 'cuda' and self.config.mixed_precision:
                with torch.amp.autocast('cuda'):
                    from_logits, to_logits, value_pred = self.model(states.clone())
                    loss = self.criterion(from_logits, to_logits, from_targets, to_targets, value_pred, value_targets)
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                # CPU or no mixed precision
                from_logits, to_logits, value_pred = self.model(states.clone())
                loss = self.criterion(from_logits, to_logits, from_targets, to_targets, value_pred, value_targets)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.update_ema()
                
                # Update progress bar with metrics
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    'lr': f"{current_lr:.2e}",
                    'grad': f"{grad_norm:.2f}" if grad_norm is not None else "N/A"
                })

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
        
        pbar.close()
        return {'loss': total_loss / num_batches}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Create progress bar for validation
        pbar = tqdm(
            val_loader,
            desc=f"Epoch {self.epoch+1} [Val]",
            unit="batch",
            dynamic_ncols=True,
            leave=False
        )
        
        with torch.no_grad():
            for states, from_targets, to_targets, value_targets in pbar:
                states = states.to(self.device)
                from_targets = from_targets.to(self.device)
                to_targets = to_targets.to(self.device)
                value_targets = value_targets.to(self.device)

                # Mark CUDA graph step boundary before model invocation
                if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                    torch.compiler.cudagraph_mark_step_begin()
                
                if self.device.type == 'cuda' and self.config.mixed_precision:
                    with torch.amp.autocast('cuda'):
                        from_logits, to_logits, value_pred = self.model(states.clone())
                        loss = self.criterion(from_logits, to_logits, from_targets, to_targets, value_pred, value_targets)
                else:
                    from_logits, to_logits, value_pred = self.model(states.clone())
                    loss = self.criterion(from_logits, to_logits, from_targets, to_targets, value_pred, value_targets)

                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar with current average loss
                pbar.set_postfix({'loss': f"{total_loss / num_batches:.4f}"})
        
        pbar.close()
        return {'loss': total_loss / num_batches}

    def train(self, examples: Union[List[TrainingExample], Dataset]) -> Dict[str, Any]:
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

        # Create progress bar for epochs
        epoch_pbar = tqdm(
            range(self.config.epochs),
            desc="Training Epochs",
            unit="epoch",
            dynamic_ncols=True
        )
        
        for epoch in epoch_pbar:
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
                        epoch_pbar.write(f"Early stopping at epoch {epoch + 1}")
                        epoch_pbar.close()
                        break
                
                # Update epoch progress bar with metrics
                epoch_pbar.set_postfix({
                    'train_loss': f"{train_metrics['loss']:.4f}",
                    'val_loss': f"{val_metrics['loss']:.4f}",
                    'best_val': f"{self.best_val_loss:.4f}",
                    'no_improve': no_improvement
                })

            training_history.append({'epoch': epoch, 'train': train_metrics, 'val': val_metrics if (epoch + 1) % self.config.validate_every == 0 else None})
        
        epoch_pbar.close()
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

    def save_model(self, path: str) -> None:
        to_save = self.ema_model if self.ema_model is not None else self.model
        if hasattr(to_save, "_orig_mod"):
            to_save = to_save._orig_mod
        torch.save(to_save.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        cleaned = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
        self.model.load_state_dict(cleaned)
        if self.ema_model:
            self.ema_model.load_state_dict(cleaned)
        print(f"Model loaded from {path}")

def load_latest_checkpoint(model: nn.Module, optimizer: optim.Optimizer) -> Optional[Dict[str, Any]]:
    """Load the latest checkpoint if it exists."""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find the latest checkpoint file
    checkpoint_files = list(Path(checkpoint_dir).glob("checkpoint_epoch_*.pt"))
    if not checkpoint_files:
        return None
    
    # Get the latest checkpoint by modification time
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    return checkpoint

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
    examples = generate_training_data_parallel(model, num_games=num_games, device=config.device)
    trainer = ChessTrainer(config)
    return trainer.train(examples)

class GraphTransformerDataset(Dataset):
    """Dataset optimized for Graph Transformer - keeps data on GPU."""

    def __init__(self, examples: List[TrainingExample], device: str = 'cuda'):
        self.examples = examples
        self.device = device
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess data to minimize device transfers."""
        # Validate examples first
        for i, ex in enumerate(self.examples):
            if not ex.validate():
                raise ValueError(f"Example {i} failed validation: {ex}")

        self.states = []
        self.from_targets = []
        self.to_targets = []
        self.value_targets = []

        for ex in self.examples:
            # Use the TrainingExample's built-in tensor conversion
            tensor_dict = ex.to_tensor(self.device)

            self.states.append(tensor_dict['state'])
            self.from_targets.append(tensor_dict['from_target'])
            self.to_targets.append(tensor_dict['to_target'])
            self.value_targets.append(tensor_dict['value_target'])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.from_targets[idx],
            self.to_targets[idx],
            self.value_targets[idx]
        )

    def get_batch_tensor(self, indices: List[int]) -> BatchData:
        """
        Get a batch of training examples as tensors.

        Args:
            indices: List of indices to include in the batch

        Returns:
            BatchData container with all tensors
        """
        batch_examples = [self.examples[i] for i in indices]
        return convert_examples_to_tensors(batch_examples, self.device)

if __name__ == "__main__":
    config = TrainingConfig()
    results = train_with_self_play(config, num_games=5)
    print(f"Training completed! Best val loss: {results['best_val_loss']}")
