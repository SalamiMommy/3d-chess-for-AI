import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json
import time
from contextlib import contextmanager
import random
import torch.nn.functional as F

from game3d.game3d import Game3D
from game3d.game.gamestate import GameState
from game3d.pieces.enums import Color, Result
from game3d.common.common import N_TOTAL_PLANES, SIZE_X, SIZE_Y, SIZE_Z
from models.resnet3d import OptimizedResNet3D, LightweightResNet3D

# ==============================================================================
# CHESS DATASET FOR TRAINING
# ==============================================================================

@dataclass
class TrainingExample:
    """Single training example from self-play."""
    state_tensor: torch.Tensor  # Board state + current player
    policy_target: torch.Tensor  # Move probabilities (soft targets)
    value_target: float          # Game outcome (-1, 0, 1)
    game_phase: float           # 0.0 (opening) to 1.0 (endgame)

class ChessDataset(Dataset):
    """Dataset for chess training examples with augmentation."""

    def __init__(self, examples: List[TrainingExample], augment: bool = True):
        self.examples = examples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        example = self.examples[idx]

        # Apply data augmentation for chess
        if self.augment and random.random() > 0.5:
            state_tensor = self._augment_state(example.state_tensor)
        else:
            state_tensor = example.state_tensor

        policy_target = example.policy_target
        value_target = torch.tensor([example.value_target], dtype=torch.float32)
        game_phase = torch.tensor([example.game_phase], dtype=torch.float32)

        return state_tensor, policy_target, value_target, game_phase

    def _augment_state(self, state: torch.Tensor) -> torch.Tensor:
        """Apply chess-specific augmentations (rotations, reflections)."""
        # Random rotation (90°, 180°, 270°) around Z-axis
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            # Rotate the spatial dimensions (last 3 dims, assuming (C, Z, Y, X))
            state = torch.rot90(state, k, dims=[2, 3])

        # Random reflection (flip along X or Y)
        if random.random() > 0.5:
            state = torch.flip(state, dims=[3])  # Flip X-axis
        elif random.random() > 0.5:
            state = torch.flip(state, dims=[2])  # Flip Y-axis

        return state

# ==============================================================================
# LOSS FUNCTIONS FOR CHESS
# ==============================================================================

class ChessLoss(nn.Module):
    """Combined loss for policy and value heads with phase weighting."""

    def __init__(self, policy_weight: float = 1.0, value_weight: float = 1.0,
                 phase_weight: float = 0.1):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.phase_weight = phase_weight

        # Value loss
        self.value_loss = nn.MSELoss()

    def forward(self, policy_pred: torch.Tensor, value_pred: torch.Tensor,
                policy_target: torch.Tensor, value_target: torch.Tensor,
                game_phase: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:

        # Policy loss (soft cross-entropy for probabilities)
        policy_loss = - (policy_target * F.log_softmax(policy_pred, dim=1)).sum(dim=1).mean()

        # Value loss with phase-based weighting
        # Middle game gets higher weight than opening/endgame
        phase_weights = 1.0 + self.phase_weight * torch.sin(game_phase * 3.14159)
        value_loss = torch.mean(phase_weights * (value_pred - value_target) ** 2)

        # Combined loss
        total_loss = (self.policy_weight * policy_loss +
                      self.value_weight * value_loss)

        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }

        return total_loss, loss_dict

class PolicyDistillationLoss(nn.Module):
    """Policy distillation for transfer learning between models."""

    def __init__(self, temperature: float = 3.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_policy: torch.Tensor, teacher_policy: torch.Tensor) -> torch.Tensor:
        """Distill knowledge from teacher to student model."""
        # Apply temperature scaling
        student_soft = F.log_softmax(student_policy / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_policy / self.temperature, dim=1)

        # KL divergence loss
        loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        return loss

# ==============================================================================
# OPTIMIZED TRAINER
# ==============================================================================

@dataclass
class TrainingConfig:
    """Configuration for chess neural network training."""
    # Model architecture
    model_type: str = "optimized"  # "optimized" or "lightweight"
    blocks: int = 15
    channels: int = 256
    n_moves: int = 10_000

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 100
    warmup_epochs: int = 5

    # Loss configuration
    policy_weight: float = 1.0
    value_weight: float = 1.0
    phase_weight: float = 0.1

    # Data configuration
    train_split: float = 0.8
    augment_data: bool = True
    max_examples: Optional[int] = None

    # Hardware configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    compile_model: bool = True

    # Logging and checkpointing
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    validate_every: int = 1

    # Advanced settings
    gradient_clipping: float = 1.0
    use_ema: bool = True  # Exponential moving average
    ema_decay: float = 0.999

class ChessTrainer:
    """Optimized trainer for chess neural networks."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create directories
        Path(config.log_dir).mkdir(exist_ok=True)
        Path(config.checkpoint_dir).mkdir(exist_ok=True)

        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)

        # Compile model for faster inference (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)

        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Loss function
        self.criterion = ChessLoss(
            policy_weight=config.policy_weight,
            value_weight=config.value_weight,
            phase_weight=config.phase_weight
        )

        # Scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        # Exponential moving average
        if config.use_ema:
            self.ema_model = self._create_ema_model()
        else:
            self.ema_model = None

        # TensorBoard writer
        self.writer = SummaryWriter(config.log_dir)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _create_model(self) -> nn.Module:
        """Create the neural network model."""
        if self.config.model_type == "optimized":
            return OptimizedResNet3D(
                blocks=self.config.blocks,
                n_moves=self.config.n_moves,
                channels=self.config.channels
            )
        else:
            return LightweightResNet3D(
                blocks=self.config.blocks,
                n_moves=self.config.n_moves,
                channels=self.config.channels
            )

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with weight decay configuration."""
        # Use AdamW for better weight decay handling
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with warmup."""
        # Cosine annealing with warmup
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs - self.config.warmup_epochs,
            eta_min=self.config.learning_rate * 0.01
        )

        if self.config.warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_epochs
            )
            return optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.config.warmup_epochs]
            )

        return main_scheduler

    def _create_ema_model(self) -> nn.Module:
        """Create exponential moving average of the model."""
        ema_model = self._create_model()
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)
        ema_model.eval()
        return ema_model

    def update_ema(self) -> None:
        """Update exponential moving average."""
        if self.ema_model is None:
            return

        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.config.ema_decay).add_(model_param.data, alpha=1 - self.config.ema_decay)

    def prepare_data(self, examples: List[TrainingExample]) -> Tuple[ChessDataset, ChessDataset]:
        """Prepare training and validation datasets."""
        if self.config.max_examples:
            examples = examples[:self.config.max_examples]

        # Split into train/validation
        split_idx = int(len(examples) * self.config.train_split)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        train_dataset = ChessDataset(train_examples, augment=self.config.augment_data)
        val_dataset = ChessDataset(val_examples, augment=False)

        return train_dataset, val_dataset

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with mixed precision."""
        self.model.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for batch_idx, (states, policy_targets, value_targets, game_phases) in enumerate(train_loader):
            # Move to device
            states = states.to(self.device, non_blocking=True)
            policy_targets = policy_targets.to(self.device, non_blocking=True)
            value_targets = value_targets.to(self.device, non_blocking=True)
            game_phases = game_phases.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                policy_pred, value_pred = self.model(states)
                loss, loss_dict = self.criterion(policy_pred, value_pred, policy_targets, value_targets, game_phases)

            # Backward with scaling
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

            # EMA update
            self.update_ema()

            # Track losses
            total_loss += loss_dict['total_loss']
            total_policy_loss += loss_dict['policy_loss']
            total_value_loss += loss_dict['value_loss']
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for states, policy_targets, value_targets, game_phases in val_loader:
                states = states.to(self.device, non_blocking=True)
                policy_targets = policy_targets.to(self.device, non_blocking=True)
                value_targets = value_targets.to(self.device, non_blocking=True)
                game_phases = game_phases.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    policy_pred, value_pred = self.model(states)
                    loss, loss_dict = self.criterion(policy_pred, value_pred, policy_targets, value_targets, game_phases)

                total_loss += loss_dict['total_loss']
                total_policy_loss += loss_dict['policy_loss']
                total_value_loss += loss_dict['value_loss']
                num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches
        }

    def train(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Train the model with early stopping."""
        train_dataset, val_dataset = self.prepare_data(examples)

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )

        training_history = []
        patience = 10  # Early stopping patience
        no_improvement = 0

        for epoch in range(self.config.epochs):
            train_metrics = self.train_epoch(train_loader)
            self.scheduler.step()

            if (epoch + 1) % self.config.validate_every == 0:
                val_metrics = self.validate(val_loader)
                self._log_epoch_metrics(epoch, train_metrics, val_metrics)

                # Checkpointing
                if (epoch + 1) % self.config.save_every == 0:
                    self._save_checkpoint(epoch, val_metrics)

                # Best model saving and early stopping
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self._save_best_model(epoch, val_metrics)
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            training_history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics if (epoch + 1) % self.config.validate_every == 0 else None,
                'lr': self.optimizer.param_groups[0]['lr']
            })

        return {
            'best_val_loss': self.best_val_loss,
            'training_history': training_history,
            'final_model_path': self._get_best_model_path(),
        }

    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float],
                           val_metrics: Dict[str, float]) -> None:
        """Log metrics to TensorBoard and console."""
        # Console logging
        print(f"Epoch {epoch+1}/{self.config.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Policy: {train_metrics['policy_loss']:.4f}, "
              f"Value: {train_metrics['value_loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Policy: {val_metrics['policy_loss']:.4f}, "
              f"Value: {val_metrics['value_loss']:.4f}")
        print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        # TensorBoard logging
        for metric_name, metric_value in train_metrics.items():
            self.writer.add_scalar(f'Train/{metric_name.capitalize()}', metric_value, epoch)

        for metric_name, metric_value in val_metrics.items():
            self.writer.add_scalar(f'Val/{metric_name.capitalize()}', metric_value, epoch)

        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }

        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def _save_best_model(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save the best model."""
        best_model_path = self._get_best_model_path()

        # Save full model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict() if self.ema_model else None,
            'config': self.config,
            'epoch': epoch,
            'metrics': metrics,
        }, best_model_path)

        print(f"Best model saved: {best_model_path}")

    def _get_best_model_path(self) -> str:
        """Get path for best model."""
        return str(Path(self.config.checkpoint_dir) / "best_model.pt")

    def load_best_model(self) -> None:
        """Load the best model."""
        best_model_path = self._get_best_model_path()
        checkpoint = torch.load(best_model_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.ema_model is not None and checkpoint.get('ema_model_state_dict'):
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

        print(f"Best model loaded from: {best_model_path}")

# ==============================================================================
# SELF-PLAY DATA GENERATION
# ==============================================================================

class SelfPlayGenerator:
    """Generate training data through self-play with the current model."""

    def __init__(self, model: nn.Module, device: str = "cuda", temperature: float = 1.0):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.temperature = temperature

    def play_game(net: torch.nn.Module, mcts_depth: int = 0) -> List[TrainingExample]:
        """
        Play a single game and return training examples.

        Args:
            net: Neural network model
            mcts_depth: MCTS simulations per move (0 = random play)

        Returns:
            List of TrainingExample objects
        """
        device = next(net.parameters()).device
        generator = SelfPlayGenerator(net, device=device, temperature=1.0)
        initial_state = GameState.start()
        return generator.generate_game(initial_state, max_moves=10000)

    def generate_game(self, initial_state: GameState, max_moves: int = 200) -> List[TrainingExample]:
        """Generate a single game of self-play data."""
        examples = []
        state = initial_state
        move_count = 0

        while not state.is_game_over() and move_count < max_moves:
            # Get model predictions
            with torch.no_grad():
                state_tensor = state.to_tensor().unsqueeze(0).to(self.device)
                policy_logits, value_pred = self.model(state_tensor)

                # Apply temperature scaling
                policy_logits = policy_logits / self.temperature

                # Convert to probabilities
                policy_probs = torch.softmax(policy_logits, dim=1)

            # Get legal moves
            legal_moves = state.legal_moves()
            if not legal_moves:
                break

            # Map policy to legal moves
            move_probs = self._map_policy_to_moves(policy_probs[0], legal_moves, state)

            # Sample move
            move = self._sample_move(legal_moves, move_probs)
            if move is None:
                break

            # Create training example
            example = TrainingExample(
                state_tensor=state.to_tensor(),
                policy_target=torch.tensor(move_probs, dtype=torch.float32),
                value_target=value_pred.item(),  # Will be updated with final outcome
                game_phase=self._estimate_game_phase(state)
            )
            examples.append(example)

            # Make move
            try:
                state = state.make_move(move)
                move_count += 1
            except Exception as e:
                print(f"Move failed in self-play: {e}")
                break

        # Update value targets with actual game outcome
        final_outcome = state.outcome()
        for example in examples:
            example.value_target = final_outcome

        return examples

    def _map_policy_to_moves(self, policy_probs: torch.Tensor, legal_moves: List,
                             state: GameState) -> List[float]:
        """Map neural network policy output to legal moves."""
        # This is a simplified mapping - you'd need a proper move indexing system
        move_probs = [0.0] * len(legal_moves)

        # For now, use uniform distribution over legal moves
        # In practice, you'd map each move to a specific index in the policy output
        uniform_prob = 1.0 / len(legal_moves)
        for i in range(len(legal_moves)):
            move_probs[i] = uniform_prob

        return move_probs

    def _sample_move(self, legal_moves: List, move_probs: List[float]) -> Optional[Any]:
        """Sample move from probability distribution."""
        if not legal_moves:
            return None

        # Use numpy for efficient sampling
        move_idx = np.random.choice(len(legal_moves), p=move_probs)
        return legal_moves[move_idx]

    def _estimate_game_phase(self, state: GameState) -> float:
        """Estimate game phase (0.0 = opening, 1.0 = endgame)."""
        # Simple heuristic based on move count and pieces remaining
        move_count = len(state.history)
        pieces_remaining = sum(1 for _ in state.board.list_occupied())

        # Normalize to [0, 1]
        phase = min(move_count / 100.0, 1.0) * 0.5 + min(pieces_remaining / 50.0, 1.0) * 0.5
        return phase

# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================

class ChessTrainingPipeline:
    """Complete training pipeline for chess neural networks."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.trainer = ChessTrainer(config)
        self.data_generator = None

    def generate_training_data(self, num_games: int, model: Optional[nn.Module] = None) -> List[TrainingExample]:
        """Generate training data through self-play."""
        if model is None:
            # Use random play for initial data generation
            model = self.trainer.model

        generator = SelfPlayGenerator(model, self.config.device)

        all_examples = []
        for game_idx in range(num_games):
            print(f"Generating game {game_idx + 1}/{num_games}")

            initial_state = GameState.start()
            game_examples = generator.generate_game(initial_state)
            all_examples.extend(game_examples)

            if (game_idx + 1) % 10 == 0:
                print(f"Generated {len(all_examples)} examples so far")

        print(f"Total examples generated: {len(all_examples)}")
        return all_examples

    def train(self, examples: Optional[List[TrainingExample]] = None,
              num_games: int = 1000) -> Dict[str, Any]:
        """Complete training pipeline."""
        if examples is None:
            print("Generating training data...")
            examples = self.generate_training_data(num_games)

        print(f"Training on {len(examples)} examples...")
        results = self.trainer.train(examples)

        return results

    def iterative_training(self, iterations: int = 5, games_per_iteration: int = 100) -> Dict[str, Any]:
        """Iterative training with self-play data generation."""
        all_results = []

        for iteration in range(iterations):
            print(f"\n=== Iteration {iteration + 1}/{iterations} ===")

            # Generate training data using current model
            examples = self.generate_training_data(games_per_iteration)

            # Train model
            results = self.trainer.train(examples)
            all_results.append(results)

            # Update data generator with new model
            self.trainer.load_best_model()

            print(f"Iteration {iteration + 1} completed. Best val loss: {results['best_val_loss']:.4f}")

        return {
            'iterations': iterations,
            'results': all_results,
            'final_model_path': self.trainer._get_best_model_path(),
        }

# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

def main():
    """Example usage of the training pipeline."""

    # Training configuration
    config = TrainingConfig(
        model_type="optimized",
        blocks=15,
        channels=256,
        n_moves=10_000,
        batch_size=32,
        learning_rate=0.001,
        epochs=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=True,
        augment_data=True,
    )

    # Create training pipeline
    pipeline = ChessTrainingPipeline(config)

    # Generate initial training data
    print("Generating initial training data...")
    initial_examples = pipeline.generate_training_data(num_games=100)

    # Train model
    print("Training model...")
    results = pipeline.train(examples=initial_examples)

    print(f"Training completed!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Final model saved at: {results['final_model_path']}")

    # Optional: iterative training
    print("\nStarting iterative training...")
    iterative_results = pipeline.iterative_training(iterations=3, games_per_iteration=50)

    return results, iterative_results

if __name__ == "__main__":
    results, iterative_results = main()
