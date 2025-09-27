"""Optimized training loop for 24GB VRAM, 64GB RAM with self-play integration."""
# training/optim_train.py

import torch
from torch.utils.data import DataLoader, Dataset
import time
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch.nn.functional as F

# Import your existing modules
from models.resnet3d import LightweightResNet3D, OptimizedResNet3D as ResNet3D
from training.checkpoint import save_checkpoint, load_latest_checkpoint

# Import the self-play and game state modules
from game3d.game.gamestate import GameState
from game3d.pieces.enums import Color, PieceType


# Training configuration
BATCH_SIZE = 256       # 24GB VRAM can handle large batches
NUM_WORKERS = 8        # 64GB RAM, so plenty for DataLoader prefetch
PIN_MEMORY = True      # Faster CPU-to-GPU transfer if using CUDA
EPOCHS = 1000          # Train for many epochs
CHECKPOINT_INTERVAL = 3600  # 1 hour in seconds
SAVE_EVERY_EPOCH = True

@dataclass
class TrainingExample:
    """Enhanced training example with metadata for chess."""
    state_tensor: torch.Tensor  # Board state (C,9,9,9)
    policy_target: torch.Tensor  # Move probabilities
    value_target: float          # Game outcome (-1, 0, 1)
    game_phase: float           # 0.0 (opening) to 1.0 (endgame)
    move_count: int             # Move number in game

class ChessDataset(Dataset):
    """Zero-copy dataset from examples list with validation."""

    def __init__(self, examples: List[TrainingExample], validate: bool = True):
        super().__init__()
        self.examples = examples
        self.validate = validate

        if validate:
            self._validate_examples()

    def _validate_examples(self):
        """Validate training examples for consistency."""
        for i, example in enumerate(self.examples):
            if not isinstance(example.state_tensor, torch.Tensor):
                raise ValueError(f"Example {i}: state_tensor must be torch.Tensor")
            if example.state_tensor.shape != (N_TOTAL_PLANES + 1, 9, 9, 9):  # +1 for current player
                raise ValueError(f"Example {i}: wrong tensor shape {example.state_tensor.shape}")
            if not isinstance(example.policy_target, torch.Tensor):
                raise ValueError(f"Example {i}: policy_target must be torch.Tensor")
            if not (-1.0 <= example.value_target <= 1.0):
                raise ValueError(f"Example {i}: value_target must be in [-1, 1], got {example.value_target}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return example.state_tensor, example.policy_target, torch.tensor(example.value_target, dtype=torch.float32)

# ==============================================================================
# SELF-PLAY INTEGRATION
# ==============================================================================

class SelfPlayGenerator:
    """Generate training data through self-play games."""

    def __init__(self, model: torch.nn.Module, device: str = "cuda", temperature: float = 1.0):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.temperature = temperature

    def generate_game(self, initial_state: GameState, max_moves: int = 200) -> List[TrainingExample]:
        """Generate a complete game of self-play data."""
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
                policy_probs = torch.softmax(policy_logits, dim=1)

            # Get legal moves and create policy target
            legal_moves = state.legal_moves()
            if not legal_moves:
                break

            # Create policy target (simplified - uniform over legal moves)
            policy_target = self._create_policy_target(policy_probs[0], legal_moves, state)

            # Create training example
            example = TrainingExample(
                state_tensor=state.to_tensor(),
                policy_target=torch.tensor(policy_target, dtype=torch.float32),
                value_target=value_pred.item(),  # Will be updated with final outcome
                game_phase=self._estimate_game_phase(state),
                move_count=move_count
            )
            examples.append(example)

            # Sample and make move
            move = self._sample_move(legal_moves, policy_target)
            if move is None:
                break

            try:
                state = state.make_move(move)
                move_count += 1
            except Exception as e:
                print(f"Move failed in self-play: {e}")
                break

        # Update all examples with final game outcome
        final_outcome = state.outcome()
        for example in examples:
            example.value_target = final_outcome

        return examples

    def _create_policy_target(self, policy_probs: torch.Tensor, legal_moves: List,
                              state: GameState) -> List[float]:
        """Create policy target for legal moves."""
        # Simplified: uniform distribution over legal moves
        # In practice, you'd map each move to a specific index in the policy output
        n_moves = len(legal_moves)
        if n_moves == 0:
            return []

        # For now, use uniform distribution
        # You could improve this by mapping moves to policy indices
        uniform_prob = 1.0 / n_moves
        return [uniform_prob] * n_moves

    def _sample_move(self, legal_moves: List, move_probs: List[float]) -> Optional[Any]:
        """Sample move from probability distribution."""
        if not legal_moves:
            return None

        move_idx = np.random.choice(len(legal_moves), p=move_probs)
        return legal_moves[move_idx]

    def _estimate_game_phase(self, state: GameState) -> float:
        """Estimate game phase (0.0 = opening, 1.0 = endgame)."""
        move_count = len(state.history)
        pieces_remaining = sum(1 for _ in state.board.list_occupied())

        # Simple heuristic
        phase = min(move_count / 100.0, 1.0) * 0.5 + min(pieces_remaining / 50.0, 1.0) * 0.5
        return phase

# ==============================================================================
# ENHANCED TRAINING LOOP WITH SELF-PLAY
# ==============================================================================

def train_model(net: torch.nn.Module, optimizer: torch.optim.Optimizer,
                dataset: List[TrainingExample], start_epoch: int = 0, start_step: int = 0,
                device: str = "cuda") -> Dict[str, Any]:
    """Enhanced training loop with self-play integration."""

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Create optimized DataLoader
    loader = DataLoader(
        ChessDataset(dataset),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )

    # Enhanced loss functions
    criterion_value = torch.nn.MSELoss()

    last_save_time = time.time()
    step = start_step
    total_loss = 0.0
    num_batches = 0

    training_stats = {
        'total_loss': [],
        'policy_loss': [],
        'value_loss': [],
        'learning_rate': []
    }

    patience = 10  # Early stopping
    no_improvement = 0
    best_val_loss = float('inf')  # Assume validation logic added if needed

    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0

        for batch_idx, batch in enumerate(loader):
            x, pi, z = batch
            x = x.to(device, non_blocking=True)
            pi = pi.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)

            # Forward pass with autocast for mixed precision
            with torch.cuda.amp.autocast():
                out_pi, out_value = net(x)

                # Compute losses (soft cross-entropy for policy)
                loss_policy = - (pi * F.log_softmax(out_pi, dim=1)).sum(dim=1).mean()
                loss_value = criterion_value(out_value.squeeze(), z)
                loss = loss_policy + loss_value

            # Backward pass with optimizations
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            optimizer.step()
            step += 1

            # Track statistics
            total_loss += loss.item()
            epoch_loss += loss.item()
            epoch_policy_loss += loss_policy.item()
            epoch_value_loss += loss_value.item()
            num_batches += 1

            # Periodic checkpointing
            if time.time() - last_save_time > CHECKPOINT_INTERVAL:
                save_checkpoint(net, optimizer, epoch, step)
                last_save_time = time.time()
                print(f"Checkpoint saved at epoch {epoch}, step {step}")

        # End of epoch processing
        avg_epoch_loss = epoch_loss / len(loader)
        avg_policy_loss = epoch_policy_loss / len(loader)
        avg_value_loss = epoch_value_loss / len(loader)

        training_stats['total_loss'].append(avg_epoch_loss)
        training_stats['policy_loss'].append(avg_policy_loss)
        training_stats['value_loss'].append(avg_value_loss)
        training_stats['learning_rate'].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch}: Loss={avg_epoch_loss:.4f}, "
              f"Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f}")

        # Early stopping (assume val_loss from validation; stub for now)
        if avg_epoch_loss < best_val_loss:
            best_val_loss = avg_epoch_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save at end of each epoch
        if SAVE_EVERY_EPOCH:
            save_checkpoint(net, optimizer, epoch, step)

    print("Training complete.")
    return training_stats

# ==============================================================================
# SELF-PLAY TRAINING LOOP
# ==============================================================================

def train_with_self_play(net: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         num_self_play_games: int = 100, games_per_epoch: int = 10,
                         device: str = "cuda") -> Dict[str, Any]:
    """Training loop that alternates between self-play and training."""

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    all_training_stats = []

    print(f"Starting self-play training with {num_self_play_games} total games...")

    for epoch in range(0, num_self_play_games, games_per_epoch):
        print(f"\n=== Epoch {epoch//games_per_epoch + 1} ===")

        # Generate self-play games
        print(f"Generating {games_per_epoch} self-play games...")
        generator = SelfPlayGenerator(net, device)

        all_examples = []
        for game_idx in range(games_per_epoch):
            initial_state = GameState.start()
            game_examples = generator.generate_game(initial_state)
            all_examples.extend(game_examples)

            if (game_idx + 1) % 5 == 0:
                print(f"  Completed game {game_idx + 1}/{games_per_epoch}")

        print(f"Generated {len(all_examples)} training examples")

        # Train on these examples
        print("Training on generated examples...")
        epoch_stats = train_model(
            net=net,
            optimizer=optimizer,
            dataset=all_examples,
            start_epoch=epoch//games_per_epoch,
            start_step=0,
            device=device
        )

        all_training_stats.append(epoch_stats)

        # Optional: Save model after each self-play epoch
        save_checkpoint(net, optimizer, epoch//games_per_epoch + 1, 0)

    print("Self-play training complete.")
    return {
        'total_epochs': len(all_training_stats),
        'final_training_stats': all_training_stats[-1] if all_training_stats else None,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

# ==============================================================================
# INTEGRATION WITH YOUR EXISTING CODE
# ==============================================================================
def load_or_init_model():
    """Load model and initialize for self-play training."""
    # Try to load your existing ResNet3D first
    try:
        net = ResNet3D(blocks=15, n_moves=10_000)
        print("Loaded existing ResNet3D model")
    except:
        # Fall back to optimized version
        net = OptimizedResNet3D(blocks=15, n_moves=10_000, channels=256)
        print("Loaded optimized ResNet3D model")

    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-5)

    # Try to load checkpoint
    checkpoint = load_latest_checkpoint(net, optimizer)
    if checkpoint is None:
        print("No checkpoint found, starting fresh training")
        return net, optimizer, 0, 0

    print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, step {checkpoint['step']}")
    return net, optimizer, checkpoint["epoch"], checkpoint["step"]

# Example usage:
if __name__ == "__main__":
    # Load model and optimizer
    net, optimizer, start_epoch, start_step = load_or_init_model_with_self_play()

    # Option 1: Train on existing dataset
    # examples = [...]  # Your existing training examples
    # train_model(net, optimizer, examples, start_epoch, start_step)

    # Option 2: Train with self-play
    results = train_with_self_play(
        net=net,
        optimizer=optimizer,
        num_self_play_games=100,
        games_per_epoch=10,
        device="cuda"
    )

    print("Training completed successfully!")
