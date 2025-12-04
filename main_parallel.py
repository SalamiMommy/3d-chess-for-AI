#!/usr/bin/env python3
"""Parallel self-play training loop for 3D chess with GPU batching and memory safety."""

# ============ SUPPRESS NUMBA DEBUG OUTPUT ============
import os
os.environ['NUMBA_WARNINGS'] = '0'
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'
os.environ['NUMBA_DEBUG'] = '0'
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
# =====================================================

# ============ CRITICAL: LOGGING SETUP MUST BE FIRST ============
import logging, sys, faulthandler, signal, traceback
from pathlib import Path

# Enable faulthandler for segfault debugging
faulthandler.enable()
faulthandler.enable(all_threads=True)

# Configure LOUD logging BEFORE any imports
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('game_errors.log'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.WARNING)


# Define custom exceptions for loud failures
class MoveValidationError(ValueError):
    """Raised when move validation fails."""
    pass

class MoveGenerationError(RuntimeError):
    """Raised when move generation fails unrecoverably."""
    pass
# ===============================================================

import multiprocessing as mp
import torch
import argparse
import pickle
import random
import time
from typing import List, Optional
from itertools import cycle
import numpy as np
import gc
import tempfile
try:
    from tqdm import tqdm
except ImportError:
    # Dummy tqdm for environments without it
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.n = 0
            print(f"Progress bar disabled (tqdm not found). Total: {self.total}")
            
        def update(self, n=1):
            self.n += n
            if self.n % 10 == 0:
                print(f"Progress: {self.n}/{self.total}")
                
        def set_postfix(self, *args, **kwargs):
            pass
            
        def close(self):
            print("Done.")
            
        def write(self, msg):
            print(msg)

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    # Now safe to import modules (logging is ready)
    from training.optim_train import ChessTrainer
    from training.parallel_self_play import generate_training_data_parallel
    from training.training_types import TrainingConfig, TrainingExample, clear_policy_pool, clear_state_pool, ReplayBuffer
    from training.opponents import AVAILABLE_OPPONENTS, create_opponent
    from game3d.common.shared_types import N_CHANNELS, SIZE, MAX_COORD_VALUE, MIN_COORD_VALUE

    # Setup ROCm optimizations
    from models.graph_transformer import setup_rocm_optimizations
    setup_rocm_optimizations()

    # Import game components for validation
    from game3d.game3d import OptimizedGame3D
    from game3d.board.board import Board
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.common.shared_types import Color, N_TOTAL_PLANES




    # CLI configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-iter", type=int, default=6)  # Increased from 24 for more data
    parser.add_argument("--num-parallel", type=int, default=6)  # Optimized for 6-core CPU
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--replay-file", type=str, default="replay_buffer.pkl")
    parser.add_argument("--max-replay", type=int, default=500000)  # Increased to utilize 64GB RAM
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opponent-types", type=str, nargs="+", default=AVAILABLE_OPPONENTS)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--model-dir", type=str, default="model_checkpoints")
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--training-mode", type=str, default="fresh", choices=["fresh", "resume"])
    parser.add_argument("--model-size", type=str, default="huge", choices=["small", "default", "large", "huge"])
    parser.add_argument("--use-mixed-precision", action="store_true", default=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2,
                       help="Gradient accumulation steps for larger effective batch size")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--validate-only", action="store_true",
                       help="Run validation only and exit")
    parser.add_argument("--cache-clear-interval", type=int, default=5,  # More aggressive cleanup
                       help="Clear caches every N iterations")
    args = parser.parse_args()

    # Generate run name if not provided
    if args.run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if args.training_mode == "resume" and args.resume_from:
            base_name = Path(args.resume_from).stem
            args.run_name = f"{base_name}_resumed_{timestamp}"
        else:
            args.run_name = f"train_{args.model_size}_{timestamp}"

    # Configuration and trainer setup
    config = TrainingConfig(
        device=args.device,
        model_type="transformer",
        model_size=args.model_size,
        mixed_precision=args.use_mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Create model
    from models.graph_transformer import create_optimized_model
    from training.training_types import POLICY_DIM

    model = create_optimized_model(args.model_size)
    model = model.to(config.device)
    print(f"Successfully created Graph Transformer model ({args.model_size})")

    trainer = ChessTrainer(config, model=model)

    # Create directories
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir = model_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer.config.log_dir = str(run_dir / "logs")
    trainer.config.checkpoint_dir = str(run_dir)
    Path(trainer.config.log_dir).mkdir(exist_ok=True)
    Path(trainer.config.checkpoint_dir).mkdir(exist_ok=True)

    # Handle model loading scenarios
    start_iteration = 0

    if args.training_mode == "resume" and args.resume_from:
        # Resume training from checkpoint with full state
        print(f"Resuming training from {args.resume_from}...")
        checkpoint = torch.load(args.resume_from, map_location=config.device)

        # Load model state
        if 'model_state_dict' in checkpoint:
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            trainer.model.load_state_dict(checkpoint)

        # Load optimizer and scheduler state if available
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'epoch' in checkpoint:
            start_iteration = checkpoint['epoch']

        print(f"Resumed training from iteration {start_iteration}")

    elif args.load_model:
        # Load model weights but start fresh training
        print(f"Loading model from {args.load_model} for fresh training...")
        trainer.load_model(args.load_model)
        print(f"Starting fresh training from loaded model")

    # NEW: Use memory-mapped replay buffer instead of in-memory list
    replay = ReplayBuffer(max_size=args.max_replay, temp_dir=str(run_dir))

    def clear_all_caches():
        """Aggressive cache clearing to prevent memory accumulation."""
        logger.info("🧹 Clearing all caches...")

        # Clear transposition table (ADD THIS)
        if hasattr(trainer.model, 'cache_manager'):
            trainer.model.cache_manager.transposition_table.clear()
            logger.info("  -> Transposition table cleared")

        # Clear effect caches (ADD THIS)
        if hasattr(trainer.model, 'cache_manager'):
            for cache in trainer.model.cache_manager._effect_cache_instances:
                if hasattr(cache, 'clear'):
                    cache.clear()
            logger.info("  -> Effect caches cleared")

        # Clear transposition table
        if hasattr(trainer.model, 'tt'):
            trainer.model.tt.clear()

        # Clear policy and state pools
        clear_policy_pool()
        clear_state_pool()

        # Clear occupancy cache
        if hasattr(trainer, 'state') and hasattr(trainer.state, 'cache_manager'):
            trainer.state.cache_manager.occupancy_cache.clear()
            trainer.state.cache_manager.move_cache.clear()

        # Force garbage collection
        gc.collect()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def validate_game_system():
        """Loud validation WITHOUT destroying cache state."""
        logger.info("[SEARCH] Validating game system integrity...")

        try:
            board = Board.startpos()
            cache = OptimizedCacheManager(board, Color.WHITE)
            game = OptimizedGame3D(board=board, cache=cache)

            # === Test 1: Move generation works at all ===
            logger.info("Testing move generation...")
            moves1 = game.state.legal_moves
            assert isinstance(moves1, np.ndarray), f"Moves not ndarray: {type(moves1)}"
            assert len(moves1) > 0, f"No moves generated: {moves1}"
            logger.info(f"[OK] Generated {len(moves1)} moves")

            # === Test 2: GameState caching (same instance) ===
            # === Test 2: GameState caching (via MoveCache) ===
            logger.info("Testing GameState caching...")
            # We don't clear local cache anymore as it's not used.
            # Instead we verify that subsequent calls return the SAME cached array from MoveCache
            
            # First call already populated the cache (moves1)
            moves2 = game.state.legal_moves
            
            # Verify it's fetching from MoveCache
            cached_moves = game.cache_manager.move_cache.get_legal_moves(game.state.color)
            assert moves2 is cached_moves, "Legal moves not matching MoveCache"
            
            # Verify identity (should be same object if cached correctly)
            # Note: generate_legal_moves might return a fresh view or the array itself depending on implementation
            # But the content should be identical
            assert np.array_equal(moves1, moves2), "Subsequent legal_moves call returned different data"
            
            logger.info("[OK] GameState caching works (via MoveCache)")

            # === Test 3: MoveCache persistence (different GameState instances) ===
            logger.info("Testing MoveCache persistence...")
            # Create NEW game with SAME cache manager
            game2 = OptimizedGame3D(board=board, cache=cache)
            moves3 = game2.state.legal_moves

            # Should hit MoveCache, not regenerate
            stats = cache.move_cache.get_statistics()
            assert stats['cache_hits'] >= 1, f"No MoveCache hits: {stats}"
            logger.info(f"[OK] MoveCache works correctly")
            logger.info(f"Final cache stats: {stats}")

            return True

        except Exception as e:
            logger.critical(f"[ALERT] GAME SYSTEM VALIDATION FAILED: {e}", exc_info=True)
            return False

    # Run validation before training
    if args.validate_only:
        success = validate_game_system()
        sys.exit(0 if success else 1)

    if not validate_game_system():
        logger.critical("Cannot start training due to validation failures")
        sys.exit(1)

    # ============ MAIN TRAINING LOOP WITH MEMORY SAFETY ============
    iteration = start_iteration
    pbar = tqdm(
        total=args.max_iter if args.max_iter else None,
        initial=start_iteration,
        desc="Training Progress",
        unit="iter",
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    
    try:
        while True:
            if args.max_iter and iteration >= args.max_iter + start_iteration:
                logger.info("Reached max iterations")
                break

            white_type = next(cycle(args.opponent_types))
            black_type = next(cycle(args.opponent_types))

            pbar.write(f"\n{'='*60}")
            pbar.write(f"ITERATION {iteration} - White: {white_type}, Black: {black_type}")
            pbar.write(f"{'='*60}")

            try:
                # Save current model state to temporary checkpoint
                temp_checkpoint = run_dir / "temp_model_for_selfplay.pt"
                torch.save(trainer.model.state_dict(), temp_checkpoint)
                logger.info(f"Saved temporary checkpoint: {temp_checkpoint}")
                
                # Generate training data with per-worker models
                fresh = generate_training_data_parallel(
                    model_checkpoint_path=str(temp_checkpoint),
                    num_games=args.games_per_iter,
                    device=args.device,
                    opponent_types=[white_type, black_type],
                    epsilon=args.epsilon,
                    num_parallel=args.num_parallel,
                    max_moves=100000,
                    model_size=args.model_size,
                )


                if not fresh:
                    logger.warning("No training examples generated this iteration")
                    continue

                pbar.write(f"  -> {len(fresh)} new examples")

                # Store in memory-mapped buffer
                replay.append(fresh)

                if len(replay) > args.max_replay:
                    # Buffer auto-manages size via overwrite
                    pbar.write(f"  -> replay buffer at capacity, will overwrite oldest")

                # Train model
                pbar.write("Training...")

                # Get examples from buffer
                # training_examples = replay.get_all_examples()  <-- REMOVED

                results = trainer.train(replay)  # Pass dataset directly
                pbar.write(f"  -> best val-loss: {results['best_val_loss']:.4f}")

                # Save replay buffer metadata (not full data)
                replay_meta_path = run_dir / "replay_metadata.pkl"
                with replay_meta_path.open("wb") as f:
                    pickle.dump({
                        'size': len(replay),
                        'max_size': args.max_replay
                    }, f)
                pbar.write(f"  -> replay metadata saved")

                # Save model checkpoint
                if iteration % args.save_interval == 0:
                    model_path = run_dir / f"model_iter_{iteration}.pt"
                    checkpoint = {
                        "iteration": iteration,
                        "model_state_dict": trainer.model.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "scheduler_state_dict": trainer.scheduler.state_dict(),
                        "best_val_loss": results['best_val_loss'],
                        "config": vars(args)
                    }
                    torch.save(checkpoint, model_path)

                    latest_path = run_dir / "model_latest.pt"
                    torch.save(checkpoint, latest_path)

                    # Simple progress indicator
                    if torch.cuda.is_available():
                        gpu_mem = torch.cuda.memory_allocated() / 1e9
                        pbar.write(f"  -> checkpoint saved (GPU: {gpu_mem:.1f}GB)")
                    else:
                        pbar.write(f"  -> checkpoint saved")

                iteration += 1
                
                # Update progress bar with detailed metrics
                gpu_mem_str = f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
                pbar.set_postfix({
                    'examples': len(fresh),
                    'replay': len(replay),
                    'val_loss': f"{results['best_val_loss']:.4f}",
                    'GPU': gpu_mem_str
                })
                pbar.update(1)

                # Periodic cache clearing
                if iteration % args.cache_clear_interval == 0:
                    clear_all_caches()

            except MoveValidationError as e:
                logger.error(f"[ERROR] Move validation failed in iteration {iteration}: {e}")
                pbar.close()
                raise

            except MoveGenerationError as e:
                logger.error(f"[ERROR] Move generation failed in iteration {iteration}: {e}")
                pbar.close()
                raise

            except Exception as e:
                logger.error(f"[ERROR] Unexpected error in iteration {iteration}: {e}")
                pbar.close()
                raise

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        pbar.close()

    except Exception as e:
        logger.critical(f"[ALERT] TRAINING CRASHED: {e}")
        pbar.close()
        # Save emergency checkpoint
        emergency_path = run_dir / "emergency_checkpoint.pt"
        torch.save({
            "iteration": iteration,
            "model_state_dict": trainer.model.state_dict(),
            "error": str(e),
            "traceback": traceback.format_exc()
        }, emergency_path)
        logger.info(f"Emergency checkpoint saved to {emergency_path}")
        sys.exit(1)

    finally:
        # Cleanup replay buffer
        pbar.close()
        replay.cleanup()
        clear_all_caches()

    # Save final model
    final_path = run_dir / "model_final.pt"
    torch.save(trainer.model.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")
    print("\nTraining complete!")
