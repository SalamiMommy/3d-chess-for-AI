#!/usr/bin/env python3
"""
Parallel self-play → training loop for 3-D chess with GPU batching.
Each iteration:
  1. generates fresh examples by parallel self-play (GPU batched)
  2. appends them to a growing replay buffer
  3. trains the net on the buffer (GPU)
  4. saves the best model and a small replay checkpoint
"""

import os, multiprocessing as mp, torch, argparse, pickle, random
from pathlib import Path
from typing import List
from itertools import cycle

# ---------- 0. Configure environment ----------
os.environ["NUMBA_DISABLE_JIT"]     = "0"
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"

# ROCm optimizations
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["HSA_ENABLE_SDMA"] = "1"  # Can help with stability

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    # ---------- 1. Import torch stuff ----------
    from training.optim_train import TrainingConfig, ChessTrainer
    from training.parallel_self_play import generate_training_data_parallel
    from training.types import TrainingExample
    from training.opponents import AVAILABLE_OPPONENTS, create_opponent

    # ---------- 2. CLI with parallel options ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-iter", type=int, default=32,
                        help="self-play games to create each iteration")
    parser.add_argument("--num-parallel", type=int, default=8,
                        help="number of games to run in parallel (GPU batch size)")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="training iterations (∞ if 0)")
    parser.add_argument("--replay-file", type=str, default="replay_buffer.pkl",
                        help="file that keeps the growing replay buffer")
    parser.add_argument("--max-replay", type=int, default=1_000_000,
                        help="max examples to keep (oldest dropped)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device for AI model (cuda/cpu)")
    parser.add_argument("--opponent-types", type=str, nargs="+",
                        default=AVAILABLE_OPPONENTS,
                        help=f"opponent types to cycle through (available: {AVAILABLE_OPPONENTS})")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="exploration rate for self-play")
    # New arguments for model saving and resuming
    parser.add_argument("--model-dir", type=str, default="model_checkpoints",
                        help="directory to save model checkpoints")
    parser.add_argument("--save-interval", type=int, default=1,
                        help="save model every N iterations")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="path to model checkpoint to resume training from")
    parser.add_argument("--resume-iteration", type=int, default=0,
                        help="iteration number to resume from (for logging)")
    # New argument for loading model but starting fresh training
    parser.add_argument("--load-model", type=str, default=None,
                        help="path to model checkpoint to load and start fresh training (ignores replay buffer and iteration)")
    args = parser.parse_args()

    # ---------- 3. config & trainer ----------
    config = TrainingConfig(device=args.device)
    trainer = ChessTrainer(config)

    # Create model directory if it doesn't exist
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Handle model loading scenarios
    start_iteration = 0
    if args.load_model:
        # Load model but start fresh training
        print(f"Loading model from {args.load_model} for fresh training...")
        trainer.load_model(args.load_model)
        print(f"Starting fresh training from loaded model")
    elif args.resume_from:
        # Resume training from checkpoint with existing state
        print(f"Loading model from {args.resume_from}...")
        trainer.load_model(args.resume_from)
        start_iteration = args.resume_iteration
        print(f"Resumed training from iteration {start_iteration}")

    # Create opponent cycles
    white_opponents = cycle(args.opponent_types)
    black_opponents = cycle(args.opponent_types)

    print(f"\n{'='*80}")
    print(f"PARALLEL TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Device: {args.device}")
    print(f"Games per iteration: {args.games_per_iter}")
    print(f"Parallel games: {args.num_parallel}")
    print(f"Available opponents: {AVAILABLE_OPPONENTS}")
    print(f"Selected opponents: {args.opponent_types}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Max replay buffer: {args.max_replay:,}")
    print(f"Model directory: {model_dir}")
    print(f"Save interval: {args.save_interval}")
    print(f"Resume from: {args.resume_from or 'None'}")
    print(f"Load model (fresh): {args.load_model or 'None'}")
    print(f"Starting iteration: {start_iteration}")
    print(f"{'='*80}\n")

    # ---------- 4. load or create replay buffer ----------
    replay_path = Path(args.replay_file)
    replay: List[TrainingExample] = []

    # Only load existing replay buffer if we're resuming, not starting fresh
    if replay_path.exists() and args.resume_from and not args.load_model:
        with replay_path.open("rb") as f:
            replay = pickle.load(f)
        print(f"Loaded {len(replay)} examples from {replay_path}")
    elif args.load_model:
        print("Starting with empty replay buffer (fresh training)")
    else:
        print("Starting with empty replay buffer")

    # ---------- 5. infinite (or limited) loop ----------
    iteration = start_iteration
    while True:
        if args.max_iter and iteration >= args.max_iter + start_iteration:
            break
        iteration += 1

        # Get next opponent pair for this iteration
        white_type = next(white_opponents)
        black_type = next(black_opponents)

        print(f"\n{'='*80}")
        print(f"ITERATION {iteration} - White: {white_type}, Black: {black_type}")
        print(f"{'='*80}")

        # 5a. parallel self-play with GPU batching
        print("Generating fresh games with parallel self-play...")
        fresh = generate_training_data_parallel(
            trainer.model,
            num_games=args.games_per_iter,
            device=args.device,
            opponent_types=[white_type, black_type],
            epsilon=args.epsilon,
            num_parallel=args.num_parallel,
        )
        print(f"  → {len(fresh)} new examples")

        # 5b. append to replay
        replay.extend(fresh)
        if len(replay) > args.max_replay:
            replay = replay[-args.max_replay:]  # keep newest
            print(f"  → replay trimmed to {len(replay)}")

        # 5c. train
        print("Training...")
        results = trainer.train(replay)
        print(f"  → best val-loss: {results['best_val_loss']:.4f}")

        # 5d. checkpoint replay buffer
        with replay_path.open("wb") as f:
            pickle.dump(replay, f)
        print(f"  → replay buffer saved to {replay_path}")

        # 5e. Save model checkpoint
        if iteration % args.save_interval == 0:
            model_path = model_dir / f"model_iter_{iteration}.pt"
            trainer.save_model(model_path)
            print(f"  → model checkpoint saved to {model_path}")

            # Also save a "latest" model for easy resuming
            latest_path = model_dir / "model_latest.pt"
            trainer.save_model(latest_path)
            print(f"  → latest model saved to {latest_path}")

        # 5f. print GPU memory stats
        if torch.cuda.is_available():
            print(f"\nGPU Memory:")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Save final model
    final_model_path = model_dir / "model_final.pt"
    trainer.save_model(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    print("\nTraining loop finished.")
