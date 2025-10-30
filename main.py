#!/usr/bin/env python3
"""
Self-play → training loop for 3-D chess.
Each iteration:
  1. generates fresh examples by self-play
  2. appends them to a growing replay buffer
  3. trains the net on the buffer
  4. saves the best model and a small replay checkpoint
"""

import os, multiprocessing as mp, torch, argparse, pickle, random
from pathlib import Path
from typing import List

# ---------- 0. Configure GPU for AI, CPU for game ----------
# Only hide GPUs from game logic, not from AI model
os.environ["NUMBA_DISABLE_JIT"]     = "0"
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    # ---------- 1. Import torch stuff ----------
    from training.optim_train import TrainingConfig, ChessTrainer
    from training.self_play import generate_training_data
    from training.types import TrainingExample

    # ---------- 2. simple CLI ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-iter", type=int, default=10,
                        help="self-play games to create each iteration")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="training iterations (∞ if 0)")
    parser.add_argument("--replay-file", type=str, default="replay_buffer.pkl",
                        help="file that keeps the growing replay buffer")
    parser.add_argument("--max-replay", type=int, default=200_000,
                        help="max examples to keep (oldest dropped)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device for AI model (cuda/cpu)")
    args = parser.parse_args()

    # ---------- 3. config & trainer ----------
    config = TrainingConfig(device=args.device)
    trainer = ChessTrainer(config)

    # ---------- 4. load or create replay buffer ----------
    replay_path = Path(args.replay_file)
    replay: List[TrainingExample] = []
    if replay_path.exists():
        with replay_path.open("rb") as f:
            replay = pickle.load(f)
        print(f"Loaded {len(replay)} examples from {replay_path}")

    # ---------- 5. infinite (or limited) loop ----------
    iteration = 0
    while True:
        if args.max_iter and iteration >= args.max_iter:
            break
        iteration += 1
        print(f"\n===== ITERATION {iteration} =====")

        # 5a. self-play
        print("Generating fresh games …")
        fresh = generate_training_data(
            trainer.model,
            num_games=args.games_per_iter,
            device=args.device  # Pass the device to self-play
        )
        print(f"  → {len(fresh)} new examples")

        # 5b. append to replay
        replay.extend(fresh)
        if len(replay) > args.max_replay:
            replay = replay[-args.max_replay :]  # keep newest
            print(f"  → replay trimmed to {len(replay)}")

        # 5c. train
        print("Training …")
        results = trainer.train(replay)
        print(f"  → best val-loss: {results['best_val_loss']:.4f}")

        # 5d. checkpoint replay buffer
        with replay_path.open("wb") as f:
            pickle.dump(replay, f)

    print("Training loop finished.")
