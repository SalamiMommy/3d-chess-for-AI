# ---------------------------------------------------------
# main.py  (UPDATED FILE)
# ---------------------------------------------------------
# import sys, pdb, traceback, faulthandler
# faulthandler.enable()
#
# _max_plane_seen = -1
#
# def trace_calls(frame, event, arg):
#     global _max_plane_seen
#     if event != 'exception':
#         return trace_calls
#     exc_type, exc_val, exc_tb = arg
#     if issubclass(exc_type, IndexError):
#         # walk back to the local vars in to_tensor()
#         while frame and frame.f_code.co_name != 'to_tensor':
#             frame = frame.f_back
#         if not frame:
#             return trace_calls
#         loc = frame.f_locals
#         ptype_val = loc.get('ptype_val', -1)
#         offset    = loc.get('offset', -1)
#         bad_idx   = ptype_val + offset
#         _max_plane_seen = max(_max_plane_seen, bad_idx)
#         print(f'[PLANE-DEBUG] ptype_val={ptype_val}  offset={offset}  → index={bad_idx}')
#         traceback.print_exception(exc_type, exc_val, exc_tb)
#         pdb.post_mortem(exc_tb)
#         sys.exit(1)
#     return trace_calls
#
# # Only enable tracing in debug mode
# if __debug__:
#     sys.settrace(trace_calls)

import argparse
from training.optim_train import TrainingConfig, train_with_self_play

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "play"])
    # Note: --mcts-depth removed as MCTS is not implemented in self_play.py.
    # If adding MCTS, reintroduce and pass to generate_training_data.
    args = parser.parse_args()

    if args.mode == "train":
        config = TrainingConfig()
        # Adjust num_games as needed (e.g., for more data).
        # This loads model (or inits), generates self-play data, and trains.
        results = train_with_self_play(config, num_games=10)
        print(f"Training completed! Best val loss: {results['best_val_loss']}")
    else:
        print("Play mode not implemented yet.")

if __name__ == "__main__":
    main()
