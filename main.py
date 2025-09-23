"""Fire up either ‘play’ or ‘train’ mode with checkpoints and hardware optimizations."""

import argparse
from models.resnet3d import ResNet3D
from training.optim_train import load_or_init_model, train_model
from training.self_play import play_game

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "play"])
    args = parser.parse_args()

    if args.mode == "train":
        net, optimizer, start_epoch, start_step = load_or_init_model()
        # Generate self-play data
        examples = play_game(net)
        # Train with optimizations
        train_model(net, optimizer, examples, start_epoch, start_step)
    else:
        print("Play mode not implemented yet.")

if __name__ == "__main__":
    main()
