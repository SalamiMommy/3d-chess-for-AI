"""Fire up either ‘play’ or ‘train’ mode."""

import argparse
from models.resnet3d import ResNet3D

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "play"])
    args = parser.parse_args()

    net = ResNet3D(blocks=15, n_moves=10_000)
    print(net)

    if args.mode == "train":
        from training.self_play import play_game
        play_game(net)          # stub
    else:
        print("Play mode not implemented yet.")

if __name__ == "__main__":
    main()
