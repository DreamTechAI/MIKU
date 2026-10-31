"""Check the local checkpoint layout without importing torch or allocating a GPU."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miku_config import MikuConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir")
    parser.add_argument("--albedo-only", action="store_true")
    args = parser.parse_args()
    config = MikuConfig()
    if args.weights_dir:
        config.checkpoints_dir = args.weights_dir
    config.pbr_enabled = not args.albedo_only
    try:
        config.validate_weights()
    except FileNotFoundError as error:
        parser.exit(1, str(error) + "\n")
    print(f"Checkpoint layout OK: {config.checkpoints_dir}")


if __name__ == "__main__":
    main()
