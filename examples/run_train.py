"""Example: python examples/run_train.py --config configs/experiments/demo.yaml --demo --output runs/demo"""

import sys
from bindmatchcopy.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
