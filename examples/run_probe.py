"""Example: python examples/run_probe.py --mode bind --checkpoint MODEL.pt --output runs/bind"""

import sys
from bindmatchcopy.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["probe", *sys.argv[1:]]))
