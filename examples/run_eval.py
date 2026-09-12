"""Example: python examples/run_eval.py --checkpoint MODEL.pt --evalset EVAL/evalset.pt --output runs/recall.json"""

import sys
from bindmatchcopy.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
