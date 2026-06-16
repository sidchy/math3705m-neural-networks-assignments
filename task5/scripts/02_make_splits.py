from __future__ import annotations

import argparse

from task5lib.splits import make_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = make_splits(args.cleaned, args.out, args.seed)
    print(report)


if __name__ == "__main__":
    main()
