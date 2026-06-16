from __future__ import annotations

import argparse

from task5lib.cleaning import clean_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = clean_all(args.raw_dir, args.out)
    print(f"cleaned {summary['total_clean_rows']} rows from {len(summary['sources'])} sources")


if __name__ == "__main__":
    main()
