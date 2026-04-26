#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Placeholder for Surya image forecast export.")
    parser.add_argument("--note", default="Not implemented yet.")
    args = parser.parse_args()
    print(args.note)


if __name__ == "__main__":
    main()
