"""
Main entry point for running the TwinTrade executable.
"""

import sys
from twintrade.cli import run_cli


def main() -> None:
    sys.exit(run_cli())


if __name__ == "__main__":
    main()
