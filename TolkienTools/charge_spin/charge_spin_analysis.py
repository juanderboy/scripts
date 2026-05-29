#!/usr/bin/env python3
"""Compatibility entry point for the charge/spin analysis module."""

import sys

from charge_spin_cli import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nchau :(")
        sys.exit(130)
