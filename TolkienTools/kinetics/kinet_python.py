#!/usr/bin/env python3
"""Compatibility entry point for TolKinet.

The implementation is split across modules in this directory. This file is
kept so existing calls to `kinet_python.py` keep working.
"""

from __future__ import annotations

from kinet_cli import main


if __name__ == "__main__":
    main()
