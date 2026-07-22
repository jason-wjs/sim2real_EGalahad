#!/usr/bin/env python3
"""Backward-compatible entry point for mjlab_g1_native conversion."""

from __future__ import annotations

import sys

from convert_to_any4hdmi import main


if __name__ == "__main__":
    raise SystemExit(main(["--source-format", "mjlab-g1-native", *sys.argv[1:]]))
