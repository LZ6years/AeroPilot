"""Run FlyAgent from `python -m src.cli`.

The installed console command `flyagent` points at the same `main()` function,
so this file keeps the module execution path convenient for development.
"""

from __future__ import annotations

from src.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())
