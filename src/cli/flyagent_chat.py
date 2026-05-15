"""Compatibility wrapper for the runtime chat CLI.

The maintained implementation lives in `src.runtime.cli.main`.  This module is
kept so older commands and tests that import `src.cli.flyagent_chat` continue to
work while CLI ownership moves under `src/runtime/cli`.
"""

from __future__ import annotations

import sys

from src.runtime.cli.main import (
    chat_repl as repl,
    create_chat_session as create_session,
    handle_slash_command,
    main as runtime_main,
    print_chat_banner as print_banner,
    print_chat_help as print_help,
    print_json,
    print_summary,
)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "chat" not in args:
        args.insert(0, "chat")
    return runtime_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
