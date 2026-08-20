"""Small, simulator-neutral helpers for durable benchmark report writes.

Benchmark execution can be interrupted while a report is being refreshed.
Writing beside the destination and replacing it atomically prevents readers
from observing a truncated JSON, Markdown, CSS, or HTML file.  This module is
intentionally limited to report artifacts; runtime memory keeps its own
append-only persistence rules.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(
    path: str | Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> str:
    """Write text through a same-directory temporary file and ``os.replace``.

    A same-directory temporary file is required because an atomic replace is
    only guaranteed on the same filesystem.  The temporary file is removed if
    encoding, flushing, or replacement fails.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return str(target)
