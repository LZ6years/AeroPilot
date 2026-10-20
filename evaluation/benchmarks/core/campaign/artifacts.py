"""Lossless post-attempt archives for public intent/tool/image reviews.

Nothing in this module is installed in a flight host. The controller calls it
only after the attempt has finished and its scene cleanup has been verified.
Public transcripts, request receipts, RGB/tool images and ordinary HTML stay
at their original paths. Repeated snapshots and diagnostic depth are retained
in a verified ZIP with exact original bytes and a restoration manifest.
"""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_DEFLATED, ZipFile

from evaluation.benchmarks.core.report_io import write_json

PROFILE = "intent_tools_images_v2"
ARCHIVE = "diagnostic_archive.zip"
MANIFEST = "diagnostic_archive.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _owned(root, relative):
    """Reject links and traversal before any archive, restoration or unlink."""
    path = root / relative
    if path.is_symlink() or root not in path.resolve().parents:
        raise ValueError("Archive member escapes its attempt: " + str(relative))
    return path


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = []

    def handle_starttag(self, tag, attrs):
        self.values.extend(value for key, value in attrs if key in {"href", "src", "data"} and value)


def _linked_files(root):
    """A human review link must keep working without extracting the archive."""
    protected = set()
    for page in root.rglob("*.html"):
        parser = _Links()
        parser.feed(page.read_text(encoding="utf-8"))
        for value in parser.values:
            parsed = urlsplit(value)
            if not parsed.scheme and parsed.path:
                protected.add((page.parent / unquote(parsed.path)).resolve())
    return protected


def _candidates(root):
    # This explicit allowlist never contains public memory, request receipts,
    # images, final topology, settings, trajectories, scores or process owners.
    for run in (root / "runs").glob("*"):
        if not run.is_dir():
            continue
        yield from (run / "debug/context_checkpoints").glob("*.json")
        yield from (run / "contexts/place_memory").glob("**/map_*.json")
        yield from (run / "artifacts").glob("**/depth/*.npy")
    yield from (root / "evaluation/cases").glob("*/audit.json")


def validate_archive(directory):
    """Verify exact bytes, not only ZIP CRCs, before deleting an original."""
    root = Path(directory).resolve()
    record = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    archive = _owned(root, record["archive"])
    if digest(archive.read_bytes()) != record["archive_sha256"]:
        raise ValueError("Diagnostic archive hash mismatch")
    with ZipFile(archive) as bundle:
        if set(bundle.namelist()) != {item["path"] for item in record["members"]}:
            raise ValueError("Diagnostic archive member mismatch")
        for item in record["members"]:
            _owned(root, item["path"])
            data = bundle.read(item["path"])
            if len(data) != item["bytes"] or digest(data) != item["sha256"]:
                raise ValueError("Diagnostic archive content mismatch: " + item["path"])
    return record


def compact_attempt(directory, *, profile):
    """Archive only a closed, independently cleaned attempt; fail without loss.

    The manifest and verified ZIP become durable before originals are removed.
    An interruption during unlink leaves both representations recoverable.
    Repeated calls never replace an existing archive or remove changed files.
    """
    root = Path(directory).resolve()
    if profile != PROFILE:
        raise ValueError("Post-attempt archive profile was not explicitly selected")
    cleanup = json.loads((root / "cleanup.json").read_text(encoding="utf-8"))
    process = json.loads((root / "process.json").read_text(encoding="utf-8"))
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8")) if (root / "receipt.json").exists() else {}
    # Older controllers could overwrite process.json with a final polling
    # heartbeat while the cleanup worker was returning. The durable receipt is
    # the attempt owner's terminal proof; a bare stale heartbeat is not.
    closed = process.get("status") == "finished" or (receipt.get("cleanup_verified") is True
             and receipt.get("process", {}).get("status") == "finished")
    if cleanup.get("verified") is not True or not closed:
        raise ValueError("Cannot archive an active or unverified attempt")
    if (root / MANIFEST).exists():
        return validate_archive(root)
    if (root / ARCHIVE).exists():
        raise ValueError("Uncommitted diagnostic archive requires operator review")
    protected = _linked_files(root)
    paths = sorted({p for p in _candidates(root) if p.is_file() and p.resolve() not in protected})
    before = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    entries = []
    temporary = root / (ARCHIVE + ".tmp")
    if temporary.exists():
        raise ValueError("Uncommitted diagnostic archive staging file")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in paths:
            relative = path.relative_to(root).as_posix()
            data = _owned(root, relative).read_bytes()
            bundle.writestr(relative, data)
            entries.append(dict(path=relative, bytes=len(data), sha256=digest(data)))
    # Verify the staged archive in memory before publishing the deletion plan.
    with ZipFile(temporary) as bundle:
        for item in entries:
            if digest(bundle.read(item["path"])) != item["sha256"]:
                raise ValueError("Staged diagnostic archive failed verification")
    os.replace(temporary, root / ARCHIVE)
    with (root / ARCHIVE).open("r+b") as durable:
        os.fsync(durable.fileno())
    record = dict(profile=PROFILE, archive=ARCHIVE, archive_sha256=digest((root / ARCHIVE).read_bytes()),
                  members=entries, before_bytes=before,
                  archived_original_bytes=sum(item["bytes"] for item in entries),
                  archive_bytes=(root / ARCHIVE).stat().st_size,
                  cleanup_verified=True, process_finished=True)
    write_json(root / MANIFEST, record)
    validate_archive(root)
    for item in entries:
        path = _owned(root, item["path"])
        if digest(path.read_bytes()) != item["sha256"]:
            raise ValueError("Closed artifact changed during archive: " + item["path"])
        path.unlink()
    return record


def restore_archive(directory, destination):
    """Restore exact archived bytes into a separate offline investigation tree.

    Copy the retained attempt tree there first when a complete original layout
    is needed. Never restore into an active attempt or overwrite changed files.
    """
    root, target = Path(directory).resolve(), Path(destination).resolve()
    if target == root or root in target.parents:
        raise ValueError("Restoration must use a separate directory")
    record = validate_archive(root)
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(root / ARCHIVE) as bundle:
        for item in record["members"]:
            path = _owned(target, item["path"])
            data = bundle.read(item["path"])
            if path.exists() and path.read_bytes() != data:
                raise ValueError("Restoration would overwrite changed content")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return record
