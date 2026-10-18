"""Download public HF datasets with mirror support and persistent resume metadata.

No model/API calls, extraction or dependency installation. Uses the official Hub
HTTP downloader (Xet disabled for mirror compatibility). Keep local_dir/.cache
to resume partial transfers. A manifest pins the commit across restarts.
"""
from __future__ import annotations

import argparse
import errno
import fnmatch
import json
import os
from pathlib import Path
import shutil
import time


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return number


def retry(operation, attempts, sleep=time.sleep):
    """Retry transient transport/server failures, never auth or disk failures."""
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if (status is not None and status < 500 and status not in (408, 429)) or (
                isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EACCES, errno.EROFS)
            ) or isinstance(exc, (ValueError, TypeError, ImportError)):
                raise
            if attempt == attempts:
                raise
            delay = min(60, 5 * 2 ** min(attempt - 1, 4))
            # Avoid printing signed download URLs or credentials from exceptions.
            print(f'{type(exc).__name__}: retry {attempt + 1}/{attempts} in {delay}s', flush=True)
            sleep(delay)


def selected_files(siblings, patterns):
    selected = [s for s in siblings if not patterns or any(
        fnmatch.fnmatchcase(s.rfilename, pattern) for pattern in patterns)]
    if not selected:
        raise ValueError('No matching files; check --include patterns')
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, help='Dataset owner/name')
    parser.add_argument('--local-dir', type=Path, required=True)
    parser.add_argument('--endpoint', choices=['https://hf-mirror.com', 'https://huggingface.co'],
                        default='https://hf-mirror.com')
    parser.add_argument('--include', action='append', help='Repeatable filename glob; quote in shell')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--workers', type=positive, default=4)
    parser.add_argument('--attempts', type=positive, default=20)
    parser.add_argument('--timeout', type=positive, default=120)
    parser.add_argument('--list-only', action='store_true', help='Metadata only, no payload download')
    args = parser.parse_args(argv)
    # Hub environment settings must precede its first import.
    os.environ['HF_ENDPOINT'] = args.endpoint
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(args.timeout)
    os.environ['HF_HUB_ETAG_TIMEOUT'] = str(args.timeout)
    from huggingface_hub import HfApi, snapshot_download

    folder = args.local_dir.expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / '.flyagent-download.json'
    revision = args.revision
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved['repo'] != args.repo or saved['requested_revision'] != args.revision:
            raise ValueError('Destination belongs to another repository/revision; use a new directory')
        revision = saved['commit']
    api = HfApi(endpoint=args.endpoint, token=False)
    info = retry(lambda: api.repo_info(args.repo, repo_type='dataset', revision=revision,
                                     files_metadata=True), args.attempts)
    files = selected_files(info.siblings, args.include)
    total = sum(f.size or 0 for f in files)
    for entry in files:
        print(f'{entry.size if entry.size is not None else "unknown"}\t{entry.rfilename}')
    print(f'Commit: {info.sha}\nSelected: {len(files)} files, {total / 1024**3:.2f} GiB', flush=True)
    if args.list_only:
        return 0
    if any(f.size is None for f in files):
        raise ValueError('Missing remote sizes; cannot check available space')
    # Conservative check: completed files reduce requirement; partial files do not.
    # Reserve headroom and leave extraction to a separate, explicit operation.
    needed = sum(f.size for f in files if not (folder / f.rfilename).is_file()
                 or (folder / f.rfilename).stat().st_size != f.size)
    if shutil.disk_usage(folder).free < needed + 1024**3:
        raise ValueError('Insufficient disk space for selected downloads plus 1 GiB reserve')
    if not manifest.exists():
        with manifest.open('x', encoding='utf-8') as stream:
            json.dump(dict(repo=args.repo, requested_revision=args.revision, commit=info.sha), stream, indent=2)
    retry(lambda: snapshot_download(repo_id=args.repo, repo_type='dataset', revision=info.sha,
          local_dir=str(folder), allow_patterns=[f.rfilename for f in files],
          endpoint=args.endpoint, token=False, max_workers=args.workers), args.attempts)
    for entry in files:
        if (folder / entry.rfilename).stat().st_size != entry.size:
            raise ValueError('Downloaded file size mismatch: ' + entry.rfilename)
    print('Download complete; size checks passed. Keep .cache for future resume.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped. Run the same command to resume.')
        raise SystemExit(130)
