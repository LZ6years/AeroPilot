"""Process-local HTTP resume adapter for huggingface_hub 1.33.0.

Keeps stable partial files across process exits without editing site-packages.
Run only one downloader per destination. Existing random-name partial files
are deliberately not adopted: another HF process may still be writing them.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re


def persistent_download(incomplete_path, destination_path, url_to_download, headers,
                        expected_size, filename, force_download, etag,
                        xet_file_data, tqdm_class=None):
    """Replace only the temporary-file policy; reuse Hub HTTP Range handling."""
    from huggingface_hub import file_download as fd
    partial, target = Path(incomplete_path), Path(destination_path)
    if target.exists() and not force_download:
        return
    if force_download:
        raise ValueError('Force download is disabled in the resume adapter')
    if expected_size is None:
        raise ValueError('Cannot safely resume without expected size')
    partial.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists() and partial.stat().st_size > expected_size:
        raise ValueError('Partial file is larger than expected; inspect it before retrying')
    offset = partial.stat().st_size if partial.exists() else 0
    fd._check_disk_space(expected_size - offset, partial.parent)
    print(f'{filename}: resume from {offset}/{expected_size} bytes', flush=True)
    # No finally/unlink: retain the prefix after errors, SIGINT or process exit.
    with partial.open('ab') as stream:
        fd.http_get(url_to_download, stream, resume_size=offset, headers=headers,
                    expected_size=expected_size, displayed_filename=filename,
                    tqdm_class=tqdm_class)
    if partial.stat().st_size != expected_size:
        raise ValueError('Downloaded size does not match metadata')
    # LFS ETags are content SHA256; do not mistake Git blob SHA1 for SHA256.
    if etag and re.fullmatch(r'[0-9a-fA-F]{64}', etag):
        digest = hashlib.sha256()
        with partial.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != etag.lower():
            raise ValueError('SHA256 mismatch; partial retained for inspection')
    fd._chmod_and_move(partial, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-dir', type=Path, required=True)
    parser.add_argument('--include', nargs='+')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
    os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')
    import fcntl
    import huggingface_hub as hub
    from huggingface_hub import file_download as fd
    if hub.__version__ != '1.33.0':
        raise RuntimeError('Adapter verified for huggingface_hub 1.33.0 only; do not silently upgrade')
    folder = args.local_dir.expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.flyagent-resume.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another resume downloader owns this directory') from None
        fd._download_to_tmp_and_move = persistent_download
        # Persist the snapshot revision so restarts do not mix upstream versions.
        revision_file = folder / '.flyagent-traveluav-revision'
        if revision_file.exists():
            revision = revision_file.read_text().strip()
        else:
            revision = hub.HfApi(token=False).repo_info(
                'wangxiangyu0814/TravelUAV_env', repo_type='dataset').sha
            revision_file.write_text(revision + '\n', encoding='utf-8')
        if not re.fullmatch('[0-9a-f]{40}', revision):
            raise ValueError('Invalid saved revision')
        hub.snapshot_download('wangxiangyu0814/TravelUAV_env', repo_type='dataset',
            revision=revision, local_dir=folder, allow_patterns=args.include,
            max_workers=args.workers, token=False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped; stable partial files retained.')
        raise SystemExit(130)
