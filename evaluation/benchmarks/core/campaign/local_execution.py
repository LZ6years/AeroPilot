"""Linux campaign recovery using persisted, process-local ownership tokens.

No process names or GPU-wide termination are used. All live group members must
still carry the launch token before a recovered supervisor may signal them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import time


def owned_members(group, token):
    """Reject reused groups and unreadable ownership rather than guessing."""
    if group <= 1 or group == os.getpgrp() or not token:
        raise ValueError("Unsafe local scene ownership")
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            if int(fields[2]) != group or fields[0] in {'Z', 'X'}:
                continue
            if int(fields[3]) != group:
                raise RuntimeError("Scene group no longer belongs to its original session")
            expected = ('FLYAGENT_PROCESS_OWNER=' + token).encode()
            if expected not in (entry / 'environ').read_bytes().split(b'\0'):
                raise RuntimeError("Local scene ownership token mismatch")
            members.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def cleanup_local(run_root):
    """Return the same verified-cleanup contract as remote scene recovery."""
    receipts = []
    for path in Path(run_root).rglob('*.process.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        group, token = record['pgid'], record['owner_token']
        for sig, timeout in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            if not owned_members(group, token):
                break
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + timeout
            while owned_members(group, token) and time.monotonic() < deadline:
                time.sleep(.05)
        if owned_members(group, token):
            raise RuntimeError('Local scene group survived cleanup')
        settings = json.loads(Path(record['settings_path']).read_text(encoding='utf-8'))
        with socket.socket() as client:
            client.settimeout(1)
            closed = client.connect_ex(('127.0.0.1', int(settings['ApiServerPort']))) != 0
        receipts.append({'scene': record['scene'], 'local_pgid': group,
                         'remote_stop': {'cleanup_verified': True},
                         'endpoint': {'status': 'unreachable' if closed else 'reachable'},
                         'local_endpoint_closed': closed})
    return receipts
