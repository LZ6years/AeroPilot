"""Pause an already running frozen dispatcher at the pilot/report handoff.

The guard observes the final pilot child's completed process receipt, before
the supervisor's remote cleanup and next dispatch. It uses the existing PAUSE
control and leaves frozen flight code and results unchanged. Optional startup
recovery resumes pre-launch SSH timeouts only with independent cleanup proof.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.benchmarks.aerial_vln.recovery import RecoveryQueue
from evaluation.benchmarks.aerial_vln.campaign import campaign_lock
from evaluation.benchmarks.aerial_vln.runner import _summary
from evaluation.benchmarks.core.report_io import atomic_write_text, write_json
from src.agent.config.model import read_env_file
from src.runtime.configs.map_config import load_map_config
from src.sim_server.remote.recovery import verify_unlaunched_scene


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def pilot_rows(manifest, state):
    return [state["tasks"][RecoveryQueue.key(task)] for task in manifest["pilot"]]


def final_child_finished(output, rows):
    """Score-ready last child has exited; its supervisor is still cleaning."""
    if sum(row["status"] == "scored" for row in rows) != 14:
        return False
    active = [row for row in rows if row["status"] == "running"]
    if len(active) != 1:
        return False
    row = active[0]
    task = row["task"]
    base = Path(output) / task["split"] / "attempts" / task["episode_id"] / f'attempt_{row["attempts"][-1]["number"]:02d}'
    process = _json(base / "process.json")
    result = _json(base / "result.json")
    if not result:
        result = _json(base / "evaluation/cases" / task["episode_id"] / "case_result.json")
    return process.get("exit_code") == 0 and not process.get("reason") and bool(result.get("metrics")) \
        and not result.get("infrastructure_error") and not result.get("metric_error")


def write_report(output, manifest, state):
    output = Path(output)
    rows = pilot_rows(manifest, state)
    finished = [row for row in rows if row["status"] == "scored"]
    summaries = {split: _summary([row["result"] for row in finished if row["task"]["split"] == split])
                 for split in ("val_seen", "val_unseen")}
    record = {"pilot_scored": len(finished), "pilot_requested": 15, "status": state["status"],
              "reason": state["reason"], "split_summaries": summaries,
              "full_seen_started": any(row["attempts"] for key, row in state["tasks"].items()
                                       if row["task"]["split"] == "val_seen"
                                       and key not in {RecoveryQueue.key(task) for task in manifest["pilot"]}),
              "pilot": [{"task": row["task"], "status": row["status"], "attempt_count": len(row["attempts"]),
                         "result": row["result"]} for row in rows]}
    write_json(output / "pilot_report.json", record)
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<style>body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:16px system-ui;color:#17212c}table{border-collapse:collapse;width:100%}td,th{padding:.7rem;text-align:left;border-bottom:1px solid #dbe2e9}pre{white-space:pre-wrap}a{color:#1764b0}</style>',
             '<title>AerialVLN-S 15 条预跑报告</title><h1>AerialVLN-S 预跑报告</h1>',
             f'<p>有效评分 {len(finished)} / 15；seen 9 条、unseen 6 条，覆盖 13 个场景。</p>',
             '<p>预跑结束后暂停；报告交付后再启动 seen 剩余 324 条。导航失败保留评分，Provider 中断不进入 SR 分母。</p>',
             '<table><tr><th>Split</th><th>已评分</th><th>SR</th><th>nDTW</th><th>平均步数</th></tr>']
    for split, summary in summaries.items():
        metrics = summary['metrics']
        parts.append(f'<tr><td>{split}</td><td>{summary["evaluated_count"]}</td><td>{metrics.get("success")}</td><td>{metrics.get("ndtw")}</td><td>{metrics.get("steps_taken")}</td></tr>')
    parts.append('</table><h2>过程图片、动作与公开说明</h2><table><tr><th>Split</th><th>Scene</th><th>Episode</th><th>状态</th><th>SR</th><th>完整尝试</th></tr>')
    for row in rows:
        task = row['task']
        result = row.get('result') or {}
        report = result.get('report') or {}
        label = html.escape(task['episode_id'])
        if report.get('path'):
            link = Path(os.path.relpath(report['path'], output)).as_posix()
            label = f'<a href="{html.escape(link, quote=True)}">{label}</a>'
        parts.append(f'<tr><td>{task["split"]}</td><td>{task.get("scene_id")}</td><td>{label}</td><td>{row["status"]}</td><td>{result.get("metrics", {}).get("success")}</td><td>{len(row["attempts"])}</td></tr>')
    storage = _json(output / 'storage_check.json')
    parts.append('</table><h2>技术检查</h2><p>使用冻结模型与源代码；每条分别核验原始图片、HTML、评分和场景清理。</p><p>' +
                 html.escape(str(state['status']) + ': ' + str(state['reason'])) + '</p>')
    if storage:
        parts.append(f'<p>当前 H 盘可用 {storage.get("free_bytes", 0)/1024**3:.2f} GiB；空间检查：{storage.get("passed")}。</p>')
    parts.append('</html>')
    atomic_write_text(output / 'pilot_report.html', '\n'.join(parts))
    return record


def recover_startup(output, manifest, *, tasks=None, stage='seen', verify=verify_unlaunched_scene, launch=subprocess.Popen):
    """Resume the selected frozen queue after explicit pre-launch cleanup proof.

    Historical failed receipts remain unchanged. The recovery receipt and queue
    event explain why resumption is now safe. Full attempts remain capped at
    three by the existing frozen queue; this observer never retries navigation.
    """
    output = Path(output)
    with campaign_lock(output):
        state = _json(output / 'progress.json')
        rows = [state['tasks'][RecoveryQueue.key(task)] for task in tasks] if tasks is not None else pilot_rows(manifest, state)
        if (output / 'PAUSE').exists() or state['status'] != 'paused' or state['reason'] != 'cleanup_unverified':
            return False
        if all(row['status'] == 'scored' for row in rows):
            return False
        dispatcher = _json(output / 'dispatcher.json')
        if dispatcher.get('pid'):
            try:
                command = psutil.Process(dispatcher['pid']).cmdline()
                if 'evaluation.benchmarks.aerial_vln.campaign' in command and 'run' in command and str(output) in command:
                    return False
            except psutil.NoSuchProcess:
                pass
        failed = [row for row in rows if row['status'] == 'retry_pending' and row['attempts']
                  and row['attempts'][-1].get('cleanup_verified') is False]
        if len(failed) != 1:
            return False
        row = failed[0]
        attempt = row['attempts'][-1]
        result = attempt.get('result') or {}
        infrastructure = result.get('infrastructure_error') or {}
        error = infrastructure.get('error') or {}
        if infrastructure.get('stage') != 'scene_setup' or error.get('error_type') != 'TimeoutExpired' \
                or not any(marker in error.get('error', '') for marker in ("['ssh',", "['scp',")):
            return False
        directory = Path(attempt['directory']).resolve()
        if output.resolve() not in directory.parents:
            return False
        scenes = list((directory / 'runs').glob('*/scenes/*'))
        if len(scenes) != 1:
            return False
        for key, value in read_env_file(manifest['configs'][row['task']['split']]['env_path']).items():
            if key.startswith('FLYAGENT_'):
                os.environ.setdefault(key, value)
        packages, _ = load_map_config(manifest['configs'][row['task']['split']]['map_config_path'])
        map_id = manifest['configs'][row['task']['split']]['scene_map'][row['task']['scene_id']]
        package = next(package for package in packages if package.map_id == map_id)
        process = attempt['process']
        proof = verify(package, scenes[0], [process, *process['descendants']])
        write_json(directory / 'startup_recovery_check.json', proof)
        if not proof['verified']:
            return False
        stamp = str(time.time_ns())
        write_json(output / 'recovery' / f'progress_before_{stamp}.json', state)
        write_json(directory / 'cleanup_recovery.json', proof)
        attempt.update(status='infrastructure_failed', cleanup_verified=True, cleanup_recovery=proof)
        row['status'] = 'retry_pending' if len(row['attempts']) < 3 else 'retry_exhausted'
        state.setdefault('recovery_events', []).append({'episode': RecoveryQueue.key(row['task']), **proof})
        state.update(status='prepared', reason=None)
        write_json(output / 'progress.json', state)
        command = [sys.executable, '-u', '-X', 'faulthandler', '-m',
                   'evaluation.benchmarks.aerial_vln.campaign', 'run', '--output', str(output), '--stage', stage]
        log_dir = output / 'recovery'
        with (log_dir / f'dispatcher_{stamp}.log').open('w', encoding='utf-8') as stdout, \
                (log_dir / f'dispatcher_{stamp}_error.log').open('w', encoding='utf-8') as stderr:
            try:
                child = launch(command, cwd=str(output / 'source'), stdout=stdout, stderr=stderr,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            except Exception:
                state.update(status='paused', reason='recovery_dispatcher_launch_failed')
                write_json(output / 'progress.json', state)
                raise
        write_json(output / 'handoff_recovery_dispatcher.json', {'pid': child.pid, 'command': command,
                   'stdout': str(stdout.name), 'stderr': str(stderr.name), 'scene': proof['scene']})
        print(json.dumps({'startup_recovered': True, 'scene': proof['scene'], 'dispatcher_pid': child.pid}), flush=True)
        return True


def dispatcher_running(output):
    """A paused dispatcher may still be flushing its final reports."""
    dispatcher = _json(Path(output) / 'dispatcher.json')
    if not dispatcher.get('pid'):
        return False
    try:
        command = psutil.Process(dispatcher['pid']).cmdline()
        return 'evaluation.benchmarks.aerial_vln.campaign' in command and 'run' in command and str(output) in command
    except psutil.NoSuchProcess:
        return False


def shared_simulator_busy(output, rows):
    """Wait for a foreign flight's dispatcher, including gaps between cases.

    Remember PID identities rather than just polling a port: another serial
    campaign may release the listener briefly before starting its next case.
    This observer never signals or edits that campaign.
    """
    output = Path(output)
    path = output / 'shared_simulator_wait.json'
    wait = _json(path)
    live = []
    for owner in wait.get('owners', []):
        try:
            process = psutil.Process(owner['pid'])
            if process.create_time() == owner['created'] and process.cmdline() == owner['command']:
                live.append(owner)
        except psutil.NoSuchProcess:
            pass
    if live:
        return True
    if wait.get('status') == 'waiting':
        write_json(path, {**wait, 'status': 'released'})
    ports = set()
    for row in rows:
        if row['status'] != 'retry_pending' or not row['attempts']:
            continue
        directory = row['attempts'][-1].get('directory')
        if not directory:
            continue
        proof = _json(Path(directory) / 'startup_recovery_check.json')
        if proof.get('reason') == 'local_tunnel_not_verified_closed':
            ports.update(tunnel['port'] for tunnel in proof.get('local_tunnels', []) if not tunnel['closed'])
    if not ports:
        return False
    owners = {}
    for connection in psutil.net_connections(kind='tcp'):
        if connection.status != psutil.CONN_LISTEN or not connection.laddr or connection.laddr.port not in ports or not connection.pid:
            continue
        try:
            listener = psutil.Process(connection.pid)
            command = listener.cmdline()
            if not command or Path(command[0]).stem.lower() != 'ssh' or '-L' not in command:
                continue
            processes = [listener, *[parent for parent in listener.parents()
                                     if parent.cmdline() and Path(parent.cmdline()[0]).stem.lower().startswith('python')]]
            for process in processes:
                owners[process.pid] = {'pid': process.pid, 'created': process.create_time(), 'command': process.cmdline()}
        except psutil.NoSuchProcess:
            continue
    if owners:
        write_json(path, {'status': 'waiting', 'reason': 'shared_simulator_in_use', 'ports': sorted(ports),
                          'owners': list(owners.values()), 'automatic_resume_after_release': True})
        print(json.dumps({'waiting_shared_simulator': True, 'owner_pids': list(owners)}), flush=True)
        return True
    return False


def recover_paused_queue(output, manifest, rows, checks, *, tasks=None, stage='seen'):
    """Share bounded connectivity rechecks between pilot and split observers."""
    if dispatcher_running(output):
        time.sleep(1)
        return True
    if shared_simulator_busy(output, rows):
        time.sleep(5)
        return True
    signature = tuple((row['task']['episode_id'], len(row['attempts'])) for row in rows if row['status'] == 'retry_pending')
    count, retry_at = checks.get(signature, (0, 0))
    if time.monotonic() < retry_at:
        time.sleep(1)
        return True
    try:
        if recover_startup(output, manifest, tasks=tasks, stage=stage):
            return True
        if shared_simulator_busy(output, rows):
            return True
    except Exception as exc:
        write_json(Path(output) / 'startup_recovery_error.json', {'type': type(exc).__name__, 'error': str(exc), 'check': count + 1})
        if count < 3:
            checks[signature] = (count + 1, time.monotonic() + (30, 120, 300)[count])
            return True
    return False


def monitor_split(output, split, *, recover=False):
    """Observe an explicitly started split after pilot report delivery.

    This mode never arms the pilot PAUSE barrier or starts another split. It
    only restarts the same frozen stage for verified startup interruptions.
    Provider outage and manual pauses retain their existing queue semantics.
    """
    output = Path(output).resolve()
    if not (output / 'pilot_report_delivered.json').is_file():
        raise ValueError('Deliver the pilot report before enabling a full split observer')
    manifest = _json(output / 'manifest.json')
    tasks = [task for task in manifest['tasks'] if task['split'] == split]
    if not tasks:
        raise ValueError('No tasks in the selected split')
    stage = 'seen' if split == 'val_seen' else 'unseen'
    write_json(output / 'handoff_control.json', {'pid': os.getpid(), 'mode': 'split_monitor',
               'split': split, 'pilot_report_delivered': True, 'automatic_seen': split == 'val_seen'})
    previous = None
    recovery_checks = {}
    while True:
        state = _json(output / 'progress.json')
        if not state:
            time.sleep(1)
            continue
        rows = [state['tasks'][RecoveryQueue.key(task)] for task in tasks]
        counts = dict(Counter(row['status'] for row in rows))
        snapshot = (state['status'], state['reason'], counts)
        if snapshot != previous:
            write_json(output / 'split_observer_status.json', {'split': split, 'campaign_status': state['status'],
                       'reason': state['reason'], 'counts': counts, 'requested': len(tasks), 'observer_pid': os.getpid()})
            previous = snapshot
        if all(row['status'] == 'scored' for row in rows) and state['status'] != 'running':
            print(json.dumps({'split_complete': split, 'scored': len(rows)}), flush=True)
            return
        if state['status'] == 'paused':
            if recover and state['reason'] == 'cleanup_unverified' and not (output / 'PAUSE').exists():
                if recover_paused_queue(output, manifest, rows, recovery_checks, tasks=tasks, stage=stage):
                    continue
            print(json.dumps({'split_paused': split, 'scored': counts.get('scored', 0), 'reason': state['reason']}), flush=True)
            return
        time.sleep(1)


def guard(output, *, recover=False):
    output = Path(output).resolve()
    manifest = _json(output / 'manifest.json')
    write_json(output / 'handoff_control.json', {"pid": os.getpid(), "requested": "report_before_full_seen",
                                               "automatic_seen": False, "pilot_count": 15})
    previous = None
    recovery_checks = {}
    armed = False
    while True:
        state = _json(output / 'progress.json')
        if not state:
            time.sleep(1)
            continue
        rows = pilot_rows(manifest, state)
        complete = all(row['status'] == 'scored' for row in rows)
        if not armed and (complete or final_child_finished(output, rows)):
            atomic_write_text(output / 'PAUSE', 'Pilot report handoff requested by user. Resume only after report delivery.\n')
            armed = True
            write_json(output / 'handoff_armed.json', {"pilot_final_child_finished": True, "full_seen_allowed": False})
        snapshot = [(row['status'], len(row['attempts'])) for row in rows] + [(state['status'], state['reason'])]
        if snapshot != previous:
            write_report(output, manifest, state)
            previous = snapshot
        if complete and state['status'] != 'running':
            record = write_report(output, manifest, state)
            write_json(output / 'pilot_report_ready.json', {"scored": 15, "report": str(output / 'pilot_report.html'),
                                                          "full_seen_started": record['full_seen_started']})
            print(json.dumps({"pilot_report_ready": True, "full_seen_started": record['full_seen_started']}), flush=True)
            return
        if state['status'] == 'paused' and not complete:
            if recover and state['reason'] == 'cleanup_unverified' and not (output / 'PAUSE').exists():
                if recover_paused_queue(output, manifest, rows, recovery_checks):
                    continue
            print(json.dumps({"pilot_paused": True, "scored": sum(row['status']=='scored' for row in rows),
                              "reason": state['reason']}), flush=True)
            return
        time.sleep(0.02 if sum(row['status'] == 'scored' for row in rows) == 14 else 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='H:/FlyAgent-Eval/aerialvln_s_20261007', type=Path)
    parser.add_argument('--recover-startup', action='store_true', help='Resume only verified pre-launch SSH timeouts; keep all other pauses.')
    parser.add_argument('--monitor-split', choices=('val_seen', 'val_unseen'), help='After report delivery, observe only this explicitly started full split.')
    args = parser.parse_args()
    # Reuse the OS lock under a separate directory so only one observer can
    # modify recovery state or arm the pilot barrier at a time.
    lock_dir = args.output.resolve() / 'handoff'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with campaign_lock(lock_dir):
        if args.monitor_split:
            monitor_split(args.output, args.monitor_split, recover=args.recover_startup)
        else:
            guard(args.output, recover=args.recover_startup)
