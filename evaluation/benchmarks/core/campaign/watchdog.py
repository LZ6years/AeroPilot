"""External, single-instance supervision of the frozen seen controller.

The watchdog reads campaign progress but never claims tasks, scores results or
changes scene state. Every restart goes through the controller's ownership and
cleanup checks. An OS task periodically relaunches this watchdog itself; an
independent file lock prevents duplicate supervisors across crashes/logins.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil
from evaluation.benchmarks.core.campaign.compat import selected_tasks
from evaluation.benchmarks.core.campaign.compat import completion_status
from evaluation.benchmarks.core.campaign.contracts import primary_config


BACKOFF_SECONDS = (30, 120, 300, 900, 1800)
CONTROLLER_STALL_SECONDS = 1200
POLL_SECONDS = 10
RECOVERABLE_REASONS = {
    'cleanup_unverified', 'interrupted_cleanup_unverified', 'host_exited',
    'host_startup_timeout', 'host_interrupted', 'no_progress', 'episode_deadline',
    'provider_unavailable', 'storage_budget', 'pilot_storage_budget', 'parallel_cuda_memory', 'parallel_rate_limit',
}


def resume_policy(state, *, manual_pause=False, alive=False, controller_error=None,
                  storage=None, explicit_resume=False, infrastructure_recovery=False, terminal_status=None):
    """Choose recovery without changing the queue or treating failures as scores."""
    if alive:
        return 'observe'
    if manual_pause or state.get('reason') == 'manual_interrupt' and not explicit_resume:
        return 'manual_pause'
    if state.get('status') in {'seen_complete', 'awaiting_image_budget_choice', 'campaign_complete'} or (
            isinstance(terminal_status, str) and state.get('status') == terminal_status):
        return state['status']
    if not state.get('tasks'):
        return 'needs_attention'
    reason = state.get('reason')
    if reason == 'infrastructure_failure':
        return 'recover' if infrastructure_recovery else 'needs_attention'
    if reason == 'controller_exception':
        transient = {'PermissionError', 'TimeoutExpired', 'ConnectionError'}
        error = controller_error or {}
        diagnostic_decode = error.get('error_type') == 'UnicodeDecodeError' and 'stderr.log' in error.get('traceback', '')
        # An operator repairing a verified frozen controller can explicitly
        # resume once. The marker is consumed at launch; a repeated code fault
        # still needs attention instead of entering an endless restart loop.
        return 'recover' if explicit_resume or error.get('error_type') in transient or diagnostic_decode else 'needs_attention'
    if reason == 'storage_or_coverage_check':
        return 'recover' if (storage or {}).get('passed') is False else 'needs_attention'
    if reason in RECOVERABLE_REASONS or reason is None or reason == 'manual_interrupt' and explicit_resume:
        return 'provider_probe' if reason == 'provider_unavailable' else 'recover'
    # Authentication/configuration, unauthorized GPU, failed technical gates,
    # unknown infrastructure and exhausted episode budgets require attention.
    return 'needs_attention'


def controller_owner_allowed(owner, output):
    """A stale-heartbeat kill requires a recorded parallel controller identity."""
    command = owner.get('command', [])
    root = (Path(output) / 'parallel/controller_source').resolve()
    for index, argument in enumerate(command):
        try:
            path = Path(argument).resolve()
        except (OSError, ValueError):
            continue
        if path.name == 'parallel_campaign.py' and root in path.parents:
            if index + 1 >= len(command) or command[index + 1] != 'run' or '--output' not in command:
                return False
            output_index = command.index('--output') + 1
            return output_index < len(command) and Path(command[output_index]).resolve() == Path(output).resolve()
    return False


class CampaignWatchdog:
    """Persist restart backoff independently of the single-writer task queue.

    PID/create-time/command receipts protect unrelated processes. A restart
    cannot dispatch a scene until the frozen controller revalidates cleanup.
    Backoff persists across watchdog restarts and resets only on valid progress.
    """

    def __init__(self, output, workers, controller, *, clock=time.time):
        self.output, self.workers, self.control, self.clock = Path(output), workers, controller, clock
        self.directory = self.output / 'parallel/watchdog'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = controller.read_json(self.output / 'manifest.json')
        if self.manifest.get('strict_workers') and workers != self.manifest['strict_workers']:
            raise ValueError('Campaign watchdog worker count differs from the frozen configuration')
        self.overlay = controller.read_json(self.output / 'parallel/execution_overlay.json')
        controller.verify_frozen(self.output, self.manifest)
        controller.verify_overlay(self.overlay, self.manifest)
        self.state = controller.read_json(self.directory / 'state.json') or {
            'failures': 0, 'next_retry_at': 0, 'restart_count': 0, 'last_scored': 0}
        self.owner = controller.owner_of(psutil.Process())
        self.last_status = None

    def event(self, kind, **fields):
        record = {'time': datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(), 'event': kind, **fields}
        def append():
            with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
        self.control.retry_file_access(append, path=self.directory / 'events.jsonl', action='append')

    def checkpoint(self, status, *, reason=None, controller=None, counts=None):
        self.state['updated_at'] = self.clock()
        self.control.write_json(self.directory / 'state.json', self.state)
        self.control.write_json(self.directory / 'status.json', {
            'status': status, 'reason': reason, 'watchdog': self.owner, 'controller': controller,
            'next_retry_at': self.state.get('next_retry_at'), 'restart_count': self.state['restart_count'],
            'failures': self.state['failures'], 'counts': counts or {},
            'interpreter': sys.executable, 'updated_at': self.state['updated_at']})
        signature = (status, reason)
        if signature != self.last_status:
            self.event('state_changed', status=status, reason=reason)
            self.last_status = signature

    def backoff(self, reason, *, identity=None):
        self.state['failures'] += 1
        delay = BACKOFF_SECONDS[min(self.state['failures'] - 1, len(BACKOFF_SECONDS) - 1)]
        self.state.update(next_retry_at=self.clock() + delay, last_dead_identity=identity)
        self.event('recovery_backoff', reason=reason, delay_seconds=delay)

    def recorded_controller(self):
        own = self.control.read_json(self.directory / 'controller_process.json')
        dispatcher = self.control.read_json(self.output / 'dispatcher.json')
        # Adopt an already running controller instead of starting a competitor.
        for owner in (own, dispatcher):
            if owner and self.control.matching_process(owner) is not None:
                return owner, True
        return own or dispatcher, False

    def observe_handoff(self, counts):
        """Supervise a deliberate drain without killing its suspended writer.

        The independent operator's PID/create-time/command receipt distinguishes
        this transition from a stalled controller. If it disappears, preserve a
        pause and release only its recorded writer so normal cleanup can finish.
        """
        directory = self.output / 'parallel/scheduling_handoff'
        transition = self.control.read_json(directory / 'status.json')
        if transition.get('status') not in {'draining', 'installing'}:
            return False
        owner = self.control.read_json(directory / 'process.json')
        command = owner.get('command', [])
        if not any(Path(arg).name == 'image_budget_order_handoff.py' for arg in command) or '--output' not in command:
            return False
        index = command.index('--output') + 1
        if index >= len(command) or Path(command[index]).resolve() != self.output.resolve():
            return False
        if self.control.matching_process(owner) is not None:
            self.checkpoint('monitoring_handoff', reason='budget_order_change', counts=counts)
        else:
            marker = self.output / 'PAUSE'
            if not marker.exists():
                self.control.atomic_write_text(marker, 'Budget order handoff process exited before completion.\n')
            controller = transition.get('controller', {})
            if controller_owner_allowed(controller, self.output):
                process = self.control.matching_process(controller)
                if process is not None and process.status() == psutil.STATUS_STOPPED:
                    process.resume()
            self.checkpoint('needs_attention', reason='budget_order_handoff_exited', counts=counts)
        return True

    def tick(self):
        state = self.control.read_json(self.output / 'progress.json')
        if state.get('fingerprint') != self.manifest['fingerprint']:
            self.checkpoint('needs_attention', reason='progress_identity_invalid')
            return False
        tasks = selected_tasks(self.manifest, tasks=[item['task'] for item in state['tasks'].values()])
        seen = [item for item in state['tasks'].values() if item['task'] in tasks]
        counts = {}
        for item in seen:
            counts[item['status']] = counts.get(item['status'], 0) + 1
        scored = counts.get('scored', 0)
        if scored > self.state['last_scored']:
            self.state.update(failures=0, next_retry_at=0, last_scored=scored)
        if self.observe_handoff(counts):
            return False
        owner, alive = self.recorded_controller()
        explicit_resume = (self.directory / 'RESUME').exists()
        policy = resume_policy(state, manual_pause=(self.output / 'PAUSE').exists(), alive=alive,
            explicit_resume=explicit_resume,
            infrastructure_recovery=self.control.ParallelRecoveryQueue.recoverable_pause(state),
            controller_error=self.control.read_json(self.output / 'parallel/controller_error.json'),
            storage=self.control.read_json(self.output / 'storage_check.json'), terminal_status=completion_status(self.manifest))
        if alive:
            heartbeat = max((path.stat().st_mtime for path in (
                self.output / 'progress.json', self.output / 'parallel/status.json') if path.exists()), default=0)
            age = self.clock() - max(heartbeat, owner.get('created', 0))
            if age > CONTROLLER_STALL_SECONDS and controller_owner_allowed(owner, self.output):
                self.control.kill_owned(owner)
                self.backoff('controller_heartbeat_stalled', identity=[owner['pid'], owner['created']])
                self.checkpoint('backoff', reason='controller_heartbeat_stalled', counts=counts)
            else:
                self.checkpoint('monitoring', controller=owner, counts=counts)
            return False
        if policy in {'seen_complete', 'awaiting_image_budget_choice', 'campaign_complete'} or policy == completion_status(self.manifest):
            self.checkpoint(policy, counts=counts)
            return True
        if policy in {'manual_pause', 'needs_attention'}:
            self.checkpoint(policy, reason=state.get('reason'), counts=counts)
            return False
        if self.manifest.get('execution', {}).get('mode') == 'local' and state.get('reason') in {
                'cleanup_unverified', 'interrupted_cleanup_unverified'}:
            self.checkpoint('needs_attention', reason=state.get('reason'), counts=counts)
            return False
        if state.get('reason') in {'storage_budget', 'pilot_storage_budget', 'storage_or_coverage_check'}:
            storage = self.control.read_json(self.output / 'storage_check.json')
            required = storage.get('required_bytes')
            if type(required) is int and required > 0:
                # Waiting for disk capacity is not a failed controller restart.
                # Keep its original forecast/margin and leave authoritative
                # progress untouched; the controller rechecks before dispatch.
                free = shutil.disk_usage(self.output).free
                self.control.write_json(self.directory / 'storage_wait.json', {
                    'free_bytes': free, 'required_bytes': required, 'passed': free >= required,
                    'checked_at': self.clock()})
                if free < required:
                    self.checkpoint('waiting_for_storage', reason=state.get('reason'), counts=counts)
                    return False
        identity = [owner.get('pid'), owner.get('created')] if owner else None
        if owner and identity != self.state.get('last_dead_identity'):
            self.backoff(state.get('reason') or 'controller_exited', identity=identity)
        if self.clock() < self.state.get('next_retry_at', 0):
            self.checkpoint('backoff', reason=state.get('reason'), counts=counts)
            return False
        if policy == 'provider_probe':
            if not self.control.provider_probe(primary_config(self.manifest)['env_path'],
                    self.directory / 'service_check.json', service_check=True):
                self.backoff('provider_unavailable', identity=identity)
                self.checkpoint('backoff', reason='provider_unavailable', counts=counts)
                return False
        # Only the controller acquires campaign.lock and writes progress. It
        # verifies frozen source, GPU UUIDs, storage, scenes and task leases.
        workers = self.manifest.get('strict_workers') or min(self.workers, state.get('workers', self.workers))
        stamp = datetime.fromtimestamp(self.clock(), timezone.utc).strftime('%Y%m%d_%H%M%S')
        logs = self.directory / 'runs'
        logs.mkdir(exist_ok=True)
        command = [sys.executable, '-u', '-X', 'utf8', self.overlay['controller_script'],
                   'run', '--output', str(self.output), '--workers', str(workers)]
        with (logs / (stamp + '.log')).open('w', encoding='utf-8') as stdout, \
                (logs / (stamp + '_error.log')).open('w', encoding='utf-8') as stderr:
            child = subprocess.Popen(command, cwd=str(self.output / 'source'), stdout=stdout, stderr=stderr,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        owner = self.control.owner_of(psutil.Process(child.pid))
        self.control.write_json(self.directory / 'controller_process.json', owner)
        self.state['restart_count'] += 1
        self.state['next_retry_at'] = 0
        self.event('controller_started', controller=owner, workers=workers)
        if explicit_resume:
            (self.directory / 'RESUME').unlink()
        self.checkpoint('monitoring', controller=owner, counts=counts)
        return False


def run_watchdog(args, controller):
    """Run until seen completes; duplicate OS launches return without mutations."""
    directory = args.output.resolve() / 'parallel/watchdog'
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        try:
            # Reuse the OS-released single-writer lock at a separate directory;
            # never hold the actual campaign queue lock in the supervisor.
            stack.enter_context(controller.campaign_lock(directory))
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return
            raise
        try:
            watcher = CampaignWatchdog(args.output.resolve(), args.workers, controller)
        except Exception as exc:
            controller.write_json(directory / 'status.json', {
                'status': 'needs_attention', 'reason': 'watchdog_initialization_failed',
                'error_type': type(exc).__name__, 'error': str(exc), 'interpreter': sys.executable})
            raise
        watcher.event('watchdog_started', watchdog=watcher.owner, interpreter=sys.executable)
        while True:
            try:
                if watcher.tick():
                    return
            except KeyboardInterrupt:
                # An explicit supervisor interruption must not silently restart
                # the controller via the next OS trigger.
                controller.atomic_write_text(args.output / 'PAUSE', 'Watchdog manually interrupted.\n')
                return
            except Exception as exc:
                # Do not clear progress or launch through missing/error evidence.
                watcher.backoff(type(exc).__name__, identity=watcher.state.get('last_dead_identity'))
                watcher.checkpoint('waiting_for_recovery', reason=type(exc).__name__)
                controller.write_json(directory / 'last_error.json', {
                    'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
            time.sleep(POLL_SECONDS)


# Read compatibility for historical diagnostics and frozen supervisor imports.
SeenWatchdog = CampaignWatchdog
