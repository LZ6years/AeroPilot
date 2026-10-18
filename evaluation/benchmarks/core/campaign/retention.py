"""Operator-selected persistence cohorts; navigation and scoring stay frozen.

Full records finish before compact records begin. Capacity is checked for the
current cohort, and compact estimates are calibrated from actual receipts.
No historical artifacts are removed by this policy.
"""
import shutil

from evaluation.benchmarks.core.campaign.recovery import ParallelRecoveryQueue


def profile(task, policy):
    return 'full' if ParallelRecoveryQueue.key(task) in policy['full_task_keys'] else 'paper_process_metrics_v1'


def candidates(tasks, queue, policy):
    remaining = [t for t in tasks if not queue.terminal(queue.state['tasks'][queue.key(t)])]
    full = [t for t in remaining if profile(t, policy) == 'full']
    return full or remaining


def budget(output, tasks, queue, policy, workers):
    stage = candidates(tasks, queue, policy)
    selected = profile(stage[0], policy) if stage else 'paper_process_metrics_v1'
    attempts = [a for t in tasks for a in queue.state['tasks'][queue.key(t)]['attempts']
                if a.get('logging_profile', 'full') == selected and a.get('retained_bytes')]
    per_case = max([a['retained_bytes'] for a in attempts] +
                   ([policy['full_seed_bytes']] if selected == 'full' else [0]))
    peak = max((a.get('temporary_depth_bytes', 0) for a in attempts), default=0)
    required = max(10 * 1024**3, len(stage) * per_case * 2 + workers * peak)
    free = shutil.disk_usage(output).free
    return dict(scope='retention_cohort', logging_profile=selected, remaining_cases=len(stage),
                total_remaining_cases=sum(not queue.terminal(queue.state['tasks'][queue.key(t)]) for t in tasks),
                observed_max_retained_bytes=per_case, observed_cases=len(attempts),
                calibration_pending=not attempts, required_bytes=required, free_bytes=free, passed=free >= required)


def validate(policy, tasks):
    keys = [ParallelRecoveryQueue.key(t) for t in tasks]
    full = policy.get('full_task_keys')
    if not isinstance(full, list) or not full or len(set(full)) != len(full) or not set(full) <= set(keys):
        raise ValueError('Retention cohort must identify unique frozen tasks')
    if type(policy.get('full_seed_bytes')) is not int or policy['full_seed_bytes'] <= 0:
        raise ValueError('Retention requires a measured full-record size')
