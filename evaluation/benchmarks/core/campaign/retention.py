"""Operator-selected persistence cohorts; navigation and scoring stay frozen.

Full records finish before compact records begin. Capacity is checked for the
current cohort, and compact estimates are calibrated from actual receipts.
No historical artifacts are removed by this policy.
"""
import shutil
from collections import defaultdict
import hashlib

from evaluation.benchmarks.core.campaign.recovery import ParallelRecoveryQueue
from evaluation.benchmarks.core.campaign.artifacts import PROFILE as ARCHIVE_PROFILE


def select_full_audit(tasks, count, seed=42):
    """Select audit retention by proportional strata and stable hash order.

    Largest remainders break ties by scene/difficulty. Outcome-independent
    selection never removes any task from the scored population.
    """
    if type(count) is not int or not 1 <= count <= len(tasks):
        raise ValueError('Full-audit count must be between one and the population size')
    buckets = defaultdict(list)
    keys = [ParallelRecoveryQueue.key(t) for t in tasks]
    if len(set(keys)) != len(keys):
        raise ValueError('Audit population contains duplicate task identities')
    for task in tasks:
        buckets[(str(task['scene_id']), str(task['difficulty']))].append(task)
    quota = {key: len(values) * count // len(tasks) for key, values in buckets.items()}
    order = sorted(buckets, key=lambda key: (-(len(buckets[key]) * count % len(tasks)), key))
    for key in order[:count - sum(quota.values())]:
        quota[key] += 1
    result = []
    for key in sorted(buckets):
        ranked = sorted(buckets[key], key=lambda t: (hashlib.sha256(
            (str(seed) + ':' + t['episode_id']).encode()).hexdigest(), ParallelRecoveryQueue.key(t)))
        result.extend(ranked[:quota[key]])
    return result


def profile(task, policy):
    return ('full' if ParallelRecoveryQueue.key(task) in policy['full_task_keys']
            else policy.get('compact_profile', 'paper_process_metrics_v1'))


def _finished_for_retention(item, queue):
    """Release the next logging cohort after all allowed attempts are closed.

    Exhausted technical failures remain unscored in campaign coverage. They
    only stop blocking the next retention cohort when every attempt is invalid
    and independently cleaned; this grants no new attempt or valid result.
    Incomplete cleanup, missing receipts and available retry budgets still block.
    """
    if queue.terminal(item):
        return True
    attempts = item.get('attempts', [])
    return (item.get('status') == 'retry_exhausted' and item.get('result') is None
            and len(attempts) >= queue.attempt_limit(item)
            and all(a.get('cleanup_verified') is True and a.get('valid_score') is False
                    and a.get('status') in {'infrastructure_failed', 'provider_failed', 'interrupted'}
                    for a in attempts))


def candidates(tasks, queue, policy):
    remaining = [t for t in tasks if not _finished_for_retention(queue.state['tasks'][queue.key(t)], queue)]
    full = [t for t in remaining if profile(t, policy) == 'full']
    return full or remaining


def budget(output, tasks, queue, policy, workers):
    stage = candidates(tasks, queue, policy)
    selected = profile(stage[0], policy) if stage else policy.get('compact_profile', 'paper_process_metrics_v1')
    attempts = [a for t in tasks for a in queue.state['tasks'][queue.key(t)]['attempts']
                if a.get('logging_profile', 'full') == selected and a.get('retained_bytes')]
    # The measured seed prevents a newly selected archive format from starting
    # with a zero-byte estimate. Old formats already occupy the measured free
    # space, but cannot predict this format's future per-case retention.
    seed = policy['full_seed_bytes'] if selected == 'full' else policy.get('compact_seed_bytes', 0)
    per_case = max([a['retained_bytes'] for a in attempts] + [seed])
    peak = max([max(a.get('temporary_depth_bytes', 0), a.get('archive_working_bytes', 0))
                for a in attempts] + [policy.get('compact_peak_bytes', 0) if selected != 'full' else 0])
    required = max(10 * 1024**3, len(stage) * per_case * 2 + workers * peak)
    free = shutil.disk_usage(output).free
    return dict(scope='retention_cohort', logging_profile=selected, remaining_cases=len(stage),
                total_remaining_cases=sum(not _finished_for_retention(queue.state['tasks'][queue.key(t)], queue)
                                          for t in tasks),
                observed_max_retained_bytes=per_case, observed_cases=len(attempts),
                measured_seed_bytes=seed, temporary_peak_bytes=peak,
                calibration_pending=not attempts, required_bytes=required, free_bytes=free, passed=free >= required)


def validate(policy, tasks):
    keys = [ParallelRecoveryQueue.key(t) for t in tasks]
    full = policy.get('full_task_keys')
    if not isinstance(full, list) or not full or len(set(full)) != len(full) or not set(full) <= set(keys):
        raise ValueError('Retention cohort must identify unique frozen tasks')
    if type(policy.get('full_seed_bytes')) is not int or policy['full_seed_bytes'] <= 0:
        raise ValueError('Retention requires a measured full-record size')
    if policy.get('compact_profile', 'paper_process_metrics_v1') not in {'paper_process_metrics_v1', ARCHIVE_PROFILE}:
        raise ValueError('Unknown compact artifact profile')
    if policy.get('compact_profile') == ARCHIVE_PROFILE:
        if any(type(policy.get(key)) is not int or policy[key] <= 0
               for key in ('compact_seed_bytes', 'compact_peak_bytes')):
            raise ValueError('Post-attempt archives require measured retained and working sizes')
