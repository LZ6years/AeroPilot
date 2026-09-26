"""Summarize existing AirSim audits offline, without SDK, model, or simulator I/O.

Example: python tests/manual/airsim_timing_report.py --batch-root <batch>
--output <new-json>. Historical records lacking stage timings are reported as
missing, never treated as zero-cost RPCs. Only selected attempts in summary.json
are aggregated, so prior retries are not counted twice.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def summarize(root: Path) -> dict:
    """Separate total case wall time from partial and complete action timings."""
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    cases = []
    for case in summary['cases']:
        audit_path = (root / case['attempt_dir'] / 'evaluation_output/evaluation/cases'
                      / case['case_id'] / 'audit.json')
        if not audit_path.exists():
            cases.append({'case_id': case['case_id'], 'audit_available': False})
            continue
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        # New diagnostics are intentionally absent from model ToolResults.
        # Canonical action audits retain both old and new envelope versions.
        results = [a.get('result', {}) for a in audit.get('actions', [])]
        diagnostics = [r.get('execution_state', {}).get('action_diagnostics') or
                       r.get('data', {}).get('action_diagnostics') for r in results]
        diagnostics = [d for d in diagnostics if d]
        totals = Counter()
        for item in diagnostics:
            totals.update(item.get('timings_sec', {}))
        elapsed = [d['timings_sec']['total'] for d in diagnostics if 'total' in d.get('timings_sec', {})]
        cases.append({'case_id': case['case_id'], 'benchmark': case['benchmark'],
                      'audit_available': True, 'case_wall_sec': case.get('elapsed_seconds'),
                      'actions': len(diagnostics), 'missing_stage_timings': len(diagnostics) - len(elapsed),
                      'action_window_sec': sum(d.get('elapsed_sec', 0) for d in diagnostics),
                      'stage_totals_sec': dict(totals),
                      'action_total_p50_sec': statistics.median(elapsed) if elapsed else None,
                      'action_total_p95_sec': sorted(elapsed)[max(0, math.ceil(len(elapsed) * .95) - 1)] if elapsed else None,
                      'failures': dict(Counter(d['failure_kind'] for d in diagnostics if d.get('failure_kind')))})
    return {'source': str(root.resolve()), 'offline_only': True,
            'timing_note': 'action_wait contains resume/submit/future_wait; total contains all stages. Do not sum nested timings. Capture/stabilization/LLM are outside this action total.',
            'cases': cases}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.batch_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(args.output)
