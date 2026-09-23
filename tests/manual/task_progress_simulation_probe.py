"""Fixed-case guidance canaries through production benchmark execution/review.

Use --benchmark aerial|travel --group baseline|guided --output-dir <new-dir>.
Baseline swaps only saved prompt/tool prose and disables reminders. Runtime,
sensors, safety, metrics and managed scene cleanup are shared by both treatments.
This diagnostic subset is not a representative benchmark evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent.planning.model_action_planner import ModelActionPlanner
from src.tools.registry import ToolRegistry
from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.aerial_vln.runner import AerialVlnBenchmarkRunner
from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
from evaluation.benchmarks.travel_uav.runner import TravelUavBenchmarkRunner
from tests.manual.task_progress_guidance_probe import BASELINE, write_json, digest, model_override

AERIAL_ID = '39GHHAVOMOP8ZV27ILH5P0TV5F6J4A'
TRAVEL_ID = 'NYCEnvironmentMegapa__c3f9fc36-7c55-4a48-8213-5adce9b35b12'


class AuditedPlanner(ModelActionPlanner):
    """Preserve provider model identity in experiment audits, without changing prompts."""

    def plan_turn(self, *args, **kwargs):
        try:
            return super().plan_turn(*args, **kwargs)
        finally:
            response = self.last_exchange.get('response')
            if isinstance(response, dict):
                response['model_identity'] = {
                    'requested_model': self.last_request.get('model'),
                    'returned_model': self.last_response.get('model')}


class BaselinePlanner(AuditedPlanner):
    """Use saved pre-change prose with today's executable tool contracts."""

    def __init__(self, config):
        super().__init__(replace(config, task_progress_reminder_interval=0))
        saved = json.loads(BASELINE.read_text(encoding='utf-8'))
        self.system_prompt = saved['system_prompt']
        self.descriptions = {t['function']['name']: t['function']['description'].split('\nExecution:', 1)[0]
                             for t in saved['tools']}

    def plan_turn(self, transcript, registry, **kwargs):
        view = ToolRegistry()
        for name in registry.names():
            spec = registry.get(name)
            if spec.wire_name in self.descriptions:
                spec = replace(spec, description=self.descriptions[spec.wire_name])
            view.register(spec)
        return super().plan_turn(transcript, view, **kwargs)


class FixedAerialRunner(AerialVlnBenchmarkRunner):
    """Select one diagnostic case after full-population difficulty assignment."""

    def __init__(self, config, planner_type):
        super().__init__(config)
        self.planner_type = planner_type

    def prepare_evaluation(self, *, require_execution_ready):
        prepared = super().prepare_evaluation(require_execution_ready=False)
        cases = tuple(c for c in prepared.cases if c.episode_id == AERIAL_ID)
        if len(cases) != 1:
            raise ValueError('Fixed AerialVLN case missing from full selection')
        selected = tuple(a for a in prepared.selection.assignments if a.episode_id == AERIAL_ID)
        selection = replace(prepared.selection, selected_assignments=selected)
        prepared = replace(prepared, cases=cases, selection=selection,
                           execution_readiness=self._execution_readiness(selection.selected_scene_ids))
        if require_execution_ready:
            self._validate_dataset_for_execution(prepared)
            if not prepared.execution_readiness['ready']:
                raise ValueError('Fixed AerialVLN scene not ready')
        return prepared

    def _make_runtime(self, run_dir, recorder, *, planner=None):
        return super()._make_runtime(run_dir, recorder,
                                    planner=planner or self.planner_type(self._model_config()))


def make_runner(benchmark, group, output, model=None, run_root=None):
    planner_type = BaselinePlanner if group == 'baseline' else AuditedPlanner
    if benchmark == 'aerial':
        config = AerialVlnEvalConfig.from_file('evaluation/benchmarks/aerial_vln/configs/aerial_vln_s_val_seen_full.example.json')
        source = json.loads(Path('evaluation/benchmarks/aerial_vln/outputs/tool_desc_v2_aerial_expand_scene12_20260922/manifest.json').read_text(encoding='utf-8'))
        config = replace(config, expected_split_sha256=source['dataset']['split_sha256'],
                         limit_per_difficulty=None, difficulties=('low', 'medium', 'high'),
                         execution_mode='managed', save_artifacts=True, enable_action_planner_debug=True,
                         output_dir=str(output), generate_html_review=True)
        if run_root is not None:
            config = replace(config, run_root=str(run_root))
        runner = FixedAerialRunner(config, planner_type)
    else:
        config = TravelUavEvalConfig.from_file('evaluation/benchmarks/travel_uav/configs/travel_uav_test_seen_quick.example.json')
        source = json.loads(Path('evaluation/benchmarks/travel_uav/outputs/tool_desc_v2_travel_expand2_20260922/manifest.json').read_text(encoding='utf-8'))
        config = replace(config, expected_index_sha256=source['dataset']['index_sha256'],
                         map_config_path='configs/maps/travel_uav_remote.local', case_ids=(TRAVEL_ID,),
                         limit_per_difficulty=None, output_dir=str(output))
        if run_root is not None:
            config = replace(config, run_root=str(run_root))
        runner = TravelUavBenchmarkRunner(config, planner_factory=planner_type)
    runner.experiment_source_model = runner._model_config().model_name
    model = model_override(replace(runner._model_config(), task_progress_reminder_interval=0 if group == 'baseline' else 5), model)
    if benchmark == 'aerial':
        runner._loaded_model_config = model
    else:
        runner._model_config_value = model
    return runner


def run(benchmark, group, output, validate_only=False, model=None, run_root=None):
    if output.exists():
        raise ValueError('Use a new output directory')
    runner = make_runner(benchmark, group, output, model=model, run_root=run_root)
    runner.prepare_evaluation(require_execution_ready=True)
    output.mkdir(parents=True)
    # Detect concurrent workspace changes that would confound paired flights.
    sources = sorted([*ROOT.glob('src/**/*.py'), *ROOT.glob('plugins/**/*.py'),
                      *ROOT.glob('plugins/**/*.json'), *ROOT.glob('configs/**/*.json')])
    fingerprint = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    start_sources = fingerprint()
    write_json(output / 'experiment.json', {'benchmark': benchmark, 'group': group,
               'baseline_sha256': digest(json.loads(BASELINE.read_text(encoding='utf-8'))),
               'config': runner.config.to_dict(), 'model': runner._model_config().redacted_summary(),
               'configured_model': runner.experiment_source_model,
               'thinking_policy': 'not explicitly set; provider default',
               'source_hashes': start_sources,
               'scope': 'fixed-case diagnostic; not a representative benchmark score'})
    started = time.monotonic()
    try:
        result = runner.validate(eval_id='validation') if validate_only else runner.run(eval_id='evaluation')
        write_json(output / 'result.json', result)
        print(json.dumps({'benchmark': benchmark, 'group': group, 'eval_dir': result['eval_dir']}), flush=True)
    except Exception as exc:
        write_json(output / 'failure.json', {'error_type': type(exc).__name__})
        raise
    finally:
        end_sources = fingerprint()
        write_json(output / 'timing.json', {'elapsed_seconds': time.monotonic() - started,
                   'changed_sources': [p for p in start_sources if start_sources[p] != end_sources[p]]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=('aerial', 'travel'), required=True)
    parser.add_argument('--group', choices=('baseline', 'guided'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--model', help='Experiment-only model override; never writes .env')
    parser.add_argument('--run-root', type=Path, help='Isolated diagnostic run directory')
    parser.add_argument('--trace-dir', type=Path, help='Diagnostic-only perception stage trace')
    args = parser.parse_args()
    if args.trace_dir:
        from tests.manual.process_diagnostics import install_grounding_trace
        install_grounding_trace(args.trace_dir)
    run(args.benchmark, args.group, args.output_dir, args.validate_only, model=args.model, run_root=args.run_root)
