"""Disk visibility and exception boundaries for incremental read results."""
import json
import threading
import types
from concurrent.futures import ThreadPoolExecutor, wait as all_done
from dataclasses import replace
from pathlib import Path

import pytest

from src.agent.transcript import AssistantTurn
from src.memory.context.contracts import (
    ContextFragment,
    InstructionEnvelope,
    StateCheckpoint,
    checkpoint_fingerprint,
)
from src.memory.context.media_refs import image_media_ref
from src.runtime.configs.read_only_execution_config import ReadOnlyExecutionConfig
from src.runtime.control.read_only_executor import ReadOnlyBatchExecutor
from src.runtime.task.session import TaskSession
from src.tools.schemas import ReadExecutionHints
from tests.support.read_only import baseline_method, job, ok
from tests.support.runtime import call, harness


class Fatal(BaseException):
    pass


def records(root, name):
    path = Path(root) / 'memory' / name
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def setup_runtime(path, workers=1):
    runtime, state, scene, _ = harness(path, [])
    runtime.read_only_execution_config = ReadOnlyExecutionConfig(workers)
    runtime.turn_executor.read_only_execution_config = runtime.read_only_execution_config
    runtime.tools.read_execution_hints = lambda *a: ReadExecutionHints()
    calls = [call('a', 'knowledge.load', knowledge_id='a'),
             call('b', 'knowledge.load', knowledge_id='b')]
    runtime.agent_transcript.append_assistant(AssistantTurn('test_turn', 'read', calls))
    return runtime, state, scene, calls


def start_v2_instruction(transcript, instruction, instruction_id):
    transcript.start_instruction(
        instruction,
        instruction_id,
        InstructionEnvelope(
            instruction_id=instruction_id,
            protocol_context="",
            identity={},
            user_preferences={},
            knowledge_catalog=(),
            static_module_guidance=(),
        ),
    )


def run(runtime, state, scene, calls):
    legacy = getattr(runtime, '_execute_native_read_calls', None)
    if callable(legacy):
        return legacy(scene, 'uav', state, 'read', calls)
    session = TaskSession(
        scene=scene,
        agent_id='uav',
        state=state,
        memory=runtime.memory,
        transcript=runtime.agent_transcript,
        checkpoints=runtime.checkpoints,
        knowledge_service=runtime.session.knowledge_service,
    )
    return runtime.turn_executor._execute_native_read_calls(
        session, 'read', calls, 'decision_0001'
    )


def test_serial_result_visible_before_next_call_and_fatal(tmp_path):
    runtime, state, scene, calls = setup_runtime(tmp_path)
    caller = threading.get_ident()
    def dispatch(*args, **kw):
        assert threading.get_ident() == caller
        if args[3]['knowledge_id'] == 'b':
            assert len(records(tmp_path, 'actions.jsonl')) == 1
            assert records(tmp_path, 'agent_transcript.jsonl')[-1]['kind'] == 'tool_result'
            raise Fatal('second failed')
        return ok('saved')
    runtime.tools.call = dispatch
    with pytest.raises(Fatal):
        run(runtime, state, scene, calls)
    assert len(records(tmp_path, 'tool_completions.jsonl')) == 1


def test_reverse_completion_is_saved_while_first_blocked(tmp_path):
    runtime, state, scene, calls = setup_runtime(tmp_path, 4)
    entered, release, saved = threading.Event(), threading.Event(), threading.Event()
    original = runtime.memory_writer.write_tool_completion
    def persist(**kwargs):
        result = original(**kwargs)
        saved.set()
        return result
    runtime.memory_writer.write_tool_completion = persist
    def dispatch(*args, **kw):
        if args[3]['knowledge_id'] == 'a':
            entered.set()
            assert release.wait(5)
        else:
            assert entered.wait(5)
        return ok(args[3]['knowledge_id'])
    runtime.tools.call = dispatch
    with ThreadPoolExecutor(1) as owner:
        future = owner.submit(run, runtime, state, scene, calls)
        try:
            assert saved.wait(5)
            assert records(tmp_path, 'tool_completions.jsonl')[0]['tool_call_id'] == 'b'
            assert not records(tmp_path, 'actions.jsonl')
        finally:
            release.set()
        result = future.result()
    assert [r['tool_call_id'] for r in result] == ['a', 'b']
    assert [r['args']['knowledge_id'] for r in records(tmp_path, 'actions.jsonl')] == ['a', 'b']


def test_all_ready_results_collected_despite_fatal(monkeypatch):
    import src.runtime.control.read_only_executor as module
    monkeypatch.setattr(module, 'wait', lambda futures, **kw: all_done(futures))
    barrier = threading.Barrier(2)
    caller = threading.get_ident()
    def fatal():
        barrier.wait(5)
        raise Fatal('first')
    def success():
        barrier.wait(5)
        return ok()
    delivered = []
    def receive(index, result):
        assert threading.get_ident() == caller
        delivered.append(index)
    with pytest.raises(Fatal, match='first'):
        ReadOnlyBatchExecutor(ReadOnlyExecutionConfig(2)).execute(
            [job('a', fatal), job('b', success), job('c', lambda: pytest.fail('submitted'))],
            on_completed=receive)
    assert delivered == [1]


def test_results_delivered_during_fatal_cleanup():
    started, release, cleanup = threading.Event(), threading.Event(), threading.Event()
    def held():
        started.set()
        assert release.wait(5)
        return ok()
    def fatal():
        assert started.wait(5)
        raise Fatal()
    delivered = []
    executor = ReadOnlyBatchExecutor(ReadOnlyExecutionConfig(2),
        lambda name, data: cleanup.set() if name == 'read_call_cancelled' else None)
    with ThreadPoolExecutor(1) as owner:
        future = owner.submit(executor.execute,
            [job('a', held), job('b', fatal), job('c', lambda: pytest.fail('submitted'))],
            on_completed=lambda i, r: delivered.append(i))
        try:
            assert cleanup.wait(5)
        finally:
            release.set()
        with pytest.raises(Fatal):
            future.result()
    assert delivered == [0]


def test_callback_failure_stops_submission_and_is_not_retried():
    error = OSError('disk unavailable')
    notices = []
    def receive(i, r):
        notices.append(i)
        raise error
    with pytest.raises(OSError) as caught:
        ReadOnlyBatchExecutor().execute([job('a', ok), job('b', lambda: pytest.fail('ran'))],
                                       on_completed=receive)
    assert caught.value is error and notices == [0]


@pytest.mark.parametrize('method', ['write_tool_completion', 'write_action'])
def test_runtime_write_failure_stops_before_next_tool(tmp_path, method):
    runtime, state, scene, calls = setup_runtime(tmp_path)
    started = []
    runtime.tools.call = lambda *a, **k: (started.append(a[3]['knowledge_id']) or ok())
    def fail(*a, **k):
        raise OSError('disk failed')
    setattr(runtime.memory_writer, method, fail)
    with pytest.raises(OSError, match='disk failed'):
        run(runtime, state, scene, calls)
    assert started == ['a']
    assert not [e for e in runtime.agent_transcript.items if e.kind == 'tool_result']


def test_receipts_audit_only_portable_media_and_old_records(tmp_path):
    from PIL import Image
    from evaluation.benchmarks.core.audit import BenchmarkAuditReader, EpisodeAuditCursor
    from evaluation.benchmarks.core.review import BenchmarkCaseReview, BenchmarkReviewRenderer
    from evaluation.benchmarks.uav_flow.postprocess import _audit_from_payload
    runtime, state, scene, calls = setup_runtime(tmp_path)
    image = tmp_path / 'evidence.png'
    Image.new('RGB', (4, 4), 'red').save(image)
    runtime.memory_writer.write_tool_completion(
        instruction_id=runtime.agent_transcript.instruction_id, decision_id='decision_0001',
        call_index=1, tool_call_id='b', tool_name='test.read', args={},
        result={'ok': True, 'data': {'_flyagent_model_inputs': [
            {'type': 'image', 'path': str(image), 'data': 'data:image/png;base64,private'}]}})
    assert 'base64,private' not in (tmp_path / 'memory/tool_completions.jsonl').read_text()
    audit = BenchmarkAuditReader().read(EpisodeAuditCursor.freeze(tmp_path))
    assert len(audit.uncommitted_tool_completions()) == 1
    assert _audit_from_payload(audit.to_dict()).tool_completions == audit.tool_completions
    case = BenchmarkCaseReview('01', 'Fake', 'test', 'No model or simulator', {}, {}, audit, ())
    # Keep exported absolute paths short even under pytest's temporary root.
    for detail in (False, True):
        output = BenchmarkReviewRenderer(detailed_audit=detail).write_case(tmp_path / 'r', case)
        page = Path(output['case_html']).read_text(encoding='utf-8')
        assert 'Completed result / not committed' in page
        assert 'completions/' in page and '无最终评分' in page
    old = replace(audit, tool_completions=())
    assert not old.uncommitted_tool_completions()


def test_receipts_do_not_change_normal_projection(tmp_path):
    from src.agent.context.projection import HistoryProjector
    runtime, state, scene, calls = setup_runtime(tmp_path)
    runtime.tools.call = lambda *a, **k: ok()
    run(runtime, state, scene, calls)
    projector = HistoryProjector()
    before = projector.project(runtime.agent_transcript, runtime.tools.registry)
    (tmp_path / 'memory/tool_completions.jsonl').unlink()
    after = projector.project(runtime.agent_transcript, runtime.tools.registry)
    assert before == after


def test_save_failure_ends_public_loop_without_next_model_request(tmp_path):
    from tests.support.runtime import done, public_runtime
    runtime, backend = public_runtime(tmp_path, [
        [call('read', 'knowledge.load', name='guide')], [done(observation='obs_0001')]])
    failure = OSError('receipt disk error')
    def fail(**kwargs):
        raise failure
    runtime._memory_writer.write_tool_completion = fail
    with pytest.raises(OSError) as caught:
        runtime.run_task('map', 'Read guidance')
    assert caught.value is failure
    assert runtime.planner.decision_indices == [1]
    assert backend.stops == ['scene']


def test_receipt_cursor_and_instruction_scope(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from itertools import count
    ticks = count()
    monkeypatch.setattr('src.memory.stores.event_store._timestamp',
        lambda: (datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=next(ticks))).isoformat())
    from evaluation.benchmarks.core.audit import BenchmarkAuditReader, EpisodeAuditCursor
    runtime, state, scene, calls = setup_runtime(tmp_path)
    transcript = runtime.agent_transcript
    start_v2_instruction(transcript, 'first', 'inst_first')
    def save(identity):
        runtime.memory_writer.write_tool_completion(
            instruction_id=identity, decision_id='decision_0001', call_index=0,
            tool_call_id='reused_between_tasks', tool_name='test.read', args={}, result={'ok': True})
    save('inst_first')
    start_v2_instruction(transcript, 'second', 'inst_second')
    save('inst_second')
    cursor = EpisodeAuditCursor.freeze(tmp_path)
    cursor = replace(cursor, started_at=records(tmp_path, 'agent_transcript.jsonl')[-1]['timestamp'])
    save('inst_second')  # Written after the frozen boundary; must not leak in.
    audit = BenchmarkAuditReader().read(cursor)
    assert len(audit.tool_completions) == 1
    assert audit.tool_completions[0]['instruction_id'] == 'inst_second'


def test_successful_receipts_are_not_shown_twice(tmp_path):
    from evaluation.benchmarks.core.audit import BenchmarkAuditReader, EpisodeAuditCursor
    runtime, state, scene, calls = setup_runtime(tmp_path)
    runtime.tools.call = lambda *a, **k: ok()
    run(runtime, state, scene, calls)
    audit = BenchmarkAuditReader().read(EpisodeAuditCursor.freeze(tmp_path))
    assert len(audit.tool_completions) == 2
    assert not audit.uncommitted_tool_completions()


@pytest.mark.parametrize('workers', [1, 4])
def test_old_and_new_runtime_produce_identical_requests_with_images(tmp_path, workers):
    import base64
    import io
    import types
    from PIL import Image
    from tests.support.planner import StubPlanner, response_with_calls
    buffer = io.BytesIO()
    Image.new('RGB', (8, 8), 'blue').save(buffer, format='PNG')
    data = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    old, _ = baseline_method()
    requests = []
    for mode in ('baseline', 'candidate'):
        runtime, state, scene, calls = setup_runtime(tmp_path / mode, workers)
        image_path = tmp_path / mode / 'image.png'
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(buffer.getvalue())
        start_v2_instruction(runtime.agent_transcript, 'Inspect', 'inst_image')
        media_ref = image_media_ref(
            image_path,
            observation_id='obs_image',
            frame_id='frame_image',
            role='current_observation',
            usage='Use as the current observation.',
            scene_width=8,
            scene_height=8,
        )
        fragment = ContextFragment('image', {}, (media_ref.to_model_input(),))
        runtime.agent_transcript.append_state(StateCheckpoint(
            checkpoint_id='checkpoint_0001',
            reason='instruction_start',
            observation_id='obs_image',
            task_plan=None,
            loaded_knowledge=(),
            fragments=(fragment,),
            fingerprint=checkpoint_fingerprint('obs_image', None, (), (fragment,)),
        ))
        runtime.agent_transcript.append_assistant(AssistantTurn('read', 'read', calls))
        if mode == 'baseline':
            runtime._execute_native_read_calls = types.MethodType(old, runtime)
        runtime.tools.call = lambda *a, **k: ok()
        run(runtime, state, scene, calls)
        planner = StubPlanner(response_with_calls(('next', 'uav_turn_relative', {'yaw_deg': 5})))
        planner.plan_turn(runtime.agent_transcript, runtime.tools.registry)
        requests.append(planner.payloads[0])
    assert requests[0] == requests[1]
    assert data in json.dumps(requests[0])
