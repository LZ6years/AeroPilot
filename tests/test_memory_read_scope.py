"""Scope correctness at the memory boundary, including inherited async context."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import pytest

from src.memory.stores import MemoryRecordReader


def setup_reader(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"items":[1]}\n{"items":[2]}\n', encoding="utf-8")
    return MemoryRecordReader(tmp_path), path


def test_nested_scope_reads_once_and_returns_independent_records(tmp_path):
    reader, path = setup_reader(tmp_path)
    original = Path.read_text
    reads = []
    def count(p, *a, **kw):
        reads.append(p)
        return original(p, *a, **kw)
    with patch.object(Path, "read_text", count), reader.read_scope():
        first = reader.read_ref("records.jsonl#1")
        first["items"].append(99)
        with reader.read_scope():
            assert reader.read_ref("records.jsonl#1") == {"items": [1]}
            assert reader.read_ref("records.jsonl#2") == {"items": [2]}
        assert reader.read_ref("records.jsonl#3") is None
    assert reads == [path]
    assert reader._read_scope.get() is None


def test_cached_reference_does_not_repeat_path_resolution(tmp_path):
    reader, _ = setup_reader(tmp_path)
    with reader.read_scope():
        reader.read_ref("records.jsonl#1")
        with patch.object(Path, "resolve", side_effect=AssertionError("repeated filesystem resolution")):
            assert reader.read_ref("records.jsonl#2") == {"items": [2]}


def test_next_build_sees_append_replace_delete_and_recreate(tmp_path):
    reader, path = setup_reader(tmp_path)
    with reader.read_scope():
        assert reader.read_ref("records.jsonl#2") == {"items": [2]}
        path.write_text('{"new":true}\n', encoding="utf-8")
        assert reader.read_ref("records.jsonl#2") == {"items": [2]}
    with reader.read_scope():
        assert reader.read_ref("records.jsonl#1") == {"new": True}
        assert reader.read_ref("records.jsonl#2") is None
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"appended":1}\n')
    with reader.read_scope():
        assert reader.read_ref("records.jsonl#2") == {"appended": 1}
    path.unlink()
    with reader.read_scope():
        assert reader.read_ref("records.jsonl#1") is None
        path.write_text('{"recreated":1}', encoding="utf-8")
        assert reader.read_ref("records.jsonl#1") == {"recreated": 1}


def test_exceptions_release_scope_and_failed_reads_are_retryable(tmp_path):
    reader, path = setup_reader(tmp_path)
    with reader.read_scope():
        with patch.object(Path, "read_text", side_effect=PermissionError):
            with pytest.raises(PermissionError):
                reader.read_ref("records.jsonl#1")
        assert reader.read_ref("records.jsonl#1") == {"items": [1]}
    path.write_text('invalid', encoding="utf-8")
    with pytest.raises(ValueError), reader.read_scope():
        reader.read_ref("records.jsonl#1")
    assert reader._read_scope.get() is None
    path.write_text('{}', encoding="utf-8")
    assert reader.read_ref("records.jsonl#1") == {}


@pytest.mark.parametrize("ref", [None, {}, "bad", "records.jsonl#bad", "records.jsonl#0", "records.jsonl#-1"])
def test_invalid_reference_behavior_unchanged(tmp_path, ref):
    reader, _ = setup_reader(tmp_path)
    with reader.read_scope():
        assert reader.read_ref(ref) is None


def test_threads_have_separate_scopes(tmp_path):
    reader, _ = setup_reader(tmp_path)
    barrier = Barrier(2)
    def work():
        with reader.read_scope():
            reader.read_ref("records.jsonl#1")
            identity = id(reader._read_scope.get()[1])
            barrier.wait(timeout=5)
            assert reader.read_ref("records.jsonl#2") == {"items": [2]}
            return identity
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(work) for _ in range(2)]
        assert len({f.result() for f in futures}) == 2
    assert reader._read_scope.get() is None


def test_async_child_does_not_share_inherited_cache(tmp_path):
    reader, path = setup_reader(tmp_path)
    async def run():
        with reader.read_scope():
            reader.read_ref("records.jsonl#1")
            path.write_text('{"child":true}', encoding="utf-8")
            async def child():
                # Even a read before entering a new scope must not use parent cache.
                assert reader.read_ref("records.jsonl#1") == {"child": True}
                with reader.read_scope():
                    return reader.read_ref("records.jsonl#1")
            assert await asyncio.create_task(child()) == {"child": True}
            assert reader.read_ref("records.jsonl#1") == {"items": [1]}
    asyncio.run(run())
    assert reader._read_scope.get() is None


def test_builder_opens_scope_and_keeps_custom_reader_compatible(tmp_path):
    from types import SimpleNamespace
    from src.perception.context_builder import ObservationContextBuilder
    reader, _ = setup_reader(tmp_path)
    builder = ObservationContextBuilder.__new__(ObservationContextBuilder)
    builder.map_memory = builder.run_memory = None
    builder._stage_context_from_state = lambda state, memory: None
    def compose(sources, stage):
        if sources.flight_memory.memory_reader is reader:
            assert reader._read_scope.get() is not None
        return {"ok": True}
    builder.composer = SimpleNamespace(compose=compose)
    for custom in (reader, object(), None):
        assert builder.build(None, SimpleNamespace(memory_reader=custom)) == {"ok": True}
    assert reader._read_scope.get() is None
    def fail(*args):
        raise RuntimeError("compose failed")
    builder.composer.compose = fail
    with pytest.raises(RuntimeError):
        builder.build(None, SimpleNamespace(memory_reader=reader))
    assert reader._read_scope.get() is None
