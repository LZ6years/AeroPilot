"""Event diagnostics restore global hooks and keep nested accounting explicit."""
import json
import sys
from types import SimpleNamespace
import pytest

from tests.manual.cuda_event_probe import capture_events, summarize_rows
from tests.manual.task_profile import TaskProfile


@pytest.mark.parametrize('sam_modules', [False, True])
def test_event_capture_scope_and_exception_cleanup(tmp_path, monkeypatch, sam_modules):
    class Tensor:
        def __init__(self, device='cpu'):
            self.device = device
        def to(self, device):
            return Tensor(device)
        def cpu(self):
            return Tensor('cpu')
        def numel(self):
            return 4
        def element_size(self):
            return 4
    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self, stream):
            pass
        def synchronize(self):
            pass
        def elapsed_time(self, end):
            return 10
    sync = lambda *a, **k: None
    torch = SimpleNamespace(Tensor=Tensor, cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1, Event=Event,
        synchronize=sync, current_stream=lambda: 'stream'))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    original = Tensor.to
    classes = {name: type(name, (), {'forward': lambda self: 42}) for name in
               ('SamVisionEncoder', 'SamPromptEncoder', 'SamMaskDecoder')}
    originals = {name: cls.forward for name, cls in classes.items()}
    monkeypatch.setitem(sys.modules, 'transformers.models.sam', SimpleNamespace(modeling_sam=SimpleNamespace(**classes)))
    profile = TaskProfile(tmp_path / 'timing.jsonl')
    with pytest.raises(RuntimeError, match='test failure'):
        with capture_events(tmp_path, sam_modules=sam_modules):
            with profile.span('vision.sam.pipeline'):
                t = Tensor().to('cuda:0')
                with profile.span('vision.sam.forward'):
                    if sam_modules:
                        for cls in classes.values():
                            assert cls().forward() == 42
                    t.to('cuda:0')  # Internal transfers are deliberately uninstrumented.
                    torch.cuda.synchronize()
                t.cpu()
            raise RuntimeError('test failure')
    assert Tensor.to is original and torch.cuda.synchronize is sync
    record = json.loads((tmp_path / 'cuda_events.json').read_text())
    assert len(record['rows']) == (7 if sam_modules else 4)
    assert all(cls.forward is originals[name] for name, cls in classes.items())
    assert record['groups']['vision.sam.forward']['stream_sec'] == .01
    assert record['groups']['vision.sam.pipeline.transfer.cpu_to_cuda:0']['bytes'] == 16


def test_grouping_does_not_flatten_nested_stage_names():
    groups = summarize_rows([
        {'stage': 'forward', 'host_sec': 2, 'stream_sec': 1.9},
        {'stage': 'forward.explicit_sync', 'host_sec': 1.8},
    ])
    assert groups['forward']['host_sec'] == 2
    assert groups['forward.explicit_sync']['host_sec'] == 1.8
