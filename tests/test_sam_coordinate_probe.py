"""The diagnostic rewrite must only add the two coordinate device arguments."""
from types import SimpleNamespace
import pytest
from tests.manual.sam_coordinate_probe import device_coordinate_method


def coordinate_fixture(self, q, k, rel_pos):
    a = torch.arange(q)
    b = torch.arange(k)
    return a, b


def changed_fixture(self, q, k, rel_pos):
    return torch.arange(q)


def test_rewrite_changes_only_coordinate_device(monkeypatch):
    monkeypatch.setitem(globals(), 'torch', SimpleNamespace(arange=lambda n, **kw: (n, kw)))
    candidate = device_coordinate_method(coordinate_fixture)
    assert candidate(None, 14, 64, SimpleNamespace(device='cuda:0')) == (
        (14, {'device': 'cuda:0'}), (64, {'device': 'cuda:0'}))
    assert coordinate_fixture(None, 14, 64, None) == ((14, {}), (64, {}))


def test_source_drift_is_rejected():
    with pytest.raises(ValueError, match='exactly two'):
        device_coordinate_method(changed_fixture)
