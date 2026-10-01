"""Device serialization and lazy model construction without requiring a GPU."""

from concurrent.futures import ThreadPoolExecutor
import threading
import sys
from types import SimpleNamespace
import pytest

from src.perception.inference_resources import inference_slot, serialized_inference, inference_observer


def test_same_device_cross_backend_exclusion_and_exception_release():
    active, peak, counter_lock = [0], [0], threading.Lock()
    start = threading.Barrier(4)
    def work(index):
        start.wait(timeout=3)
        try:
            with inference_slot("test:0", "backend" + str(index)):
                with counter_lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                # A nested lazy load is reentrant, not a second active slot.
                with inference_slot("test:0", "load"):
                    pass
                with counter_lock:
                    active[0] -= 1
                if index == 1:
                    raise ValueError("inference failed")
        except ValueError:
            return "error"
        return "ok"
    with ThreadPoolExecutor(4) as pool:
        values = list(pool.map(work, range(4)))
    assert peak[0] == 1 and values.count("error") == 1
    with inference_slot("test:0", "after_error"):
        pass


def test_distinct_devices_overlap_and_observer_failure_releases():
    barrier = threading.Barrier(2)
    def work(device):
        with inference_slot(device, "forward"):
            barrier.wait(timeout=3)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(work, ["test:1", "test:2"]))
    def fail(event, payload):
        if event == "device_execution_finished":
            raise RuntimeError("observer failed")
    with pytest.raises(RuntimeError), inference_observer(fail), inference_slot("test:3", "forward"):
        pass
    with ThreadPoolExecutor(1) as pool:
        pool.submit(work_without_barrier, "test:3").result(timeout=3)


def work_without_barrier(device):
    with inference_slot(device, "check"):
        return True


@pytest.mark.parametrize("backend_name", ["grounding", "sam"])
def test_concurrent_direct_load_constructs_model_once(monkeypatch, backend_name):
    from plugins.capabilities.visual_grounding.measure_visible_depth.scripts.backends.grounding_dino import GroundingDinoBackend
    from plugins.capabilities.visual_grounding.measure_visible_depth.scripts.backends.sam import SamBackend
    counts = {"processor": 0, "model": 0}
    class Processor:
        @classmethod
        def from_pretrained(cls, *a):
            counts["processor"] += 1
            return cls()
    class Model:
        @classmethod
        def from_pretrained(cls, *a):
            counts["model"] += 1
            return cls()
        def eval(self):
            pass
        def to(self, device):
            assert device == "cpu"
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=Processor, AutoModelForZeroShotObjectDetection=Model, SamProcessor=Processor, SamModel=Model))
    backend = (GroundingDinoBackend("fake", "cpu", box_threshold=.4, text_threshold=.2)
               if backend_name == "grounding" else SamBackend("fake", "cpu"))
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: backend._load(), range(12)))
    assert counts == {"processor": 1, "model": 1}
    assert all(result[1] is results[0][1] for result in results)


def test_service_cache_construction_is_atomic(monkeypatch):
    from plugins.capabilities.visual_grounding.measure_visible_depth.scripts import service
    instances = []
    class Grounder:
        def __init__(self, *a, **k):
            instances.append(self)
    monkeypatch.setattr(service, "_SERVICE_CACHE", {})
    monkeypatch.setattr(service, "GroundingDinoBackend", Grounder)
    settings = {"grounding_backend": "grounding_dino", "grounding_model": "fake",
                "segmentation_backend": "none", "segmentation_model": "fake", "device": "cpu"}
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: service.service_from_settings(settings), range(20)))
    assert len(instances) == 1 and all(value is results[0] for value in results)


def test_empty_sam_does_not_initialize_device(monkeypatch):
    from plugins.capabilities.visual_grounding.measure_visible_depth.scripts.backends.sam import SamBackend
    backend = SamBackend("unused", "auto")
    monkeypatch.setattr(backend, "_torch", lambda: (_ for _ in ()).throw(AssertionError("No work")))
    assert backend.segment(None, []) == []


@pytest.mark.parametrize("backend_name", ["grounding", "sam"])
def test_failed_device_load_is_not_published(monkeypatch, backend_name):
    from plugins.capabilities.visual_grounding.measure_visible_depth.scripts.backends.grounding_dino import GroundingDinoBackend
    from plugins.capabilities.visual_grounding.measure_visible_depth.scripts.backends.sam import SamBackend
    class Processor:
        @staticmethod
        def from_pretrained(*args):
            return object()
    class Model:
        @classmethod
        def from_pretrained(cls, *args):
            return cls()
        def eval(self):
            pass
        def to(self, device):
            raise RuntimeError("device failed")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=Processor, AutoModelForZeroShotObjectDetection=Model, SamProcessor=Processor, SamModel=Model))
    backend = (GroundingDinoBackend("fake", "cpu", box_threshold=.4, text_threshold=.2)
               if backend_name == "grounding" else SamBackend("fake", "cpu"))
    with pytest.raises(RuntimeError, match="device failed"):
        backend._load()
    assert backend._model is None and backend._processor is None
    with ThreadPoolExecutor(1) as pool:
        assert pool.submit(work_without_barrier, "cpu").result(timeout=3)
