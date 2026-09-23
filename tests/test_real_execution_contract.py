"""Production execution cannot silently fall back to synthetic flight."""

from pathlib import Path
import pytest

from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
from src.sim_server.backends.airsim_backend import AirSimBackend
from src.sim_server.backends.unrealcv_backend import UnrealCvBackend


@pytest.mark.parametrize("config,required", [
    (UavFlowEvalConfig, {}),
    (AerialVlnEvalConfig, {"dataset_root": "fixture"}),
    (TravelUavEvalConfig, {"index_path": "fixture"}),
])
def test_explicit_execution_required_and_dry_rejected(config, required):
    with pytest.raises(ValueError, match="execution_mode"):
        config.from_dict(required)
    with pytest.raises(ValueError, match="dry execution was removed"):
        config.from_dict({**required, "execution_mode": "dry"})
    assert config.from_dict({**required, "execution_mode": "managed"}).execution_mode == "managed"


@pytest.mark.parametrize("backend", [AirSimBackend, UnrealCvBackend])
def test_default_backend_has_no_synthetic_adapter(backend):
    assert backend().uav is None  # Real adapter is bound to the scene at start.


def test_production_has_no_fake_execution_switches():
    root = Path(__file__).resolve().parents[1]
    for folder in ("src", "evaluation/benchmarks"):
        for path in (root / folder).rglob("*.py"):
            if "outputs" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for obsolete in ("NullUavPrimitiveAdapter", "use_real_uav", "use_real_client", "def _dry_"):
                assert obsolete not in text, (path, obsolete)
