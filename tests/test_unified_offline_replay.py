"""End-to-end scripted audit/report gate; no external service is constructed."""

from pathlib import Path
import subprocess
import sys

from tests.support.review import run_replay


def test_transcript_projection_cold_process_has_no_eager_planner_cycle():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tests/manual/native_tool_transcript_probe.py"],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "probe_ok" in result.stdout


def test_scripted_replay_strict_audit_tracking_and_truthful_html(tmp_path):
    rgb = Path(__file__).parent / "fixtures/visual_description_images/frame_000001.png"
    output = tmp_path / "replay"
    report = run_replay(output, rgb)
    assert report["violations"] == []
    assert report["trace_integrity"] == "ok"
    assert report["real_provider_called"] is False
    page = Path(report["case_html"]).read_text(encoding="utf-8")
    assert "Scripted offline replay" in page
    assert "Sensor RGB — not a captured Provider input" in page
    assert "Knowledge receipt" in page
    assert "Applied plan update" in page
    assert "review_assets" in page
    assert (output / "review.html").is_file()
    assert "data:image" not in page
