import pytest
from tests.manual.gpu_display_context import classify_gpu_sample


@pytest.mark.parametrize("kind,memory,expected", [("G", "13 MiB", [2]), ("C+G", "13 MiB", [1, 2]), ("G", "100 MiB", [1, 2]), ("G", "N/A", [1, 2])])
def test_only_small_display_context_is_exempt(kind, memory, expected):
    sample = {"gpus": [{"index": 1, "processes": [{"pgid": 123, "type": kind, "used_memory": memory}]}],
              "owned": [{"pid": 123, "expected_gpu": 2, "observed_gpus": [1, 2]}]}
    updated = classify_gpu_sample(sample)
    assert updated["owned"][0]["observed_gpus"] == expected
    assert sample["owned"][0]["observed_gpus"] == [1, 2]
