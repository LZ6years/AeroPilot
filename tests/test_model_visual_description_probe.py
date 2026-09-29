"""Unit checks for the manual real-engine visual description probe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from src.agent.config import ModelConfig


def load_probe_module():
    path = Path("tests") / "manual" / "model_visual_description_probe.py"
    spec = importlib.util.spec_from_file_location("model_visual_description_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_caption_client():
    def _client(config, payload):
        assert payload["model"] == config.model_name
        user_content = payload["messages"][1]["content"]
        assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        return {
            "parsed": {
                "visual_description": {
                    "phase": {"name": "search_visual_target", "meaning": "寻找建筑入口"},
                    "phase_relevance": "可见建筑底层入口候选。",
                    "scene_summary": "建筑底层入口候选位于画面中景。",
                    "spatial_overview": "midground-center: 建筑底层开口。",
                    "relevant_objects": [
                        {
                            "name": "建筑底层开口",
                            "position": "midground-center",
                            "role": "target_candidate",
                            "phase_relevance": "与入口目标同类。",
                            "description": "矩形开口位于建筑底层。",
                            "confidence": "high",
                        }
                    ],
                    "free_space": ["前景铺装区域开阔"],
                    "landmarks": ["背景高层建筑"],
                    "phase_relevant_cues": ["建筑底层存在矩形开口"],
                    "uncertainty": ["未见明确入口标识"],
                    "confidence": "medium",
                }
            },
            "duration_sec": 0.12,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    return _client


def test_visual_description_probe_defaults_to_fixture_images():
    probe = load_probe_module()

    images = probe.fixture_images(limit=5)

    assert len(images) == 5
    assert images[0].parts[-3:] == ("fixtures", "visual_description_images", "frame_000001.png")


def test_visual_description_probe_normalizes_future_custom_module_schema():
    probe = load_probe_module()
    image_path = Path("tests/fixtures/visual_description_images/frame_000001.png")

    normalized = probe.normalize_visual_description(
        {
            "visual_description": {
                "phase": {"name": "search_visual_target", "meaning": "寻找入口或地标"},
                "phase_relevance": "画面中有可作为定位参照的树和车辆。",
                "scene_summary": "城市街区中景有树和车辆。",
                "spatial_overview": {"foreground": "开阔地面", "midground": "车辆和树"},
                "relevant_objects": [
                    {
                        "name": "黄色校车",
                        "position": "midground_left",
                        "role": "vehicle",
                        "phase_relevance": "可作为定位参照",
                        "description": "位于树左侧",
                        "confidence": "HIGH",
                    },
                    "灰色铺装地面",
                ],
                "free_space": ["前景中央铺装地面开阔", "中景车辆间通行条件良好"],
                "landmarks": ["中景绿树"],
                "navigation_relevant_cues": [
                    {"cue_type": "landmark", "description": "树可作为视觉参照"},
                    "车辆靠近建筑底层边缘",
                ],
                "uncertainty": "没有看到明确入口标识",
                "confidence": "HIGH",
            }
        },
        image_path,
    )

    description = normalized["visual_description"]
    assert description["schema_version"] == "phase_visual_summary_v1"
    assert description["strategy"] == "qwen_vl_phase_caption_v3"
    assert description["source_image"] == str(image_path)
    assert description["phase"]["name"] == "search_visual_target"
    assert description["spatial_overview"] == "foreground: 开阔地面；midground: 车辆和树"
    assert description["relevant_objects"][0]["position"] == "midground-left"
    assert description["relevant_objects"][0]["role"] == "vehicle"
    assert description["relevant_objects"][0]["phase_relevance"] == "可作为定位参照"
    assert description["relevant_objects"][1]["position"] == "unknown"
    assert description["free_space"] == ["前景中央铺装地面开阔", "中景车辆间可见空隙明显"]
    assert description["landmarks"] == ["中景绿树"]
    assert description["phase_relevant_cues"] == ["landmark: 树可作为视觉参照", "车辆邻近建筑底层边缘"]
    assert description["uncertainty"] == ["没有看到明确入口标识"]
    assert description["confidence"] == "high"


def test_visual_description_probe_builds_context_with_encoded_model_input(tmp_path):
    probe = load_probe_module()
    image_path = Path("tests/fixtures/visual_description_images/frame_000001.png")

    context = probe.build_description_context(
        image_path,
        tmp_path / "case",
        1,
        model_config=ModelConfig(model_name="qwen3-vl-flash", api_key="secret-key", base_url="https://example.test/v1"),
        caption_client=fake_caption_client(),
    )

    fragment = context["fragments"][0]
    assert fragment["model_content"]["visual_processing"]["status"] == "captioned"
    assert fragment["model_content"]["visual_description"]["phase"]["name"] == "search_visual_target"
    assert fragment["model_inputs"][0]["type"] == "text"
    assert fragment["model_inputs"][0]["role"] == "visual_caption"
    assert "data:image/png;base64," not in fragment["model_inputs"][0]["text"]


def test_visual_description_probe_outputs_redacted_logs(tmp_path):
    probe = load_probe_module()
    config = ModelConfig(
        model_name="qwen3-vl-plus",
        api_key="secret-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    results = [
        {
            "image_name": "frame_000001.png",
            "source_image": "tests/fixtures/visual_description_images/frame_000001.png",
            "model_input": {"type": "image", "encoding": "data_url", "mime_type": "image/png", "data_chars": 123},
            "visual_description": {
                "schema_version": "phase_visual_summary_v1",
                "strategy": "qwen_vl_phase_caption_v3",
                "source_image": "tests/fixtures/visual_description_images/frame_000001.png",
                "phase": {"name": "search_visual_target", "meaning": "寻找入口或地标"},
                "phase_relevance": "可见树和建筑边界，可作为定位参照。",
                "scene_summary": "当前阶段相关地标可见。",
                "spatial_overview": "前景是铺装地面，中景有车辆，远景是高楼。",
                "relevant_objects": [
                    {
                        "name": "建筑边界",
                        "position": "background-center",
                        "role": "boundary",
                        "phase_relevance": "可辅助定位",
                        "description": "两侧建筑形成视觉边界",
                        "confidence": "high",
                    }
                ],
                "free_space": ["前景铺装地面开阔"],
                "landmarks": ["中景树"],
                "phase_relevant_cues": ["开阔地面是视觉线索"],
                "uncertainty": ["没有深度信息"],
                "confidence": "medium",
            },
            "call_metrics": {
                "duration_sec": 1.23,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "raw": {}},
            },
        }
    ]
    assessment = probe.assess_descriptions(results)
    redacted_payloads = [
        {
            "source_image": "tests/fixtures/visual_description_images/frame_000001.png",
            "api_key_present": True,
            "payload": {"has_image_part": True},
        }
    ]

    paths = probe.write_outputs(
        tmp_path,
        config,
        results,
        assessment,
        console_lines=["ok"],
        redacted_payloads=redacted_payloads,
        write_redacted_payloads=True,
    )

    text = paths["json"].read_text(encoding="utf-8")
    redacted_text = (paths["redacted_payloads"] / "frame_000001.json").read_text(encoding="utf-8")
    assert "phase_visual_summary_v1" in text
    assert "call_metrics" in text
    assert "token_usage" in text
    assert "secret-key" not in json.dumps(json.loads(text))
    assert "secret-key" not in redacted_text
    assert "data:image/png;base64," not in redacted_text


def test_visual_description_assessment_flags_unstable_schema():
    probe = load_probe_module()
    results = [
        {
            "visual_description": {
                "relevant_objects": [
                    {
                        "name": "bad object",
                        "position": "leftish",
                        "role": "navigation_command",
                        "phase_relevance": "reference",
                        "description": "visible",
                        "confidence": "high",
                    }
                ],
                "free_space": ["foreground area", "适合飞行机动"],
                "landmarks": ["tree"],
                "phase_relevant_cues": ["recommended_next_observation: move_forward"],
                "uncertainty": ["unclear entrance"],
            },
            "call_metrics": {"duration_sec": 0.5, "usage": {"total_tokens": 12}},
        }
    ]

    assessment = probe.assess_descriptions(results)

    assert assessment["format_consistency"]["all_relevant_objects_valid"] is False
    assert assessment["format_consistency"]["all_list_fields_are_strings"] is True
    assert assessment["format_consistency"]["no_action_suggestion_terms"] is False
    assert assessment["format_consistency"]["no_soft_action_phrases"] is False
    assert assessment["soft_action_phrase_count"] >= 2
    assert assessment["format_consistency"]["all_layout_text_is_chinese"] is True
    assert assessment["unknown_position_count"] == 0


def test_visual_description_assessment_flags_english_layout_text():
    probe = load_probe_module()
    results = [
        {
            "visual_description": {
                "scene_summary": "Urban street scene with buildings and vehicles.",
                "spatial_overview": "Foreground paved area.",
                "relevant_objects": [],
                "free_space": [],
                "landmarks": [],
                "phase_relevant_cues": [],
                "uncertainty": [],
            },
            "call_metrics": {"duration_sec": 0.5, "usage": {"total_tokens": 12}},
        }
    ]

    assessment = probe.assess_descriptions(results)

    assert assessment["english_layout_text_count"] == 2
    assert assessment["format_consistency"]["all_layout_text_is_chinese"] is False


def test_visual_description_assessment_allows_position_labels_in_layout_text():
    probe = load_probe_module()
    results = [
        {
            "visual_description": {
                "scene_summary": "城市街景，入口候选不清晰。",
                "spatial_overview": "foreground-left: 铺装地面；midground-center: 车辆；background-right: 建筑立面",
                "relevant_objects": [],
                "free_space": [],
                "landmarks": [],
                "phase_relevant_cues": [],
                "uncertainty": [],
            },
            "call_metrics": {"duration_sec": 0.5, "usage": {"total_tokens": 12}},
        }
    ]

    assessment = probe.assess_descriptions(results)

    assert assessment["english_layout_text_count"] == 0
    assert assessment["format_consistency"]["all_layout_text_is_chinese"] is True


def test_visual_description_assessment_flags_target_candidate_misuse_for_entrance_phase():
    probe = load_probe_module()
    results = [
        {
            "visual_description": {
                "phase": {"name": "search_visual_target", "meaning": "寻找建筑入口"},
                "relevant_objects": [
                    {
                        "name": "黄色校车",
                        "position": "midground-left",
                        "role": "target_candidate",
                        "phase_relevance": "颜色明显，可作为参照",
                        "description": "位于画面左侧的车辆",
                        "confidence": "high",
                    },
                    {
                        "name": "建筑底层入口阴影区",
                        "position": "background-left",
                        "role": "target_candidate",
                        "phase_relevance": "与入口目标同类",
                        "description": "底层门廊轮廓不清晰",
                        "confidence": "medium",
                    },
                ],
                "free_space": [],
                "landmarks": [],
                "phase_relevant_cues": [],
                "uncertainty": [],
            },
            "call_metrics": {"duration_sec": 0.5, "usage": {"total_tokens": 12}},
        }
    ]

    assessment = probe.assess_descriptions(results)

    assert assessment["target_candidate_misuse_count"] == 1
    assert assessment["format_consistency"]["no_target_candidate_misuse"] is False
    assert assessment["unknown_position_count"] == 0


def test_visual_description_assessment_reports_unknown_position_rate():
    probe = load_probe_module()
    results = [
        {
            "visual_description": {
                "relevant_objects": [
                    {
                        "name": "entrance",
                        "position": "unknown",
                        "role": "target_candidate",
                        "phase_relevance": "candidate",
                        "description": "unclear",
                        "confidence": "medium",
                    },
                    {
                        "name": "tree",
                        "position": "midground-left",
                        "role": "landmark",
                        "phase_relevance": "reference",
                        "description": "visible",
                        "confidence": "high",
                    },
                ],
                "free_space": [],
                "landmarks": [],
                "phase_relevant_cues": [],
                "uncertainty": [],
            },
            "call_metrics": {"duration_sec": 0.5, "usage": {"total_tokens": 12}},
        }
    ]

    assessment = probe.assess_descriptions(results)

    assert assessment["unknown_position_count"] == 1
    assert assessment["unknown_position_rate"] == 0.5


def test_visual_description_markdown_uses_absolute_renderable_image_links(tmp_path):
    probe = load_probe_module()
    config = ModelConfig(model_name="qwen3-vl-flash", api_key="secret-key", base_url="https://example.test/v1")
    image_path = Path("tests/fixtures/visual_description_images/frame_000001.png")
    results = [
        {
            "image_name": image_path.name,
            "source_image": str(image_path),
            "visual_description": {
                "phase_relevance": "high",
                "scene_summary": "phase-relevant visual context",
                "spatial_overview": "foreground-center free space",
                "relevant_objects": [],
                "free_space": [],
                "landmarks": [],
                "phase_relevant_cues": [],
                "uncertainty": [],
                "confidence": "medium",
            },
            "call_metrics": {"duration_sec": 0.5, "usage": {"total_tokens": 12}},
        }
    ]
    assessment = probe.assess_descriptions(results)

    paths = probe.write_outputs(
        tmp_path,
        config,
        results,
        assessment,
        console_lines=[],
        redacted_payloads=[],
        write_redacted_payloads=False,
    )

    markdown = paths["markdown"].read_text(encoding="utf-8")
    expected = image_path.resolve().as_posix()
    assert f"]({expected})" in markdown


def test_visual_description_probe_usage_summary_supports_qwen_usage_shapes():
    probe = load_probe_module()

    usage = probe._usage_summary({"input_tokens": "11", "output_tokens": 7})

    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == 18


def test_visual_description_probe_defaults_to_qwen_vl_flash(tmp_path, monkeypatch):
    probe = load_probe_module()
    for key in [
        "FLYAGENT_MODEL_API_KEY",
        "FLYAGENT_MULTIMODAL_MODEL_NAME",
        "FLYAGENT_VISUAL_DESCRIPTION_MODEL_NAME",
    ]:
        monkeypatch.delenv(key, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("FLYAGENT_MODEL_API_KEY=secret-key\n", encoding="utf-8")

    config = probe.build_visual_description_config(env_path)

    assert config.model_name == "qwen3-vl-flash"
    assert config.api_key == "secret-key"


def test_visual_description_prompt_is_phase_aware_and_navigation_bounded():
    probe = load_probe_module()
    config = ModelConfig(model_name="qwen3-vl-flash", api_key="secret-key", base_url="https://example.test/v1")
    payload = probe.build_description_payload(
        config,
        {
            "active_instruction": "寻找建筑入口",
            "current_phase": {"name": "search_target", "goal": "寻找入口候选"},
            "fragments": [],
        },
        Path("tests/fixtures/visual_description_images/frame_000001.png"),
    )

    prompt = payload["messages"][0]["content"]
    assert "phase_visual_summary_v1" in prompt
    assert "qwen_vl_phase_caption_v3" in prompt
    assert "relevant_objects" in prompt
    assert "phase_relevant_cues" in prompt
    assert "position" in prompt
    assert "foreground-left" in prompt and "background-right" in prompt
    assert "不要默认 unknown" in prompt
    assert "phase_relevance" in prompt
    assert "所有自然语言必须使用中文" in prompt
    assert "禁止输出英文描述" in prompt
    assert "target_candidate 只给入口" in prompt
    assert "车辆用 vehicle" in prompt
    assert "天气" in prompt and "禁止输出" in prompt
    assert "悬停" in prompt and "绕行" in prompt
    assert "recommended_next_observation" in prompt
    assert "occlusions_or_risks" not in prompt
