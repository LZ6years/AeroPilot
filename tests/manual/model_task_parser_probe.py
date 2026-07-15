"""Manual probe for testing the real model-backed M3 task parser.

This file is not a pytest test. It calls the configured model API and prints
the parsed semantic JSON for several multilingual instructions.

Configure .env first:

    FLYAGENT_TASK_PARSER_MODE=model
    FLYAGENT_MODEL_PROVIDER=openai_compatible
    FLYAGENT_MODEL_NAME=<your-model-name>
    FLYAGENT_MODEL_API_KEY=<your-api-key>
    FLYAGENT_MODEL_BASE_URL=<your-compatible-base-url>

Recommended PowerShell setup for readable Chinese output:

    $env:PYTHONIOENCODING='utf-8'

Run the default instruction set:

    python tests/manual/model_task_parser_probe.py --env-path .env

Run one custom instruction:

    python tests/manual/model_task_parser_probe.py --env-path .env --instruction "前方寻找红色车辆，靠近但保持安全距离"

Run several custom instructions in one call:

    python tests/manual/model_task_parser_probe.py --env-path .env --instruction "左转寻找建筑入口，并停在入口前" --instruction "观察右侧区域，如果目标不明确就悬停等待确认"

Use this probe when you want to compare model semantic JSON stability.
For the full M3+M4 dry-run chain, use:

    python demos/memory_system_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.semantic.task_parser import TaskParser
from src.runtime.configs.model_config import load_model_config


DEFAULT_INSTRUCTIONS = [
    "前面右转会有一个黑色的路灯，右转面向路灯后停下",
    "前方寻找红色车辆，靠近但保持安全距离",
    "左转寻找建筑入口，并停在入口前",
    "观察右侧区域，如果目标不明确就悬停等待确认",
    "Turn left, find the building entrance, and hover in front of it.",
    "Search ahead for a white delivery truck, face it, and hover.",
]

HELP_EPILOG = """
Operation examples:

  Test the default model-backed instruction set:
    $env:PYTHONIOENCODING='utf-8'
    python tests/manual/model_task_parser_probe.py --env-path .env

  Test one instruction:
    python tests/manual/model_task_parser_probe.py --env-path .env --instruction "前方寻找红色车辆，靠近但保持安全距离"

  Test several instructions in one run:
    python tests/manual/model_task_parser_probe.py --env-path .env --instruction "左转寻找建筑入口，并停在入口前" --instruction "观察右侧区域，如果目标不明确就悬停等待确认"

Instruction examples:
  - 前面右转会有一个黑色的路灯，右转面向路灯后停下
  - 前方寻找红色车辆，靠近但保持安全距离
  - 左转寻找建筑入口，并停在入口前
  - 观察右侧区域，如果目标不明确就悬停等待确认
  - 沿着道路向前观察，找到蓝色指示牌后面向它悬停
  - 绕过前方障碍物，寻找开阔区域并悬停等待下一步指令
  - 找到最近的黄色施工牌，面向它但不要靠得太近
  - Search ahead for a white delivery truck, face it, and hover.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FlyAgent real model task parser probe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG,
    )
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--instruction", action="append", default=None)
    parser.add_argument("--show-examples", action="store_true", help="Print default instruction examples before running.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_config = load_model_config(args.env_path)
    parser = TaskParser(mode="model", model_config=model_config)
    instructions = args.instruction or DEFAULT_INSTRUCTIONS
    if args.show_examples:
        print_instruction_examples()
    print("Loaded model config:")
    print(json.dumps(model_config.redacted_summary(), indent=2, ensure_ascii=False))
    print("\nAPI key is redacted and will not be printed.\n")
    print("Each result should be one validated TaskParseResult JSON object.")
    print("Phase names are open semantic labels, and every phase should include index/total.\n")
    for index, instruction in enumerate(instructions, start=1):
        print("=" * 80)
        print(f"Case {index}")
        print(f"Instruction: {instruction}")
        try:
            result = parser.parse(instruction)
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
            print("Phase skill trace:")
            print(json.dumps(parser.last_parse_trace, indent=2, ensure_ascii=False))
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")


def print_instruction_examples() -> None:
    print("Instruction examples:")
    for index, instruction in enumerate(DEFAULT_INSTRUCTIONS, start=1):
        print(f"  {index}. {instruction}")
    print()


if __name__ == "__main__":
    main()
