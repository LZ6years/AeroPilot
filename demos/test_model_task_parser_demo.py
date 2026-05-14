"""Manual demo for testing the real model-backed M3 task parser.

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

    python test_model_task_parser_demo.py --env-path .env

Run one custom instruction:

    python test_model_task_parser_demo.py --env-path .env --instruction "鍓嶆柟瀵绘壘绾㈣壊杞﹁締锛岄潬杩戜絾淇濇寔瀹夊叏璺濈"

Run several custom instructions in one call:

    python test_model_task_parser_demo.py --env-path .env --instruction "宸﹁浆瀵绘壘寤虹瓚鍏ュ彛锛屽苟鍋滃湪鍏ュ彛鍓? --instruction "瑙傚療鍙充晶鍖哄煙锛屽鏋滅洰鏍囦笉鏄庣‘灏辨偓鍋滅瓑寰呯‘璁?

Use this demo when you want to compare model semantic JSON stability.
For the full M3+M4 dry-run chain, use:

    python demo_memory_system.py
"""

from __future__ import annotations

import argparse
import json

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
    python test_model_task_parser_demo.py --env-path .env

  Test one instruction:
    python test_model_task_parser_demo.py --env-path .env --instruction "鍓嶆柟瀵绘壘绾㈣壊杞﹁締锛岄潬杩戜絾淇濇寔瀹夊叏璺濈"

  Test several instructions in one run:
    python test_model_task_parser_demo.py --env-path .env --instruction "宸﹁浆瀵绘壘寤虹瓚鍏ュ彛锛屽苟鍋滃湪鍏ュ彛鍓? --instruction "瑙傚療鍙充晶鍖哄煙锛屽鏋滅洰鏍囦笉鏄庣‘灏辨偓鍋滅瓑寰呯‘璁?

Instruction examples:
  - 鍓嶉潰鍙宠浆浼氭湁涓€涓粦鑹茬殑璺伅锛屽彸杞潰鍚戣矾鐏悗鍋滀笅
  - 鍓嶆柟瀵绘壘绾㈣壊杞﹁締锛岄潬杩戜絾淇濇寔瀹夊叏璺濈
  - 宸﹁浆瀵绘壘寤虹瓚鍏ュ彛锛屽苟鍋滃湪鍏ュ彛鍓?  - 瑙傚療鍙充晶鍖哄煙锛屽鏋滅洰鏍囦笉鏄庣‘灏辨偓鍋滅瓑寰呯‘璁?  - 娌跨潃閬撹矾鍚戝墠瑙傚療锛屾壘鍒拌摑鑹叉寚绀虹墝鍚庨潰鍚戝畠鎮仠
  - 缁曡繃鍓嶆柟闅滅鐗╋紝瀵绘壘寮€闃斿尯鍩熷苟鎮仠绛夊緟涓嬩竴姝ユ寚浠?  - 鎵惧埌鏈€杩戠殑榛勮壊鏂藉伐鐗岋紝闈㈠悜瀹冧絾涓嶈闈犲緱澶繎
  - Search ahead for a white delivery truck, face it, and hover.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FlyAgent real model task parser demo",
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
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")


def print_instruction_examples() -> None:
    print("Instruction examples:")
    for index, instruction in enumerate(DEFAULT_INSTRUCTIONS, start=1):
        print(f"  {index}. {instruction}")
    print()


if __name__ == "__main__":
    main()
