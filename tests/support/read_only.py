"""Read-only execution job/result fixtures."""

import ast
import copy
import hashlib
import subprocess

from src.env_client import ActionResult
from src.agent.transcript import AgentToolCall, ToolResult
from src.plugins.contracts.skill import (
    SKILL_ARTIFACT_REFS_KEY,
    SKILL_MODEL_INPUTS_KEY,
)
from src.runtime.task.turn_executor import (
    _pop_dict_list,
    _pop_string_list,
    _result_artifact_refs,
)
from src.runtime.control.read_only_executor import ReadJob, ReadOnlyBatchExecutor
from src.tools.schemas import ReadExecutionHints


def job(name, invoke, hints=None):
    return ReadJob(name, "test.read", hints or ReadExecutionHints(), invoke)


def ok(value="ok"):
    return ActionResult(True, "test.read", value)


def baseline_method():
    """Load the frozen pre-refactor read executor for equivalence tests."""

    import src.runtime.runners.task_runtime as module

    source = subprocess.check_output(
        ["git", "show", "fbbec23:src/runtime/runners/task_runtime.py"],
        text=True,
        encoding="utf-8",
    )
    tree = ast.parse(source.lstrip("\ufeff"))
    runtime_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TaskRuntime"
    )
    method = next(
        node
        for node in runtime_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_execute_native_read_calls"
    )
    namespace = dict(vars(module))
    namespace.update(
        copy=copy,
        AgentToolCall=AgentToolCall,
        ToolResult=ToolResult,
        ReadJob=ReadJob,
        ReadOnlyBatchExecutor=ReadOnlyBatchExecutor,
        SKILL_ARTIFACT_REFS_KEY=SKILL_ARTIFACT_REFS_KEY,
        SKILL_MODEL_INPUTS_KEY=SKILL_MODEL_INPUTS_KEY,
        _pop_dict_list=_pop_dict_list,
        _pop_string_list=_pop_string_list,
        _result_artifact_refs=_result_artifact_refs,
    )
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            "<fbbec23-runtime>",
            "exec",
        ),
        namespace,
    )
    return namespace["_execute_native_read_calls"], hashlib.sha256(
        source.encode()
    ).hexdigest()


__all__ = ["baseline_method", "job", "ok"]
