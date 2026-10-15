"""First-attempt real-provider mode admission; never execute flight actions."""
import json

from src.agent.planning.task_plan import TaskPlan
from src.llm import ModelRequest
from src.tools.diagnostics import exception_diagnostic
from src.tools.validation import validate_tool_args


def check_transition(phase, previous, candidate, args):
    """Check requested lifecycle semantics in addition to accepting valid JSON."""
    expected_operation = "replace" if phase in {"create", "reorder"} else "patch"
    if args["operation"] != expected_operation:
        raise ValueError(f"{phase} requires operation={expected_operation}")
    if candidate.revision != previous.revision + 1:
        raise ValueError("Admission revision did not advance exactly once")
    if phase == "create":
        if len(candidate.steps) != 2 or any(step.status != "pending" for step in candidate.steps):
            raise ValueError("Create must save exactly two pending steps")
        return
    old = {step.id: step for step in previous.steps}
    current = {step.id: step for step in candidate.steps}
    if set(old) != set(current):
        raise ValueError("Admission changed stable step IDs")
    first, second = previous.steps
    if phase == "advance":
        if [step.status for step in candidate.steps] != ["completed", "in_progress"]:
            raise ValueError("Advance did not complete first and activate second")
        if any(current[key].description != old[key].description for key in old):
            raise ValueError("Status patch changed a description")
    elif phase == "correct":
        if current[second.id].description == second.description:
            raise ValueError("Description correction did not change the second description")
        if any(current[key].status != old[key].status for key in old):
            raise ValueError("Description correction changed a status")
    elif phase == "reorder":
        if tuple(candidate.steps) != tuple(reversed(previous.steps)):
            raise ValueError("Reorder must reverse the complete snapshot without other changes")
    elif phase == "cancel":
        active_id = next(step.id for step in previous.steps if step.status == "in_progress")
        if current[active_id].status != "cancelled":
            raise ValueError("Cancel did not cancel the active step")
        if any(current[key].description != old[key].description for key in old):
            raise ValueError("Cancel changed a description")
        if any(current[key] != old[key] for key in old if key != active_id):
            raise ValueError("Cancel changed another step")


def run_plan_admission(root, client, model, tools, image_url, image_source, image_sha256, write):
    """Three independent groups; any first-attempt failure closes admission."""
    spec = next(tool["function"] for tool in tools if tool["function"]["name"] == "runtime_update_plan")
    if spec.get("strict") is not True:
        raise ValueError("Plan admission requires the frozen strict:true tool")
    parameters = {"tool_choice": "auto", **model.generation_parameters()}
    receipts = []
    record = {"status": "running", "plan_probe_status": "running", "groups_required": 3,
              "model": model.redacted_summary(), "image_source": image_source,
              "image_sha256": image_sha256, "receipts": receipts}
    prompt = (root / "source/src/agent/planning/prompts/agent_tool_loop_system.txt").read_text(encoding="utf-8")
    instructions = {
        "create": "Inspect the real image and create exactly two pending inspection steps using operation=replace. Mention a visible feature in a description.",
        "advance": "Use operation=patch: complete the first step and activate the second, changing only statuses. Use null descriptions and null explanation.",
        "correct": "Use operation=patch: clarify the second step description by including how to verify its visible feature. Change only that description; status and explanation must be null.",
        "reorder": "Use operation=replace: reverse the two steps in the saved snapshot; preserve their IDs, descriptions and statuses exactly. explanation must be null.",
        "cancel": "Use operation=patch: cancel the currently in_progress step only. description and explanation must be null.",
    }
    for group in range(1, 4):
        plan = TaskPlan()
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": [
            {"type": "text", "text": "Interface admission only; no flight will execute."},
            {"type": "image_url", "image_url": {"url": image_url}}]}]
        for phase, instruction in instructions.items():
            messages.append({"role": "user", "content": "CURRENT TASK STATE\n" + json.dumps({"task_plan": plan.to_dict()}, ensure_ascii=False)
                + "\n" + instruction + " Submit exactly one runtime_update_plan call. All fields are required; unused patch fields are null."})
            entry = {"group": group, "phase": phase, "attempt": 1, "plan_before": plan.to_dict()}
            receipts.append(entry)
            # Save the exact multimodal request and raw provider response; no repair.
            request_path = root / "admission_requests" / f"group_{group}_{phase}.json"
            write(request_path, {"messages": messages, "tools": tools, "parameters": parameters})
            entry["request_path"] = str(request_path)
            try:
                response = client.request(ModelRequest(messages=messages, tools=tools, parameters=parameters))
                entry.update(tool_calls=response.tool_calls, response=response.data, usage=response.usage,
                             transport=response.transport_trace)
                if response.data.get("model") != model.model_name:
                    raise ValueError("Admission returned a different model")
                calls = response.tool_calls
                if len(calls) != 1 or calls[0]["function"]["name"] != spec["name"]:
                    raise ValueError("Admission must submit exactly one plan call")
                args = json.loads(calls[0]["function"]["arguments"])
                entry["args"] = args
                validate_tool_args(args, spec["parameters"])
                candidate = plan.updated(args)
                check_transition(phase, plan, candidate, args)
            except Exception as exc:
                entry.update(status="rejected", diagnostic=exception_diagnostic(exc,
                    phase="admission", execution="not_started"), error=str(exc), plan=plan.to_dict())
                record.update(status="failed", plan_probe_status="failed")
                write(root / "admission.json", record)
                raise RuntimeError("Plan admission failed; selected navigation cases were not started") from exc
            plan = candidate
            entry.update(status="ok", plan=plan.to_dict())
            messages.append({"role": "assistant", "content": response.text or None, "tool_calls": calls})
            messages.append({"role": "tool", "tool_call_id": calls[0]["id"],
                "content": json.dumps({"status": "ok", "revision": plan.revision})})
            # Retain native call history while keeping one latest task-state block.
            messages = [m for m in messages if not (m["role"] == "user" and isinstance(m["content"], str)
                        and m["content"].startswith("CURRENT TASK STATE"))]
            write(root / "admission.json", record)
    record.update(status="passed", plan_probe_status="passed", groups_completed=3)
    write(root / "admission.json", record)
    return record


def run_no_plan_admission(root, client, model, tools, image_url, image_source, image_sha256, write):
    """Three single native calls on saved RGB; validate only, never dispatch.

    Plan lifecycle admission is explicitly inapplicable when its tool is absent.
    This exercises rotation, translation and knowledge argument shapes with the
    same frozen provider tool surface that subsequent navigation will receive.
    """
    specs = {tool["function"]["name"]: tool["function"] for tool in tools}
    if "runtime_update_plan" in specs:
        raise ValueError("Off-mode admission unexpectedly exposes the plan tool")
    phases = [
        ("rotation", "uav_turn_relative", "Request a 10 degree yaw rotation."),
        ("translation", "uav_move_body", "Request vx=1, vy=0, vz=0 metres/second for duration=1 second."),
        ("knowledge", "knowledge_load", "Load basic_flight_control knowledge.")]
    if not all(name in specs for _, name, _ in phases):
        raise ValueError("Off-mode admission is missing a required native tool")
    prompt_dir = root / "source/src/agent/planning/prompts"
    parameters = {"tool_choice": "auto", **model.generation_parameters()}
    record = {"status": "running", "plan_mode": "off", "plan_probe_status": "not_applicable",
              "native_probe_status": "running", "groups_required": 3, "receipts": [],
              "model": model.redacted_summary(), "image_source": image_source, "image_sha256": image_sha256,
              "execution": "not_started", "notice": "API contract admission only; no tools are dispatched."}
    for group, (phase, name, instruction) in enumerate(phases, 1):
        messages = [{"role": "system", "content": (prompt_dir / path).read_text(encoding="utf-8")}
                    for path in ("agent_tool_loop_system.txt", "tool_user_system.txt")]
        messages += [{"role": "user", "content": [
            {"type": "text", "text": "Saved real RGB for interface admission. No flight or tool will execute."},
            {"type": "image_url", "image_url": {"url": image_url}}]},
            {"role": "system", "content": 'CURRENT TASK STATE\n{"task_plan":null,"loaded_knowledge":[]}'},
            {"role": "user", "content": instruction + f" Submit exactly one {name} call."}]
        request_path = root / "admission_requests" / f"native_{group}_{phase}.json"
        write(request_path, {"messages": messages, "tools": tools, "parameters": parameters})
        entry = {"group": group, "phase": phase, "attempt": 1, "request_path": str(request_path)}
        record["receipts"].append(entry)
        try:
            response = client.request(ModelRequest(messages=messages, tools=tools, parameters=parameters))
            entry.update(tool_calls=response.tool_calls, response=response.data, usage=response.usage,
                         transport=response.transport_trace)
            if response.data.get("model") != model.model_name:
                raise ValueError("Admission returned a different model")
            if len(response.tool_calls) != 1 or response.tool_calls[0]["function"]["name"] != name:
                raise ValueError(f"Admission expected exactly one {name} call")
            call = response.tool_calls[0]
            if not call.get("id"):
                raise ValueError("Admission omitted the native tool call ID")
            args = json.loads(call["function"]["arguments"])
            validate_tool_args(args, specs[name]["parameters"])
            entry.update(status="ok", args=args, execution="not_started")
        except Exception as exc:
            entry.update(status="rejected", error=str(exc), diagnostic=exception_diagnostic(
                exc, phase="admission", execution="not_started"))
            record.update(status="failed", native_probe_status="failed")
            write(root / "admission.json", record)
            raise RuntimeError("No-plan native admission failed; navigation was not started") from exc
        write(root / "admission.json", record)
    record.update(status="passed", native_probe_status="passed", groups_completed=3)
    write(root / "admission.json", record)
    return record
