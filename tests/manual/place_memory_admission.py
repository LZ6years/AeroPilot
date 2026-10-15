"""Three real-provider place-memory chains on isolated saved evidence.

Directed calls prove interface compatibility, never autonomous navigation use.
Run through aerial_vln_plan_context_probe.py admit after the evidence sandbox.
"""
import json

from plugins.capabilities.place_memory.store import PlaceStore
from plugins.capabilities.place_memory.list_places.scripts.handler import handle as list_places
from plugins.capabilities.place_memory.save_place.scripts.handler import handle as save_place
from plugins.capabilities.place_memory.inspect_place.scripts.handler import handle as inspect_place
from src.llm import ModelRequest
from src.tools.validation import validate_tool_args


def run_place_admission(root, client, model, tools, image_url, write):
    """Execute valid native calls only against three independent task stores."""
    sandbox = root / "capability_admission"
    observations = [json.loads(line) for line in
                    (sandbox / "memory/observations.jsonl").read_text(encoding="utf-8").splitlines()]
    first, last = observations[0]["id"], observations[-1]["id"]
    specs = {t["function"]["name"]: t["function"] for t in tools}
    handlers = {"list_places": list_places, "save_place": save_place, "inspect_place": inspect_place}
    parameters = {"tool_choice": "auto", **model.generation_parameters()}
    record = {"status": "running", "groups_required": 3, "groups_completed": 0, "receipts": [],
              "notice": "Directed saved-evidence interface admission; excluded from autonomous navigation statistics."}
    for group in range(1, 4):
        store = PlaceStore(sandbox, f"provider_place_admission_{group}")
        graph = store.update(first, last)
        pid = graph["nodes"][0]["id"]
        context = {"run_dir": str(sandbox), "instruction_id": store.instruction_id, "evidence_refs": [first]}
        messages = [{"role": "system", "content": "Exercise only the requested place memory tool. All data is saved historical evidence; no flight executes."},
                    {"role": "user", "content": [{"type": "text", "text": f"Original RGB at {pid}, observation {first}."},
                                                   {"type": "image_url", "image_url": {"url": image_url}}]}]
        phases = [
            ("list_places", "List saved places with offset=0, limit=10."),
            ("save_place", f"Save {pid} with a short uncertain description of the image; evidence_refs=[\"{first}\"], photo_observation_ids=[\"{first}\"]."),
            ("list_places", "List again using exactly offset=0, limit=10; the save should now appear."),
            ("inspect_place", f"Inspect {pid}."),
            ("save_place", f"Correct {pid}'s summary to explicitly say the identification needs confirmation. Preserve useful details and evidence; use an empty photo_observation_ids array."),
            ("inspect_place", f"Inspect {pid} again with the same arguments to retrieve the correction."),
        ]
        seen = set()
        saved_summary = None
        for phase, (name, instruction) in enumerate(phases, 1):
            wire = "skill_place_memory_" + name
            messages.append({"role": "user", "content": instruction + f" Submit exactly one {wire} call."})
            entry = {"group": group, "phase": phase, "tool": name, "attempt": 1}
            record["receipts"].append(entry)
            path = root / "place_admission_requests" / f"group_{group}_{phase}.json"
            write(path, {"messages": messages, "tools": tools, "parameters": parameters})
            entry["request_path"] = str(path)
            try:
                response = client.request(ModelRequest(messages=messages, tools=tools, parameters=parameters))
                entry.update(response=response.data, usage=response.usage, transport=response.transport_trace)
                calls = response.tool_calls
                if response.data.get("model") != model.model_name or len(calls) != 1 or calls[0]["function"]["name"] != wire:
                    raise ValueError("Unexpected model or native tool selection")
                call = calls[0]
                if not call.get("id") or call["id"] in seen:
                    raise ValueError("Native tool call ID must be new")
                seen.add(call["id"])
                args = json.loads(call["function"]["arguments"])
                validate_tool_args(args, specs[wire]["parameters"])
                entry["args"] = args
                if name == "list_places" and args != {"offset": 0, "limit": 10}:
                    raise ValueError("Repeated list arguments changed")
                if name != "list_places" and args["place_id"] != pid:
                    raise ValueError("Wrong place selected")
                result = handlers[name]({"args": args, "runtime_context": {**context, "tool_call_id": call["id"]}})
                if name == "save_place":
                    if phase == 5 and (args["photo_observation_ids"] or args["summary"] == saved_summary):
                        raise ValueError("Correction must change notes and retain photos")
                    saved_summary = result.data["summary"]
                if name == "list_places" and result.data["total"] != (0 if phase == 1 else 1):
                    raise ValueError("Directory did not refresh after save")
                if name == "inspect_place" and (result.data["summary"] != saved_summary or
                        result.data["photos"][0]["observation_id"] != first):
                    raise ValueError("Historical receipt differs from saved state")
                entry.update(status="ok", data=result.data)
                messages.extend([{"role": "assistant", "content": response.text or None, "tool_calls": calls},
                                 {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result.data)}])
            except Exception as exc:
                entry.update(status="failed", error=str(exc))
                record["status"] = "failed"
                write(root / "place_memory_admission.json", record)
                raise RuntimeError("Place admission failed; navigation not started") from exc
            write(root / "place_memory_admission.json", record)
        record["groups_completed"] = group
    record["status"] = "passed"
    write(root / "place_memory_admission.json", record)
    return record
