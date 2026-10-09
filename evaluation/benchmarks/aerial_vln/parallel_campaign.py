"""Run frozen seen episodes on two threads sharing one perception model cache.

The external controller is the sole progress writer and scene-cleanup owner.
A supervised host process runs isolated Runtime/Provider/Transcript objects in
threads; only capability model instances are shared through their existing
cache and CUDA gate. Rendering happens in the controller, outside the threads.
Remote simulators use GPU 0 and GPU 2; GPU 1 is explicitly excluded.

Commands: handoff stops the old serial dispatcher at a completed-child boundary;
run verifies/freeze-records this extension, runs four technical trial episodes,
then continues seen. watch supervises only the seen controller with persistent
restart backoff. host is a private subprocess entry. PAUSE remains the
manual interruption control. The original frozen source directory is untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from urllib.parse import unquote
import uuid

import psutil

REMOTE_GPU_INSPECTION = """import json,os,subprocess,sys
import xml.etree.ElementTree as ET
scenes=json.loads(sys.argv[1])
result=subprocess.run(['nvidia-smi','-q','-x'],capture_output=True,text=True,timeout=8)
if result.returncode: raise RuntimeError(result.stderr.strip())
gpus=[]
for index,gpu in enumerate(ET.fromstring(result.stdout).findall('gpu')):
 processes=[]
 for process in gpu.findall('processes/process_info'):
  pid=int(process.findtext('pid'))
  try: group=os.getpgid(pid)
  except ProcessLookupError: continue
  processes.append({'pid':pid,'pgid':group,'type':process.findtext('type')})
 gpus.append({'index':index,'name':gpu.findtext('product_name'),'uuid':gpu.findtext('uuid'),
              'memory_used':gpu.findtext('fb_memory_usage/used'),'processes':processes})
owned=[]
for scene in scenes:
 observed=[gpu['index'] for gpu in gpus if any(p['pgid']==scene['pid'] for p in gpu['processes'])]
 owned.append({**scene,'observed_gpus':observed})
print(json.dumps({'gpus':gpus,'owned':owned}))
"""

# NVIDIA's index is not necessarily the Vulkan enumeration index: a software
# adapter may occur between NVIDIA devices. Select the existing NVIDIA ICD,
# map UUIDs to Vulkan indices before UE starts, and restrict CUDA by UUID too.
REMOTE_VULKAN_BINDING = r"""import ctypes as C,json,os,pathlib,subprocess,sys,uuid
index=int(sys.argv[1])
if index not in (0,2): raise ValueError('Only GPU 0 and 2 are authorized')
response=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader'],
                        capture_output=True,text=True,timeout=8,check=True)
rows=[line.split(', ') for line in response.stdout.strip().splitlines()]
row=next(row for row in rows if int(row[0])==index)
domain,bus,slot=row[2].split(':')
device,function=slot.split('.')
icds=[]
for directory in ('/usr/share/vulkan/icd.d','/etc/vulkan/icd.d'):
 for path in pathlib.Path(directory).glob('*.json'):
  manifest=json.loads(path.read_text())
  if 'nvidia' in manifest.get('ICD',{}).get('library_path','').lower(): icds.append(str(path))
if not icds: raise RuntimeError('Existing NVIDIA Vulkan ICD is required')
environment={'VK_ICD_FILENAMES':':'.join(sorted(icds)), 'VK_DRIVER_FILES':':'.join(sorted(icds)),
             'CUDA_VISIBLE_DEVICES':row[1]}
os.environ.update(environment)
class App(C.Structure):
 _fields_=[('sType',C.c_uint32),('pNext',C.c_void_p),('name',C.c_char_p),('version',C.c_uint32),
           ('engine',C.c_char_p),('engineVersion',C.c_uint32),('apiVersion',C.c_uint32)]
class Create(C.Structure):
 _fields_=[('sType',C.c_uint32),('pNext',C.c_void_p),('flags',C.c_uint32),('app',C.POINTER(App)),
           ('layers',C.c_uint32),('layerNames',C.c_void_p),('extensions',C.c_uint32),('extensionNames',C.c_void_p)]
class Identity(C.Structure):
 _fields_=[('sType',C.c_uint32),('pNext',C.c_void_p),('deviceUUID',C.c_ubyte*16),
           ('driverUUID',C.c_ubyte*16),('luid',C.c_ubyte*8),('nodeMask',C.c_uint32),('luidValid',C.c_uint32)]
class Properties(C.Structure):
 _fields_=[('sType',C.c_uint32),('pNext',C.c_void_p),('properties',C.c_ubyte*4096)]
library=C.CDLL('libvulkan.so.1')
library.vkCreateInstance.argtypes=[C.POINTER(Create),C.c_void_p,C.POINTER(C.c_void_p)]
library.vkEnumeratePhysicalDevices.argtypes=[C.c_void_p,C.POINTER(C.c_uint32),C.POINTER(C.c_void_p)]
library.vkGetPhysicalDeviceProperties2.argtypes=[C.c_void_p,C.POINTER(Properties)]
library.vkDestroyInstance.argtypes=[C.c_void_p,C.c_void_p]
app=App(sType=0,name=b'FlyAgent GPU ownership probe',apiVersion=(1<<22)|(1<<12))
create=Create(sType=1,app=C.pointer(app))
instance=C.c_void_p()
result=library.vkCreateInstance(C.byref(create),None,C.byref(instance))
if result: raise RuntimeError('vkCreateInstance failed: '+str(result))
try:
 count=C.c_uint32()
 result=library.vkEnumeratePhysicalDevices(instance,C.byref(count),None)
 if result: raise RuntimeError('Vulkan enumeration failed: '+str(result))
 devices=(C.c_void_p*count.value)()
 result=library.vkEnumeratePhysicalDevices(instance,C.byref(count),devices)
 if result: raise RuntimeError('Vulkan enumeration failed: '+str(result))
 visible=[]
 for device in devices:
  identity=Identity(sType=1000071004)
  properties=Properties(sType=1000059001,pNext=C.addressof(identity))
  library.vkGetPhysicalDeviceProperties2(device,C.byref(properties))
  visible.append('GPU-'+str(uuid.UUID(bytes=bytes(identity.deviceUUID))))
finally: library.vkDestroyInstance(instance,None)
if visible.count(row[1])!=1: raise RuntimeError('GPU UUID is not uniquely visible to Vulkan')
print(json.dumps({'remote_gpu':index,'uuid':row[1],'pci_bus_id':row[2],
                  'environment':environment,'visible_vulkan_uuids':visible,'vulkan_index':visible.index(row[1])}))
"""

DEFAULT_OUTPUT = Path("H:/FlyAgent-Eval/aerialvln_s_20261007")
SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]

# Bootstrap before importing any project module. A real run must load Runtime,
# prompts, plugins and evaluation from the original frozen source, even when
# this external launcher lives in the working repository.
if __name__ == "__main__":
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("command", nargs="?")
    early.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    bootstrap, _ = early.parse_known_args()
    if bootstrap.command in {"run", "host", "watch"}:
        frozen = bootstrap.output.resolve() / "source"
        if not frozen.is_dir():
            raise RuntimeError("A prepared frozen campaign is required")
        sys.path.insert(0, str(frozen))
        os.chdir(frozen)
    else:
        sys.path.insert(0, str(REPO))

from evaluation.benchmarks.aerial_vln.campaign import (
    _attempt_dir, _render_reports, campaign_lock, environment_diagnostics,
    provider_probe, storage_check,
)
from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.aerial_vln.runner import AerialVlnBenchmarkRunner
from evaluation.benchmarks.core.process_review import write_process_review
from evaluation.benchmarks.core.report_io import (
    atomic_write_text as _atomic_write_text, write_json as _write_json,
)
from evaluation.benchmarks.core.supervision import cleanup_owned, cleanup_verified
from src.agent.config import load_model_config
from src.agent.config.model import read_env_file
from src.runtime.configs.map_config import load_map_config
from src.sim_server.remote.control import RemoteSceneControl


def _extension(name, original):
    """Load separately frozen orchestration helpers without mixing Runtime code."""
    path = Path(globals().get("_FROZEN_EXTENSION_DIR", SCRIPT.parent)) / name
    if not path.is_file():
        path = REPO / original
    spec = importlib.util.spec_from_file_location(
        "src.sim_server.remote.parallel_startup_inspection" if name == "startup_recovery.py"
        else "parallel_campaign_queue_extension", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ParallelRecoveryQueue = _extension(
    "parallel_recovery.py", "evaluation/benchmarks/aerial_vln/parallel_recovery.py"
).ParallelRecoveryQueue


FILE_ACCESS_BACKOFF = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def retry_file_access(operation, *, path=None, action="read"):
    """Recover short Windows sharing conflicts without hiding persistent faults.

    Readers can race an atomic replacement, and open readers can briefly block
    a writer's replacement. Retry the complete operation, preserving the shared
    utility's atomic writes. This budget is independent of Provider/episode
    attempts; persistent access errors still reach the controller checkpoint.
    """
    started = time.monotonic()
    for attempt in range(len(FILE_ACCESS_BACKOFF) + 1):
        try:
            value = operation()
        except OSError as exc:
            transient = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in {32, 33}
            if not transient or attempt == len(FILE_ACCESS_BACKOFF):
                raise
            delay = FILE_ACCESS_BACKOFF[attempt]
            print(json.dumps({"event": "file_access_retry", "path": str(path) if path else None,
                  "operation": action, "attempt": attempt + 1, "error_type": type(exc).__name__,
                  "winerror": getattr(exc, "winerror", None), "backoff_seconds": delay}), flush=True)
            time.sleep(delay)
        else:
            if attempt:
                print(json.dumps({"event": "file_access_recovered", "path": str(path) if path else None,
                      "operation": action, "attempts": attempt + 1,
                      "elapsed_seconds": time.monotonic() - started}), flush=True)
            return value


def write_json(path, value):
    """Reuse atomic JSON persistence with the metadata access retry budget."""
    return retry_file_access(lambda: _write_json(path, value), path=path, action="write")


def atomic_write_text(path, content, *, encoding="utf-8"):
    return retry_file_access(lambda: _atomic_write_text(path, content, encoding=encoding),
                             path=path, action="write")


def read_json(path):
    """Missing/partial legacy artifacts wait for a poll; sharing conflicts retry."""
    try:
        return json.loads(retry_file_access(lambda: Path(path).read_text(encoding="utf-8-sig"), path=path))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def valid_result(result):
    return bool(result.get("metrics")) and not result.get("infrastructure_error") and not result.get("metric_error")


def owner_of(process):
    return {"pid": process.pid, "created": process.create_time(), "command": process.cmdline()}


def read_process_diagnostic(path):
    """Keep native/legacy stderr readable without turning decoding into a crash.

    Python children use UTF-8, but native dependencies and previous hosts may
    emit locale-specific progress bytes. Preserve those bytes as escape text
    for classification while the original log remains unchanged on disk.
    """
    path = Path(path)
    return retry_file_access(path.read_bytes, path=path).decode('utf-8', errors='backslashreplace')


def matching_process(owner):
    """PID reuse never authorizes terminating an unrelated process."""
    try:
        process = psutil.Process(owner["pid"])
        if process.create_time() == owner.get("created") and process.cmdline() == owner.get("command"):
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def kill_owned(owner):
    process = matching_process(owner)
    if process is not None:
        process.kill()
        process.wait(timeout=15)


def validate_gpu_binding(binding, expected):
    """Reject unverified GPU selection before modifying any execution catalog."""
    visible = binding.get("visible_vulkan_uuids", [])
    index = binding.get("vulkan_index")
    environment = binding.get("environment", {})
    device_uuid = binding.get("uuid", "")
    if expected not in {0, 2} or binding.get("remote_gpu") != expected \
            or not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", device_uuid) \
            or not isinstance(index, int) or index < 0 or index >= len(visible) \
            or visible[index] != device_uuid or visible.count(device_uuid) != 1 \
            or set(environment) != {"VK_ICD_FILENAMES", "VK_DRIVER_FILES", "CUDA_VISIBLE_DEVICES"} \
            or not environment.get("VK_DRIVER_FILES") \
            or environment["VK_DRIVER_FILES"] != environment["VK_ICD_FILENAMES"] \
            or environment["CUDA_VISIBLE_DEVICES"] != device_uuid:
        raise ValueError("Remote Vulkan/CUDA GPU binding is not verified")
    return binding


def discover_gpu_bindings(map_config_path):
    """Read UUIDs through the existing driver; no packages or system changes."""
    packages, _ = load_map_config(map_config_path)
    control = RemoteSceneControl()
    target = control.target(packages[0])
    if any(control.target(package) != target for package in packages):
        raise ValueError("All parallel maps must use the same remote host")
    bindings = {}
    for lane, index in (("A", 0), ("B", 2)):
        response = control.runner.run(target, ["python3", "-c", REMOTE_VULKAN_BINDING, str(index)], timeout_sec=30)
        bindings[lane] = validate_gpu_binding(json.loads(response), index)
    return bindings


def gpu_violation(sample):
    """Any owned process on an unauthorized GPU stops new dispatch immediately."""
    return any(set(item.get("observed_gpus", [])) - {0, 2} for item in sample.get("owned", []))


def lane_catalog(catalog, lane, binding):
    """Isolate ports and select remote GPU 0/2, keeping GPU 1 excluded."""
    if lane not in {"A", "B"}:
        raise ValueError("Unknown execution lane")
    data = json.loads(json.dumps(catalog))
    offset = 0 if lane == "A" else 1
    validate_gpu_binding(binding, 0 if lane == "A" else 2)
    data["port_policy"] = {"control_port_start": 35001 + offset, "max_scene_num": 1}
    for package in data["maps"]:
        if package.get("connection", {}).get("mode") != "remote":
            raise ValueError("Parallel campaign requires remote managed scenes")
        package["airsim"]["port"] = 30001 + offset
        args = package["launch"].get("args", [])
        package["launch"]["args"] = [arg for arg in args if not arg.lower().startswith("-graphicsadapter")]
        package["launch"]["args"].append("-GraphicsAdapter=" + str(binding["vulkan_index"]))
        package["launch"]["env"] = {**package["launch"].get("env", {}), **binding["environment"]}
        package["launch"]["existing_process_policy"] = "reject"
    return data


def verify_frozen(output, manifest):
    for relative, digest in manifest["source_hashes"].items():
        path = output / "source" / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen source changed: " + relative)
    if load_model_config(manifest["configs"]["val_seen"]["env_path"]).redacted_summary() != manifest["model"]:
        raise ValueError("Effective model configuration changed")
    for key, value in read_env_file(manifest["configs"]["val_seen"]["env_path"]).items():
        if key.startswith("FLYAGENT_"):
            os.environ.setdefault(key, value)
    os.environ.update(TEMP=str(output / "temp"), TMP=str(output / "temp"))
    (output / "temp").mkdir(exist_ok=True)


def prepare_extension(output, manifest):
    """Freeze the additive controller, queue and read-only recovery helper."""
    files = {
        "parallel_campaign.py": SCRIPT,
        "parallel_recovery.py": SCRIPT.with_name("parallel_recovery.py")
        if SCRIPT.with_name("parallel_recovery.py").exists()
        else REPO / "evaluation/benchmarks/aerial_vln/parallel_recovery.py",
        "startup_recovery.py": SCRIPT.with_name("startup_recovery.py")
        if SCRIPT.with_name("startup_recovery.py").exists()
        else REPO / "src/sim_server/remote/recovery.py",
        "parallel_watchdog.py": SCRIPT.with_name("parallel_watchdog.py")
        if SCRIPT.with_name("parallel_watchdog.py").exists()
        else REPO / "evaluation/benchmarks/aerial_vln/parallel_watchdog.py",
    }
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    revision = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = output / "parallel/controller_source" / revision[:16]
    destination.mkdir(parents=True, exist_ok=True)
    for name, path in files.items():
        target = destination / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError("Parallel controller snapshot changed")
        if not target.exists():
            target.write_bytes(path.read_bytes())
    maps = read_json(manifest["configs"]["val_seen"]["map_config_path"])
    bindings = discover_gpu_bindings(manifest["configs"]["val_seen"]["map_config_path"])
    lanes = {}
    for lane in ("A", "B"):
        path = output / "parallel" / ("maps_" + lane + ".json")
        catalog = lane_catalog(maps, lane, bindings[lane])
        write_json(path, catalog)
        lanes[lane] = {"map_config_path": str(path),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "local_port": 35001 if lane == "A" else 35002,
                       "remote_port": 30001 if lane == "A" else 30002, "remote_gpu": 0 if lane == "A" else 2}
    overlay = {"version": 2, "base_fingerprint": manifest["fingerprint"], "remote_gpu_bindings": bindings,
               "controller_revision": revision, "source_hashes": hashes,
               "controller_script": str(destination / SCRIPT.name),
               "forbidden_remote_gpus": [1], "local_perception_device": "cuda:0",
               "lanes": lanes, "html_profile": manifest["configs"]["val_seen"]["report_profile"],
               "thread_html_rendering": False, "created_at": utc_now()}
    write_json(output / "parallel/execution_overlay.json", overlay)
    return overlay


def verify_overlay(overlay, manifest):
    if overlay["base_fingerprint"] != manifest["fingerprint"] or overlay["forbidden_remote_gpus"] != [1]:
        raise ValueError("Execution overlay does not belong to this campaign")
    for lane, info in overlay["lanes"].items():
        path = Path(info["map_config_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError("Lane configuration changed")
        expected = lane_catalog(read_json(manifest["configs"]["val_seen"]["map_config_path"]), lane,
                                overlay["remote_gpu_bindings"][lane])
        if read_json(path) != expected or info["remote_gpu"] != (0 if lane == "A" else 2):
            raise ValueError("Only transport isolation and GPU 0/2 are allowed")
    directory = Path(overlay["controller_script"]).parent
    for name, digest in overlay["source_hashes"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Parallel extension snapshot changed")


def cached_models(settings, cache_root):
    """Identify complete local snapshots; downloading or changing weights is forbidden."""
    records = []
    for key in ("grounding_model", "segmentation_model"):
        model = settings[key]
        directory = Path(cache_root) / ("models--" + model.replace("/", "--"))
        reference = directory / "refs/main"
        if not reference.is_file():
            raise RuntimeError("Model is not available in the local cache: " + model)
        revision = reference.read_text(encoding="utf-8").strip()
        snapshot = directory / "snapshots" / revision
        if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
            raise RuntimeError("Local model snapshot is incomplete: " + model)
        files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in snapshot.iterdir() if path.is_file()}
        records.append({"model": model, "revision": revision, "snapshot": str(snapshot),
                        "reference": str(reference), "files": files})
    return records


def offline_models(output, overlay):
    """Reuse existing weights without Hub metadata HEAD requests or TLS retries."""
    from huggingface_hub.constants import HF_HUB_CACHE
    settings = read_json(output / "source/plugins/capabilities/visual_grounding/measure_visible_depth/capability.json")["settings"]
    actual = cached_models(settings, HF_HUB_CACHE)
    if overlay.get("model_cache") and overlay["model_cache"] != actual:
        raise RuntimeError("Local perception model weights changed since parallel freeze")
    overlay["model_cache"] = actual
    overlay["host_environment"] = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    write_json(output / "parallel/execution_overlay.json", overlay)


def trial_tasks(tasks, state):
    """Four untouched seen episodes covering the difficulty range."""
    pending = [task for task in tasks if state["tasks"][ParallelRecoveryQueue.key(task)]["status"] == "pending"]
    chosen = []
    for difficulty in ("low", "medium", "high", "high"):
        found = next((task for task in pending if task not in chosen and task.get("difficulty") == difficulty), None)
        if found is not None:
            chosen.append(found)
    chosen.extend(task for task in pending if task not in chosen)
    return chosen[:4]


def model_snapshot():
    """Human-only model identity/memory evidence, never added to model inputs."""
    import torch
    from plugins.capabilities.visual_grounding.measure_visible_depth.scripts import service
    with service._SERVICE_CACHE_LOCK:
        services = list(service._SERVICE_CACHE.values())
        identities = [{"service": id(item),
                       "grounding_model": id(item.grounding_backend._model) if item.grounding_backend._model else None,
                       "segmentation_model": id(item.segmentation_backend._model)
                       if item.segmentation_backend and item.segmentation_backend._model else None}
                      for item in services]
    return {"host_pid": os.getpid(), "host_created": psutil.Process().create_time(),
            "services": identities, "service_count": len(identities),
            "allocated_bytes": torch.cuda.memory_allocated(0),
            "reserved_bytes": torch.cuda.memory_reserved(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0)}


def warm_perception():
    """Load the unchanged capability settings once, under existing CUDA gates."""
    from plugins.capabilities.visual_grounding.measure_visible_depth.scripts.service import service_from_settings
    capability = Path("plugins/capabilities/visual_grounding/measure_visible_depth/capability.json")
    service = service_from_settings(read_json(capability)["settings"])
    service.grounding_backend._load()
    if service.segmentation_backend is not None:
        service.segmentation_backend._load()
    return model_snapshot()


def execute_thread(output, manifest, request, environment):
    """One isolated Runtime graph; full HTML is rendered later by the parent."""
    from src.perception.inference_resources import inference_observer
    task, attempt = request["task"], request["attempt"]
    directory = _attempt_dir(output, task, attempt["number"])
    write_json(directory / "environment.json", {**environment, "lane": request["lane"], "host_pid": os.getpid()})
    config = replace(AerialVlnEvalConfig.from_dict(manifest["configs"]["val_seen"]),
                     episode_ids=(task["episode_id"],), output_dir=str(directory / "evaluation"),
                     run_root=str(directory / "runs"), map_config_path=request["map_config_path"],
                     generate_html_review=False)
    events = directory / "inference_resources.jsonl"

    def observe(event, payload):
        with events.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"timestamp": utc_now(), "event": event, **payload}) + "\n")

    with inference_observer(observe):
        runner = AerialVlnBenchmarkRunner(config)
        prepared = runner.prepare_evaluation(require_execution_ready=True)
        case = prepared.cases[0]
        if case.episode_id != task["episode_id"]:
            raise ValueError("Worker selected a different episode")
        result = runner.run_case(case, directory / "evaluation",
                                 assignment=prepared.selection.assignment_by_episode[case.episode_id],
                                 provenance=prepared.provenance)
    write_json(directory / "thread_evidence.json", {
        "episode_id": case.episode_id, "lane": request["lane"],
        "instruction_sha256": hashlib.sha256(case.instruction.encode("utf-8")).hexdigest(),
        "runtime_source": str(Path(sys.modules["src.runtime.factory"].__file__).resolve()),
        "thread_finished": True})
    write_json(directory / "result.json", result)
    return result


def host_main(args):
    """Persistent two-thread host; a dead controller cannot leave it running."""
    output, session = args.output.resolve(), args.session.resolve()
    manifest = read_json(output / "manifest.json")
    overlay = read_json(output / "parallel/execution_overlay.json")
    verify_frozen(output, manifest)
    verify_overlay(overlay, manifest)
    environment = environment_diagnostics()
    if not environment["cuda_available"]:
        raise RuntimeError("CUDA unavailable in parallel host")
    environment["shared_models"] = warm_perception()
    write_json(session / "host_environment.json", environment)
    parent = {"pid": args.parent_pid, "created": args.parent_created}
    processed, futures, descendants = set(), {}, {}
    last_status = 0
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="aerial-vln") as pool:
        while True:
            try:
                controller = psutil.Process(parent["pid"])
                if controller.create_time() != parent["created"]:
                    os._exit(70)
            except psutil.NoSuchProcess:
                os._exit(70)
            now = time.monotonic()
            if now - last_status >= 1:
                for process in psutil.Process().children(recursive=True):
                    try:
                        descendants[process.pid] = owner_of(process)
                    except psutil.Error:
                        pass
                write_json(session / "host_status.json", {
                    "pid": os.getpid(), "timestamp": utc_now(), "active": list(futures),
                    "descendants": list(descendants.values()), "shared_models": model_snapshot()})
                last_status = now
            for lane in list(futures):
                request, future, started = futures[lane]
                if not future.done():
                    continue
                try:
                    result, exception = future.result(), None
                except Exception as exc:
                    result = {}
                    exception = {"error_type": type(exc).__name__, "error": str(exc),
                                 "traceback": traceback.format_exc()}
                write_json(session / "results" / (request["id"] + ".json"), {
                    "id": request["id"], "lane": lane, "task": request["task"],
                    "attempt": request["attempt"], "result": result, "exception": exception,
                    "thread_finished": True, "elapsed_seconds": time.monotonic() - started,
                    "shared_models": model_snapshot()})
                del futures[lane]
            if (session / "STOP").exists():
                if not futures:
                    return
            else:
                for lane in ("A", "B")[:args.workers]:
                    request = read_json(session / "commands" / (lane + ".json"))
                    if not request or request["id"] in processed or lane in futures:
                        continue
                    if request["lane"] != lane or request["task"]["split"] != "val_seen":
                        raise ValueError("Worker request violates lane or split")
                    if request["map_config_path"] != overlay["lanes"][lane]["map_config_path"]:
                        raise ValueError("Worker request has an unknown map configuration")
                    processed.add(request["id"])
                    futures[lane] = (request, pool.submit(execute_thread, output, manifest, request, environment), time.monotonic())
            time.sleep(0.1)


def attempt_result(directory, task):
    result = read_json(directory / "result.json")
    return result or read_json(directory / "evaluation/cases" / task["episode_id"] / "case_result.json")


def finish_attempt(output, manifest, task, attempt, process, *, reason=None):
    """Verify this lane's ownership, render the original HTML and clean depth."""
    directory = _attempt_dir(output, task, attempt["number"])
    map_path = attempt.get("map_config_path", manifest["configs"][task["split"]]["map_config_path"])
    result = attempt_result(directory, task)
    old_cleanup = read_json(directory / "cleanup.json")
    if old_cleanup and not old_cleanup.get("verified") and not (directory / "cleanup_initial.json").exists():
        write_json(directory / "cleanup_initial.json", old_cleanup)
    try:
        receipts = cleanup_owned(directory / "runs", map_path, process.get("descendants", []))
        verified = cleanup_verified(receipts)
        proof = None
        if not verified:
            scenes = list((directory / "runs").glob("*/scenes/*"))
            tunnels = [owner for owner in process.get("descendants", [])
                       if "-L" in owner.get("command", []) and
                       owner["command"][owner["command"].index("-L") + 1].startswith(
                           "127.0.0.1:" + str(attempt.get("local_port", 35001)) + ":")]
            if len(scenes) == 1:
                packages, _ = load_map_config(map_path)
                map_id = manifest["configs"][task["split"]]["scene_map"][task["scene_id"]]
                package = next(item for item in packages if item.map_id == map_id)
                recovery = _extension("startup_recovery.py", "src/sim_server/remote/recovery.py")
                verifier = (recovery.verify_failed_launch if (scenes[0] / "remote-scene-spec.json").exists()
                            else recovery.verify_unlaunched_scene)
                # Historical SSH stop failure is not permanent proof of a
                # residual scene. Only the lifecycle owner may replace it with
                # fresh pre-launch absence evidence after the lane has ended.
                recheck = {} if (scenes[0] / "remote-scene-spec.json").exists() else {"recheck_cleanup": True}
                proof = verifier(package, scenes[0], tunnels,
                    completed_attempt=True, local_endpoint=("127.0.0.1", attempt.get("local_port", 35001)),
                    **recheck)
                verified = proof["verified"]
        write_json(directory / "cleanup.json", {"verified": verified, "receipts": receipts, "startup_proof": proof})
    except Exception as exc:
        verified = False
        write_json(directory / "cleanup.json", {"verified": False, "error_type": type(exc).__name__, "error": str(exc)})
    valid = valid_result(result)
    # A durable terminal result survives host death; incomplete TLS trajectories
    # have infrastructure metadata and cannot satisfy valid_result.
    process = {**process, "reason": None if valid else reason, "status": "finished"}
    write_json(directory / "process.json", process)
    case_dir = directory / "evaluation/cases" / task["episode_id"]
    if valid and verified:
        try:
            result["report"] = write_process_review(case_dir, result,
                profile=manifest["configs"][task["split"]]["report_profile"])
            result.pop("report_error", None)
        except Exception as exc:
            result["report_error"] = {"error_type": type(exc).__name__, "error": str(exc)}
        write_json(case_dir / "case_result.json", result)
        write_json(directory / "result.json", result)
    depth_bytes = 0
    if valid and verified and result.get("report") and not result.get("report_error"):
        owned = (directory / "runs").resolve()
        for path in owned.rglob("*.npy"):
            if owned in path.resolve().parents and any("depth" in part.lower() for part in path.parts):
                depth_bytes += path.stat().st_size
                path.unlink()
    retained = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    receipt = {"result": result, "cleanup_verified": verified, "valid_score": valid,
               "process": process, "directory": str(directory), "retained_bytes": retained,
               "temporary_depth_bytes": depth_bytes}
    write_json(directory / "receipt.json", receipt)
    return receipt


def activity_time(directory):
    latest = 0
    for path in (directory / "runs").rglob("*"):
        try:
            if path.is_file() and path.suffix in {".json", ".jsonl", ".png"}:
                latest = max(latest, path.stat().st_mtime_ns)
        except FileNotFoundError:
            pass
    return latest


def repair_pending_reports(output, manifest, queue):
    """Retry presentation independently; existing scores never trigger a flight."""
    pending = []
    for item in queue.state["tasks"].values():
        result = item.get("result")
        if not result or not (result.get("report_error") or not result.get("report")):
            continue
        task = item["task"]
        directory = Path(result["run_dir"]).parents[1]
        case_dir = directory / "evaluation/cases" / task["episode_id"]
        try:
            result["report"] = write_process_review(case_dir, result,
                profile=manifest["configs"][task["split"]]["report_profile"])
            result.pop("report_error", None)
            write_json(case_dir / "case_result.json", result)
            write_json(directory / "result.json", result)
            queue.save()
        except Exception as exc:
            pending.append({"episode": queue.key(task), "error": str(exc)})
    write_json(output / "parallel/report_pending.json", {"pending": pending, "checked_at": utc_now()})


def storage_budget(output, queue, tasks, workers):
    if not storage_check(output, queue, tasks):
        return False
    record = read_json(output / "storage_check.json")
    peaks = [attempt.get("temporary_depth_bytes", 0)
             for item in queue.state["tasks"].values() for attempt in item["attempts"]]
    required = max(10 * 1024**3, record["remaining_cases"] * record["observed_max_retained_bytes"] * 2
                   + workers * max(peaks, default=0))
    record.update(required_bytes=required, concurrent_attempts=workers,
                  passed=record["free_bytes"] >= required)
    write_json(output / "storage_check.json", record)
    return record["passed"]


def coverage_check(output, manifest, queue, *, started=None, initial_scored=0):
    """Verify canonical coverage and publish measured throughput/storage totals."""
    expected = {queue.key(task) for task in manifest["tasks"]}
    if set(queue.state["tasks"]) != expected or len(expected) != len(manifest["tasks"]):
        raise ValueError("Campaign task coverage changed")
    counts = {split: dict(Counter(item["status"] for item in queue.state["tasks"].values()
                                  if item["task"]["split"] == split))
              for split in ("val_seen", "val_unseen")}
    for split, number in manifest["counts"].items():
        if sum(counts[split].values()) != number:
            raise ValueError("Split episode count changed")
    sizes = [item["attempts"][-1].get("retained_bytes", 0)
             for item in queue.state["tasks"].values() if item["status"] == "scored"]
    elapsed = time.monotonic() - started if started is not None else 0
    scored = counts["val_seen"].get("scored", 0)
    throughput = (scored - initial_scored) * 3600 / elapsed if elapsed > 0 else None
    write_json(output / "parallel/progress_check.json", {
        "counts": counts, "checked_at": utc_now(), "elapsed_seconds": elapsed,
        "valid_episodes_per_hour": throughput, "retained_bytes": sum(sizes),
        "mean_retained_bytes": sum(sizes) / len(sizes) if sizes else 0,
        "remaining_hours_estimate": (manifest["counts"]["val_seen"] - scored) / throughput
        if throughput and scored - initial_scored >= 4 else None})


def remote_gpu_sample(manifest, output, active, overlay):
    """Read NVML process groups; inspect only identities owned by these lanes."""
    scenes = []
    for lane, request in active.items():
        directory = _attempt_dir(output, request["task"], request["attempt"]["number"])
        for path in (directory / "runs").glob("*/state/scene_instance.json"):
            scene = read_json(path)
            if scene.get("process_ownership") == "remote_managed" and scene.get("remote_process_id"):
                scenes.append({"lane": lane, "pid": scene["remote_process_id"],
                               "expected_gpu": overlay["lanes"][lane]["remote_gpu"]})
    if not scenes:
        return None
    packages, _ = load_map_config(overlay["lanes"]["A"]["map_config_path"])
    control = RemoteSceneControl()
    response = control.runner.run(control.target(packages[0]),
        ["python3", "-c", REMOTE_GPU_INSPECTION, json.dumps(scenes)], timeout_sec=15)
    return {**json.loads(response), "checked_at": utc_now()}


def validate_trial(output, queue, tasks, snapshots, *, gpu_samples=None, workers=2):
    errors, images = [], 0
    for task in tasks:
        item = queue.state["tasks"][queue.key(task)]
        if item["status"] != "scored" or not item["attempts"][-1].get("cleanup_verified"):
            errors.append("episode_or_cleanup:" + queue.key(task))
            continue
        directory = _attempt_dir(output, task, item["attempts"][-1]["number"]).resolve()
        page = directory / "evaluation/cases" / task["episode_id"] / "review.html"
        if not page.exists():
            errors.append("missing_report:" + queue.key(task))
            continue
        html = page.read_text(encoding="utf-8")
        if "data:image/" in html:
            errors.append("inline_image:" + queue.key(task))
        for reference in re.findall(r'<img\b[^>]*\bsrc=["\']([^"\']+)', html, re.IGNORECASE):
            image = (page.parent / unquote(reference)).resolve()
            if directory not in image.parents or not image.is_file():
                errors.append("image_scope:" + reference)
            images += 1
        result = item["result"]
        run_path = Path(result.get("run_dir", "")).resolve()
        evidence = read_json(directory / "thread_evidence.json")
        if directory not in run_path.parents or evidence.get("episode_id") != task["episode_id"]:
            errors.append("transcript_scope:" + queue.key(task))
        else:
            history = run_path / "memory/agent_transcript.jsonl"
            instructions = [json.loads(line)["payload"].get("instruction", "")
                            for line in history.read_text(encoding="utf-8").splitlines()
                            if line.strip() and json.loads(line).get("kind") == "user_instruction"] if history.exists() else []
            if len(instructions) != 1 or hashlib.sha256(instructions[0].encode("utf-8")).hexdigest() != evidence.get("instruction_sha256"):
                errors.append("transcript_instruction:" + queue.key(task))
    identities = [snapshot.get("services", []) for snapshot in snapshots]
    if not identities or any(len(items) != 1 or not items[0]["grounding_model"]
                             or not items[0]["segmentation_model"] for items in identities):
        errors.append("shared_model_count")
    else:
        # A process restart may legitimately construct a fresh single cache;
        # identities must remain stable within each supervised host.
        hosts = {}
        for snapshot, items in zip(snapshots, identities):
            previous = hosts.setdefault((snapshot.get("host_pid"), snapshot.get("host_created")), items)
            if previous != items:
                errors.append("shared_model_identity_changed")
    if gpu_samples is not None:
        observations = [item for sample in gpu_samples for item in sample.get("owned", [])]
        for lane, gpu in (("A", 0), ("B", 2))[:workers]:
            # SDL can retain a graphics context on the display's GPU 0 while
            # Vulkan renders on GPU 2. Both are authorized; GPU 1 never is.
            if not any(item["lane"] == lane and gpu in item["observed_gpus"]
                       and not set(item["observed_gpus"]) - {0, 2} for item in observations):
                errors.append("remote_gpu_unverified:" + lane)
        if any(1 in item["observed_gpus"] for item in observations):
            errors.append("forbidden_remote_gpu")
    return {"passed": not errors, "errors": errors, "episodes": len(tasks),
            "image_references": images, "model_snapshot": snapshots[-1] if snapshots else {},
            "remote_gpu_samples": gpu_samples, "workers": workers,
            "navigation_sr_is_gate": False, "checked_at": utc_now()}


def service_recovery(output, manifest, queue):
    """Drain first, then make bounded probes; a user pause interrupts backoff."""
    for delay in (30, 120, 300):
        queue.state["service_backoff_seconds"] = delay
        queue.save()
        until = time.monotonic() + delay
        while time.monotonic() < until:
            if (output / "PAUSE").exists():
                return queue.pause("manual_interrupt")
            time.sleep(min(1, max(0, until - time.monotonic())))
        if provider_probe(manifest["configs"]["val_seen"]["env_path"],
                          output / "parallel/service_check.json", service_check=True):
            queue.state["consecutive_provider_failures"] = 0
            queue.save()
            return True
    return queue.pause("provider_unavailable")


def watchdog_reason(request, now, deadline, no_progress):
    """Deterministic attempt watchdog; file activity never extends the deadline."""
    if now - request["started"] > deadline:
        return "episode_deadline"
    if now - request["progress_at"] > no_progress:
        return "no_progress"
    return None


def write_exit_status(output, queue, *, workers, gate, host_pid):
    """Publish the final state to both readers; a stopped host has no lanes."""
    finished = utc_now()
    state = {"status": queue.state["status"], "reason": queue.state["reason"],
             "counts": dict(Counter(item["status"] for item in queue.state["tasks"].values()
                                    if item["task"]["split"] == "val_seen")), "finished_at": finished}
    write_json(output / "parallel/exit.json", state)
    write_json(output / "parallel/status.json", {**state, "workers": workers, "lanes": {},
               "trial_passed": gate, "host_pid": host_pid, "updated_at": finished})


def run_controller(args):
    """One queue writer and one supervised host, with technical trial gating."""
    output = args.output.resolve()
    manifest = read_json(output / "manifest.json")
    verify_frozen(output, manifest)
    if (output / "PAUSE").exists():
        raise RuntimeError("Campaign has a pending manual pause")
    if not (output / "pilot_report_delivered.json").exists():
        raise RuntimeError("The pilot report must be delivered before full seen")
    tasks = [task for task in manifest["tasks"] if task["split"] == "val_seen"]
    with campaign_lock(output):
        overlay = getattr(args, "_overlay", None) or prepare_extension(output, manifest)
        args._overlay = overlay
        globals()["_FROZEN_EXTENSION_DIR"] = Path(overlay["controller_script"]).parent
        verify_overlay(overlay, manifest)
        if discover_gpu_bindings(manifest["configs"]["val_seen"]["map_config_path"]) != overlay["remote_gpu_bindings"]:
            raise RuntimeError("Remote GPU UUID or Vulkan mapping changed since freeze")
        offline_models(output, overlay)
        queue = retry_file_access(lambda: ParallelRecoveryQueue(
            output / "progress.json", manifest["tasks"], manifest["fingerprint"],
            save_retry=lambda operation: retry_file_access(operation, path=output / "progress.json", action="write")),
            path=output / "progress.json")
        # Recovery kills only the previous shared host, once, before examining
        # each lane independently. Never use serial _cleanup on a live host.
        previous = read_json(output / "parallel/host_process.json")
        if previous:
            kill_owned(previous)
        def recover(task, attempt):
            process = read_json(_attempt_dir(output, task, attempt["number"]) / "process.json")
            if not process and attempt.get("session"):
                process = {**previous, "descendants": read_json(
                    Path(attempt["session"]) / "host_status.json").get("descendants", [])}
            return finish_attempt(output, manifest, task, attempt,
                                  process,
                                  reason="host_interrupted")
        if not queue.recover_leases(recover):
            return
        repair_pending_reports(output, manifest, queue)
        coverage_check(output, manifest, queue)
        trial_path = output / "parallel/trial_selection.json"
        selection = read_json(trial_path)
        if not selection:
            selection = {"tasks": trial_tasks(tasks, queue.state), "started_at": utc_now()}
            write_json(trial_path, selection)
        trial = selection["tasks"]
        gate_record = read_json(output / "parallel/trial_check.json")
        gate = gate_record.get("passed", False) and gate_record.get("controller_revision") == overlay["controller_revision"]
        snapshots = read_json(output / "parallel/trial_snapshots.json").get("snapshots", [])
        if not storage_budget(output, queue, tasks, args.workers):
            queue.pause("storage_budget")
            return
        session = output / "parallel/sessions" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        session.mkdir(parents=True)
        command = [sys.executable, "-u", "-X", "utf8", "-X", "faulthandler", overlay["controller_script"], "host",
                   "--output", str(output), "--session", str(session), "--workers", str(args.workers),
                   "--parent-pid", str(os.getpid()), "--parent-created", str(psutil.Process().create_time())]
        stdout = (session / "stdout.log").open("w", encoding="utf-8")
        stderr = (session / "stderr.log").open("w", encoding="utf-8")
        child = subprocess.Popen(command, cwd=str(output / "source"), stdout=stdout, stderr=stderr,
                                 env={**os.environ, **overlay["host_environment"]},
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        host = owner_of(psutil.Process(child.pid))
        write_json(output / "parallel/host_process.json", {**host, "session": str(session)})
        write_json(output / "dispatcher.json", {**owner_of(psutil.Process()), "stage": "seen",
                   "mode": "parallel", "workers": args.workers, "interpreter": sys.executable})
        queue.state.pop("pause_cause", None)
        queue.state.update(status="running", reason=None, execution_mode="parallel", workers=args.workers)
        queue.save()
        active, completed, service_due = {}, 0, False
        gpu_samples = read_json(output / "parallel/gpu_samples.json").get("samples", [])
        last_gpu_check = 0
        fallback = False
        rate_limits = 0
        started = time.monotonic()
        initial_scored = sum(item["status"] == "scored" and item["task"]["split"] == "val_seen"
                             for item in queue.state["tasks"].values())
        fatal = None
        host_was_ready = False
        try:
            while True:
                host_status = read_json(session / "host_status.json")
                descendants = host_status.get("descendants", [])
                now = time.monotonic()
                host_was_ready = host_was_ready or (session / "host_environment.json").exists()
                for lane in list(active):
                    request = active[lane]
                    directory = _attempt_dir(output, request["task"], request["attempt"]["number"])
                    fact = read_json(session / "results" / (request["id"] + ".json"))
                    process = {**host, "descendants": descendants, "lane": lane,
                               "status": "running", "elapsed_seconds": now - request["started"]}
                    write_json(directory / "process.json", process)
                    if fact:
                        if fact["task"] != request["task"] or fact["attempt"] != request["attempt"]:
                            raise ValueError("Host receipt lease identity mismatch")
                        process.update(elapsed_seconds=fact["elapsed_seconds"], exit_code=0)
                        if fact.get("exception"):
                            write_json(directory / "worker_exception.json", fact["exception"])
                        receipt = finish_attempt(output, manifest, request["task"], request["attempt"], process)
                        action = queue.accept(request["task"], request["attempt"]["number"], receipt)
                        snapshots.append(fact["shared_models"])
                        write_json(output / "parallel/trial_snapshots.json", {"snapshots": snapshots[-8:]})
                        completed += 1
                        del active[lane]
                        _render_reports(output, queue)
                        print(json.dumps({"episode": queue.key(request["task"]), "lane": lane,
                              "attempt": request["attempt"]["number"], "action": action}), flush=True)
                        if action == "service_check":
                            service_due = True
                        if action == "pause":
                            fatal = queue.state["reason"]
                        exception = json.dumps(fact.get("exception") or {}).lower()
                        if args.workers == 2 and "out of memory" in exception and "cuda" in exception:
                            fallback = True
                            fatal = "parallel_cuda_memory"
                        provider = receipt["result"].get("infrastructure_error") or {}
                        provider_error = provider.get("error") or {}
                        if provider_error.get("status_code") == 429:
                            rate_limits += 1
                        elif action == "scored":
                            rate_limits = 0
                        if args.workers == 2 and rate_limits >= 2:
                            fallback = True
                            fatal = "parallel_rate_limit"
                        continue
                    activity = activity_time(directory)
                    if activity != request.get("activity"):
                        request.update(activity=activity, progress_at=now)
                    fatal = fatal or watchdog_reason(request, now, args.episode_deadline, args.no_progress)
                if (output / "PAUSE").exists():
                    fatal = "manual_interrupt"
                if child.poll() is not None:
                    fatal = fatal or "host_exited"
                    error_text = read_process_diagnostic(session / "stderr.log").lower()
                    if args.workers == 2 and "cuda" in error_text and "out of memory" in error_text:
                        fallback = True
                        fatal = "parallel_cuda_memory"
                if now - started > 900 and not (session / "host_environment.json").exists():
                    fatal = "host_startup_timeout"
                if fatal:
                    break
                if active and now - last_gpu_check >= (60 if gate else 15):
                    try:
                        sample = remote_gpu_sample(manifest, output, active, overlay)
                        if sample:
                            gpu_samples.append(sample)
                            write_json(output / "parallel/gpu_samples.json",
                                       {"samples": gpu_samples[-20:] if gate else gpu_samples})
                            if gpu_violation(sample):
                                fatal = "unauthorized_remote_gpu"
                                write_json(output / "parallel/gpu_violation.json", sample)
                    except Exception as exc:
                        write_json(output / "parallel/gpu_sample_error.json",
                                   {"error": str(exc), "checked_at": utc_now()})
                    last_gpu_check = now
                if fatal:
                    break
                if service_due and not active:
                    if not service_recovery(output, manifest, queue):
                        fatal = queue.state["reason"]
                        break
                    service_due = False
                if completed >= 25:
                    if not storage_budget(output, queue, tasks, args.workers):
                        fatal = "storage_or_coverage_check"
                        break
                    verify_overlay(overlay, manifest)
                    coverage_check(output, manifest, queue, started=started, initial_scored=initial_scored)
                    repair_pending_reports(output, manifest, queue)
                    completed = 0
                if not gate and all(queue.state["tasks"][queue.key(task)]["status"] == "scored" for task in trial):
                    repair_pending_reports(output, manifest, queue)
                    check = validate_trial(output, queue, trial, snapshots, gpu_samples=gpu_samples, workers=args.workers)
                    check["elapsed_seconds"] = now - started
                    check["controller_revision"] = overlay["controller_revision"]
                    write_json(output / "parallel/trial_check.json", check)
                    if not check["passed"]:
                        fatal = "parallel_technical_check"
                        break
                    gate = True
                    print(json.dumps({"parallel_trial_passed": True, "episodes": len(trial)}), flush=True)
                if gate:
                    candidates = tasks
                else:
                    # Two technical waves, rather than allowing the third
                    # trial to begin before both first-wave lanes finish.
                    waves = [trial[index:index + 2] for index in range(0, len(trial), 2)]
                    candidates = next((wave for wave in waves if any(
                        queue.state["tasks"][queue.key(task)]["status"] != "scored" for task in wave)), [])
                if (session / "host_environment.json").exists() and not service_due:
                    for lane in ("A", "B")[:args.workers]:
                        if lane in active:
                            continue
                        info = overlay["lanes"][lane]
                        lease = queue.claim(candidates, lane, metadata={
                            "map_config_path": info["map_config_path"], "local_port": info["local_port"],
                            "remote_port": info["remote_port"], "remote_gpu": info["remote_gpu"], "session": str(session)})
                        if not lease:
                            continue
                        task, attempt = lease
                        request = {"id": uuid.uuid4().hex, "lane": lane, "task": task,
                                   "attempt": attempt, "map_config_path": info["map_config_path"]}
                        write_json(session / "commands" / (lane + ".json"), request)
                        active[lane] = {**request, "started": now, "progress_at": now}
                counts = dict(Counter(item["status"] for item in queue.state["tasks"].values()
                                      if item["task"]["split"] == "val_seen"))
                write_json(output / "parallel/status.json", {"status": queue.state["status"],
                    "reason": queue.state["reason"], "counts": counts, "workers": args.workers,
                    "lanes": {lane: request["task"] for lane, request in active.items()},
                    "trial_passed": gate, "host_pid": child.pid, "updated_at": utc_now()})
                if not active and (session / "host_environment.json").exists():
                    if all(queue.state["tasks"][queue.key(task)]["status"] == "scored" for task in tasks):
                        queue.state.update(status="seen_complete", reason=None)
                        queue.save()
                        break
                    if not any(queue.state["tasks"][queue.key(task)]["status"] in {"pending", "retry_pending"}
                               for task in candidates):
                        fatal = "episode_attempts_exhausted"
                        break
                time.sleep(1)
        except KeyboardInterrupt:
            fatal = "manual_interrupt"
        except Exception as exc:
            fatal = "controller_exception"
            write_json(output / "parallel/controller_error.json", {
                "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        finally:
            if fatal:
                kill_owned(host)
            else:
                atomic_write_text(session / "STOP", "Controller finished seen.\n")
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    kill_owned(host)
            child.wait(timeout=15)
            stdout.close()
            stderr.close()
            descendants = read_json(session / "host_status.json").get("descendants", [])
            for request in active.values():
                receipt = finish_attempt(output, manifest, request["task"], request["attempt"],
                    {**host, "descendants": descendants, "elapsed_seconds": time.monotonic() - request["started"]},
                    reason=fatal or "host_interrupted")
                queue.accept(request["task"], request["attempt"]["number"], receipt)
            if fatal:
                queue.pause("cleanup_unverified" if queue.state.get("reason") == "cleanup_unverified" else fatal)
            _render_reports(output, queue)
            repair_pending_reports(output, manifest, queue)
            coverage_check(output, manifest, queue, started=started, initial_scored=initial_scored)
            write_exit_status(output, queue, workers=args.workers, gate=gate, host_pid=child.pid)
            print(json.dumps({"status": queue.state["status"], "reason": queue.state["reason"]}), flush=True)
        clean = queue.state.get("reason") not in {"cleanup_unverified", "interrupted_cleanup_unverified"}
        if fallback and clean and not (output / "PAUSE").exists():
            write_json(output / "parallel/fallback.json", {
                "reason": fatal, "previous_workers": args.workers, "workers": 1, "created_at": utc_now()})
            return {"workers": 1, "reason": fatal}
        if fatal in {"host_exited", "no_progress", "episode_deadline"} and host_was_ready and clean \
                and not (output / "PAUSE").exists():
            return {"workers": args.workers, "reason": fatal}


def completed_boundary(output, state):
    """Find the exited serial child while its parent is finishing cleanup."""
    active = [item for item in state["tasks"].values() if item["status"] == "running"]
    if len(active) != 1:
        return None
    item = active[0]
    directory = _attempt_dir(output, item["task"], item["attempts"][-1]["number"])
    process = read_json(directory / "process.json")
    if process.get("exit_code") != 0 or process.get("reason"):
        return None
    if not valid_result(attempt_result(directory, item["task"])):
        return None
    return item


def handoff(args):
    """Briefly suspend only our dispatcher to prove a safe no-flight boundary."""
    output = args.output.resolve()
    if (output / "PAUSE").exists():
        raise RuntimeError("An existing manual pause must be preserved")
    dispatcher = read_json(output / "dispatcher.json")
    parent = psutil.Process(dispatcher["pid"])
    owner = owner_of(parent)
    if "evaluation.benchmarks.aerial_vln.campaign" not in owner["command"] or "run" not in owner["command"] \
            or str(output) not in owner["command"]:
        raise ValueError("Only this campaign's serial dispatcher may be handed off")
    observer_info = read_json(output / "handoff_control.json")
    observer = psutil.Process(observer_info["pid"])
    observer_owner = owner_of(observer)
    if str(output) not in observer_owner["command"] or not any(
            Path(arg).name == "aerial_vln_pilot_handoff.py" for arg in observer_owner["command"]):
        raise ValueError("Observer ownership mismatch")
    observer.terminate()
    observer.wait(timeout=15)
    token = "Parallel handoff " + uuid.uuid4().hex
    print(json.dumps({"handoff_waiting": True, "dispatcher_pid": parent.pid}), flush=True)
    while matching_process(owner) is not None:
        state = read_json(output / "progress.json")
        item = completed_boundary(output, state)
        if item is None:
            if state.get("status") == "paused":
                raise RuntimeError("Serial campaign paused before a safe handoff")
            time.sleep(0.02)
            continue
        parent = matching_process(owner)
        if parent is None:
            break
        parent.suspend()
        try:
            # Suspending this parent cannot suspend the independent child.
            # If a new flight was already spawned, resume and wait for it.
            live_flights = []
            for child in parent.children(recursive=True):
                try:
                    command = child.cmdline()
                    if "evaluation.benchmarks.aerial_vln.campaign" in command and "child" in command:
                        live_flights.append(child.pid)
                except psutil.NoSuchProcess:
                    pass
            fresh = completed_boundary(output, read_json(output / "progress.json"))
            if live_flights or fresh is None or fresh["task"] != item["task"]:
                continue
            atomic_write_text(output / "PAUSE", token + "\n")
            write_json(output / "parallel/handoff.json", {"token": token, "dispatcher": owner,
                       "observer": observer_owner, "boundary_episode": item["task"], "requested_at": utc_now()})
        finally:
            parent.resume()
        break
    while matching_process(owner) is not None:
        time.sleep(0.2)
    with campaign_lock(output):
        state = read_json(output / "progress.json")
        if any(item["status"] == "running" for item in state["tasks"].values()):
            raise RuntimeError("Serial task remained running at handoff")
        if state.get("reason") not in {None, "manual_interrupt"}:
            raise RuntimeError("Serial cleanup did not finish successfully")
        marker = output / "PAUSE"
        if not marker.exists() or marker.read_text(encoding="utf-8").strip() != token:
            raise RuntimeError("Handoff pause marker changed; preserve it")
        marker.unlink()
        write_json(output / "parallel/handoff_ready.json", {"verified": True, "completed_at": utc_now(),
                   "seen_scored": sum(item["status"] == "scored" and item["task"]["split"] == "val_seen"
                                      for item in state["tasks"].values())})
    print(json.dumps({"handoff_ready": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "host", "handoff", "watch"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--parent-created", type=float)
    parser.add_argument("--episode-deadline", type=float, default=7200)
    parser.add_argument("--no-progress", type=float, default=900)
    args = parser.parse_args()
    if args.command == "watch":
        _extension("parallel_watchdog.py", "evaluation/benchmarks/aerial_vln/parallel_watchdog.py").run_watchdog(
            args, sys.modules[__name__])
    elif args.command == "host":
        host_main(args)
    elif args.command == "handoff":
        handoff(args)
    else:
        recoveries = 0
        while True:
            recovery = run_controller(args)
            if not recovery:
                break
            recoveries += 1
            if recoveries > 3:
                break
            args.workers = recovery["workers"]
            # A bounded host restart remains interruptible. Episode attempts
            # are still independently capped at three by the durable queue.
            delay = (30, 120, 300)[recoveries - 1]
            until = time.monotonic() + delay
            while time.monotonic() < until:
                if (args.output / "PAUSE").exists():
                    return
                time.sleep(min(1, max(0, until - time.monotonic())))


if __name__ == "__main__":
    main()
