"""Run frozen benchmark campaigns on two isolated threads with shared resources.

The external controller is the sole progress writer and scene-cleanup owner.
A supervised host process runs isolated Runtime/Provider/Transcript objects in
threads; only capability model instances are shared through their existing
cache and CUDA gate. Rendering happens in the controller, outside the threads.
Remote simulators use GPU 0 and GPU 2; GPU 1 is explicitly excluded.

Commands: prepare freezes selected cases; validate checks selections offline;
run completes configured stages after a technical trial; watch supervises the
pinned controller with persistent restart backoff. host is a private entry.
handoff retains the dataset-owned historical serial handoff. PAUSE is the manual
interruption control. The frozen navigation source stays authoritative.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
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

DEFAULT_OUTPUT = Path("G:/FlyAgent-Eval/benchmark_campaign")
SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4] if SCRIPT.name == "parallel.py" else SCRIPT.parent

# Bootstrap before importing any project module. A real run must load Runtime,
# prompts, plugins and evaluation from the original frozen source, even when
# this external launcher lives in the working repository.
if __name__ == "__main__":
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("command", nargs="?")
    early.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    bootstrap, _ = early.parse_known_args()
    if bootstrap.command in {"run", "host", "watch"} or (SCRIPT.parent / "campaign").is_dir():
        root = bootstrap.output.resolve()
        overlay_path = root / "parallel/execution_overlay.json"
        execution = json.loads(overlay_path.read_text(encoding="utf-8")) if overlay_path.exists() else {}
        frozen = Path(execution.get("execution_source", root / "source")).resolve()
        if root not in frozen.parents:
            raise RuntimeError("Execution source must belong to this campaign")
        for relative, digest in execution.get("execution_source_hashes", {}).items():
            target = (frozen / relative).resolve()
            if frozen not in target.parents or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Execution source hash mismatch: " + relative)
        if not frozen.is_dir():
            raise RuntimeError("A prepared frozen campaign is required")
        sys.path.insert(0, str(frozen))
        os.chdir(frozen)
    else:
        sys.path.insert(0, str(REPO))

    # A controller snapshot contains its own evaluation-only shared package.
    # Install that namespace after selecting frozen source and before importing
    # any flight code. This also supports campaigns frozen before this package
    # existed, without replacing Runtime, provider or metric implementations.
    package_dir = SCRIPT.parent / "campaign"
    if package_dir.is_dir():
        frozen_overlay = json.loads((bootstrap.output.resolve() / "parallel/execution_overlay.json").read_text(encoding="utf-8"))
        for relative, expected in frozen_overlay["source_hashes"].items():
            if relative.startswith(("campaign/", "adapters/")) and hashlib.sha256((SCRIPT.parent / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen campaign helper changed: " + relative)
        import importlib
        importlib.import_module("evaluation.benchmarks.core")
        package_name = "evaluation.benchmarks.core.campaign"
        package_spec = importlib.util.spec_from_file_location(package_name, package_dir / "__init__.py", submodule_search_locations=[str(package_dir)])
        package_module = importlib.util.module_from_spec(package_spec)
        sys.modules[package_name] = package_module
        package_spec.loader.exec_module(package_module)
        sys.modules[package_name + ".parallel"] = sys.modules[__name__]
        package_module.parallel = sys.modules[__name__]
        for relative in frozen_overlay["source_hashes"]:
            if not relative.startswith("adapters/") or not relative.endswith(".py"):
                continue
            dataset = Path(relative).stem
            parent_name = "evaluation.benchmarks." + dataset
            parent_module = importlib.import_module(parent_name)
            module_name = parent_name + ".campaign_adapter"
            spec = importlib.util.spec_from_file_location(module_name, SCRIPT.parent / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            parent_module.campaign_adapter = module

from evaluation.benchmarks.core.campaign.support import (
    _attempt_dir, campaign_lock, environment_diagnostics, provider_probe, storage_check,
)
from evaluation.benchmarks.core.campaign.contracts import task_config, primary_config
from evaluation.benchmarks.core.campaign import compat
from evaluation.benchmarks.core.campaign.recovery import ParallelRecoveryQueue
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
    return (type((result.get("metrics") or {}).get("success")) in {bool, int, float}
            and not result.get("infrastructure_error") and not result.get("metric_error")
            and not ParallelRecoveryQueue.infrastructure_record(result)
            and result.get("final_score_available") is not False)


def campaign_tasks(manifest):
    """Select the declared stage; legacy stage conventions stay in compat."""
    return compat.selected_tasks(manifest)




def install_report_readers(overlay, manifest):
    """Bind only hashed evaluation readers; keep frozen flight imports intact.

    Failed historical audits may need a reader repair after flights have begun.
    A separately frozen reporting extension permits this without replacing any
    Runtime, provider, prompt, tool, knowledge or dataset configuration.
    """
    readers = compat.report_readers(manifest)
    if not readers:
        return
    import importlib

    directory = Path(overlay["controller_script"]).parent
    for filename, name in readers.items():
        if not name.startswith("evaluation."):
            raise ValueError("Report extensions must be evaluation modules")
        path = directory / filename
        expected = overlay["source_hashes"].get(filename)
        if not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Unverified reporting extension: " + filename)
        package_name, attribute = name.rsplit(".", 1)
        package = importlib.import_module(package_name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        setattr(package, attribute, module)


def _render_reports(output, queue):
    return compat.render_report(output, read_json(output / "manifest.json"), queue)


def render_reports(output, queue):
    """Report failures stay pending and cannot interrupt the mixed task queue."""
    try:
        result = _render_reports(output, queue)
    except Exception as exc:
        write_json(output / "parallel/report_error.json", {"pending": True,
                   "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc(), "updated_at": utc_now()})
        return None
    write_json(output / "parallel/report_error.json", {"pending": False, "updated_at": utc_now()})
    return result


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
    if load_model_config(primary_config(manifest)["env_path"]).redacted_summary() != manifest["model"]:
        raise ValueError("Effective model configuration changed")
    for key, value in read_env_file(primary_config(manifest)["env_path"]).items():
        if key.startswith("FLYAGENT_"):
            os.environ.setdefault(key, value)
    os.environ.update(TEMP=str(output / "temp"), TMP=str(output / "temp"))
    (output / "temp").mkdir(exist_ok=True)


def prepare_extension(output, manifest, *, dataset_order=None):
    """Freeze the additive controller, queue and read-only recovery helper."""
    existing_overlay = read_json(output / "parallel/execution_overlay.json")
    files = {
        "parallel_campaign.py": SCRIPT,
        "parallel_recovery.py": Path(sys.modules[ParallelRecoveryQueue.__module__].__file__),
        "startup_recovery.py": SCRIPT.with_name("startup_recovery.py")
        if SCRIPT.with_name("startup_recovery.py").exists()
        else REPO / "src/sim_server/remote/recovery.py",
    }
    package_dir = Path(compat.__file__).parent
    files["parallel_watchdog.py"] = package_dir / "watchdog.py"
    files.update({"campaign/" + path.name: path for path in package_dir.glob("*.py")})
    datasets = {compat.dataset_name(manifest, task) for task in manifest.get("tasks", [{}])}
    for dataset in datasets:
        adapter = compat.adapter_for(manifest, {"dataset": dataset})
        files["adapters/" + dataset + ".py"] = Path(sys.modules[type(adapter).__module__].__file__)
    for name, path in compat.extra_files(manifest, REPO).items():
        files[name] = SCRIPT.with_name(name) if SCRIPT.with_name(name).is_file() else Path(path)
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    revision = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = output / "parallel/controller_source" / revision[:16]
    destination.mkdir(parents=True, exist_ok=True)
    for name, path in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError("Parallel controller snapshot changed")
        if not target.exists():
            target.write_bytes(path.read_bytes())
    maps = read_json(primary_config(manifest)["map_config_path"])
    bindings = discover_gpu_bindings(primary_config(manifest)["map_config_path"])
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
               "controller_script": str(destination / "parallel_campaign.py"),
               "forbidden_remote_gpus": [1], "local_perception_device": "cuda:0",
               "lanes": lanes, "html_profile": compat.report_profile(manifest, primary_config(manifest)),
               "thread_html_rendering": False, "created_at": utc_now()}
    logging = compat.logging_settings(manifest)
    overlay["logging_profile"] = logging["profile"]
    overlay["logging_adapter"] = logging.get("adapter")
    field, order = compat.group_policy(manifest)
    if field:
        overlay["scheduling_policy"] = {"mode": "budget_sequential", "group_field": field, "budget_order": order,
                                        "technical_trial_scope": "each_budget", "promotion_requires": "all_scored"}
        # Operator scheduling changes live outside the frozen navigation manifest.
        # Refreezing readers must preserve an already selected dataset order.
        if dataset_order is None:
            dataset_order = read_json(output / "parallel/execution_overlay.json").get("scheduling_policy", {}).get("dataset_order")
        if dataset_order:
            if len(set(dataset_order)) != len(dataset_order) or set(dataset_order) != datasets:
                raise ValueError("Dataset order must cover the campaign datasets exactly once")
            mode = existing_overlay.get("scheduling_policy", {}).get("mode", "budget_dataset_sequential")
            if mode != "dataset_budget_sequential":
                mode = "budget_dataset_sequential"
            overlay["scheduling_policy"].update(mode=mode, dataset_order=list(dataset_order),
                                                  technical_trial_scope="each_budget_dataset")
        overlay["scheduling_policy_sha256"] = hashlib.sha256(json.dumps(overlay["scheduling_policy"], sort_keys=True).encode()).hexdigest()
    for name in ("execution_source", "execution_source_hashes", "execution_source_sha256", "recovery_revision",
                 "recovery_policy", "recovery_validation_required"):
        if name in existing_overlay:
            overlay[name] = existing_overlay[name]
    write_json(output / "parallel/execution_overlay.json", overlay)
    return overlay


def verify_overlay(overlay, manifest):
    if overlay["base_fingerprint"] != manifest["fingerprint"] or overlay["forbidden_remote_gpus"] != [1]:
        raise ValueError("Execution overlay does not belong to this campaign")
    if overlay.get("scheduling_policy_sha256") and hashlib.sha256(json.dumps(
            overlay["scheduling_policy"], sort_keys=True).encode()).hexdigest() != overlay["scheduling_policy_sha256"]:
        raise ValueError("Frozen scheduling policy changed")
    for lane, info in overlay["lanes"].items():
        path = Path(info["map_config_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError("Lane configuration changed")
        expected = lane_catalog(read_json(primary_config(manifest)["map_config_path"]), lane,
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
    if not visual_grounding_enabled(output / "source"):
        overlay["model_cache"] = []
        overlay["host_environment"] = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        write_json(output / "parallel/execution_overlay.json", overlay)
        return
    from huggingface_hub.constants import HF_HUB_CACHE
    settings = read_json(output / "source/plugins/capabilities/visual_grounding/measure_visible_depth/capability.json")["settings"]
    actual = cached_models(settings, HF_HUB_CACHE)
    if overlay.get("model_cache") and overlay["model_cache"] != actual:
        raise RuntimeError("Local perception model weights changed since parallel freeze")
    overlay["model_cache"] = actual
    overlay["host_environment"] = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    write_json(output / "parallel/execution_overlay.json", overlay)


def trial_tasks(tasks, state):
    """Choose untouched cases representing the available difficulty labels."""
    pending = [task for task in tasks if state["tasks"][ParallelRecoveryQueue.key(task)]["status"] == "pending"]
    chosen = []
    for label in dict.fromkeys(task.get("difficulty") for task in pending):
        found = next(task for task in pending if task.get("difficulty") == label)
        chosen.append(found)
    chosen.extend(task for task in pending if task not in chosen)
    return chosen[:4]


def visual_grounding_enabled(source=None):
    """External preload follows the frozen capability configuration."""
    from src.plugins import PluginRegistry
    config = (Path(source) if source else Path.cwd()) / "configs/plugins.json"
    return any(c.call_name == "skill.visual_grounding.measure_visible_depth"
               for c in PluginRegistry.from_config(config).skill_capability_catalog.capabilities)


def model_snapshot():
    """Human-only model identity/memory evidence, never added to model inputs."""
    import torch
    module = sys.modules.get("plugins.capabilities.visual_grounding.measure_visible_depth.scripts.service")
    services, identities = [], []
    if module is not None:
        with module._SERVICE_CACHE_LOCK:
            services = list(module._SERVICE_CACHE.values())
    if services:
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
    if not visual_grounding_enabled():
        return model_snapshot()
    from plugins.capabilities.visual_grounding.measure_visible_depth.scripts.service import service_from_settings
    capability = Path("plugins/capabilities/visual_grounding/measure_visible_depth/capability.json")
    service = service_from_settings(read_json(capability)["settings"])
    service.grounding_backend._load()
    if service.segmentation_backend is not None:
        service.segmentation_backend._load()
    return model_snapshot()


def execute_thread(output, manifest, request, environment):
    """Execute through the dataset adapter with an isolated Runtime/run tree."""
    task, attempt = request["task"], request["attempt"]
    directory = _attempt_dir(output, task, attempt["number"])
    write_json(directory / "execution_phase.json", {"phase": "bootstrap", "at": utc_now()})
    write_json(directory / "environment.json", {**environment, "lane": request["lane"], "host_pid": os.getpid()})
    data = deepcopy(task_config(manifest, task))
    data["map_config_path"] = request["map_config_path"]
    if manifest.get("context_configs") and "image_budget" in task:
        data["context_config_path"] = manifest["context_configs"][str(task["image_budget"])]
    data.update(deepcopy(manifest.get("configuration_overrides", {}).get(task.get("configuration_id"), {})))
    from src.perception.inference_resources import inference_observer
    def observe(event, payload):
        with (directory / "inference_resources.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"timestamp": utc_now(), "event": event, **payload}) + "\n")
    from contextlib import nullcontext
    overlay = read_json(output / "parallel/execution_overlay.json")
    scope = nullcontext()
    if overlay.get("recovery_policy"):
        from src.llm.recovery import RecoveryPolicy, recovery_scope
        began = time.monotonic()
        def progress(value):
            write_json(directory / "provider_recovery.json", {**value, "updated_at": utc_now(),
                       "recovery_revision": overlay["recovery_revision"]})
        scope = recovery_scope(RecoveryPolicy(**overlay["recovery_policy"], progress=progress,
            cancelled=lambda: (output / "PAUSE").exists() or
            time.monotonic()-began >= request.get("episode_deadline", 7200)))
    from src.sim_server.remote.ssh import ssh_observer
    def ssh_progress(value):
        with (directory / "ssh_operations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**value, "timestamp": utc_now()}) + "\n")
    with inference_observer(observe), scope, ssh_observer(ssh_progress):
        write_json(directory / "execution_phase.json", {"phase": "navigation", "at": utc_now()})
        result = compat.adapter_for(manifest, task).execute(data, task, directory)
    result["recovery_revision"] = overlay.get("recovery_revision", "original")
    # The adapter validates the original instruction/truth before executing.
    # This receipt records only evidence ownership for later technical checks.
    instruction_hash = task.get("instruction_sha256")
    if instruction_hash is None and result.get("run_dir"):
        history = Path(result["run_dir"]) / "memory/agent_transcript.jsonl"
        instructions = [json.loads(line)["payload"].get("instruction", "") for line in history.read_text(encoding="utf-8").splitlines()
                        if line.strip() and json.loads(line).get("kind") == "user_instruction"] if history.exists() else []
        instruction_hash = hashlib.sha256(instructions[0].encode("utf-8")).hexdigest() if len(instructions) == 1 else None
    write_json(directory / "thread_evidence.json", {"episode_id": task["episode_id"], "lane": request["lane"],
        "instruction_sha256": instruction_hash, "runtime_source": str(Path(sys.modules["src.runtime.factory"].__file__).resolve()),
        "image_budget": task.get("image_budget"), "thread_finished": True})
    write_json(directory / "result.json", result)
    return result


def host_main(args):
    """Persistent two-thread host; a dead controller cannot leave it running."""
    output, session = args.output.resolve(), args.session.resolve()
    manifest = read_json(output / "manifest.json")
    overlay = read_json(output / "parallel/execution_overlay.json")
    verify_frozen(output, manifest)
    verify_overlay(overlay, manifest)
    if overlay.get("recovery_policy"):
        from src.llm.recovery import RecoveryPolicy
        from src.sim_server.remote.ssh import ssh_observer, ssh_deadline
        import src.llm.client
        expected_source = Path(overlay["execution_source"]).resolve()
        if expected_source not in Path(src.llm.client.__file__).resolve().parents:
            raise RuntimeError("Recovery runtime source mismatch before dispatch")
    if overlay.get("logging_adapter"):
        logging_adapter = _extension(overlay["logging_adapter"], "evaluation/experiments/compact_logging.py")
        logging_adapter.install()
    environment = environment_diagnostics()
    if not environment["cuda_available"]:
        raise RuntimeError("CUDA unavailable in parallel host")
    environment["shared_models"] = warm_perception()
    write_json(session / "host_environment.json", environment)
    parent = {"pid": args.parent_pid, "created": args.parent_created}
    processed, futures, descendants = set(), {}, {}
    last_status = 0
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="benchmark-campaign") as pool:
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
                    if request["lane"] != lane or request["task"] not in campaign_tasks(manifest):
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
    map_path = attempt.get("map_config_path", task_config(manifest, task)["map_config_path"])
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
                map_id = task_config(manifest, task)["scene_map"][task["scene_id"]]
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
    with (directory / "cleanup_history.jsonl").open("a", encoding="utf-8") as history:
        history.write(json.dumps({"checked_at": utc_now(), **read_json(directory / "cleanup.json")}) + "\n")
    valid = valid_result(result)
    # A durable terminal result survives host death; incomplete TLS trajectories
    # have infrastructure metadata and cannot satisfy valid_result.
    process = {**process, "reason": None if valid else reason, "status": "finished"}
    write_json(directory / "process.json", process)
    case_dir = directory / "evaluation/cases" / task["episode_id"]
    terminal_provider = manifest.get("terminal_provider_errors", False) and bool(ParallelRecoveryQueue.provider_error(result))
    if (valid or terminal_provider) and verified:
        try:
            result["report"] = write_process_review(case_dir, result,
                profile=compat.report_profile(manifest, task_config(manifest, task)))
            compat.attach_review(case_dir / "review.html", result, manifest)
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
    heartbeat = directory / "provider_recovery.json"
    latest = heartbeat.stat().st_mtime_ns if heartbeat.exists() else 0
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
                profile=compat.report_profile(manifest, task_config(manifest, task)))
            compat.attach_review(case_dir / "review.html", result, manifest)
            result.pop("report_error", None)
            write_json(case_dir / "case_result.json", result)
            write_json(directory / "result.json", result)
            queue.save()
        except Exception as exc:
            pending.append({"episode": queue.key(task), "error": str(exc)})
    write_json(output / "parallel/report_pending.json", {"pending": pending, "checked_at": utc_now()})


def storage_budget(output, queue, tasks, workers):
    overlay = read_json(output / "parallel/execution_overlay.json")
    if overlay.get("logging_profile") == "paper_process_metrics_v1":
        # Historical full-debug attempts are already included in disk usage;
        # they do not predict future retention under the new logging profile.
        attempts = [a for v in queue.state["tasks"].values() for a in v["attempts"]
                    if a.get("logging_profile") == overlay["logging_profile"] and a.get("retained_bytes")]
        all_remaining = sum(not queue.terminal(queue.state["tasks"][queue.key(t)]) for t in tasks)
        stage, storage_tasks = (configuration_stage(manifest=read_json(output / "manifest.json"), tasks=tasks,
                                                    queue=queue, policy=overlay["scheduling_policy"])
                                if overlay.get("scheduling_policy", {}).get("dataset_order") else (None, tasks))
        remaining = sum(not queue.terminal(queue.state["tasks"][queue.key(t)]) for t in storage_tasks)
        per_case = max((a["retained_bytes"] for a in attempts), default=0)
        peak = max((a.get("temporary_depth_bytes", 0) for a in attempts), default=0)
        required = max(10 * 1024**3, remaining * per_case * 2 + workers * peak)
        free = shutil.disk_usage(output).free
        record = {"free_bytes": free, "required_bytes": required, "remaining_cases": remaining,
                  "scope": "configuration_stage" if stage is not None else "campaign", "configuration_stage": stage,
                  "total_remaining_cases": all_remaining,
                  "full_campaign_required_bytes": max(10 * 1024**3, all_remaining * per_case * 2 + workers * peak),
                  "observed_max_retained_bytes": per_case, "observed_cases": len(attempts),
                  "logging_profile": overlay["logging_profile"], "calibration_pending": not attempts,
                  "concurrent_attempts": workers, "passed": free >= required}
        write_json(output / "storage_check.json", record)
        return record["passed"]
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
              for split in manifest["counts"]}
    for split, number in manifest["counts"].items():
        if sum(counts[split].values()) != number:
            raise ValueError("Split episode count changed")
    sizes = [item["attempts"][-1].get("retained_bytes", 0)
             for item in queue.state["tasks"].values() if item["status"] == "scored"]
    elapsed = time.monotonic() - started if started is not None else 0
    scored = sum(queue.state["tasks"][queue.key(t)]["status"] == "scored" for t in campaign_tasks(manifest))
    throughput = (scored - initial_scored) * 3600 / elapsed if elapsed > 0 else None
    # Persist wall-clock checkpoints so ETA includes retry time and downtime.
    # A deliberate accounting correction starts a fresh measurement interval.
    stamp = time.time()
    history_path = output / "parallel/throughput_history.json"
    history = read_json(history_path).get("checkpoints", [])
    if history and scored < history[-1]["scored"]:
        history = []
    history = [point for point in history if stamp - point["timestamp"] <= 7200]
    history.append({"timestamp": stamp, "scored": scored})
    write_json(history_path, {"checkpoints": history})
    gained = scored - history[0]["scored"]
    wall_seconds = stamp - history[0]["timestamp"]
    rolling_rate = gained * 3600 / wall_seconds if gained >= 4 and wall_seconds > 0 else None
    stage, stage_tasks = configuration_stage(manifest, campaign_tasks(manifest), queue,
                                            read_json(output / "parallel/execution_overlay.json").get("scheduling_policy"))
    remaining_stage = sum(queue.state["tasks"][queue.key(t)]["status"] != "scored" for t in stage_tasks)
    write_json(output / "parallel/progress_check.json", {
        "counts": counts, "checked_at": utc_now(), "elapsed_seconds": elapsed,
        "valid_episodes_per_hour": throughput, "retained_bytes": sum(sizes),
        "mean_retained_bytes": sum(sizes) / len(sizes) if sizes else 0,
        "configuration_stage": stage, "recent_valid_episodes_per_hour": rolling_rate,
        "recent_window_seconds": wall_seconds, "recent_valid_episodes": gained,
        "stage_remaining_hours_estimate": remaining_stage / rolling_rate if rolling_rate else None,
        "remaining_hours_estimate": (len(campaign_tasks(manifest)) - scored) / rolling_rate if rolling_rate else None})
    compat.write_comparison(output, queue, manifest)
    render_reports(output, queue)


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
        if not queue.terminal(item) or not item["attempts"][-1].get("cleanup_verified"):
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
        manifest = read_json(output / "manifest.json")
        errors.extend(queue.key(task) + ":" + e for e in compat.check_requests(run_path, task, manifest))
        if manifest.get("request_audit_gate"):
            request_checks = _extension("request_audit.py", "tests/manual/aerial_vln_plan_context_acceptance.py").request_checks
            for path in run_path.glob("debug/tool_loop/*/decision_*.json"):
                for model_round in read_json(path).get("model_rounds", []):
                    issues, _ = request_checks(model_round, require_map=True,
                        plan_mode=manifest.get("plan_mode", "optional"), require_place_list=True)
                    errors.extend(queue.key(task) + ":" + issue for issue in issues)
    identities = [snapshot.get("services", []) for snapshot in snapshots]
    needs_models = visual_grounding_enabled(output / "source") if (output / "source/configs/plugins.json").exists() else True
    if not needs_models:
        if not identities or any(items for items in identities):
            errors.append("disabled_model_loaded")
    elif not identities or any(len(items) != 1 or not items[0]["grounding_model"]
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
        if provider_probe(primary_config(manifest)["env_path"],
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
    """Publish final coverage in the configured stage, with no stopped lanes."""
    finished = utc_now()
    manifest = read_json(output / "manifest.json")
    keys = {queue.key(t) for t in campaign_tasks(manifest)} if manifest.get("tasks") else set(queue.state["tasks"])
    state = {"status": queue.state["status"], "reason": queue.state["reason"],
             "counts": dict(Counter(item["status"] for key, item in queue.state["tasks"].items() if key in keys)), "finished_at": finished}
    write_json(output / "parallel/exit.json", state)
    write_json(output / "parallel/status.json", {**state, "workers": workers, "lanes": {},
               "trial_passed": gate, "host_pid": host_pid, "updated_at": finished})


def dispatch_candidates(manifest, candidates, lane, active, workers, queue):
    """Limit active tasks independently of available lane worker threads.

    A serial replay can keep both original GPU lanes available while admitting
    exactly one task, in manifest order. Default campaigns retain their policy.
    """
    limit = manifest.get("dispatch_concurrency", workers)
    if type(limit) is not int or not 1 <= limit <= workers:
        raise ValueError("Invalid dispatch concurrency")
    assignments = manifest.get("task_lane_assignments", {})
    if assignments and (set(assignments) != {t["episode_id"] for t in manifest["tasks"]}
                        or any(value not in ("A", "B")[:workers] for value in assignments.values())):
        raise ValueError("Invalid fixed task lane assignments")
    if len(active) >= limit:
        return []
    eligible = [task for task in candidates if not queue.terminal(queue.state["tasks"][queue.key(task)])]
    if limit == 1:
        eligible = eligible[:1]
    return [task for task in eligible if not assignments or assignments[task["episode_id"]] == lane]


def configuration_stage(manifest, tasks, queue, policy=None):
    """Advance a configuration group only after every task has a valid score."""
    field, order = compat.group_policy(manifest)
    if field is None:
        return None, tasks
    if not order or len(set(order)) != len(order) or set(order) != {t[field] for t in tasks}:
        raise ValueError("Configuration stages must cover every configured group exactly once")
    datasets = (policy or {}).get("dataset_order")
    if datasets and (len(set(datasets)) != len(datasets) or set(datasets) != {t["dataset"] for t in tasks}):
        raise ValueError("Dataset stages must cover the cohort exactly once")
    stages = [((value, dataset), [t for t in tasks if t[field] == value and t["dataset"] == dataset])
              for value in order for dataset in datasets] if datasets else [
                  (value, [t for t in tasks if t[field] == value]) for value in order]
    # Preserve stage identity and retry barriers; only change traversal priority.
    if datasets and (policy or {}).get("mode") == "dataset_budget_sequential":
        stages.sort(key=lambda stage: (datasets.index(stage[0][1]), order.index(stage[0][0])))
    if any(not stage for _, stage in stages):
        raise ValueError("Every configured stage requires tasks")
    for value, stage in stages:
        if any(queue.state["tasks"][queue.key(task)]["status"] != "scored" for task in stage):
            return value, stage
    return stages[-1]


image_budget_stage = configuration_stage


def recovery_candidates(output, overlay, candidates, queue):
    """Admit only two fixed cases until the new recovery revision is verified."""
    if not overlay.get("recovery_validation_required"):
        return candidates
    path = output / "parallel/recovery_validation.json"
    record = read_json(path)
    revision = overlay["recovery_revision"]
    if record.get("recovery_revision") == revision and record.get("passed"):
        return candidates
    selected = [item for item in queue.state["tasks"].values() if any(
        a.get("recovery_revision") == revision for a in item["attempts"])]
    if len(selected) >= 2:
        passed = all(item["status"] == "scored" and any(
            a.get("recovery_revision") == revision and a.get("valid_score") and a.get("cleanup_verified")
            for a in item["attempts"]) for item in selected)
        write_json(path, {"recovery_revision": revision, "passed": passed,
                         "tasks": [i["task"] for i in selected], "checked_at": utc_now()})
        return candidates if passed else [i["task"] for i in selected]
    return candidates


def technical_check_path(output, budget):
    """Budget-local checks allow its full cohort before another budget's pilot."""
    label = "_".join(str(v) for v in budget) if isinstance(budget, tuple) else budget
    name = "trial_check.json" if budget is None else f"trial_check_images_{label}.json"
    return output / "parallel" / name


def stage_candidates(stage_tasks, trial, gate, queue):
    """Run two-case technical waves, then admit only the active stage cohort."""
    if gate:
        return stage_tasks
    waves = [trial[index:index + 2] for index in range(0, len(trial), 2)]
    return next((wave for wave in waves if any(
        not queue.terminal(queue.state["tasks"][queue.key(task)]) for task in wave)), [])


def run_controller(args):
    """One queue writer and one supervised host, with technical trial gating."""
    output = args.output.resolve()
    manifest = read_json(output / "manifest.json")
    if manifest.get("strict_workers") and args.workers != manifest["strict_workers"]:
        raise ValueError("This experiment requires exactly two execution lanes")
    verify_frozen(output, manifest)
    if (output / "PAUSE").exists():
        raise RuntimeError("Campaign has a pending manual pause")
    if manifest.get("version", 1) == 1 and not manifest.get("technical_trial") and not (output / "pilot_report_delivered.json").exists():
        raise RuntimeError("The pilot report must be delivered before full seen")
    if compat.requires_delivered_pilot(manifest) and not read_json(output / "capture_validation.json").get("passed"):
        raise RuntimeError("RGB/depth capture validation must pass before navigation")
    tasks = campaign_tasks(manifest)
    with campaign_lock(output):
        overlay = (getattr(args, "_overlay", None) or read_json(output / "parallel/execution_overlay.json")
                   or prepare_extension(output, manifest))
        args._overlay = overlay
        globals()["_FROZEN_EXTENSION_DIR"] = Path(overlay["controller_script"]).parent
        verify_overlay(overlay, manifest)
        install_report_readers(overlay, manifest)
        if discover_gpu_bindings(primary_config(manifest)["map_config_path"]) != overlay["remote_gpu_bindings"]:
            raise RuntimeError("Remote GPU UUID or Vulkan mapping changed since freeze")
        offline_models(output, overlay)
        queue = retry_file_access(lambda: ParallelRecoveryQueue(
            output / "progress.json", manifest["tasks"], manifest["fingerprint"],
            terminal_provider_errors=manifest.get("terminal_provider_errors", False),
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
            selection = {"tasks": manifest.get("technical_trial") or trial_tasks(tasks, queue.state), "started_at": utc_now()}
            write_json(trial_path, selection)
        all_trial = selection["tasks"]
        policy = overlay.get("scheduling_policy", {})
        budget, stage_tasks = configuration_stage(manifest, tasks, queue, policy)
        group_field, _ = compat.group_policy(manifest)
        trial = [task for task in all_trial if task in stage_tasks]
        if not trial:
            raise ValueError("Every execution stage requires a technical trial")
        gate_record = read_json(technical_check_path(output, budget))
        gate = (gate_record.get("passed", False) and gate_record.get("controller_revision") == overlay["controller_revision"]
                and gate_record.get("scheduling_policy_sha256") == overlay.get("scheduling_policy_sha256"))
        if compat.requires_delivered_pilot(manifest) and not (output / "pilot_report_delivered.json").exists():
            gate = False
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
        from evaluation.benchmarks.core.campaign.cleanup import CleanupChecks
        cleanups = CleanupChecks()
        drain_reason = None
        active, completed, service_due = {}, 0, False
        gpu_samples = read_json(output / "parallel/gpu_samples.json").get("samples", [])
        last_gpu_check = 0
        last_progress_check = time.monotonic()
        fallback = False
        rate_limits = 0
        started = time.monotonic()
        initial_scored = sum(item["status"] == "scored" and item["task"] in tasks
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
                        if lane not in cleanups.jobs:
                            # Captured leases are immutable copies; cleanup workers
                            # never receive or mutate the shared recovery queue.
                            cleanups.start(lane, lambda r=request, p=process: finish_attempt(
                                output, manifest, r["task"], r["attempt"], p))
                            request["cleanup_pending"] = True
                        receipt = cleanups.poll(lane)
                        if receipt is None:
                            continue
                        action = queue.accept(request["task"], request["attempt"]["number"], receipt)
                        snapshots.append(fact["shared_models"])
                        write_json(output / "parallel/trial_snapshots.json", {"snapshots": snapshots[-8:]})
                        completed += 1
                        del active[lane]
                        render_reports(output, queue)
                        print(json.dumps({"episode": queue.key(request["task"]), "lane": lane,
                              "attempt": request["attempt"]["number"], "action": action}), flush=True)
                        if action == "service_check":
                            service_due = True
                        if action == "pause":
                            # Stop dispatch, but allow the other flight to finish.
                            drain_reason = queue.state["reason"]
                        exception = json.dumps(fact.get("exception") or {}).lower()
                        if args.workers == 2 and "out of memory" in exception and "cuda" in exception:
                            fallback = True
                            fatal = "parallel_cuda_memory"
                        provider_error = queue.provider_error(receipt.get("result"))
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
                if drain_reason and not active:
                    fatal = drain_reason
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
                if now - last_progress_check >= 300:
                    coverage_check(output, manifest, queue, started=started, initial_scored=initial_scored)
                    last_progress_check = now
                next_budget, next_tasks = configuration_stage(manifest, tasks, queue, policy)
                if next_budget != budget:
                    budget, stage_tasks = next_budget, next_tasks
                    trial = [task for task in all_trial if task in stage_tasks]
                    if not trial:
                        raise ValueError("Every image budget requires a technical trial")
                    gate_record = read_json(technical_check_path(output, budget))
                    gate = (gate_record.get("passed", False) and gate_record.get("controller_revision") == overlay["controller_revision"]
                            and gate_record.get("scheduling_policy_sha256") == overlay.get("scheduling_policy_sha256"))
                    print(json.dumps({"configuration_stage": budget, "episodes": len(stage_tasks)}), flush=True)
                if not gate and all(queue.terminal(queue.state["tasks"][queue.key(task)]) for task in trial):
                    repair_pending_reports(output, manifest, queue)
                    check = validate_trial(output, queue, trial, snapshots, gpu_samples=gpu_samples, workers=args.workers)
                    check["elapsed_seconds"] = now - started
                    check["controller_revision"] = overlay["controller_revision"]
                    check["scheduling_policy_sha256"] = overlay.get("scheduling_policy_sha256")
                    check["configuration_stage"] = budget
                    write_json(technical_check_path(output, budget), check)
                    write_json(output / "parallel/trial_check.json", check)
                    if not check["passed"]:
                        fatal = "parallel_technical_check"
                        break
                    if compat.requires_delivered_pilot(manifest):
                        compat.write_comparison(output, queue, manifest, pilot=True)
                        if not storage_budget(output, queue, tasks, args.workers):
                            fatal = "pilot_storage_budget"
                            break
                        write_json(output / "pilot_report_delivered.json", {"report": str(output / "pilot_report.html"), "ready_at": utc_now()})
                    if budget is not None and not storage_budget(output, queue, tasks, args.workers):
                        fatal = "pilot_storage_budget"
                        break
                    gate = True
                    print(json.dumps({"parallel_trial_passed": True, "episodes": len(trial)}), flush=True)
                candidates = stage_candidates(stage_tasks, trial, gate, queue)
                candidates = recovery_candidates(output, overlay, candidates, queue)
                if (session / "host_environment.json").exists() and not service_due and not drain_reason:
                    for lane in ("A", "B")[:args.workers]:
                        if lane in active:
                            continue
                        info = overlay["lanes"][lane]
                        admitted = recovery_candidates(output, overlay, candidates, queue)
                        eligible = dispatch_candidates(manifest, admitted, lane, active, args.workers, queue)
                        lease = queue.claim(eligible, lane, metadata={
                            "map_config_path": info["map_config_path"], "local_port": info["local_port"],
                            "remote_port": info["remote_port"], "remote_gpu": info["remote_gpu"], "session": str(session),
                            "logging_profile": overlay.get("logging_profile", "full"),
                            "recovery_revision": overlay.get("recovery_revision", "original")})
                        if not lease:
                            continue
                        task, attempt = lease
                        request = {"id": uuid.uuid4().hex, "lane": lane, "task": task,
                                   "attempt": attempt, "map_config_path": info["map_config_path"],
                                   "episode_deadline": args.episode_deadline}
                        write_json(session / "commands" / (lane + ".json"), request)
                        active[lane] = {**request, "started": now, "progress_at": now}
                        if manifest.get("dispatch_concurrency"):
                            with (session / "dispatch_audit.jsonl").open("a", encoding="utf-8") as audit:
                                audit.write(json.dumps({"timestamp": utc_now(), "active": {
                                    name: value["task"]["episode_id"] for name, value in active.items()},
                                    "limit": manifest["dispatch_concurrency"]}) + "\n")
                counts = dict(Counter(item["status"] for item in queue.state["tasks"].values()
                                      if item["task"] in tasks))
                write_json(output / "parallel/status.json", {"status": queue.state["status"],
                    "reason": queue.state["reason"], "counts": counts, "workers": args.workers,
                    "lanes": {lane: request["task"] for lane, request in active.items()},
                    "cleanup_pending": [lane for lane in active if active[lane].get("cleanup_pending")],
                    "drain_reason": drain_reason,
                    "trial_passed": gate, "image_budget_stage": budget[0] if isinstance(budget, tuple) else budget,
                    "dataset_stage": budget[1] if isinstance(budget, tuple) else None,
                    "configuration_stage": budget, "configuration_field": group_field,
                    "scheduling_mode": policy.get("mode", "manifest_order"),
                    "stage_counts": dict(Counter(queue.state["tasks"][queue.key(task)]["status"] for task in stage_tasks)),
                    "host_pid": child.pid, "updated_at": utc_now()})
                if not active and (session / "host_environment.json").exists():
                    if all(queue.terminal(queue.state["tasks"][queue.key(task)]) for task in tasks):
                        queue.state.update(status=compat.completion_status(manifest), reason=None)
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
                "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
                "controller_revision": overlay["controller_revision"]})
        finally:
            cleanups.close()
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
            render_reports(output, queue)
            repair_pending_reports(output, manifest, queue)
            coverage_check(output, manifest, queue, started=started, initial_scored=initial_scored)
            write_exit_status(output, queue, workers=args.workers, gate=gate, host_pid=child.pid)
            print(json.dumps({"status": queue.state["status"], "reason": queue.state["reason"]}), flush=True)
        clean = queue.state.get("reason") not in {"cleanup_unverified", "interrupted_cleanup_unverified"}
        if fallback and clean and not manifest.get("strict_workers") and not (output / "PAUSE").exists():
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
    """Dispatch the optional dataset-owned legacy handoff command."""
    from importlib import import_module
    manifest = read_json(args.output.resolve() / "manifest.json")
    dataset = compat.dataset_name(manifest)
    module = import_module(f"evaluation.benchmarks.{dataset}.campaign_handoff")
    return module.handoff(args)


def main(*, default_dataset=None, default_output=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate", "run", "host", "handoff", "watch"))
    parser.add_argument("--output", type=Path, default=default_output or DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default=default_dataset)
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--plugin-config", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--parent-created", type=float)
    parser.add_argument("--episode-deadline", type=float, default=7200)
    parser.add_argument("--no-progress", type=float, default=900)
    args = parser.parse_args()
    if args.command in {"prepare", "validate"}:
        if not args.dataset or not args.config:
            parser.error("prepare/validate require a dataset and at least one --config")
        from evaluation.benchmarks.core.campaign.prepare import prepare, validate
        result = (validate(args.config, args.dataset, root=REPO) if args.command == "validate"
                  else prepare(args.output, args.config, args.dataset, root=REPO, plugin_config=args.plugin_config))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    # Public entries resume the pinned snapshot instead of replacing a running
    # cohort's control code with whatever happens to be in today's worktree.
    snapshot_root = args.output.resolve() / "parallel/controller_source"
    if args.command in {"run", "watch"} and snapshot_root not in SCRIPT.resolve().parents:
        manifest = read_json(args.output.resolve() / "manifest.json")
        verify_frozen(args.output.resolve(), manifest)
        overlay = read_json(args.output.resolve() / "parallel/execution_overlay.json")
        if not overlay:
            with campaign_lock(args.output.resolve()):
                overlay = prepare_extension(args.output.resolve(), manifest)
        verify_overlay(overlay, manifest)
        command = [sys.executable, "-u", "-X", "utf8", overlay["controller_script"], *sys.argv[1:]]
        os.execv(sys.executable, command)
    if args.command == "watch":
        from evaluation.benchmarks.core.campaign.watchdog import run_watchdog
        run_watchdog(args, sys.modules[__name__])
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
