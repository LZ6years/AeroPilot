"""Frozen-image component timing; no model, simulator or inference backend.

Ordinary timings and instrumented stage/memory diagnostics are separate.
Historical repeats represent reuse opportunities, not recorded encode calls.
"""
import argparse
import base64
import hashlib
import html
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import tracemalloc
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import PIL
from PIL import Image
import psutil
from src.perception.frame_payload import frame_bytes, _validate_dimensions
from src.memory.context.modules.image import latest_rgb_model_inputs_from_frame
from tests.manual.experiment_sources import source_hashes
from tests.manual.gpu_probe_environment import check_interpreter, other_project_jobs
from tests.manual.process_diagnostics import supervise, write_json
from tests.manual.read_only_concurrency_probe import percentile
from tests.manual.export_benchmark_review import validate_bundle
from tests.manual.review_browser_probe import inspect


def speed_gate(repeated, missed):
    """Engineering thresholds, not confidence intervals or navigation claims."""
    b, c = repeated['baseline'], repeated['candidate']
    mb, mc = missed['baseline'], missed['candidate']
    return (b['p50_sec'] - c['p50_sec'] >= .001 and c['p50_sec'] <= b['p50_sec'] * .8
            and c['p95_sec'] <= b['p95_sec'] * 1.1
            and mc['p50_sec'] - mb['p50_sec'] <= max(mb['p50_sec'] * .1, .0005))


def captured_fixture_request(frame, directory):
    """Explicit constructed input with historical pixels; interception only."""
    from types import SimpleNamespace
    from src.agent.mission_state import MissionState
    from src.agent.transcript import AgentTranscript
    from src.agent.planning.model_action_planner import ModelActionPlanner
    from src.runtime.configs.model_config import ModelConfig
    from src.memory.flight_memory import FlightMemory
    from src.memory.stores import MemoryEventWriter, MemoryRecordReader
    from src.perception.context_builder import ObservationContextBuilder
    from src.tools.registry import default_registry
    writer=MemoryEventWriter(directory)
    state=MissionState('component_fixture','Offline image encoding equivalence fixture')
    memory=FlightMemory('component_fixture','fixture_map','fixture_scene','fixture_agent',
                        active_instruction=state.user_instruction,memory_reader=MemoryRecordReader(directory))
    observation=writer.write_observation({'frames':{'rgb':frame},'modalities':['rgb']})
    memory.append_observation_ref(observation['observation_ref'])
    memory.append_frame_refs(observation['frame_refs'])
    builder=ObservationContextBuilder()
    captures=[]
    class Captured(Exception): pass
    class OfflinePlanner(ModelActionPlanner):
        def _request(self,payload):
            captures.append(payload)
            raise Captured
    planner=OfflinePlanner(ModelConfig(api_key='offline-only',model_name='offline-only'),client=object())
    contexts=[]
    for capacity in (0,32*1024*1024):
        builder.composer.registry.modules['image'].configure({'rgb_encoding_cache_max_bytes':capacity})
        context=builder.build(state,memory)
        assert builder.build(state,memory)==context
        contexts.append(context)
        transcript=AgentTranscript(SimpleNamespace(write_transcript_event=lambda **kw:None))
        transcript.start_instruction(state.user_instruction,'fixture_instruction')
        transcript.append_state(context,observation['observation_ref']['id'])
        try:
            planner.plan_turn(transcript,default_registry(),decision_index=6)
        except Captured:
            pass
        else:
            raise AssertionError('Offline interception was not reached')
    assert contexts[0]==contexts[1] and captures[0]==captures[1]
    return dict(constructed_input=True, context_equal=True, request_equal=True,
                request_sha256=hashlib.sha256(json.dumps(captures[0],sort_keys=True).encode()).hexdigest())


def comparison_child(root, sample, trial):
    from src.perception.frame_payload import RgbEncodingCache
    check_interpreter()
    manifest=read(root/'manifest.json')
    assert all(sha(root/s['frame_file'])==s['frame_sha256'] for s in manifest['samples'])
    frames=[read(root/s['frame_file']) for s in manifest['samples']]
    for f,s in zip(frames,manifest['samples']):
        assert hashlib.sha256(frame_bytes(f)).hexdigest()==s['rgb_sha256']
    if len({s['rgb_sha256'] for s in manifest['samples']})<=4:
        raise ValueError('Historical sequence does not exceed cache capacity; do not manufacture inputs')
    target=frames[sample-1]
    cache=RgbEncodingCache(32*1024*1024)
    rows=[]
    for mode in ('repeated','forced_miss','sequence'):
        cache.clear()
        for n in range(31):
            selected=frames[n%len(frames)] if mode=='sequence' else target
            result={}
            for variant in (('baseline','candidate') if n%2==0 else ('candidate','baseline')):
                if variant=='candidate' and mode=='forced_miss':
                    cache.clear()  # Separate miss overhead from eviction/clear timing.
                start=time.perf_counter()
                result[variant]=latest_rgb_model_inputs_from_frame(selected,**({'cache':cache} if variant=='candidate' else {}))
                rows.append(dict(mode=mode,variant=variant,repeat=n+1,first=n==0,wall_sec=time.perf_counter()-start))
            assert result['baseline']==result['candidate']
            # Byte equality is stricter, but also explicitly verify pixels and size.
            decoded=[Image.open(BytesIO(base64.b64decode(result[v][0]['data'].split(',',1)[1]))) for v in ('baseline','candidate')]
            assert decoded[0].size==decoded[1].size and decoded[0].tobytes()==decoded[1].tobytes()
            for image in decoded: image.close()
    stats=cache.stats()
    groups={mode:{variant:distribution([r['wall_sec'] for r in rows if not r['first'] and r['mode']==mode and r['variant']==variant])
                  for variant in ('baseline','candidate')} for mode in ('repeated','forced_miss','sequence')}
    tracemalloc.start()
    try:
        memory_cache=RgbEncodingCache(32*1024*1024)
        for frame in frames: memory_cache.encode(frame)
        _,peak=tracemalloc.get_traced_memory()
    finally: tracemalloc.stop()
    directory=root/f's{sample}t{trial}'
    fixture=captured_fixture_request(target,directory/'constructed_fixture')
    result=dict(sample=sample,trial=trial,rows=rows,groups=groups,cache_stats=stats,
                ordinary_calls_per_variant=len(rows)//2,ordinary_source_reads=len(rows),
                baseline_encode_calls=len(rows)//2,python_peak_bytes=peak,rss_after_bytes=psutil.Process().memory_info().rss,
                diagnostic_memory_cache=memory_cache.stats(),fixture=fixture,
                speed_pass=speed_gate(groups['repeated'],groups['forced_miss']))
    write_json(directory/'result.json',result)


def run_comparison(args):
    check_interpreter()
    if other_project_jobs(ROOT): raise RuntimeError('Other project experiments are active')
    source=args.input_manifest.resolve().parent
    old=read(args.input_manifest)
    if not read(source/'summary.json')['cache_experiment_eligible']:
        raise ValueError('Phase one did not meet the cache entry gate')
    root,review=args.root.resolve(),args.review_root.resolve()
    if root.exists() or review.exists(): raise ValueError('Use new directories')
    root.mkdir(parents=True); review.mkdir(parents=True)
    manifest={**old,'state':'preparing','source_hashes':source_hashes(),'phase_one':str(source),
              'effective_cache_bytes':32*1024*1024,'production_default_bytes':0}
    try:
        for s in manifest['samples']:
            f=read(source/s['frame_file'])
            for key in ('artifact_path','path','artifact_ref'):
                value=f.get(key)
                if value and Path(value).is_file():
                    target=root/Path(value).name
                    shutil.copyfile(value,target); f[key]=str(target)
            write_json(root/s['frame_file'],f)
            assert hashlib.sha256(frame_bytes(f)).hexdigest()==s['rgb_sha256']
            s['frame_sha256']=sha(root/s['frame_file'])
            shutil.copyfile(source/f'{s["index"]:02d}.png',root/f'{s["index"]:02d}.png')
        for relative in manifest['source_hashes']:
            target=root/'source'/relative
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/relative,target)
        manifest['state']='running'; write_json(root/'manifest.json',manifest)
        samples=[]
        for n in range(1,7):
            trials=[]
            for trial in range(1,4):
                if other_project_jobs(ROOT): raise RuntimeError('Another experiment started; pausing')
                directory=root/f's{n}t{trial}'
                status=supervise([sys.executable,'-u','-X','faulthandler',str(root/'source/tests/manual/image_context_probe.py'),
                                  'compare_child','--root',str(root),'--sample',str(n),'--trial',str(trial)],directory)
                if status['exit_code']: raise RuntimeError(f'Comparison child failed: {directory.name}')
                trials.append(read(directory/'result.json'))
            groups={mode:{v:distribution([r['wall_sec'] for t in trials for r in t['rows'] if not r['first'] and r['mode']==mode and r['variant']==v])
                          for v in ('baseline','candidate')} for mode in ('repeated','forced_miss','sequence')}
            row=dict(sample=n,trials=trials,groups=groups,speed_pass=all(t['speed_pass'] for t in trials) and speed_gate(groups['repeated'],groups['forced_miss']))
            samples.append(row)
            print(json.dumps({'sample':n,'groups':groups,'speed_pass':row['speed_pass']}),flush=True)
        summary=dict(state='passed',correctness_pass=True,recommend_task_validation=all(s['speed_pass'] for s in samples),
                     production_enabled=False,samples=samples)
        write_json(root/'summary.json',summary); write_json(review/'summary.json',summary)
        links=[]
        for n,s in enumerate(samples,1):
            directory=review/'cases'/f'{n:02d}'
            directory.mkdir(parents=True); shutil.copyfile(root/f'{n:02d}.png',directory/'rgb.png')
            page='<!doctype html><meta charset="utf-8"><h1>RGB cache comparison</h1><p>Component replay; constructed request fixture, no model call.</p><div class="phase-timeline"><img src="rgb.png"></div><pre>'+html.escape(json.dumps(s,indent=2))+'</pre>'
            for name in ('review','audit_review'): (directory/f'{name}.html').write_text(page,encoding='utf-8')
            links.append(f'<li><a href="cases/{n:02d}/review.html">Sample {n}</a> / <a href="cases/{n:02d}/audit_review.html">audit</a></li>')
        (review/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>图片编码缓存对照</h1><p>生产默认关闭；仅离线组件验证，不是完整导航加速。</p><ul>'+''.join(links)+'</ul><a href="summary.json">Full results</a>',encoding='utf-8')
        (review/'browser_validation').mkdir()
        pages=[f'cases/{n:02d}/{kind}.html' for n in (1,4) for kind in ('review','audit_review')]
        for name,exe in (('chrome','C:/Program Files/Google/Chrome/Application/chrome.exe'),('edge','C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')):
            write_json(review/'browser_validation'/f'{name}.json',inspect(Path(exe),review,pages,name))
        assert all(sha(p)==v for p,v in manifest['input_hashes'].items())
        assert all(sha(root/'source'/p)==v for p,v in manifest['source_hashes'].items())
        manifest['state']='passed'; write_json(root/'manifest.json',manifest)
        write_json(review/'resource_validation.json',validate_bundle(review))
    except BaseException as exc:
        manifest.update(state='paused',error=repr(exc)); write_json(root/'manifest.json',manifest)
        raise


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_frame(decision, run, records):
    """Select via the recorded observation -> frame reference, never final state."""
    obs = [r for r in records['observations'] if r['id'] == decision['checkpoint_observation_id']]
    if len(obs) != 1:
        raise ValueError('Missing or ambiguous recorded observation')
    refs = [r for r in obs[0]['frame_refs'] if r['modality'] == 'rgb']
    if len(refs) != 1:
        raise ValueError('Need exactly one recorded RGB frame')
    relative, number = refs[0]['ref'].split('#')
    if relative != 'memory/frames.jsonl':
        raise ValueError('Unexpected frame reference file')
    record = records['frames'][int(number) - 1]
    if record['id'] != refs[0]['id'] or record['observation_id'] != obs[0]['id']:
        raise ValueError('Frame/observation reference mismatch')
    frame = {**record, **record.get('metadata', {})}
    for key in ('path', 'artifact_path', 'artifact_ref'):
        value = frame.get(key)
        if value and not Path(value).is_absolute():
            frame[key] = str(run / value)
    frame.update(selection_role='current_observation', input_id='rgb_current')
    frame_bytes(frame)  # Fail closed on missing/corrupt transport.
    return frame


def distribution(values):
    return dict(count=len(values), p50_sec=statistics.median(values), p95_sec=percentile(values, .95))


def eligible(samples, opportunities):
    return any(r['adjacent_repeats'] > 0 for r in opportunities) and sum(
        s['conversion']['p50_sec'] >= .002 for s in samples) >= 3


def measured_stages(frame):
    """Same conversion operations as production, checked against its output."""
    stamps = [time.perf_counter()]
    raw = frame_bytes(frame)
    stamps.append(time.perf_counter())
    with Image.open(BytesIO(raw)) as source:
        pixels = source.convert('RGB')
    _validate_dimensions(frame, *pixels.size)
    stamps.append(time.perf_counter())
    output = BytesIO()
    pixels.save(output, format='PNG')
    stamps.append(time.perf_counter())
    url = 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii')
    stamps.append(time.perf_counter())
    stages = dict(zip(('read', 'decode_rgb', 'png', 'base64'),
                      (b-a for a,b in zip(stamps, stamps[1:]))))
    from src.memory.context.modules import image as module
    with patch.object(module, 'rgb_data_url', return_value=(url, pixels.size)):
        began = time.perf_counter()
        latest_rgb_model_inputs_from_frame(frame)
        stages['input_assembly'] = time.perf_counter() - began
    return url, stages, len(raw)


def child(root, sample, trial):
    check_interpreter()
    manifest = read(root / 'manifest.json')
    selected = manifest['samples'][sample - 1]
    assert sha(root/selected['frame_file'])==selected['frame_sha256']
    frame = read(root / selected['frame_file'])
    if hashlib.sha256(frame_bytes(frame)).hexdigest() != selected['rgb_sha256']:
        raise ValueError('Frozen RGB changed')
    ordinary, diagnostic = [], []
    reference = None
    for n in range(31):
        began = time.perf_counter()
        result = latest_rgb_model_inputs_from_frame(frame)
        elapsed = time.perf_counter() - began
        if reference is None:
            reference = result
        assert result == reference
        ordinary.append(elapsed)
    for _ in range(30):
        url, stages, byte_count = measured_stages(frame)
        assert url == reference[0]['data']
        diagnostic.append(stages)
    tracemalloc.start()
    try:
        latest_rgb_model_inputs_from_frame(frame)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    result = dict(sample=sample, trial=trial, executable=sys.executable,
        python=sys.version, pillow=PIL.__version__, first_sec=ordinary[0], ordinary=ordinary[1:],
        stages=diagnostic, warm=distribution(ordinary[1:]),
        conversion=distribution([sum(row[k] for k in ('decode_rgb','png','base64')) for row in diagnostic]),
        raw_bytes_per_call=byte_count, ordinary_encode_calls=31, diagnostic_encode_calls=30,
        memory_probe_encode_calls=1, python_peak_bytes=peak,
        rss_after_bytes=psutil.Process().memory_info().rss,
        output_sha256=hashlib.sha256(reference[0]['data'].encode()).hexdigest())
    write_json(root / f's{sample}t{trial}' / 'result.json', result)


def run(args):
    check_interpreter()
    if other_project_jobs(ROOT): raise RuntimeError('Other project experiments are active')
    root, review = args.root.resolve(), args.review_root.resolve()
    if root.exists() or review.exists():
        raise ValueError('Use new directories')
    frozen = read(args.input_manifest)
    root.mkdir(parents=True)
    review.mkdir(parents=True)
    manifest = dict(state='preparing', samples=[], opportunities=[], input_hashes={}, source_hashes=source_hashes(),
                    limitations='Component replay only; historical full-context builds remain unavailable. No navigation or model requests.')
    try:
        groups = {}
        for n, sample in enumerate(frozen['samples'], 1):
            run_dir = Path(sample['run_dir'])
            if str(run_dir) not in groups:
                records = {name: [json.loads(line) for line in (run_dir / f'memory/{name}.jsonl').read_text(encoding='utf-8').splitlines()]
                           for name in ('observations', 'frames')}
                groups[str(run_dir)] = records
                sequence = []
                for path in sorted((run_dir / 'debug/tool_loop').glob('*/decision_*.json')):
                    manifest['input_hashes'][str(path)] = sha(path)
                    frame = resolve_frame(read(path), run_dir, records)
                    sequence.append(hashlib.sha256(frame_bytes(frame)).hexdigest())
                manifest['opportunities'].append(dict(case_id=sample['case_id'], decisions=len(sequence),
                    adjacent_repeats=sum(a==b for a,b in zip(sequence,sequence[1:])), hashes=sequence))
                for name in records:
                    path = run_dir / f'memory/{name}.jsonl'
                    manifest['input_hashes'][str(path)] = sha(path)
            path = Path(sample['decision_path'])
            if sha(path) != frozen['input_hashes'][str(path)]:
                raise ValueError('Prior frozen decision changed')
            frame = resolve_frame(read(path), run_dir, groups[str(run_dir)])
            raw = frame_bytes(frame)
            asset = root / f'{n:02d}.png'
            asset.write_bytes(raw)
            # Preserve the transport form. Files are relocated explicitly;
            # in-memory base64 is retained, not replaced with a disk fast path.
            mapping = {}
            for key in ('artifact_path','path','artifact_ref'):
                if isinstance(frame.get(key), str) and Path(frame[key]).is_file():
                    original = Path(frame[key])
                    manifest['input_hashes'][str(original)] = sha(original)
                    target = root / f'{n:02d}_{key}.bin'
                    shutil.copyfile(original, target)
                    mapping[frame[key]] = str(target)
                    frame[key] = str(target)
            write_json(root / f'{n:02d}.json', frame)
            manifest['samples'].append(dict(index=n, benchmark=sample['benchmark'], case_id=sample['case_id'],
                decision_index=sample['decision_index'], frame_file=f'{n:02d}.json',
                frame_sha256=sha(root / f'{n:02d}.json'), rgb_sha256=hashlib.sha256(raw).hexdigest(), path_mapping=mapping))
        for relative in manifest['source_hashes']:
            path = root / 'source' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, path)
        manifest['state'] = 'running'
        write_json(root / 'manifest.json', manifest)
        rows = []
        for sample in range(1, 7):
            trials = []
            for trial in range(1,4):
                directory = root / f's{sample}t{trial}'
                command = [sys.executable, '-u', '-X', 'faulthandler',
                    str(root / 'source/tests/manual/image_context_probe.py'), 'child', '--root', str(root),
                    '--sample', str(sample), '--trial', str(trial)]
                status = supervise(command, directory)
                if status['exit_code']:
                    raise RuntimeError(f'Child failed: {directory.name}')
                trials.append(read(directory / 'result.json'))
            assert len({t['output_sha256'] for t in trials}) == 1
            row = dict(sample=sample, trials=trials, warm=distribution([x for t in trials for x in t['ordinary']]),
                conversion=distribution([sum(d[k] for k in ('decode_rgb','png','base64')) for t in trials for d in t['stages']]))
            rows.append(row)
            print(json.dumps({k:row[k] for k in ('sample','warm','conversion')}), flush=True)
        gate = eligible(rows, manifest['opportunities'])
        summary = dict(state='passed', cache_experiment_eligible=gate, samples=rows, opportunities=manifest['opportunities'])
        write_json(root / 'summary.json', summary)
        write_json(review / 'summary.json', summary)
        manifest['state'] = 'passed'
        write_json(root / 'manifest.json', manifest)
        links = []
        for n, row in enumerate(rows,1):
            folder = review / 'cases' / f'{n:02d}'
            folder.mkdir(parents=True)
            shutil.copyfile(root / f'{n:02d}.png', folder / 'rgb.png')
            body = '<!doctype html><meta charset="utf-8"><h1>RGB component replay</h1><div class="phase-timeline"><img src="rgb.png"></div><pre>' + html.escape(json.dumps(row, indent=2)) + '</pre>'
            for name in ('review.html','audit_review.html'):
                (folder / name).write_text(body, encoding='utf-8')
            links.append(f'<li><a href="cases/{n:02d}/review.html">Sample {n}</a></li>')
        (review / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>图片上下文：第一阶段测量</h1><p>仅组件重放，不是完整上下文或导航计时。</p><p>Cache experiment eligible: '+str(gate)+'</p><ul>'+''.join(links)+'</ul><a href="summary.json">Raw results</a>', encoding='utf-8')
        (review / 'browser_validation').mkdir()
        pages = [f'cases/{n:02d}/{kind}.html' for n in (1,4) for kind in ('review','audit_review')]
        for name, exe in (('chrome','C:/Program Files/Google/Chrome/Application/chrome.exe'),('edge','C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')):
            write_json(review/'browser_validation'/f'{name}.json',inspect(Path(exe),review,pages,name))
        assert all(sha(p)==v for p,v in manifest['input_hashes'].items())
        assert all(sha(root/'source'/p)==v for p,v in manifest['source_hashes'].items())
        write_json(review/'resource_validation.json',validate_bundle(review))
    except BaseException as exc:
        manifest.update(state='paused', error=repr(exc))
        write_json(root/'manifest.json',manifest)
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','child','compare','compare_child'))
    parser.add_argument('--input-manifest',type=Path)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--review-root',type=Path)
    parser.add_argument('--sample',type=int)
    parser.add_argument('--trial',type=int)
    args=parser.parse_args()
    if args.mode in ('child','compare_child'):
        os.chdir(ROOT)
    if args.mode=='child': child(args.root,args.sample,args.trial)
    elif args.mode=='compare_child': comparison_child(args.root,args.sample,args.trial)
    elif args.mode=='compare': run_comparison(args)
    else: run(args)
