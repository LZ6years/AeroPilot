"""Fixed-context V1/V2 comparison and three frozen, visible-evidence-only excerpts.

The capability probe uses one original RGB and strict JSON Schema. No fallback
to JSON Object or another model is attempted. All artifacts live in a new run.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import sys
import hashlib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from plugins.research.flight_experience.calls import Calls,write
from plugins.research.flight_experience.contracts import obj,STRING,STRINGS,CHANGE,REVIEW,prompt,validate_review
from plugins.research.flight_experience import read_run,run_experiment,ExperimentSettings
from plugins.research.flight_experience.store import Store,dumps,digest
from plugins.research.flight_experience.maintenance import validate_change
from src.llm import DefaultModelClient
from src.agent.config import load_model_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('capability','compare','segments'),required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    old=ROOT/'tests/manual_outputs/flight_exp_5x2_20260927'
    original_dir=old/'experience/04/diagnostic'
    original=json.loads((original_dir/'experiment.json').read_text(encoding='utf-8'))
    bundle=read_run(Path(original['source']))
    model=load_model_config(ROOT/'.env')
    settings=replace(model.to_provider_settings(),model_name='qwen3.8-flash',timeout_sec=180,provider_timeout_retries=1)
    reflection,reviewer=DefaultModelClient(settings),DefaultModelClient(settings)
    parameters={'thinking':{'type':model.thinking_mode}} if model.thinking_mode else {}
    if args.mode=='capability':
        if (output/'capability.json').exists():
            raise ValueError('Capability report already exists')
        calls=Calls(output/'capability',parameters=parameters)
        result={'status':'failed','model':settings.model_name,'timeout_sec':180}
        try:
            answer=calls.ask('capability',reflection,'Describe the supplied historical RGB briefly. Return JSON matching the schema.',
                {'purpose':'strict schema multimodal capability check'},obj(description=STRING,visible_categories=STRINGS),
                bundles={'run':bundle},shown={'run':['obs_0003']},allow_repair=False)
            result.update(status='supported',answer=answer)
        except Exception as exc:
            result.update(error_type=type(exc).__name__,category=getattr(exc,'category',None),http_status=getattr(exc,'status_code',None))
        write(output/'capability.json',result)
        print(dumps(result),flush=True)
        return
    if json.loads((output/'capability.json').read_text(encoding='utf-8'))['status']!='supported':
        raise ValueError('Strict multimodal capability not verified')
    if args.mode=='compare':
        database=Store(output/'comparison.sqlite')
        database.initialize()
        imported=database.import_experiment(original_dir)
        cards=[database.get(eid) for eid in database.list_ids()]
        sid=cards[0]['evidence'][0]['source_id']
        shown={sid:sorted({o for c in cards for l in c['evidence'] for o in l['observation_ids']})}
        refs=[r for c in cards for l in c['evidence'] for r in l['evidence_refs']]
        evidence={sid:bundle.excerpt(shown[sid],refs)}
        material={'experiences':cards,'original_evidence':evidence}
        write(output/'comparison_manifest.json',{'material':material,'shown':shown,'input_sha256':digest(material),
            'source_sha256':bundle.sources,'unit':'one fixed group of four historical candidates','model':settings.model_name,
            'timeout_sec':180,'output_schema':CHANGE,'parameters':parameters})
        v1=ROOT/'plugins/research/flight_experience/prompts/v1'
        prompts={'v1':(v1/'boundary.txt').read_text(encoding='utf-8')+(v1/'maintain.txt').read_text(encoding='utf-8'),'v2':prompt('maintain')}
        for variant,system in prompts.items():
            directory=output/'comparison'/variant
            if directory.exists():
                raise ValueError('Do not overwrite comparison attempts')
            calls=Calls(directory,parameters=parameters,prompt_version=variant)
            result={'variant':variant,'status':'failed'}
            try:
                change=calls.ask('reflection',reflection,system,material,CHANGE,bundles={sid:bundle},shown=shown)
                write(directory/'proposal.json',change)
                validate_change(change,cards,{sid:replace(bundle,timeline=evidence[sid]['timeline'],observations={o:bundle.observations[o] for o in shown[sid]})},shown)
                review=calls.ask('review',reviewer,prompt('review'),{'before':cards,'proposed_change':change,'original_evidence':evidence},
                    REVIEW,bundles={sid:bundle},shown=shown)
                validate_review(review)
                result.update(status='completed',review=review)
            except Exception as exc:
                result.update(error_type=type(exc).__name__,category=getattr(exc,'category',None))
            write(directory/'result.json',result)
            print(dumps(result),flush=True)
        return
    # Selection criteria use visible action/intent logs only, never benchmark metrics.
    selected=[]
    for case,topic,kind in [(1,'requested_vs_executed','clipped'),(9,'turn_then_target_unseen','turn'),(3,'discover_vs_arrive','node')]:
        old_result=json.loads((old/f'experience/{case:02d}/diagnostic/experiment.json').read_text(encoding='utf-8'))
        source=read_run(Path(old_result['source']))
        chosen=None
        for i,row in enumerate(source.timeline):
            if kind=='clipped' and row.get('kind')=='tool_result' and row.get('safety_interventions'):
                chosen=i;break
            if kind=='turn' and row.get('kind')=='tool_result' and row.get('tool')=='uav.turn_relative' and row.get('status')=='ok':
                chosen=i;break
            if kind=='node' and row.get('kind')=='assistant_turn' and any(w in (row.get('unverified_intent') or '').lower() for w in ('尽头','end of','路口')):
                chosen=i;break
        if chosen is None:
            raise ValueError(f'No matching visible event for fixed case {case}; do not replace case')
        checkpoints=[(i,r['observation_id']) for i,r in enumerate(source.timeline) if r['kind']=='state_checkpoint' and r.get('observation_id') in source.images]
        before=[v for v in checkpoints if v[0]<=chosen][-2:]
        after=[v for v in checkpoints if v[0]>chosen][:2]
        observations=list(dict.fromkeys(o for _,o in before+after))
        selected.append({'case':case,'topic':topic,'source':str(source.source),'event_id':source.timeline[chosen]['id'],
            'observations':observations,'source_sha256':source.sources,
            'image_sha256':{o:source.observations[o]['sha256'] for o in observations}})
    manifest=output/'segments_manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8'))!=selected:
            raise ValueError('Frozen manifest changed')
    else:
        write(manifest,selected)
    for segment in selected:
        source=read_run(Path(segment['source']))
        destination=output/'segments'/segment['topic']
        if destination.exists():
            raise ValueError('Do not overwrite segment experiment')
        # Initial generation sees only the fixed selection. Reviewer may inspect
        # additional original counterevidence through its independent index call.
        result=run_experiment(source,destination,reflection=reflection,reviewer=reviewer,parameters=parameters,
            settings=ExperimentSettings(observation_ids=tuple(segment['observations']),reflection_model=settings.model_name,
                review_model=settings.model_name,library_path=output/'new_experiences.sqlite',checkpoint_directory=output/'generation_checkpoints'))
        print(dumps({'topic':segment['topic'],'status':result['status'],'accepted':len(result['accepted']),'published':result['published']}),flush=True)


if __name__=='__main__':
    main()
