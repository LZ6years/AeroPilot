"""Compare old/new rejection feedback on six captured Flash failures; execute no tools."""

import argparse
import copy
import html
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent.planning.model_action_planner import _parse_assistant_turn, AgentProtocolError
from src.agent.transcript import AgentToolCall, AssistantTurn, ToolResult, TranscriptEvent
from src.runtime.control.actions.terminal import environmental_evidence_refs, apply_terminal
from src.agent.config import load_model_config
from src.tools.registry import default_registry
from src.llm import ModelRequest, create_model_client
from src.llm.audit import snapshot_request
from src.llm.providers.openai_compatible import public_response_snapshot
from tests.manual.tool_selection_ablation_probe import restore_request, wire_check
from tests.manual.process_diagnostics import write_json


def prior_events(run, before_id):
    """Use canonical recorded events, excluding the rejected terminal request and future evidence."""
    events = []
    for line in (run/'memory/agent_transcript.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line); kind, payload = row['kind'], row['payload']
        if kind == 'provider_turn_rejected' and any(c.get('tool_call_id')==before_id for c in payload.get('tool_calls',[])):
            break
        if kind == 'assistant_turn':
            if any(c['tool_call_id']==before_id for c in payload['tool_calls']):
                break
            payload = AssistantTurn(**{**payload, 'tool_calls':[AgentToolCall(**c) for c in payload['tool_calls']]})
        elif kind == 'tool_result':
            payload = ToolResult(**payload)
        events.append(TranscriptEvent(kind,payload))
    return events


def prepare(run, index):
    registry = default_registry()
    read = lambda n: json.loads(next(run.glob(f'debug/tool_loop/*/decision_{n:04}.json')).read_text(encoding='utf-8'))['model_rounds'][0]
    rejected, following = read(index), read(index+1)
    call = rejected['response']['message']['tool_calls'][0]
    refs = environmental_evidence_refs(prior_events(run,call['id']),registry)
    old = restore_request(following['request'],following['request_media'],run)
    new = copy.deepcopy(old)
    expected = call['function']['name']
    if index in (9,14,15,16):
        try:
            _parse_assistant_turn(rejected['response']['message'],registry)
        except AgentProtocolError as exc:
            feedback = str(exc)
        else:
            raise ValueError('Expected an invalid JSON capture')
        old_feedback = 'tool uav_move_body arguments are not valid JSON'
    else:
        try:
            apply_terminal(SimpleNamespace(completed=False,stopped=False),json.loads(call['function']['arguments']),
                           completed=True,available_evidence=refs)
        except ValueError as exc:
            feedback = str(exc)
        else:
            raise ValueError('Expected rejected terminal references')
        old_feedback = 'terminal evidence must reference prior environmental evidence in this instruction'
        matched = [m for m in new['messages'] if m['role']=='tool' and m.get('tool_call_id')==call['id']]
        assert len(matched)==1
        content=json.loads(matched[0]['content']); content['message']=feedback
        matched[0]['content']=json.dumps(content,ensure_ascii=False)
    slots = [m for m in new['messages'] if m['role']=='user' and m.get('content')==old_feedback]
    assert slots
    slots[-1]['content']=feedback
    return {'decision':index,'expected_tool':expected,'available_evidence':sorted(refs),
            'original_arguments':call['function']['arguments'],'variants':{'old':old,'improved':new},
            'request_media':following['request_media']}


def run_probe(run, output):
    if output.exists():
        raise ValueError('Use a new directory')
    samples=[prepare(run,i) for i in (9,14,15,16,20,21)]
    config=load_model_config()
    if config.model_name!='qwen3.8-flash':
        raise ValueError('Expected the configured Flash model')
    output.mkdir(parents=True)
    manifest=[]
    for sample in samples:
        manifest.append({**{k:v for k,v in sample.items() if k!='variants'},
                         'requests':{k:snapshot_request(v)[0] for k,v in sample['variants'].items()}})
    write_json(output/'manifest.json',{'source_run':str(run.resolve()),'samples':manifest,'planned_requests':36})
    client=create_model_client(config.to_provider_settings()); rows=[]; status='running'
    try:
        for sample in samples:
            for repeat in range(3):
                for group in (('old','improved') if repeat%2==0 else ('improved','old')):
                    payload=sample['variants'][group]
                    request=ModelRequest(messages=payload['messages'],tools=payload['tools'],
                        parameters={k:v for k,v in payload.items() if k not in ('model','messages','tools')})
                    started=time.monotonic(); response=client.request(request)
                    public=public_response_snapshot(response.data); check=wire_check(public,payload)
                    calls=public['message'].get('tool_calls',[])
                    same_tool=len(calls)==1 and calls[0]['function']['name']==sample['expected_tool']
                    refs_valid=None; same_args=None
                    if same_tool and not check['issues']:
                        args=json.loads(calls[0]['function']['arguments'])
                        if sample['expected_tool']=='runtime_report_completed':
                            refs_valid=bool(args['evidence_refs']) and set(args['evidence_refs']).issubset(sample['available_evidence'])
                        else:
                            # Analysis-only hypothesis; NEVER execute or repair returned actions.
                            hypothesis=json.loads(sample['original_arguments'].replace(': 00.',': 0.'))
                            same_args=args==hypothesis
                    rows.append({'decision':sample['decision'],'group':group,'repeat':repeat+1,
                                 'response':public,'returned_model':response.data.get('model'),
                                 'elapsed_seconds':time.monotonic()-started,'wire_check':check,
                                 'same_tool':same_tool,'terminal_refs_valid':refs_valid,
                                 'args_match_decimal_hypothesis':same_args,'transport':response.transport_trace,
                                 'tools_executed':False})
                    write_json(output/'results.json',{'status':status,'results':rows})
                    print(json.dumps({'completed':len(rows),'decision':sample['decision'],'group':group,
                                      'issues':check['issues'],'same_tool':same_tool,'valid_refs':refs_valid}),flush=True)
        status='completed'
    except Exception as exc:
        status='failed'; write_json(output/'failure.json',{'error_type':type(exc).__name__,'completed':len(rows)})
        raise
    finally:
        write_json(output/'results.json',{'status':status,'results':rows})
        (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><h1>Feedback recovery; no actions executed</h1><pre>'+html.escape(json.dumps({'status':status,'results':rows},ensure_ascii=False,indent=2))+'</pre>',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); run_probe(args.run_dir,args.output_dir)
