"""Real simulator/API probe: run with the explicit flyagent interpreter.

Uses the production provider and runtime. It does not inject scripted decisions.
The controller heartbeat must remain connected during each autonomous task.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

from websockets.sync.client import connect


def request(base, path, body=None, method=None):
    data=json.dumps(body).encode() if body is not None else None
    with urlopen(Request(base+path,data=data,method=method,headers={'Content-Type':'application/json'}),timeout=15) as response:
        return json.load(response)


class Controller:
    def __init__(self,url):
        self.url=url;self.stop=threading.Event();self.ready=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.error=None
    def run(self):
        try:
            with connect(self.url) as socket:
                self.socket=socket;self.ready.set()
                while not self.stop.wait(.5):
                    socket.send(json.dumps({'type':'ping'}))
                    try:socket.recv(timeout=.01)
                    except TimeoutError:pass
        except Exception as exc:self.error=str(exc);self.ready.set()
    def start(self):
        self.thread.start();self.ready.wait(10)
        if self.error:raise RuntimeError(self.error)
    def close(self):
        self.stop.set();self.thread.join(10)


def wait_state(base,path,predicate,timeout=300):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=request(base,path)
        if predicate(value):return value
        if value['phase']=='failed':raise RuntimeError(value['error'])
        time.sleep(.25)
    raise TimeoutError('Session did not reach expected state')


def measure_preview(base,path,seconds=12):
    """JPEG-arrival latency: source RPC/encode + time until complete HTTP part."""
    samples=[];arrivals=[];started=time.monotonic()
    with urlopen(base+path+'/preview.mjpeg',timeout=15) as response:
        while time.monotonic()-started<seconds:
            line=response.readline()
            if not line.startswith(b'--frame'):continue
            headers={}
            while True:
                line=response.readline()
                if line==b'\r\n':break
                key,value=line.decode().split(':',1);headers[key.lower()]=value.strip()
            size=int(headers['content-length']);remaining=size
            while remaining:
                chunk=response.read(remaining)
                if not chunk:raise EOFError('Preview stream ended')
                remaining-=len(chunk)
            arrived=time.time();capture=datetime.fromisoformat(headers['x-captured-at']).timestamp()
            samples.append(max(0,(arrived-capture)*1000)+float(headers['x-capture-ms']))
            arrivals.append(time.monotonic())
    ordered=sorted(samples)
    return {'frames':len(samples),'arrival_fps':(len(arrivals)-1)/(arrivals[-1]-arrivals[0]),
            'rpc_encode_http_p95_ms':ordered[min(len(ordered)-1,int(len(ordered)*.95))],
            'measurement':'AirSim RPC + JPEG encode + complete local HTTP JPEG arrival; excludes browser paint'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='http://127.0.0.1:8000')
    parser.add_argument('--session',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--instruction',default='请在当前位置向上飞行 3 米，然后沿机头方向前进 2 米，保持悬停并报告任务完成。')
    parser.add_argument('--disconnect-test',action='store_true')
    parser.add_argument('--stop-test',action='store_true')
    args=parser.parse_args()
    import torch
    report={'diagnostics':{'sys.executable':sys.executable,'torch.__version__':torch.__version__,
            'torch.version.cuda':torch.version.cuda,'torch.cuda.is_available()':torch.cuda.is_available()},'session_id':args.session}
    path='/api/sessions/'+args.session
    initial=wait_state(args.base,path,lambda s:s['phase']=='ready')
    controller=Controller(args.base.replace('http','ws',1)+path+'/ws');controller.start()
    try:
        report['preview_before']=measure_preview(args.base,path)
        before=request(args.base,path)['event_seq']
        request(args.base,path+'/instructions',{'instruction':args.instruction,'request_id':uuid.uuid4().hex})
        if args.stop_test:
            deadline=time.monotonic()+60
            while not any(e['kind']=='planner_request_started' for e in request(args.base,path+'/events?after='+str(before))):
                if time.monotonic()>deadline:raise TimeoutError('No planner request started')
                time.sleep(.1)
            request(args.base,path+'/control',{'action':'pause'})
            wait_state(args.base,path,lambda s:s.get('control',{}).get('status')=='paused')
            # Measure the continuously rendered stream during an active paused task.
            report['preview_during_paused_task']=measure_preview(args.base,path)
            request(args.base,path+'/control',{'action':'stop'})
        if args.disconnect_test:
            request(args.base,path+'/control',{'action':'pause'})
            paused=wait_state(args.base,path,lambda s:s.get('control',{}).get('status')=='paused')
            report['pause']=paused['control']
            request(args.base,path+'/control',{'action':'resume'})
            controller.close();started=time.monotonic()
            paused=wait_state(args.base,path,lambda s:s.get('control',{}).get('status')=='paused',timeout=90)
            report['disconnect_pause_elapsed_sec']=time.monotonic()-started
            events=request(args.base,path+'/events?after='+str(before))
            window_count=sum(e['kind']=='physical_window_recorded' for e in events)
            time.sleep(1)
            assert sum(e['kind']=='physical_window_recorded' for e in request(args.base,path+'/events?after='+str(before)))==window_count
            controller=Controller(args.base.replace('http','ws',1)+path+'/ws');controller.start()
            assert request(args.base,path)['control']['desired']=='paused'
            report['reconnect_stays_paused']=True
            request(args.base,path+'/control',{'action':'resume'})
        final=wait_state(args.base,path,lambda s:not s['busy'] and s['summary'] is not None)
        report['summary']=final['summary'];report['preview_after']=measure_preview(args.base,path)
        report['run_id']=final['run_id']
        timeline=request(args.base,'/api/runs/'+final['run_id']+'/timeline')
        report['decisions']=len(timeline['decisions']);report['observations']=len(timeline['observations'])
        report['all_decisions_have_observation']=all(d['observation_id'] for d in timeline['decisions'])
        current_events=request(args.base,path+'/events?after='+str(before))
        report['physical_windows_this_instruction']=sum(e['kind']=='physical_window_recorded' for e in current_events)
        report['discarded_decisions_this_instruction']=sum(e['kind']=='operator_decision_discarded' for e in current_events)
        report['model_rgb_dimensions']=sorted({(f['width'],f['height']) for o in timeline['observations'] for f in o['frames'] if f['modality']=='rgb'})
    finally:
        controller.close()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
