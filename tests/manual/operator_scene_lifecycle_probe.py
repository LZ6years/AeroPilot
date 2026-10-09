"""Attach to an existing CityEnviron, close clients, verify the owner survives."""

import argparse
import dataclasses
import json
import time
from pathlib import Path

from src.adapters.airsim.client import probe_airsim_endpoint
from src.runtime.configs.map_config import load_map_config
from src.sim_server.server import SimulationServer
from src.sim_server.preview import PreviewConfig
from tests.manual.operator_workbench_probe import request, wait_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--owner-session', help='Also close the managed API owner and verify its port is released')
    args=parser.parse_args()
    maps,ports=load_map_config('configs/maps/airvln_maps.json')
    package=next(m for m in maps if m.map_id=='cityenviron_win64')
    package=dataclasses.replace(package,airsim={**package.airsim,'port':args.port})
    server=SimulationServer([package],ports,run_dir=args.output.parent/'attach-scene',execution_mode='attach',preview_config=PreviewConfig())
    scene=server.start_scene(package.map_id)
    preview=server.open_preview(scene.scene_instance_id)
    try:
        deadline=time.monotonic()+15
        while preview.latest() is None:
            if time.monotonic()>deadline:raise TimeoutError(preview.stats())
            time.sleep(.1)
        report={'scene':scene.to_dict(),'preview':preview.stats()}
        assert scene.process_ownership=='external' and scene.process_id is None
    finally:
        server.stop_scene(scene.scene_instance_id)
    alive,detail=probe_airsim_endpoint(scene.control_host,scene.control_port)
    report.update({'external_scene_survived':alive,'probe':detail,
                  'preview_services_released':not server.preview_services,'backend_clients_released':not server.backends,
                  'ports_released':not server._reserved_ports,'owned_processes_released':not server.local_scene_control.manager._processes})
    assert alive
    if args.owner_session:
        base='http://127.0.0.1:8000';path='/api/sessions/'+args.owner_session
        request(base,path,method='DELETE')
        closed=wait_state(base,path,lambda s:s['phase']=='closed',timeout=30)
        report['managed_owner_final_phase']=closed['phase']
        still_alive,detail=probe_airsim_endpoint(scene.control_host,scene.control_port)
        report['managed_rpc_port_released']=not still_alive
        report['managed_close_probe']=detail
        assert not still_alive
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
