"""Independent reader frame selection and bounded original evidence loading."""
from dataclasses import replace
from copy import deepcopy

from .contracts import SELECTION, prompt


def select(calls, stage, client, bundles, required, *, extra=None, limit=8):
    schema=deepcopy(SELECTION)
    selection=schema['properties']['selections']['items']['properties']
    selection['source_id']={'type':'string','enum':sorted(bundles)}
    observations=sorted({obs for bundle in bundles.values() for obs in bundle.images})
    if not observations:
        raise ValueError('No available original frames')
    selection['observation_ids']['items']={'type':'string','enum':observations}
    result = calls.ask(stage, client, prompt('select'), {
        'indices':{sid:b.temporal_index() for sid,b in bundles.items()},
        'required':required, 'image_budget':limit, 'material':extra}, schema)
    shown = {sid:set(obs) for sid,obs in required.items()}
    for entry in result['selections']:
        if entry['source_id'] not in bundles:
            raise ValueError('Unknown selection source')
        shown.setdefault(entry['source_id'],set()).update(entry['observation_ids'])
    if sum(map(len,shown.values())) > limit:
        raise ValueError('Selection exceeds image budget')
    if any(set(obs)-set(bundles[sid].images) for sid,obs in shown.items()):
        raise ValueError('Selection references unavailable original images')
    return {sid:sorted(obs) for sid,obs in shown.items()}


def materials(bundles, shown, refs):
    evidence, supplied = {}, {}
    for sid,bundle in bundles.items():
        observations = shown.get(sid,[])
        part = bundle.excerpt(observations, refs.get(sid,[]))
        evidence[sid] = part
        supplied[sid] = replace(bundle,timeline=part['timeline'],
            observations={o:bundle.observations[o] for o in observations})
    return evidence,supplied
