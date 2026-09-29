"""Publish newly generated V2 experiences after independent review and provenance checks."""
import json
from pathlib import Path
import time

from .store import Store, digest
from .experiment import _validate_candidate
from .contracts import VERSION, validate_review


def publish_generated(directory, database):
    """Old artifact imports remain candidate-only; V2 publication is explicit here.

    Returns exact published IDs and versions. All accepted items are committed in
    one transaction, after original source hashes and actual review images match.
    """
    directory=Path(directory)
    result=json.loads((directory/'experiment.json').read_text(encoding='utf-8'))
    if result.get('protocol_version')!=VERSION or result.get('status')!='completed':
        raise ValueError('Only completed V2 generation is eligible')
    store=Store(database)
    store.initialize()
    imported=store.import_experiment(directory)
    import_id=imported['import_id']
    prepared=[]
    for candidate in result['accepted']:
        positions=[i for i,c in enumerate(result['candidates']) if c['id']==candidate['id']]
        if len(positions)!=1:
            raise ValueError('Unknown accepted identity')
        eid=digest({'import':import_id,'offset':positions[0]})[:24]
        current=store.get(eid)
        review=next((r for r in reversed(result['reviews']) if r['id']==candidate['id']),None)
        if not review or review['decision']!='approve':
            raise ValueError('Missing independent approval')
        validate_review({k:review[k] for k in ('decision','reason','evidence_support','condition_boundaries','counterexamples','decision_value','revision_requests')})
        sid=current['evidence'][0]['source_id']
        bundle=store.check_source(sid)
        issues=_validate_candidate(candidate,bundle,set(review['image_ids']),bundle.evidence_ids)
        if issues:
            raise ValueError('Invalid publication evidence')
        if current['state'] not in {'candidate','published'}:
            raise ValueError('Existing experience is withdrawn, archived or superseded')
        review_calls=[c for c in result.get('calls',[]) if c.get('stage','').startswith('review_'+candidate['id']+'_')]
        audit=review_calls[-1] if review_calls else {}
        review={**review,'protocol_version':VERSION,'configured_model':result.get('review_model'),
            'returned_model':audit.get('returned_model'),'prompt_sha256':audit.get('prompt_sha256'),
            'audit_path':str(directory/'calls.json'),'publication_source':'independent_generation_review'}
        prepared.append((eid,candidate,sid,review,(current['state'],current['version'],current['published_version'])))
    published=[]
    with store.connect(write=True) as db:
        for eid,candidate,sid,review,base in prepared:
            current=store._get(db,eid)
            if (current['state'],current['version'],current['published_version']) != base:
                raise ValueError('Version conflict before generation publication')
            if current['state']=='published':
                published.append({'id':eid,'version':current['published_version']})
                continue
            version=store._version(db,eid,candidate,[{'source_id':sid,'evidence_refs':candidate['evidence_refs'],
                'observation_ids':candidate['observation_ids'],'relation':'support'}])
            db.execute("UPDATE experiences SET state='published',published_version=? WHERE id=?",(version,eid))
            store._review(db,eid,version,'published',review['reason'],'model_review:v2',review)
            published.append({'id':eid,'version':version})
    return published
