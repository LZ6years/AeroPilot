"""V2 first-generation reflection, independent review and one bounded revision."""
from pathlib import Path
import time

from .calls import Calls, write
from .contracts import GENERATION, REVIEW, prompt, validate_review
from .readers import select, materials


def run_generation(bundle, output, *, reflection, reviewer, settings, parameters=None):
    # Imported at call time to keep the existing public helpers/report adapter.
    from .experiment import prepare_output, _validate_candidate, _write_report
    from .publication import publish_generated
    output = prepare_output(bundle, Path(output))
    calls = Calls(output, cache=settings.checkpoint_directory, parameters=parameters,
        prices={role+'_'+direction+'_price':getattr(settings,role+'_'+direction+'_price')
            for role in ('reflection','review') for direction in ('input','output')})
    result = {'protocol_version':'v2','status':'running','source':str(bundle.source),
        'experiment_only':True,'closed_loop_benefit':'not_evaluated','candidates':[],
        'reviews':[],'accepted':[],'calls':calls.records,'published':[],
        'reflection_model':settings.reflection_model,'review_model':settings.review_model,
        'frame_selection':'explicit_observation_ids' if settings.observation_ids else 'model_index_selection',
        'revisions':[],'estimated_cost_usd':None}
    started = time.monotonic()
    images = {}
    bundles = {'run':bundle}
    try:
        shown = {'run':list(settings.observation_ids)} if settings.observation_ids else select(
            calls,'reflection_select',reflection,bundles,{},limit=settings.max_images)
        if not shown.get('run') or set(shown['run'])-set(bundle.images):
            raise ValueError('No valid original frames selected')
        evidence,supplied = materials(bundles,shown,{})
        generated = calls.ask('reflection',reflection,prompt('generate'),
            {'original_evidence':evidence,'candidate_limit':settings.max_candidates}, GENERATION,bundles=bundles,shown=shown)
        candidates = generated['candidates']
        if len(candidates)>settings.max_candidates or len({c['id'] for c in candidates})!=len(candidates):
            raise ValueError('Candidate budget or identity violation')
        result['candidates'] = candidates
        for candidate in candidates:
            cid = candidate['id']
            problems = _validate_candidate(candidate,bundle,set(shown['run']),supplied['run'].evidence_ids)
            if problems:
                result['reviews'].append({'id':cid,'decision':'defer','validator_issues':problems})
                continue
            current = candidate
            for revision in range(2):
                label = f'{cid}_{revision}'
                review_shown = select(calls,'review_select_'+label,reviewer,bundles,
                    {'run':current['observation_ids']},extra=current,limit=settings.max_images)
                review_evidence,review_supplied = materials(bundles,review_shown,{'run':current['evidence_refs']})
                review = calls.ask('review_'+label,reviewer,prompt('review'),
                    {'candidate':current,'original_evidence':review_evidence},REVIEW,bundles=bundles,shown=review_shown)
                validate_review(review)
                problems = _validate_candidate(current,bundle,set(review_shown['run']),review_supplied['run'].evidence_ids)
                if problems:
                    review = {**review,'decision':'defer','validator_issues':problems}
                result['reviews'].append({'id':cid,'round':revision,**review,'image_ids':review_shown['run']})
                if review['decision']=='approve':
                    result['accepted'].append(current)
                    break
                if review['decision']!='revise' or revision==1:
                    if review['decision']=='revise':
                        result['reviews'][-1]['decision']='defer'
                        result['reviews'][-1]['bounded_revision_exhausted']=True
                    break
                revised = calls.ask('revision_'+label,reflection,prompt('revise'),
                    {'candidate':current,'review':review,'original_evidence':review_evidence,'candidate_limit':1},
                    GENERATION,bundles=bundles,shown=review_shown)['candidates']
                result['revisions'].append({'id':cid,'candidates':revised})
                if not revised:
                    result['reviews'].append({'id':cid,'decision':'defer','reason':'No supported revision'})
                    break
                if len(revised)!=1 or revised[0]['id']!=cid:
                    raise ValueError('Revision changed candidate identity/count')
                current = revised[0]
                problems = _validate_candidate(current,bundle,set(review_shown['run']),review_supplied['run'].evidence_ids)
                if problems:
                    result['reviews'].append({'id':cid,'decision':'defer','validator_issues':problems})
                    break
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',accepted=[],error_type=type(exc).__name__,category=getattr(exc,'category',None))
        if type(exc) is ValueError:
            result['validation_issue']=str(exc)
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        costs=[c.get('estimated_cost_usd') for c in calls.records if c['status']!='reused']
        result['estimated_cost_usd']=sum(costs) if costs and all(c is not None for c in costs) else None
        write(output/'experiment.json',result)
        # Media are persisted by the shared audit; link original bytes by digest.
        for obs,info in bundle.observations.items():
            matches=list((output/'media').glob(str(info.get('sha256'))+'.*')) if (output/'media').exists() else []
            if matches:
                images[obs]=matches[0].relative_to(output).as_posix()
        if result['status']=='completed' and result['accepted']:
            try:
                result['published']=publish_generated(output,settings.library_path or output.with_suffix('.sqlite'))
            except Exception as exc:
                result.update(status='failed',publication_error=type(exc).__name__)
        write(output/'experiment.json',result)
        write(output/'human_review.json',{'status':'optional_audit','automatic_review':True,'notes':[]})
        _write_report(output,bundle,result,images)
    return result
