"""V2 synthetic-fixture contracts: automatic publication never bypasses evidence."""
import json
import sqlite3
import threading
import time
import pytest
from tests.test_flight_experience import source,FakeClient,_run,_review,_selection
from plugins.research.flight_experience.store import Store,dumps
from plugins.research.flight_experience.maintenance import Worker,publish
from plugins.research.flight_experience.calls import Calls
from plugins.research.flight_experience.contracts import CHANGE,prompt


def test_truncated_output_is_not_checkpointed(tmp_path):
    from src.llm import ModelResponse
    from plugins.research.flight_experience.contracts import obj,STRING
    class Truncated:
        def request(self,request):
            return ModelResponse({'choices':[{'finish_reason':'length','message':{'content':'{"text":"partial"}'}}],
                'usage':{'prompt_tokens':100,'completion_tokens':10}})
    calls=Calls(tmp_path)
    with pytest.raises(ValueError,match='truncated'):
        calls.ask('reflection',Truncated(),'prompt',{},obj(text=STRING))
    assert calls.records[0]['usage']['completion_tokens']==10
    assert calls.records[0]['status']=='failed'
    assert not list((tmp_path/'checkpoints').glob('*.json'))


def test_generation_budgets_are_bounded():
    from plugins.research.flight_experience import ExperimentSettings
    for settings in ({'max_images':9},{'max_candidates':4}):
        with pytest.raises(ValueError):
            ExperimentSettings(**settings)


def test_selection_schema_enumerates_original_source_and_images(source,tmp_path):
    from plugins.research.flight_experience import read_run
    from plugins.research.flight_experience.readers import select
    client=FakeClient([_selection('material/c1')])
    with pytest.raises(ValueError,match='enum'):
        select(Calls(tmp_path), 'review_select',client,{'run':read_run(source)},{'run':['o1']})
    properties=client.requests[0].parameters['response_format']['json_schema']['schema']['properties']['selections']['items']['properties']
    assert properties['source_id']['enum']==['run']
    assert properties['observation_ids']['items']['enum']==['o1']


@pytest.fixture
def library(source,tmp_path):
    _run(source,tmp_path/'experiment')
    store=Store(tmp_path/'library.sqlite');store.initialize()
    ids=store.import_experiment(tmp_path/'experiment')['experience_ids']
    return store,ids[0],tmp_path/'experiment'


def change_for(card,**updates):
    body={k:card[k] for k in ('title','decision_question','applicability','observed_evidence','recommendation','limitations')}
    body['tags']={k:card.get('tags',{}).get(k,[]) for k in ('decision','action','observation','anomaly','platform','control','sensor')}
    body.update(updates)
    body['tags']={k:body['tags'].get(k,[]) for k in ('decision','action','observation','anomaly','platform','control','sensor')}
    return {'operation':'revise','reason':'Synthetic fixture evidence','outputs':[{'target_id':card['id'],'body':body,
        'evidence':card['evidence']}],'retire_ids':[],'conflict_ids':[]}


def run_job(store,eid,path,change=None,review=None):
    store.enqueue([eid]);sid=store.get(eid)['evidence'][0]['source_id']
    worker=Worker(store,FakeClient([change or change_for(store.get(eid))]),
        FakeClient([_selection(sid),review or _review()]),path/'audits')
    return worker.run_once(),worker


def publish_fixture(store,eid,path,**updates):
    result,_=run_job(store,eid,path,change_for(store.get(eid),**updates))
    assert result['status']=='completed',result
    assert result.get('published'),result
    return result


def test_import_readonly_immutable_scope_and_search(library,tmp_path):
    store,eid,experiment=library
    assert store.import_experiment(experiment)['duplicate']
    assert store.get(eid)['state']=='candidate'
    assert not store.search('motion',scope='experiment')
    assert store.search('motion',maintenance=True)
    with pytest.raises(ValueError):store.import_experiment(experiment,scope='production')
    publish_fixture(store,eid,tmp_path,recommendation='街道尽头 road end',tags={'sensor':['rgb']})
    for q in ('街道尽头','ROAD END'):assert store.search(q,scope='experiment')
    assert not store.search('road')
    assert not store.search('road',scope='experiment',filters={'sensor':'lidar'})
    assert store.search('road',scope='experiment',filters={'platform':'unknown'})[0]['unknown_conditions']==['platform']
    assert not store.search('road',scope='experiment',budget=10)
    assert not store.search('xyzunknown',scope='experiment')
    with store.connect(write=True) as db:
        with pytest.raises(sqlite3.IntegrityError):db.execute("UPDATE versions SET body_json='{}'")
        with pytest.raises(sqlite3.IntegrityError):db.execute('DELETE FROM versions')
    with store.connect() as db:
        with pytest.raises(sqlite3.OperationalError):db.execute("UPDATE experiences SET state='candidate'")


@pytest.mark.parametrize('decision',['defer','discard'])
def test_unapproved_never_publishes(library,tmp_path,decision):
    store,eid,_=library
    result,_=run_job(store,eid,tmp_path,review=_review(decision))
    assert result['status']=='completed' and not result.get('published')
    assert store.get(eid)['state']=='candidate'
    with pytest.raises(ValueError):publish(store,result['proposal_id'],actor='human',reason='override')


@pytest.mark.parametrize('final',['approve','revise'])
def test_bounded_revision_reaudited(library,tmp_path,final):
    store,eid,_=library;card=store.get(eid);sid=card['evidence'][0]['source_id']
    store.enqueue([eid])
    worker=Worker(store,FakeClient([change_for(card),change_for(card,recommendation='Narrow motion advice')]),
        FakeClient([_selection(sid),_review('revise'),_selection(sid),_review(final)]),tmp_path/'revision')
    result=worker.run_once()
    assert result['status']=='completed',result
    assert len(worker.reflection.requests)==2
    if final=='approve':assert result['published'] and store.get(eid)['recommendation']=='Narrow motion advice'
    else:assert result['review']['decision']=='defer' and store.get(eid)['version']==1


def test_unchanged_group_reuse_without_publishing(library,tmp_path):
    store,eid,_=library
    change={'operation':'unchanged','reason':'No supported update','outputs':[],'retire_ids':[],'conflict_ids':[]}
    first,_=run_job(store,eid,tmp_path,change)
    assert first['published']==[] and store.get(eid)['state']=='candidate'
    store.enqueue([eid]);worker=Worker(store,FakeClient([]),FakeClient([]),tmp_path/'audits')
    second=worker.run_once();assert second['group_reused'] and not worker.reflection.requests


def test_review_timeout_resumes_successful_reflection(library,tmp_path):
    store,eid,_=library;card=store.get(eid);sid=card['evidence'][0]['source_id']
    store.enqueue([eid])
    worker=Worker(store,FakeClient([change_for(card)]),FakeClient([_selection(sid),TimeoutError('SECRET')]),tmp_path/'resume')
    assert worker.run_once()['status']=='failed'
    store.enqueue([eid]);resumed=Worker(store,FakeClient([]),FakeClient([_review()]),tmp_path/'resume')
    result=resumed.run_once();assert result['published'] and not resumed.reflection.requests
    records=json.loads(next((tmp_path/'resume'/result['job_id']).rglob('calls.json')).read_text(encoding='utf-8'))
    assert [c['status'] for c in records[:2]]==['reused','reused']
    assert 'SECRET' not in ''.join(p.read_text(encoding='utf-8') for p in (tmp_path/'resume').rglob('calls.json'))


def test_format_repair_cannot_change_semantics(library,tmp_path):
    store,eid,_=library;bad=change_for(store.get(eid));bad['outputs'][0]['body']['tags']['action']='forward'
    fixed=json.loads(dumps(bad));fixed['outputs'][0]['body']['tags']['action']=['forward']
    calls=Calls(tmp_path/'format')
    assert calls.ask('reflection',FakeClient([bad,fixed]),prompt('maintain'),{},CHANGE)==fixed
    assert len(calls.records)==2
    changed=json.loads(dumps(fixed));changed['outputs'][0]['body']['recommendation']='Invented advice'
    with pytest.raises(ValueError,match='changed content'):
        Calls(tmp_path/'badrepair').ask('reflection',FakeClient([bad,changed]),prompt('maintain'),{},CHANGE)


def test_invalid_reference_not_repaired_or_reviewed(library,tmp_path):
    store,eid,_=library;change=change_for(store.get(eid));change['outputs'][0]['evidence'][0]['evidence_refs'].append('fabricated')
    result,worker=run_job(store,eid,tmp_path,change)
    assert result['status']=='failed' and not worker.reviewer.requests and store.get(eid)['state']=='candidate'


def test_missing_source_blocks_calls(library,source,tmp_path):
    store,eid,_=library;(source/'frames.jsonl').unlink()
    result,worker=run_job(store,eid,tmp_path)
    assert result['status']=='failed' and not worker.reflection.requests
    assert 'unavailable' in store.audit_sources().values()


def test_publication_rechecks_source(library,source,tmp_path):
    store,eid,_=library;sid=store.get(eid)['evidence'][0]['source_id']
    class Changing(FakeClient):
        def request(self,r):
            answer=super().request(r)
            if r.parameters['response_format']['json_schema']['name']=='flight_experience_review_0':(source/'frames.jsonl').unlink()
            return answer
    store.enqueue([eid])
    result=Worker(store,FakeClient([change_for(store.get(eid))]),Changing([_selection(sid),_review()]),tmp_path/'changed').run_once()
    assert result['status']=='failed' and store.get(eid)['state']=='candidate'


def test_search_during_model_and_lease_exclusion(library,tmp_path):
    store,eid,_=library;publish_fixture(store,eid,tmp_path)
    entered,release=threading.Event(),threading.Event()
    class Blocking(FakeClient):
        def request(self,r):
            entered.set();assert release.wait(10)
            return super().request(r)
    sid=store.get(eid)['evidence'][0]['source_id'];store.enqueue([eid]);results=[]
    worker=Worker(store,Blocking([change_for(store.get(eid))]),FakeClient([_selection(sid),_review()]),tmp_path/'blocking',lease_seconds=1)
    thread=threading.Thread(target=lambda:results.append(worker.run_once()));thread.start()
    try:
        assert entered.wait(5)
        start=time.monotonic();assert store.search('motion',scope='experiment')[0]['version']==2
        assert time.monotonic()-start<1
        assert Worker(store,None,None,tmp_path/'other').claim() is None
    finally:release.set();thread.join(10)
    assert results[0]['published'] and store.get(eid)['version']==3


def test_lease_recovery_and_dedup(library,tmp_path):
    store,eid,_=library;ids=store.enqueue([eid],request_key='batch');assert store.enqueue([eid],request_key='batch')==ids
    dead=Worker(store,None,None,tmp_path/'dead',lease_seconds=.01);attempt=dead.claim();time.sleep(.03)
    result,_=run_job(store,eid,tmp_path);assert result['published']
    with store.connect() as db:assert db.execute('SELECT state FROM attempts WHERE id=?',(attempt['attempt_id'],)).fetchone()[0]=='interrupted'


def test_concurrent_version_change_requeues(library,tmp_path):
    store,eid,_=library;card=store.get(eid)
    class Changing(FakeClient):
        def request(self,r):
            with store.connect(write=True) as db:store._version(db,eid,card,card['evidence'])
            return super().request(r)
    store.enqueue([eid]);sid=card['evidence'][0]['source_id']
    result=Worker(store,Changing([change_for(card)]),FakeClient([_selection(sid),_review()]),tmp_path/'conflict').run_once()
    assert result['status']=='conflict' and store.jobs()[-1]['state']=='pending' and not store.proposals()


@pytest.mark.parametrize('operation',['merge','split','archive','withdraw','conflict'])
def test_maintenance_operations_keep_history(library,tmp_path,operation):
    store,eid,_=library;card=store.get(eid)
    with store.connect(write=True) as db:
        db.execute("INSERT INTO experiences(id) VALUES ('duplicate')");store._version(db,'duplicate',card,card['evidence'])
    change=change_for(card);change['operation']=operation
    if operation=='merge':change['retire_ids']=['duplicate']
    elif operation=='split':
        change['outputs'][0]['target_id']=None
        other=json.loads(dumps(change['outputs'][0]));other['body']['applicability']+=' different conditions'
        change['outputs'].append(other);change['retire_ids']=[eid]
    else:
        change['outputs']=[]
        if operation=='conflict':change['conflict_ids']=[eid,'duplicate']
        else:change['retire_ids']=[eid]
    result,_=run_job(store,eid,tmp_path,change);assert result['status']=='completed',result
    assert store.get(eid,1)
    if operation=='merge':assert store.get('duplicate')['state']=='superseded'
    if operation=='split':assert len(result['published'])==2
    if operation=='archive':assert store.get(eid)['state']=='archived'
    if operation=='withdraw':assert store.get(eid)['state']=='withdrawn'


def test_lost_counterexample_rejected(library,tmp_path):
    store,eid,_=library;card=store.get(eid)
    with store.connect(write=True) as db:store._version(db,eid,card,[*card['evidence'],{**card['evidence'][0],'relation':'counterexample'}])
    change=change_for(store.get(eid));change['outputs'][0]['evidence']=card['evidence']
    result,worker=run_job(store,eid,tmp_path,change);assert result['status']=='failed' and not worker.reviewer.requests


def test_publication_rolls_back(library,tmp_path):
    store,eid,_=library
    with store.connect(write=True) as db:db.execute("CREATE TRIGGER block_publish BEFORE UPDATE OF state ON experiences WHEN NEW.state='published' BEGIN SELECT RAISE(ABORT,'fixture'); END")
    result,_=run_job(store,eid,tmp_path)
    assert result['status']=='failed' and store.get(eid)['version']==1
    assert store.proposals()[0]['state']=='awaiting_auto'


def test_migration_rollback_preserves_old_versions(library):
    store,eid,_=library;before=store.get(eid)
    with store.connect(write=True) as db:
        db.execute('DROP TABLE maintenance_groups');db.execute("UPDATE meta SET value='1'")
        db.execute("CREATE TRIGGER fail_migration BEFORE UPDATE ON meta BEGIN SELECT RAISE(ABORT,'fixture'); END")
    with pytest.raises(sqlite3.IntegrityError):store.initialize()
    with store.connect(write=True) as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_groups'").fetchone()
        assert db.execute('SELECT value FROM meta').fetchone()[0]=='1';db.execute('DROP TRIGGER fail_migration')
    store.initialize();assert store.get(eid)==before


def test_legacy_approval_not_auto_published(library):
    store,eid,_=library;jid=store.enqueue([eid])[0];change=change_for(store.get(eid))
    with store.connect(write=True) as db:db.execute('INSERT INTO proposals VALUES (?,?,?,?,?,?,?)',('old',jid,'{}',dumps(change),dumps({'decision':'approve'}),'awaiting_human',time.time()))
    with pytest.raises(ValueError):publish(store,'old',actor='model',reason='test',automatic=True)


def test_plugin_dispatch_and_provider_projection(library,tmp_path):
    from src.plugins import PluginRegistry
    from src.tools.registry import default_registry
    from src.tools.dispatcher import ToolDispatcher
    from src.llm import ModelRequest,ProviderSettings
    from src.llm.providers.openai_compatible import encode_request
    store,eid,_=library;publish_fixture(store,eid,tmp_path)
    assert 'skill.flight_experience.search' not in PluginRegistry.from_config().skill_capability_catalog.by_call_name()
    registry=PluginRegistry.discover({'plugins':{'roots':['plugins/capabilities'],'enabled':['capabilities.flight_experience']}})
    _,cap=registry.skill_capability_catalog.by_call_name()['skill.flight_experience.search'];cap.settings.update(database_path=str(store.path),scope='experiment')
    from src.plugins.skills.dispatcher import SkillCapabilityDispatcher
    tools=default_registry(registry);dispatcher=ToolDispatcher(
        None, registry=tools,
        capability_dispatcher=SkillCapabilityDispatcher(registry),
    )
    result=dispatcher.call('scene','agent',cap.call_name,{'query':'motion'});assert result.ok and result.data['experiences'][0]['version']==2
    with pytest.raises(ValueError):dispatcher.call('s','a',cap.call_name,{'query':'motion','limit':6})
    wire=encode_request(ProviderSettings(model_name='fixture'),ModelRequest(messages=[],tools=[tools.get(cap.call_name).to_provider_tool()]))
    assert wire['tools'][0]['function']['name']=='skill_flight_experience_search' and str(store.path) not in dumps(wire)


def test_missing_db_and_output_boundaries(library,source,tmp_path):
    missing=Store(tmp_path/'missing.sqlite')
    with pytest.raises(sqlite3.OperationalError):missing.search('road')
    assert not missing.path.exists()
    store,_,experiment=library
    for output in (source/'audit',experiment/'audit',tmp_path):
        with pytest.raises(ValueError):Worker(store,None,None,output)


def test_generation_publication_is_idempotent_after_report_update(source,tmp_path):
    from plugins.research.flight_experience.publication import publish_generated
    result,_,_=_run(source,tmp_path/'generated')
    assert result['published']
    assert publish_generated(tmp_path/'generated',tmp_path/'generated.sqlite')==result['published']
    store=Store(tmp_path/'generated.sqlite')
    assert len(store.list_ids())==1 and store.get(store.list_ids()[0])['version']==2
    store.set_state(store.list_ids()[0],'withdrawn',actor='test',reason='Invalidated after publication')
    with pytest.raises(ValueError,match='withdrawn'):
        publish_generated(tmp_path/'generated',tmp_path/'generated.sqlite')
    assert store.get(store.list_ids()[0])['state']=='withdrawn'


def test_checkpoint_invalidation_on_model_or_input(tmp_path):
    from src.llm import ProviderSettings
    from plugins.research.flight_experience.contracts import obj,STRING
    schema=obj(answer=STRING)
    first=FakeClient([{'answer':'original'}]);first.settings=ProviderSettings(model_name='one')
    Calls(tmp_path/'a',cache=tmp_path/'cache').ask('stage',first,'system',{'evidence':'A'},schema)
    second=FakeClient([{'answer':'different model'}]);second.settings=ProviderSettings(model_name='two')
    assert Calls(tmp_path/'b',cache=tmp_path/'cache').ask('stage',second,'system',{'evidence':'A'},schema)['answer']=='different model'
    third=FakeClient([{'answer':'different evidence'}]);third.settings=ProviderSettings(model_name='one')
    assert Calls(tmp_path/'c',cache=tmp_path/'cache').ask('stage',third,'system',{'evidence':'B'},schema)['answer']=='different evidence'


def test_malformed_review_cannot_publish(library,tmp_path):
    store,eid,_=library
    result,_=run_job(store,eid,tmp_path,review={'decision':'approve','reason':'missing assessments'})
    assert result['status']=='failed' and store.get(eid)['state']=='candidate'


def test_missing_rgb_and_image_budget_prevent_model_call(library,tmp_path):
    store,eid,_=library;card=store.get(eid)
    with store.connect(write=True) as db:
        store._version(db,eid,card,[{**card['evidence'][0],'observation_ids':[f'obs{i}' for i in range(9)]}])
    result,worker=run_job(store,eid,tmp_path)
    assert result['status']=='failed' and not worker.reflection.requests
