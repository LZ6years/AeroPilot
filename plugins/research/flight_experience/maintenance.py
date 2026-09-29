"""Leased offline maintenance worker and independent evidence-backed review.

V2 independently approved changes publish automatically after evidence/version
checks. No model call holds a SQLite transaction. A heartbeat fences stale workers;
publication checks the exact versions and states seen by the reviewer.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time
import uuid

from src.llm import ModelRequest
from src.llm.audit import snapshot_request
from .experiment import _validate_candidate, write_json
from .contracts import CHANGE, REVIEW, VERSION, prompt, prompt_manifest, validate_review
from .calls import Calls
from .readers import select, materials
from .store import Store, dumps, normalize_body, digest, tokens


OPERATIONS = {"add", "supplement", "revise", "merge", "split", "conflict", "archive", "withdraw", "unchanged"}
PROPOSAL_CONTRACT = prompt('maintain')


def bases(cards):
    return {c["id"]: {"version": c["version"], "state": c["state"], "published_version": c["published_version"]} for c in cards}


def check_bases(db, base):
    for eid, expected in base.items():
        actual = Store._get(db, eid)
        if any(actual[k] != v for k, v in expected.items()):
            raise ValueError("Version conflict; submit fresh maintenance")


def validate_change(change, cards, bundles, shown):
    """Structural/provenance checks; independent review still owns semantic quality."""
    if not isinstance(change, dict) or set(change) != {"operation", "reason", "outputs", "retire_ids", "conflict_ids"}:
        raise ValueError("Invalid change shape")
    op = change["operation"]
    if op not in OPERATIONS or not isinstance(change["reason"], str) or not change["reason"].strip():
        raise ValueError("Invalid operation/reason")
    by_id = {c["id"]: c for c in cards}
    for key in ("retire_ids", "conflict_ids"):
        values = change[key]
        if not isinstance(values, list) or any(not isinstance(x, str) for x in values) or len(set(values)) != len(values) or set(values) - set(by_id):
            raise ValueError("Unknown/duplicate affected ID")
    outputs = change["outputs"]
    if not isinstance(outputs, list) or len(outputs) > 6:
        raise ValueError("Invalid outputs")
    if op in {"unchanged", "conflict", "archive", "withdraw"} and outputs:
        raise ValueError("Non-content operation cannot rewrite content")
    if op in {"add", "supplement", "revise", "merge"} and len(outputs) != 1:
        raise ValueError("Operation requires one output")
    if op == "split" and len(outputs) < 2:
        raise ValueError("Split requires multiple outputs")
    if op not in {"merge", "split", "archive", "withdraw"} and change["retire_ids"]:
        raise ValueError("Unexpected retirement")
    if op in {"merge", "split", "archive", "withdraw"} and not change["retire_ids"]:
        raise ValueError("Retirement IDs required")
    if (op == "conflict" and len(change["conflict_ids"]) < 2) or (op != "conflict" and change["conflict_ids"]):
        raise ValueError("Invalid conflict IDs")
    targets, all_links = set(), []
    for output in outputs:
        if not isinstance(output, dict) or set(output) != {"target_id", "body", "evidence"}:
            raise ValueError("Invalid output")
        target = output["target_id"]
        if target is not None and (target not in by_id or target in targets):
            raise ValueError("Unknown/duplicate target")
        if op in {"add", "split"} and target is not None:
            raise ValueError("New output required")
        if op in {"supplement", "revise"} and target is None:
            raise ValueError("Existing target required")
        if target in change["retire_ids"]:
            raise ValueError("Cannot retire an output target")
        if target is not None:
            targets.add(target)
        output["body"] = normalize_body(output["body"])
        links = output["evidence"]
        if not isinstance(links, list) or not links:
            raise ValueError("Evidence required")
        for link in links:
            if not isinstance(link, dict) or set(link) != {"source_id", "evidence_refs", "observation_ids", "relation"}:
                raise ValueError("Invalid evidence link")
            sid = link["source_id"]
            if sid not in bundles or link["relation"] not in {"support", "counterexample"}:
                raise ValueError("Unknown source/relation")
            bundle = bundles[sid]
            candidate = {"id": "proposal", **output["body"], **link}
            problems = _validate_candidate(candidate, bundle, set(shown[sid]), bundle.evidence_ids)
            if problems:
                raise ValueError("Invalid evidence: " + ",".join(problems))
            all_links.append(link)
    # A rewrite must not silently lose the very evidence/restrictions being compared.
    affected = targets | set(change["retire_ids"])
    if outputs:
        for eid in affected:
            for old in by_id[eid]["evidence"]:
                if not any(new["source_id"] == old["source_id"] and new["relation"] == old["relation"]
                    and set(old["evidence_refs"]) <= set(new["evidence_refs"])
                    and set(old["observation_ids"]) <= set(new["observation_ids"]) for new in all_links):
                    raise ValueError("Original evidence or counterexample lost")
    return change


class Worker:
    """One leased worker per database. lease_seconds controls crash recovery only."""
    def __init__(self, store, reflection, reviewer, output, *, parameters=None, lease_seconds=120):
        self.store, self.reflection, self.reviewer = store, reflection, reviewer
        self.output = Path(output).resolve()
        self.parameters = parameters or {}
        if lease_seconds <= 0:
            raise ValueError("Worker lease must be positive")
        with store.connect() as db:
            protected = [Path(r[0]).resolve() for r in db.execute("SELECT root FROM sources UNION SELECT path FROM imports")]
        if any(self.output.is_relative_to(root) or root.is_relative_to(self.output) for root in protected):
            raise ValueError("Maintenance output must be separate from original runs and experiments")
        self.lease_seconds = lease_seconds
        self.owner = uuid.uuid4().hex

    def claim(self):
        now = time.time()
        with self.store.connect(write=True) as db:
            if db.execute("SELECT 1 FROM jobs WHERE state='running' AND lease_until>?", (now,)).fetchone():
                return None
            # Expired attempts stay visible instead of being replaced by a retry.
            db.execute("UPDATE attempts SET state='interrupted',ended=? WHERE state='running' AND job_id IN "
                "(SELECT id FROM jobs WHERE state='running' AND lease_until<=?)", (now, now))
            db.execute("UPDATE jobs SET state='pending',owner=NULL WHERE state='running' AND lease_until<=?", (now,))
            row = db.execute("SELECT * FROM jobs WHERE state='pending' ORDER BY created,id LIMIT 1").fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET state='running',owner=?,lease_until=?,attempts=attempts+1,error=NULL WHERE id=?",
                (self.owner, now+self.lease_seconds, row["id"]))
            aid = uuid.uuid4().hex
            db.execute("INSERT INTO attempts(id,job_id,started,state) VALUES (?,?,?,'running')", (aid, row["id"], now))
            return {**dict(row), "attempt_id": aid}

    @contextmanager
    def heartbeat(self, jid):
        stop = threading.Event()
        def beat():
            while not stop.wait(max(.05, self.lease_seconds/3)):
                try:
                    with self.store.connect(write=True) as db:
                        db.execute("UPDATE jobs SET lease_until=? WHERE id=? AND owner=? AND state='running'",
                            (time.time()+self.lease_seconds, jid, self.owner))
                except Exception:
                    # Commit checks the lease; a failed heartbeat cannot publish.
                    return
        thread = threading.Thread(target=beat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=5)

    def _materials(self, eid):
        target = self.store.get(eid)
        query = target.get('decision_question') or target['recommendation']
        related = self.store.search(query + ' ' + dumps(target.get('tags',{})), maintenance=True, limit=10, budget=100000)
        cards = [target]
        bundles, shown = {}, {}
        def include(card):
            proposed = {sid: set(obs) for sid, obs in shown.items()}
            for link in card["evidence"]:
                proposed.setdefault(link["source_id"], set()).update(link["observation_ids"])
            if sum(map(len, proposed.values())) > 8:
                return False
            for link in card["evidence"]:
                sid = link["source_id"]
                if sid not in bundles:
                    bundles[sid] = self.store.check_source(sid)
                problems = _validate_candidate({**card, **link}, bundles[sid], set(bundles[sid].images), bundles[sid].evidence_ids)
                if problems:
                    raise ValueError("Unavailable cited evidence")
            shown.clear()
            shown.update({sid: sorted(obs) for sid, obs in proposed.items()})
            return True
        if not include(target):
            raise ValueError("Target evidence exceeds eight-image budget")
        for entry in related:
            if len(cards)>=3:
                break
            if entry["id"] == eid or any(c["id"] == entry["id"] for c in cards):
                continue
            card = self.store.get(entry["id"])
            # Prefer explicit decision identity. Legacy entries fall back to text;
            # lexical overlap is candidate selection, never approval to merge.
            target_decisions = set(target.get('tags',{}).get('decision',[]))
            other_decisions = set(card.get('tags',{}).get('decision',[]))
            if target_decisions and other_decisions and not target_decisions & other_decisions:
                continue
            if include(card):
                cards.append(card)
        # Complete cited events plus neighboring windows; no summary substitutes.
        evidence = {}
        for sid, bundle in bundles.items():
            refs = [ref for c in cards for l in c["evidence"] if l["source_id"] == sid for ref in l["evidence_refs"]]
            evidence[sid] = bundle.excerpt(shown[sid], refs)
        return cards, bundles, shown, evidence

    def run_once(self):
        """Resume successful stages by exact input digest; auto-publish only V2 review."""
        job=self.claim()
        if not job:
            return None
        directory=self.output/job['id']/job['attempt_id']
        directory.mkdir(parents=True,exist_ok=False)
        calls=Calls(directory,cache=self.output/'checkpoints',parameters=self.parameters)
        result={'job_id':job['id'],'attempt_id':job['attempt_id'],'status':'running','protocol_version':VERSION,'estimated_cost_usd':None}
        stage='evidence_loading'
        target_id=json.loads(job['payload_json'])['id']
        try:
            with self.heartbeat(job['id']):
                cards,bundles,shown,evidence=self._materials(target_id)
                base=bases(cards)
                fingerprint=digest({'base':base,'evidence':evidence,'shown':shown,'prompts':prompt_manifest(),'schemas':[CHANGE,REVIEW],
                    'parameters':self.parameters,'models':[{k:getattr(getattr(c,'settings',None),k,None) for k in
                    ('provider','base_url','model_name','timeout_sec','provider_timeout_retries')} for c in (self.reflection,self.reviewer)]})
                with self.store.connect() as db:
                    existing=db.execute('SELECT p.* FROM maintenance_groups g JOIN proposals p ON p.id=g.proposal_id WHERE g.fingerprint=?',(fingerprint,)).fetchone()
                write_json(directory/'before.json',cards)
                write_json(directory/'evidence.json',evidence)
                if existing:
                    pid=existing['id']
                    review=json.loads(existing['review_json'])
                    result.update(group_reused=True,proposal_id=pid,review=review)
                else:
                    stage='reflection'
                    change=calls.ask('reflection',self.reflection,PROPOSAL_CONTRACT,
                        {'experiences':cards,'decision_question':cards[0].get('decision_question'), 'original_evidence':evidence},
                        CHANGE,bundles=bundles,shown=shown)
                    for round_no in range(2):
                        stage='proposal_validation'
                        _,supplied=materials(bundles,shown,{sid:[r for c in cards for l in c['evidence'] if l['source_id']==sid for r in l['evidence_refs']] for sid in bundles})
                        validate_change(change,cards,supplied,shown)
                        write_json(directory/f'proposal_{round_no}.json',change)
                        write_json(directory/'proposal.json',change)
                        required={sid:sorted(set(shown.get(sid,[])) | {o for out in change['outputs'] for l in out['evidence'] if l['source_id']==sid for o in l['observation_ids']}) for sid in bundles}
                        stage='review_select'
                        review_shown=select(calls,f'review_select_{round_no}',self.reviewer,bundles,required,extra=change)
                        refs={sid:[r for c in cards for l in c['evidence'] if l['source_id']==sid for r in l['evidence_refs']] +
                            [r for out in change['outputs'] for l in out['evidence'] if l['source_id']==sid for r in l['evidence_refs']] for sid in bundles}
                        review_evidence,review_supplied=materials(bundles,review_shown,refs)
                        validate_change(change,cards,review_supplied,review_shown)
                        stage='review'
                        review=calls.ask(f'review_{round_no}',self.reviewer,prompt('review'),
                            {'before':cards,'proposed_change':change,'original_evidence':review_evidence},REVIEW,bundles=bundles,shown=review_shown)
                        validate_review(review)
                        review={**review,'protocol_version':VERSION,'shown':review_shown,
                            'audit_path':str(directory/'calls.json'),'round':round_no,
                            'configured_model':getattr(getattr(self.reviewer,'settings',None),'model_name',None),
                            'returned_model':calls.records[-1].get('returned_model'),
                            'prompt_sha256':calls.records[-1].get('prompt_sha256'),
                            'publication_source':'independent_maintenance_review'}
                        write_json(directory/f'review_{round_no}.json',review)
                        if review['decision']!='revise':
                            break
                        if round_no==1:
                            review.update(decision='defer',bounded_revision_exhausted=True)
                            break
                        stage='revision'
                        change=calls.ask('revision',self.reflection,prompt('revise'),
                            {'before':cards,'proposal':change,'review':review,'original_evidence':review_evidence},
                            CHANGE,bundles=bundles,shown=review_shown)
                        shown=review_shown
                    pid=uuid.uuid4().hex
                    stage='commit'
                    with self.store.connect(write=True) as db:
                        if not db.execute("SELECT 1 FROM jobs WHERE id=? AND owner=? AND state='running' AND lease_until>?",(job['id'],self.owner,time.time())).fetchone():
                            raise ValueError('Worker lease lost')
                        check_bases(db,base)
                        db.execute('INSERT INTO proposals VALUES (?,?,?,?,?,?,?)',(pid,job['id'],dumps(base),dumps(change),dumps(review),
                            'awaiting_auto' if review['decision']=='approve' else review['decision'],time.time()))
                        db.execute('INSERT INTO maintenance_groups VALUES (?,?)',(fingerprint,pid))
                    result.update(proposal_id=pid,review=review)
                stage='publication'
                with self.store.connect() as db:
                    state=db.execute('SELECT state FROM proposals WHERE id=?',(pid,)).fetchone()[0]
                if state=='awaiting_auto':
                    result['published']=publish(self.store,pid,actor='model_review:v2',reason=review['reason'],automatic=True,
                        worker_owner=(job['id'],self.owner))
                result['status']='completed'
                with self.store.connect(write=True) as db:
                    db.execute("UPDATE jobs SET state='completed' WHERE id=? AND owner=?",(job['id'],self.owner))
        except Exception as exc:
            result.update(status='failed',error_stage=stage,error_type=type(exc).__name__)
            if type(exc) is ValueError and stage in {'proposal_validation','commit','publication'}:
                result['validation_issue']=str(exc)
            if type(exc) is ValueError and str(exc).startswith('Version conflict'):
                result['status']='conflict'
            with self.store.connect(write=True) as db:
                db.execute("UPDATE jobs SET state=?,error=? WHERE id=? AND owner=? AND state='running'",
                    (result['status'],type(exc).__name__,job['id'],self.owner))
        finally:
            write_json(directory/'result.json',result)
            with self.store.connect(write=True) as db:
                db.execute("UPDATE attempts SET state=?,ended=?,error=?,audit_path=? WHERE id=? AND state='running'",
                    (result['status'],time.time(),result.get('error_type'),str(directory),job['attempt_id']))
        if result['status']=='conflict':
            self.store.enqueue([target_id])
        return result



def publish(store, proposal_id, *, actor, reason, automatic=False, worker_owner=None):
    """Materialize a reviewed change atomically; historical human gates stay manual."""
    if not actor.strip() or not reason.strip():
        raise ValueError("Publication actor and reason required")
    with store.connect() as db:
        row = db.execute("SELECT * FROM proposals WHERE id=?", (proposal_id,)).fetchone()
    if not row or row["state"] != ("awaiting_auto" if automatic else "awaiting_human"):
        raise ValueError("Proposal is not independently approved in the requested publication mode")
    change, base = json.loads(row["change_json"]), json.loads(row["base_json"])
    audit_review = json.loads(row['review_json'])
    if automatic and (audit_review.get('protocol_version') != VERSION or audit_review.get('decision') != 'approve'):
        raise ValueError('Automatic publication requires V2 independent approval')
    # Revalidate raw provenance at publication, including non-content operations.
    cards = [store.get(eid, state["version"]) for eid, state in base.items()]
    sids = {l["source_id"] for c in cards for l in c["evidence"]} | {l["source_id"] for o in change["outputs"] for l in o["evidence"]}
    bundles = {sid: store.check_source(sid) for sid in sids}
    shown = audit_review.get('shown') if automatic else None
    if shown is None:
        shown = {sid: sorted({obs for c in cards for l in c["evidence"] if l["source_id"] == sid for obs in l["observation_ids"]}) for sid in sids}
    validate_change(change, cards, bundles, shown)
    published = []
    with store.connect(write=True) as db:
        if worker_owner and not db.execute("SELECT 1 FROM jobs WHERE id=? AND owner=? AND state='running' AND lease_until>?",
            (*worker_owner,time.time())).fetchone():
            raise ValueError('Worker lease lost before publication')
        if db.execute("SELECT state FROM proposals WHERE id=?", (proposal_id,)).fetchone()[0] != ("awaiting_auto" if automatic else "awaiting_human"):
            raise ValueError("Proposal already handled")
        check_bases(db, base)
        for output in change["outputs"]:
            eid = output["target_id"] or uuid.uuid4().hex
            db.execute("INSERT OR IGNORE INTO experiences(id) VALUES (?)", (eid,))
            version = store._version(db, eid, output["body"], output["evidence"])
            db.execute("UPDATE experiences SET published_version=?,state='published' WHERE id=?", (version, eid))
            store._review(db, eid, version, "published", reason, actor, {"proposal_id": proposal_id})
            published.append({"id": eid, "version": version})
        for eid in change["retire_ids"]:
            state = "withdrawn" if change['operation']=='withdraw' else ("archived" if change["operation"] == "archive" else "superseded")
            db.execute("UPDATE experiences SET state=? WHERE id=?", (state, eid))
            store._review(db, eid, base[eid]["version"], state, reason, actor, {"proposal_id": proposal_id})
            for new in published:
                db.execute("INSERT INTO relations VALUES (?,?,?,?)", (eid, new["id"], "replaced_by", proposal_id))
        for a in change["conflict_ids"]:
            for b in change["conflict_ids"]:
                if a < b:
                    db.execute("INSERT INTO relations VALUES (?,?,?,?)", (a, b, "conflict", proposal_id))
        db.execute("UPDATE proposals SET state='published' WHERE id=?", (proposal_id,))
        store._review(db, None, None, "proposal_published", reason, actor, {"proposal_id": proposal_id})
    return published
