"""Versioned prompts and strict schemas, shared by generation and maintenance."""
from __future__ import annotations

import hashlib
from pathlib import Path
from jsonschema import Draft202012Validator

VERSION = 'v2'
PROMPTS = Path(__file__).parent / 'prompts' / VERSION


def prompt(name):
    return (PROMPTS / 'boundary.txt').read_text(encoding='utf-8') + '\n' + (PROMPTS / f'{name}.txt').read_text(encoding='utf-8')


def prompt_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def prompt_manifest():
    return {p.name:prompt_hash(p.read_text(encoding='utf-8')) for p in sorted(PROMPTS.glob('*.txt'))}


def obj(**properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def array(items):
    return {'type': 'array', 'items': items}


STRING = {'type': 'string'}
STRINGS = array(STRING)
TAGS = obj(**{k: STRINGS for k in ('decision', 'action', 'observation', 'anomaly', 'platform', 'control', 'sensor')})
BODY = obj(title=STRING, decision_question=STRING, applicability=STRING,
    observed_evidence=STRING, recommendation=STRING, limitations=STRING, tags=TAGS)
CANDIDATE = obj(id=STRING, **BODY['properties'], evidence_refs=STRINGS, observation_ids=STRINGS)
GENERATION = obj(candidates=array(CANDIDATE))
LINK = obj(source_id=STRING, evidence_refs=STRINGS, observation_ids=STRINGS,
    relation={'type': 'string', 'enum': ['support', 'counterexample']})
CHANGE = obj(operation={'type': 'string', 'enum': ['add','supplement','revise','merge','split','conflict','archive','withdraw','unchanged']},
    reason=STRING, outputs=array(obj(target_id={'type': ['string', 'null']}, body=BODY, evidence=array(LINK))),
    retire_ids=STRINGS, conflict_ids=STRINGS)
REVIEW = obj(decision={'type': 'string', 'enum': ['approve','revise','defer','discard']}, reason=STRING,
    evidence_support=STRING, condition_boundaries=STRING, counterexamples=STRING,
    decision_value=STRING, revision_requests=STRINGS)
SELECTION = obj(selections=array(obj(source_id=STRING, observation_ids=STRINGS)), reason=STRING)


def validate(value, schema):
    """Format only; callers must separately check provenance and operation meaning."""
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda x: str(x.absolute_path))
    if errors:
        # Paths/types only: no untrusted payload or credentials in exception text.
        raise ValueError('; '.join(f'{list(e.absolute_path)}: {e.validator}' for e in errors[:8]))


def validate_review(value):
    validate(value, REVIEW)
    for key in ('reason', 'evidence_support', 'condition_boundaries', 'counterexamples', 'decision_value'):
        if not value[key].strip():
            raise ValueError('Empty review assessment: '+key)
    if value['decision'] == 'revise' and not value['revision_requests']:
        raise ValueError('Revision requires actionable requests')
