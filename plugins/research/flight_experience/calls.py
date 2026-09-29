"""Audited strict model calls with content-addressed stage checkpoints.

Cache reuse is exact across evidence bytes, prompt/schema, model connection and
generation parameters. Failed calls are never cached. Format repair is bounded
and accepted only when equal to an unambiguous structural normalization.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from src.llm import ModelRequest
from src.llm.audit import snapshot_request
from .contracts import VERSION, prompt, prompt_hash, prompt_manifest, validate
from .store import digest, dumps


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def structural_normalization(value, schema):
    """Only scalar-to-list wrapping at schema-declared arrays; no semantic repair."""
    if schema.get('type') == 'array':
        if isinstance(value, str) and schema['items'].get('type') == 'string':
            return [value]
        if isinstance(value, list):
            return [structural_normalization(v, schema['items']) for v in value]
    if isinstance(value, dict) and schema.get('type') == 'object':
        return {k: structural_normalization(v, schema.get('properties', {}).get(k, {})) for k,v in value.items()}
    return value


class Calls:
    """One attempt's call audit; optional shared checkpoint directory for retries."""
    def __init__(self, directory, *, cache=None, parameters=None, prices=None, prompt_version=VERSION):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache = Path(cache) if cache else self.directory / 'checkpoints'
        self.parameters = parameters or {}
        self.prices = prices or {}
        self.prompt_version = prompt_version
        self.records = []

    def _media(self, items):
        records = []
        for item in items:
            row = {k:v for k,v in item.items() if k != 'data'}
            if isinstance(item.get('data'), bytes):
                relative = f"media/{item['sha256']}{item['suffix']}"
                path = self.directory / relative
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(item['data'])
                row['asset_ref'] = relative
            records.append(row)
        return records

    def _request(self, stage, client, system, content, schema):
        request = ModelRequest(messages=[{'role':'system','content':system}, {'role':'user','content':content}],
            parameters={**self.parameters, 'response_format':{'type':'json_schema', 'json_schema':{
                'name':'flight_experience_'+stage.replace('-','_'), 'strict':True, 'schema':schema}}})
        snapshot, media = snapshot_request({'messages': request.messages, **request.parameters})
        row = {'stage':stage, 'prompt_version':self.prompt_version, 'prompt_sha256':prompt_hash(system), 'schema_sha256':digest(schema),
            'caller_request':snapshot, 'request_media':self._media(media), 'status':'running', 'usage':None, 'estimated_cost_usd':None}
        self.records.append(row)
        write(self.directory/'calls.json', self.records)
        tick = time.monotonic()
        try:
            response = client.request(request)
            row.update(status='received', response_text=response.text, usage=response.usage,
                returned_model=response.data.get('model'), transport=response.transport_trace)
            row['finish_reason']=(response.data.get('choices') or [{}])[0].get('finish_reason')
            role='review' if stage.startswith('review') else 'reflection'
            ip,op=self.prices.get(role+'_input_price'),self.prices.get(role+'_output_price')
            usage=response.usage or {}
            if ip is not None and op is not None and isinstance(usage.get('prompt_tokens'),int) and isinstance(usage.get('completion_tokens'),int):
                row['estimated_cost_usd']=(usage['prompt_tokens']*ip+usage['completion_tokens']*op)/1e6
            if response.request_snapshot:
                row.update(provider_request=response.request_snapshot, provider_media=self._media(response.media))
            if row['finish_reason']=='length':
                raise ValueError('Provider output truncated')
            return response.text
        except Exception as exc:
            row.update(status='failed', error_type=type(exc).__name__, category=getattr(exc,'category',None),
                http_status=getattr(exc,'status_code',None))
            raise
        finally:
            row['elapsed_seconds'] = time.monotonic()-tick
            write(self.directory/'calls.json', self.records)

    def ask(self, stage, client, system, material, schema, *, bundles=None, shown=None, allow_repair=True):
        content = [{'type':'text','text':dumps(material)}]
        for sid, observations in (shown or {}).items():
            for obs in observations:
                content.extend([{'type':'text','text':f'HISTORICAL RGB source_id={sid} observation_id={obs}'}, bundles[sid].image_part(obs)])
        settings = getattr(client, 'settings', None)
        identity = {k:getattr(settings,k,None) for k in ('provider','model_name','base_url','timeout_sec','provider_timeout_retries','provider_connection_retries')}
        key = digest({'stage':stage,'system':system,'content':content,'schema':schema,'parameters':self.parameters,'model':identity,'prompt_manifest':prompt_manifest()})
        checkpoint = self.cache / (key+'.json')
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding='utf-8'))
            validate(saved['value'],schema)
            self.records.append({'stage':stage,'status':'reused','checkpoint_key':key,
                'origin_audit':saved['audit'], 'prompt_version':self.prompt_version,'prompt_sha256':prompt_hash(system),
                'elapsed_seconds':0, 'usage':None,'estimated_cost_usd':None})
            write(self.directory/'calls.json',self.records)
            return saved['value']
        raw = self._request(stage, client, system, content, schema)
        try:
            value = json.loads(raw)
            validate(value,schema)
        except (ValueError, TypeError):
            if not allow_repair:
                raise
            # Recover parseable fenced JSON only. Malformed syntax is not guessed.
            stripped = (raw or '').strip()
            if stripped.startswith('```') and stripped.endswith('```'):
                stripped = '\n'.join(stripped.splitlines()[1:-1])
            expected = structural_normalization(json.loads(stripped),schema)
            validate(expected,schema)  # Missing/new evidence cannot be repaired.
            repaired = self._request(stage+'_format_repair',client,prompt('repair'),
                [{'type':'text','text':dumps({'original_response':raw,'error':'Output did not match the supplied schema'})}],schema)
            value = json.loads(repaired)
            validate(value,schema)
            if value != expected:
                raise ValueError('Format repair changed content or citations')
        write(checkpoint,{'key':key,'value':value,'audit':str(self.directory/'calls.json')})
        return value
