"""Export recorded benchmark cases to portable, Windows-safe review directories.

No model or simulator is used. Originals remain immutable. Run with --batch-root
and --output-dir; re-running only refreshes cases whose source evidence changed.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.benchmarks.core import (
    BenchmarkAuditLog, BenchmarkCaseReview, BenchmarkReviewRenderer, FrameAssetResolver,
)
from evaluation.benchmarks.core.review import ReviewSection
from evaluation.benchmarks.core.report_io import atomic_write_text

MAX_PATH = 240
EXPORT_VERSION = 1


def write_json(path: Path, value) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def checked(path: Path) -> Path:
    if len(str(path.resolve())) > MAX_PATH:
        raise ValueError(f'Review path exceeds {MAX_PATH} characters: {path}')
    return path


class References(HTMLParser):
    """Read actual resource attributes, never strings inside raw request JSON."""

    def __init__(self):
        super().__init__()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ('src', 'href') and value:
                self.refs.append((tag, name, value))


def local_path(page: Path, value: str) -> Path | None:
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    return (page.parent / unquote(parsed.path)).resolve()


def validate_bundle(root: Path) -> dict:
    """Fail on missing, external-to-bundle, or overlong local HTML resources."""
    root = root.resolve()
    images = 0
    files = list(root.rglob('*'))
    for path in files:
        checked(path)
        if path.suffix != '.html':
            continue
        parser = References()
        parser.feed(path.read_text(encoding='utf-8'))
        for tag, _attr, value in parser.refs:
            target = local_path(path, value)
            if target is None:
                continue
            checked(target)
            if not target.is_relative_to(root) or not target.is_file():
                raise ValueError(f'Invalid review resource: {path} -> {value}')
            if tag == 'img':
                images += 1
    return {'image_references': images, 'files': sum(p.is_file() for p in files),
            'max_path_length': max((len(str(p)) for p in files), default=len(str(root)))}


def portable_links(page: Path, root: Path) -> None:
    """Copy renderer links to original run diagnostics into this export only."""
    text = page.read_text(encoding='utf-8')
    parser = References()
    parser.feed(text)
    for _tag, _attr, value in parser.refs:
        source = local_path(page, value)
        if source is None or source.is_relative_to(root):
            continue
        if not source.is_file():
            raise ValueError(f'Missing source resource: {source}')
        # Path digest preserves distinct audit records without long original names.
        filename = hashlib.sha256(str(source).encode()).hexdigest()[:20] + source.suffix
        target = checked(page.parent / 'files' / filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        relative = target.relative_to(page.parent).as_posix()
        text = text.replace('="' + html.escape(value, quote=True) + '"', '="' + relative + '"')
    atomic_write_text(page, text)


def export_case(source: Path, output: Path, number: str) -> dict:
    """Rebuild both views using recorded inputs and original displayed case ID."""
    destination = output / 'cases' / number
    checked(destination / 'review_assets/provider_inputs' / ('f' * 64 + '.png'))
    audit_bytes = (source / 'audit.json').read_bytes()
    result_bytes = (source / 'case_result.json').read_bytes()
    fingerprint = hashlib.sha256(audit_bytes + result_bytes).hexdigest()
    receipt_path = destination / 'export.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('source_sha256') == fingerprint and receipt.get('version') == EXPORT_VERSION:
            return receipt
        # Recovery attempts get a separate directory; never mix old and new assets.
        raise ValueError(f'Export destination contains different evidence: {destination}')
    data = json.loads(audit_bytes)
    result = json.loads(result_bytes)
    audit = BenchmarkAuditLog(**{f.name: data[f.name] for f in fields(BenchmarkAuditLog) if f.name in data})
    assets = FrameAssetResolver().materialize(audit, destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('audit.json', 'trace.json', 'trajectory.json', 'metrics.json', 'case_result.json'):
        if (source / name).is_file():
            shutil.copyfile(source / name, checked(destination / name))
    instruction = next((e['payload'].get('instruction', '') for e in audit.agent_transcript
                        if e.get('kind') == 'user_instruction'), 'Instruction unavailable')
    metrics = result.get('metrics') or {}
    incomplete = bool(result.get('infrastructure_error')) or result.get('status') == 'error'
    sections = [ReviewSection('Trajectory', 'trajectory_ned', {
        'actual': metrics.get('trajectory_ned', []), 'reference': metrics.get('reference_path_ned', [])})]
    if incomplete:
        sections.insert(0, ReviewSection('Incomplete run · 无最终评分', 'text',
                                        'Infrastructure or execution failure; not a completed navigation score.', primary=True))
    review = BenchmarkCaseReview(
        case_id=result.get('episode_id', source.name), benchmark_name='Recorded flight review',
        protocol_variant=result.get('protocol_variant', 'unknown'), instruction=instruction,
        outcome={k: result.get(k) for k in ('status', 'completed', 'stopped', 'stopped_reason', 'error')},
        metrics={} if incomplete else {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
        audit=audit, frame_assets=assets, sections=tuple(sections))
    for detailed in (True, False):
        rendered = BenchmarkReviewRenderer(detailed_audit=detailed).write_case(
            output, review, directory_name=number)
        portable_links(Path(rendered['case_html']), output.resolve())
    receipt = {'version': EXPORT_VERSION, 'source': str(source), 'source_sha256': fingerprint,
               'case_id': review.case_id, 'incomplete': incomplete,
               'review': f'cases/{number}/review.html', 'audit': f'cases/{number}/audit_review.html'}
    if (source / 'audit.json').read_bytes() != audit_bytes or (source / 'case_result.json').read_bytes() != result_bytes:
        raise ValueError('Source evidence changed during export')
    write_json(receipt_path, receipt)
    return receipt


def export_batch(batch_root: Path, output: Path) -> dict:
    """Export all attempts; emphasize the latest while retaining prior failures."""
    batch_root, output = batch_root.resolve(), output.resolve()
    if output == batch_root or output.is_relative_to(batch_root) or batch_root.is_relative_to(output):
        raise ValueError('Use an independent review output directory')
    checked(output / 'cases/20/review_assets/provider_inputs' / ('f' * 64 + '.png'))
    output.mkdir(parents=True, exist_ok=True)
    identity = output / 'source_batch.json'
    if identity.exists() and json.loads(identity.read_text(encoding='utf-8'))['batch_root'] != str(batch_root):
        raise ValueError('Review output belongs to another batch')
    write_json(identity, {'batch_root': str(batch_root)})
    manifest = json.loads((batch_root / 'manifest.json').read_text(encoding='utf-8'))
    rows = []
    for i, item in enumerate(manifest['cases'], 1):
        prefix = f'{i:02d}_{item["benchmark"]}_{item["case_id"]}'
        original = batch_root / 'cases' / prefix
        attempts = ([original] if original.exists() else []) + sorted(original.parent.glob(prefix + '_recovery_*'))
        links = []
        for attempt_index, attempt in enumerate(attempts):
            # Latest uses the short numeric folder. Older attempts remain separate.
            number = f'{i:02d}' if attempt_index == len(attempts) - 1 else f'{i:02d}-a{attempt_index + 1}'
            source = attempt / 'evaluation_output/evaluation/cases' / item['case_id']
            if (source / 'audit.json').exists() and (source / 'case_result.json').exists():
                # Choose stable attempt-specific storage if this number already represents an older attempt.
                receipt = output / 'cases' / number / 'export.json'
                if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8')).get('source') != str(source):
                    number = f'{i:02d}-a{attempt_index + 1}'
                exported = export_case(source, output, number)
                links.append(exported)
            elif (attempt / 'process.json').exists():
                dest = output / 'cases' / number
                dest.mkdir(parents=True, exist_ok=True)
                diagnostics = {}
                for name in ('process.json', 'supervision.json'):
                    if (attempt / name).exists():
                        diagnostics[name] = json.loads((attempt / name).read_text(encoding='utf-8'))
                page = '<!doctype html><meta charset="utf-8"><h1>无最终评分 · interrupted run</h1><pre>' + html.escape(json.dumps(diagnostics, ensure_ascii=False, indent=2)) + '</pre>'
                # Keep any captured decisions/images from the parent crash review.
                original_page = attempt / 'review.html'
                if original_page.exists():
                    original = original_page.read_text(encoding='utf-8')
                    refs = References()
                    refs.feed(original)
                    for _tag, _attr, value in refs.refs:
                        target = local_path(original_page, value)
                        if target is not None:
                            relative = Path(os.path.relpath(target, dest)).as_posix()
                            original = original.replace('="' + html.escape(value, quote=True) + '"',
                                                        '="' + html.escape(relative, quote=True) + '"')
                    page += original
                atomic_write_text(dest / 'review.html', page)
                atomic_write_text(dest / 'audit_review.html', page)
                portable_links(dest / 'review.html', output)
                portable_links(dest / 'audit_review.html', output)
                links.append({'review': f'cases/{number}/review.html', 'audit': f'cases/{number}/audit_review.html', 'incomplete': True})
        latest_status = attempts[-1] / 'status.json' if attempts else None
        status = json.loads(latest_status.read_text(encoding='utf-8')) if latest_status and latest_status.exists() else {'status': 'pending'}
        rows.append({**item, 'index': i, 'status': status['status'], 'attempts': links})
    def render_rows(selected):
        table = []
        for row in selected:
            links = ' | '.join(f'<a href="{a["review"]}">Timeline {j + 1}</a> · <a href="{a["audit"]}">完整审计 {j + 1}</a>' for j, a in enumerate(row['attempts'])) or '尚未生成'
            table.append('<tr>' + ''.join('<td>' + html.escape(str(row[k])) + '</td>' for k in ('index', 'case_id', 'difficulty', 'status')) + '<td>' + links + '</td></tr>')
        return '<table><tr><th>#</th><th>Case</th><th>Difficulty</th><th>Status</th><th>Review attempts (latest last)</th></tr>' + ''.join(table) + '</table>'
    style = '<style>body{font:16px system-ui;margin:32px;background:#f5f7fa}td,th{padding:12px;border-bottom:1px solid #ccc;text-align:left}a{color:#1459b3}table{background:white;border-collapse:collapse}</style>'
    groups = [('review', rows), ('index', rows)] + [(name + '_summary', [r for r in rows if r['benchmark'] == name]) for name in dict.fromkeys(r['benchmark'] for r in rows)]
    for name, selected in groups:
        page = '<!doctype html><meta charset="utf-8">' + style + '<h1>Flight review · 图像与完整审计</h1><p>Prior failed attempts are retained; incomplete runs have no final score.</p>' + ('<p>TravelUAV: 7 historical easy / 3 new hard.</p>' if any(r['benchmark'] == 'travel' for r in selected) else '') + render_rows(selected)
        atomic_write_text(output / (name + '.html'), page)
    validation = validate_bundle(output)
    result = {'source_batch': str(batch_root), 'cases': rows, 'validation': validation}
    write_json(output / 'export_summary.json', result)
    return result


def refresh_reviews(root: Path, review_root: Path | None = None) -> None:
    """Persist presentation failures without changing navigation results.

    Kept with the exporter so its validation does not depend on a particular
    benchmark scheduler or that scheduler's model-admission configuration.
    """
    destination = (review_root or root.parent / 'reviews' / root.name).resolve()
    try:
        export_batch(root, destination)
    except Exception as exc:
        write_json(root / 'review_failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise
    write_json(root / 'review_export.json', {'output_dir': str(destination)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = export_batch(args.batch_root, args.output_dir)
    print(json.dumps(result['validation']))
