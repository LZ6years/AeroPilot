"""Experimental SAM coordinate placement only; never edits installed libraries.

Candidate preserves the installed get_rel_pos algorithm and changes only the
device of its two arange calls. AST shape is verified to reject source drift.
This is not enabled in production. Exact CUDA tensor equivalence is checked
before any timed pipeline experiment; non-equivalence stops the experiment.
"""
import ast
from contextlib import contextmanager
import inspect
import json
import textwrap
from unittest.mock import patch


def device_coordinate_method(original):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == 'torch' and node.func.attr == 'arange':
                if node.keywords or len(node.args) != 1:
                    raise ValueError('SAM coordinate method changed; unsupported arange')
                node.keywords.append(ast.keyword(arg='device', value=ast.Attribute(value=ast.Name(id='rel_pos', ctx=ast.Load()), attr='device', ctx=ast.Load())))
                changed += 1
    if changed != 2:
        raise ValueError('Expected exactly two SAM coordinate arange calls')
    ast.fix_missing_locations(tree)
    scope = dict(original.__globals__)
    exec(compile(tree, '<diagnostic SAM device coordinates>', 'exec'), scope)
    return scope[original.__name__]


def validate_coordinates(directory):
    import torch
    from transformers.models.sam.modeling_sam import SamVisionAttention
    original = SamVisionAttention.get_rel_pos
    candidate = device_coordinate_method(original)
    generator = torch.Generator(device='cuda').manual_seed(913)
    checks = []
    with torch.no_grad():
        for q, k in ((14, 14), (64, 64), (13, 17), (17, 13)):
            positions = torch.randn(2 * max(q, k) - 1, 64, device='cuda', generator=generator)
            equal = torch.equal(original(None, q, k, positions), candidate(None, q, k, positions))
            checks.append({'query_size': q, 'key_size': k, 'exactly_equal': equal})
    (directory / 'coordinate_validation.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    if not all(c['exactly_equal'] for c in checks):
        raise ValueError('CUDA coordinate outputs differ; do not proceed')


@contextmanager
def candidate_coordinates():
    from transformers.models.sam.modeling_sam import SamVisionAttention
    with patch.object(SamVisionAttention, 'get_rel_pos', device_coordinate_method(SamVisionAttention.get_rel_pos)):
        yield
