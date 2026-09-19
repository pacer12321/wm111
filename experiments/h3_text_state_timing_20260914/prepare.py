"""Create an isolated D4 copy with AST-verified text-state timing scopes only."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path('/cache/zhonghao/h3/text_state_timing_v1')
MODEL_REL = Path('vllm_omni/diffusion/models/minimax_h3')
ORIGINAL = Path('/cache/zhonghao/h3/dualstream_v1/candidate/vllm-omni')


def function(text, cls, name):
    tree = ast.parse(text)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)


def assignment_name(node):
    return node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None


def wrap(text, start, end, expression):
    lines = text.splitlines(keepends=True)
    indent = len(lines[start-1]) - len(lines[start-1].lstrip())
    block = [' ' * indent + 'with ' + expression + ':\n'] + ['    ' + line for line in lines[start-1:end]]
    return ''.join(lines[:start-1] + block + lines[end:])


class RemoveObservers(ast.NodeTransformer):
    def visit_With(self, node):
        node = self.generic_visit(node)
        if len(node.items) == 1:
            expr = node.items[0].context_expr
            if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id in ('_text_scope', '_text_branch_scope'):
                return node.body
        return node

    def visit_ImportFrom(self, node):
        return None if node.module == 'text_state_timer' else node

    def visit_Expr(self, node):
        expr = node.value
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == '_install_text_forward':
            return None
        return self.generic_visit(node)


def checked(original, instrumented):
    assert ast.dump(ast.parse(original)) == ast.dump(RemoveObservers().visit(ast.parse(instrumented)))
    compile(instrumented, '<text-state-observer>', 'exec')
    return instrumented


def instrument_op(original):
    fn = function(original, 'BidirectionalLinearBranch', 'forward_head_shard')
    first = next(n for n in fn.body if assignment_name(n) == 'text_slice')
    last = next(n for n in fn.body if assignment_name(n) == 'text_state')
    result = wrap(original, first.lineno, last.end_lineno, '_text_scope()')
    result += '\nfrom .text_state_timer import text_scope as _text_scope\n'
    return checked(original, result)


def instrument_transformer(original):
    fn = function(original, 'MiniMaxH3Attention', '_run_openvdn_ulysses')
    selected = [(n, 'T' if assignment_name(n) == 'readout_head' else 'S') for n in fn.body
                if assignment_name(n) in ('readout_head', 'source_readout_head')
                and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == 'forward_head_shard']
    assert len(selected) == 2
    result = original
    for n, side in sorted(selected, key=lambda p: -p[0].lineno):
        result = wrap(result, n.lineno, n.end_lineno, f'_text_branch_scope("{side}", self.d_layer_index)')
    result += '\nfrom .text_state_timer import branch_scope as _text_branch_scope, install_forward as _install_text_forward\n_install_text_forward(MiniMaxH3DiTModel)\n'
    return checked(original, result)


def main():
    assert Path(__file__).resolve().parent == ROOT
    sys.path.insert(0, '/cache/zhonghao/h3/color_trial_v1/b')
    import b_trial_gates as b
    sys.path.insert(0, '/cache/zhonghao/h3/dualstream_v1/deploy_d')
    import d_trial_gates as d
    host = b.profiles.require_host()
    policy_sha = 'a412150f34a5f3c564e908e5ba85dcebe32ca641b2d139ce6963cbfc5d3949cd'
    policy = d.policy_gate(policy_sha)
    sample = b.sample_profile('shirt_red_couple_124')
    manifest = {'host': host, 'original_proofs': {'B': b.source_code_manifest(), 'D': d.source_code_manifest(policy['value'])},
                'copies': {}, 'source_video': str(sample.source), 'prompt': sample.prompt,
                'source_sha256': b.profiles.digest(sample.source), 'generation': b.GENERATION,
                'physical_cards': [0,1,2,3], 'world_size': 4, 'policy_sha256': policy_sha,
                'measurement': 'only_text_state_T_and_S_all_layers_all_steps', 'original_statement_AST_preserved': True}
    destination = ROOT / 'candidates/D/vllm-omni'
    assert not destination.exists()
    shutil.copytree(ORIGINAL, destination, ignore=shutil.ignore_patterns('__pycache__', '.git'))
    for name, convert in [('openvdn_npu.py', instrument_op), ('minimax_h3_transformer.py', instrument_transformer)]:
        p = destination / MODEL_REL / name
        p.write_text(convert(p.read_text()))
    adapter = destination / MODEL_REL / 'dual_stream_adapter.py'
    text = adapter.read_text()
    assert text.count('world not in (1, 8)') == 1
    adapter.write_text(text.replace('world not in (1, 8)', 'world not in (1, 4, 8)'))
    shutil.copy2(ROOT / 'text_state_timer.py', destination / MODEL_REL / 'text_state_timer.py')
    manifest['copies']['D'] = {'vendor': str(destination), 'original': str(ORIGINAL),
                              'files': {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in destination.rglob('*.py')}}
    manifest['diagnostic_code'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in list(ROOT.glob('*.py')) + list(ROOT.glob('*.sh'))}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'prepared': True, 'scope': manifest['measurement'], 'AST_preserved': True}))


if __name__ == '__main__':
    main()
