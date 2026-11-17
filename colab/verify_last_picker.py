"""Run bounded acceptance tests; do not run a research training series."""
import argparse
import ast
import io
import json
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nmmo2.raid_nmmo.scaling_last_picker import provenance, runtime


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    modules = ['test_scaling_last_picker', 'test_train_last_picker', 'test_ready_prep',
               'test_last_picker', 'test_certified_battle']
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('nmmo2.raid_nmmo.'+name)
                               for name in modules)
    cases = list(flatten(suite))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise SystemExit(1)
    notebook = json.loads((REPO/'colab/last_picker_scaling.ipynb').read_text())
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']), filename=f'notebook-cell-{i}')
            assert cell['execution_count'] is None and not cell['outputs']
    resume = next(t.resume_evidence for t in cases if hasattr(t, 'resume_evidence'))
    report = dict(kind='bounded tests only; no Colab job or scientific training series started',
                  runtime=runtime(), provenance=provenance(), tests_run=result.testsRun,
                  skipped=[dict(test=str(t), reason=reason) for t, reason in result.skipped],
                  failures=0, errors=0, cpu_resume=resume, notebook_code_cells_parse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    args.out.with_suffix('.txt').write_text(stream.getvalue())
    print(json.dumps(resume, indent=2))


if __name__ == '__main__':
    main()
