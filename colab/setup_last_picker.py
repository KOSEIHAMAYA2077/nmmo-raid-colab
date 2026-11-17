"""Install an isolated Python environment for CPU/CUDA; never start training."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def commands(venv, repo, torch_backend='cu126'):
    python = venv / 'bin' / 'python'
    uv = [sys.executable, '-m', 'uv']
    return [
        [sys.executable, '-m', 'pip', 'install', 'uv==0.9.5'],
        uv + ['venv', '--python', '3.10.16', str(venv)],
        uv + ['pip', 'install', '--python', str(python), 'setuptools==75.8.0',
              'wheel==0.45.1', 'numpy==1.23.3', 'cython==3.3.1'],
        uv + ['pip', 'install', '--python', str(python), 'torch==2.8.0',
              '--index-url', 'https://download.pytorch.org/whl/' + torch_backend,
              '--reinstall-package', 'torch'],
        uv + ['pip', 'install', '--python', str(python), '--no-build-isolation',
              '-r', str(repo / 'nmmo2' / 'requirements.txt')],
        uv + ['pip', 'check', '--python', str(python)],
        [str(python), '-c', 'import nmmo, torch, numpy; '
         'import importlib.metadata as m; '
         'print({p:m.version(p) for p in ("nmmo","torch","numpy")}); '
         'assert m.version("nmmo")=="2.1.2"; assert numpy.__version__=="1.23.3"; '
         'assert torch.__version__.split("+")[0]=="2.8.0"'],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, default=Path('/content/nmmo-scaling-env'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--torch-backend', choices=('cpu', 'cu126'), default='cu126',
                        help='cu126 supports both CPU and CUDA profiling on a GPU runtime')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan = commands(args.venv.resolve(), repo, args.torch_backend)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not shutil.which('g++'):
        parser.error('NMMO Cython extension needs g++; use a standard Colab Linux runtime')
    for command in plan:
        # Reusing a previously created environment is safe; do not clear it.
        if 'venv' in command and (args.venv / 'bin' / 'python').exists():
            continue
        subprocess.run(command, cwd=repo, check=True)


if __name__ == '__main__':
    main()
