"""Check existing Nemotron dependencies without installing/rebuilding anything."""

import argparse
import importlib
import sys
from importlib.metadata import version

from environment import configure_libraries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true', help='Also require a visible CUDA device.')
    args = parser.parse_args()
    print(f'Python: {sys.executable}', flush=True)
    configure_libraries()
    failures = []
    for module, package in [('torch', 'torch'), ('transformers', 'transformers'),
                            ('causal_conv1d', 'causal-conv1d'), ('mamba_ssm', 'mamba-ssm')]:
        try:
            loaded = importlib.import_module(module)
            found = version(package)
            if package == 'transformers' and found != '4.57.6':
                raise RuntimeError(f'Expected 4.57.6, found {found}')
            print(f'OK {package}=={found}', flush=True)
            if module == 'torch':
                print(f'Torch path: {loaded.__file__} | built with CUDA: {loaded.version.cuda}', flush=True)
        except Exception as error:  # noqa: BLE001 — report each dependency failure, then exit nonzero.
            failures.append(module)
            print(f'FAIL {module}: {type(error).__name__}: {error}', file=sys.stderr)
    if args.cuda and 'torch' not in failures:
        import torch
        if not torch.cuda.is_available():
            failures.append('CUDA')
            print('FAIL CUDA is not available', file=sys.stderr)
        else:
            print(f'CUDA {torch.version.cuda}: {torch.cuda.device_count()} GPU(s)')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
