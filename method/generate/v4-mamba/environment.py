"""Make the Z3 library shipped by the Python wheel visible to TileLang/TVM."""

import ctypes
import os
from importlib.metadata import PackageNotFoundError, distribution


def configure_libraries():
    try:
        library = distribution('z3-solver').locate_file('z3/lib/libz3.so')
    except PackageNotFoundError:
        return
    if not library.is_file():
        return
    directory = str(library.parent)
    current = os.environ.get('LD_LIBRARY_PATH', '')
    if directory not in current.split(':'):
        os.environ['LD_LIBRARY_PATH'] = directory + (':' + current if current else '')
    # LD_LIBRARY_PATH changes alone do not update an already-running loader.
    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
