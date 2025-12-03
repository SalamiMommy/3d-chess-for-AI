
import sys
from types import ModuleType

# Mock numba module
numba = ModuleType("numba")

def njit(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def prange(*args):
    return range(*args)

numba.njit = njit
numba.prange = prange

# Inject into sys.modules
sys.modules["numba"] = numba
