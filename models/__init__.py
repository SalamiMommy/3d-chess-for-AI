"""Models package initialization - suppress Numba warnings."""
import os

# Suppress Numba debug output
os.environ['NUMBA_WARNINGS'] = '0'
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'
os.environ['NUMBA_BOUNDSCHECK'] = '0'
