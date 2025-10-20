#!/usr/bin/env python3
# --------------  main_stub.py  -----------------
import os, multiprocessing as mp

# 1.  hide GPUs from **every** child
for e in ("ROCM_VISIBLE_DEVICES",
          "HIP_VISIBLE_DEVICES",
          "CUDA_VISIBLE_DEVICES"):
    os.environ[e] = ""

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# 2.  spawn **before** any heavy module
if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    # 3.  NOW it is safe to import the real main
    import main
    main
