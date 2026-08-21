"""Resumable runner: compute & cache analytic synaptic fields for all LCs + LPLCs.

Each neuron's fields are saved to fields/<neuron>_analytic.npz. Re-running skips any
neuron already cached, so this can be interrupted and resumed. The notebook
(downstream_plotting.ipynb) then loads these caches for plotting.

Run from this folder with the project's venv:
    ../.venv/Scripts/python.exe compute_fields.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))   # make flywire_tools importable
os.chdir(HERE)                                         # data CSVs are read by relative name

import pandas as pd
from flywire_tools.connectome import Connectome
import field_plotting as fp

LC_HOPS = 6      # T3 used 5; LCs need one more hop to reach the L1-L5/R7/R8 inputs
LPLC_HOPS = 7    # LPLCs need two more


def main():
    connectome = Connectome()
    vnt = pd.read_csv("visual_neuron_types.csv")
    lc_types = fp.neuron_types(vnt, "LC")
    lplc_types = fp.neuron_types(vnt, "LPLC")
    neuron_hops = {**{lc: LC_HOPS for lc in lc_types},
                   **{lp: LPLC_HOPS for lp in lplc_types}}
    print(f"{len(neuron_hops)} neuron types to compute "
          f"({len(lc_types)} LC + {len(lplc_types)} LPLC)")
    for num, (neuron, hops) in enumerate(neuron_hops.items(), 1):
        fn = os.path.join("fields_us", f"{neuron}_analytic.npz")
        if os.path.exists(fn):
            print(f"[{num}/{len(neuron_hops)}] skip {neuron} (cached)")
            continue
        print(f"[{num}/{len(neuron_hops)}] computing {neuron} (max_hops={hops})...")
        try:
            fp.compute_or_load(connectome, neuron, hops, cache_dir="fields_us")
            print(f"    saved {neuron}")
        except Exception as exc:
            print(f"    FAILED {neuron}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
