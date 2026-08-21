"""Resumable runner: compute & cache DOWNSTREAM analytic synaptic fields for LCs + LPLCs.

For each neuron the field is the forward-flow projection from the L1-L5/R7/R8 input
channels onto that neuron, indexed by the input channel's retinal coordinate. Results are
saved to fields_ds/<neuron>_analytic.npz (resumable; already-cached neurons are skipped).

Hops: LCs use max_hops=3 (first contact with the LC from the L1-R8 input channels),
LPLCs use one more (4).

Run from this folder with the project's venv:
    ../.venv/Scripts/python.exe compute_downstream_sensitivity.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))   # make flywire_tools importable
os.chdir(HERE)                                         # data CSVs are read by relative name

import pandas as pd
from flywire_tools.connectome import Connectome
import field_plotting as fp

LC_HOPS = 3      # first contact with the LC from the L1-R8 input channels
LPLC_HOPS = 4    # one more than the LCs


SIDES = ["right", "left"]   # compute each optic lobe independently (ground truth)


def main():
    connectome = Connectome()
    vnt = pd.read_csv("visual_neuron_types.csv")
    lc_types = fp.neuron_types(vnt, "LC")
    lplc_types = fp.neuron_types(vnt, "LPLC")
    neuron_hops = {**{lc: LC_HOPS for lc in lc_types},
                   **{lp: LPLC_HOPS for lp in lplc_types}}
    print(f"{len(neuron_hops)} neuron types x {len(SIDES)} sides to compute "
          f"({len(lc_types)} LC + {len(lplc_types)} LPLC)")
    for side in SIDES:
        cache_dir = f"fields_ds_sensitivity_{side}"
        print(f"=== side={side} -> {cache_dir} ===")
        for num, (neuron, hops) in enumerate(neuron_hops.items(), 1):
            fn = os.path.join(cache_dir, f"{neuron}_analytic.npz")
            if os.path.exists(fn):
                print(f"[{side} {num}/{len(neuron_hops)}] skip {neuron} (cached)")
                continue
            print(f"[{side} {num}/{len(neuron_hops)}] computing {neuron} (max_hops={hops})...")
            try:
                fp.compute_downstream_or_load(connectome, neuron, hops,
                                              cache_dir=cache_dir, side=side, 
                                              normalize=False)
                print(f"    saved {neuron}")
            except Exception as exc:
                print(f"    FAILED {neuron}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
