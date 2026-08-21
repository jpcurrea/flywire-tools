"""Resumable runner: OLD Monte Carlo (networkx-iterating) downstream synaptic fields.

Reproduces the legacy pipeline (examples/LC_doc/preprocess_LMC_downstream_fields.py) so the
result can be compared, apples-to-apples, against the analytic downstream fields:

  for each eye side and each retinal input channel (L1-L5, R7, R8):
    - build the DOWNSTREAM paths FROM that channel's cells (side-restricted),
    - run the real Paths.monte_carlo (nested networkx graph walk),
    - get_pathway_information(top_pathways=LC/LPLC, stopping_points=LC/LPLC),
    - get_synaptic_fields(stopping_points=[channel]) -> a field per LC target type,
      indexed by the channel cell's retinal coordinate.

The per-(channel) fields (keyed by LC type) are then transposed into per-LC caches
fields_ds_mcsim_<side>/<LC>_mcsim.npz keyed by channel (+ an OR-combined 'all'), matching
the analytic all_fields layout so downstream_plotting_mc.ipynb can load them the same way.

NOTE ON COST: Paths.monte_carlo runs `reps` walks PER STARTING CELL (~789 for L1) over the
full downstream graph. At reps=1e5, max_hops=5 each channel's path HDF5 is multiple GB and the
walk takes a long time. Run this offline. Use REPS / MAX_HOPS below (or run_channel args) to
scale down for testing.

Run from analytic_results/ with the project's venv:
    ../.venv/Scripts/python.exe compute_downstream_mc_sim.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.chdir(HERE)

import numpy as np
import pandas as pd
from fafbseg import flywire
from flywire_tools.connectome import Connectome, SynapticField, SynapticFields
import field_plotting as fp

SIDES = ["right", "left"]
REPS = 10_000            # walks PER starting cell (legacy default 1e4)
MAX_HOPS = 5              # legacy get_paths default
COLUMN_FILE = fp.COLUMN_INFO_CSV     # visual_column_info.csv -> both eyes + matching grid
RAW_PREFIX = "_rawmc_"               # per-channel intermediate cache name


def _targets():
    vnt = pd.read_csv("visual_neuron_types.csv")
    return list(fp.neuron_types(vnt, "LC")) + list(fp.neuron_types(vnt, "LPLC"))


def run_channel(connectome, side, channel, targets, reps=REPS, max_hops=MAX_HOPS,
                column_file=COLUMN_FILE):
    """Old-MC downstream field for one (side, channel). Returns SynapticFields keyed by LC."""
    starting_ids = flywire.NeuronCriteria(cell_type=channel, side=side)
    paths = connectome.get_paths(starting_ids, direction="downstream",
                                 max_hops=max_hops, skip_recurrents=False, rerun=True)
    scratch = os.path.join(f"fields_ds_mcsim_{side}", f"_mc_{channel}.h5")
    paths.monte_carlo(reps=int(reps), direction="downstream", save_fn=scratch)
    paths.get_pathway_information(top_pathways=targets, stopping_points=targets)
    fields = paths.get_synaptic_fields(stopping_points=[channel], side=side,
                                       conditional=True, column_file=column_file)
    # free the (large) monte-carlo HDF5
    try:
        paths._h5file.close()
    except Exception:
        pass
    if os.path.exists(scratch):
        try:
            os.remove(scratch)
        except Exception:
            pass
    return fields


def assemble_side(side, channels, cache_dir):
    """Transpose per-channel LC-keyed fields into per-LC channel-keyed SynapticFields (+ 'all')."""
    # load per-channel raw caches
    raw = {}
    for ch in channels:
        fn = os.path.join(cache_dir, f"{RAW_PREFIX}{ch}.npz")
        if os.path.exists(fn):
            raw[ch] = fp.load_SynapticFields(fn)
    if not raw:
        return
    # every LC present in any channel
    lcs = sorted({lc for ch in raw for lc in raw[ch].keys()})
    ref = next(iter(next(iter(raw.values())).values()))
    width, height = ref.shape[-2:]
    for lc in lcs:
        grids = {}
        none_reach = np.ones((1, width, height), dtype=np.float64)
        for ch in channels:
            if ch in raw and lc in raw[ch]:
                f = raw[ch][lc]
                grids[ch] = SynapticField(np.asarray(f)[:1], ps=f.ps, qs=f.qs)
                none_reach *= (1.0 - np.clip(np.asarray(grids[ch]), 0.0, 1.0))
        if not grids:
            continue
        all_field = SynapticField((1.0 - none_reach).astype(np.float32), ps=ref.ps, qs=ref.qs)
        grids["all"] = all_field
        SynapticFields(grids).save(os.path.join(cache_dir, f"{lc}_mcsim.npz"))


def main(sides=SIDES, channels=fp.STARTING_POINTS, reps=REPS, max_hops=MAX_HOPS):
    connectome = Connectome()
    targets = _targets()
    for side in sides:
        cache_dir = f"fields_ds_mcsim_{side}"
        os.makedirs(cache_dir, exist_ok=True)
        print(f"=== side={side} -> {cache_dir} (reps={reps}, max_hops={max_hops}) ===")
        for ch in channels:
            raw_fn = os.path.join(cache_dir, f"{RAW_PREFIX}{ch}.npz")
            if os.path.exists(raw_fn):
                print(f"[{side} {ch}] skip (cached)")
                continue
            print(f"[{side} {ch}] computing old-MC downstream field...")
            try:
                fields = run_channel(connectome, side, ch, targets,
                                     reps=reps, max_hops=max_hops)
                fields.save(raw_fn)
                print(f"    saved raw {ch} ({len(fields)} LC targets)")
            except Exception as exc:
                print(f"    FAILED {side} {ch}: {type(exc).__name__}: {exc}")
        assemble_side(side, channels, cache_dir)
        print(f"    assembled per-LC caches for side={side}")


if __name__ == "__main__":
    main()
