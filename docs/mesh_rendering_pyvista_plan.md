# Mesh rendering plan — PyVista (replacing Plotly `Mesh3d` / neuroglancer for Section 8d/8e)

Goal: replace the slow Plotly `Mesh3d` per-hop rendering (Section 8d) and the glitchy
neuroglancer scenes (Section 8e) with a fast, interactive, VTK-based PyVista renderer that
colours each cell by its per-hop output contribution (OC) on a magma scale.

The meshes are **already cached locally** at `matrix_cache/meshes_lod4.pkl` (written by the
Section 8d cell), so no re-download is required. This plan reuses that cache and adds a
one-time export step so the geometry is also available as portable `.ply` files (useful later
for Blender or any external tool).

---

## 0. Prerequisites

- Install PyVista into the active env (`.venv`, Python 3.12.9):

  ```powershell
  pip install pyvista
  ```

- Off-screen rendering (for saving PNGs without a window) additionally needs a VTK-capable
  backend; PyVista bundles it, but for headless PNG export set `pv.OFF_SCREEN = True`.
- Reuses existing objects from the **Section 8 base cell** and Section 5:
  `o0`, `Pr`, `N_HOPS`, `sorted_info`, `CACHE_DIR`, plus the mesh cache pickle.

---

## 1. Load the cached meshes (no re-download)

- Read `matrix_cache/meshes_lod{MESH_LOD}.pkl` → `{root_id: (vertices, faces)}`.
- If a root_id is missing from the cache, fetch it once via
  `fafbseg.flywire.get_mesh_neuron(..., lod=MESH_LOD)` and update the pickle (same logic as the
  current 8d cell), so the notebook still works on a fresh clone.
- Helper `to_polydata(V, F)` converting `(vertices, faces)` → `pyvista.PolyData` with the VTK
  face format (prepend each triangle with a `3`): `faces = np.hstack([np.full((n,1), 3), F]).ravel()`.

## 2. One-time export to `.ply` (portability / future Blender use)

- New helper `export_meshes_to_ply(out_dir="matrix_cache/meshes_ply")`:
  - For each cached cell, write `<root_id>.ply` (binary) via `PolyData.save()`.
  - Skip files that already exist so it's cheap to re-run.
- This is the "store locally" step — geometry now lives as standard `.ply` alongside the pickle.

## 3. Per-hop node occupancy (same as 8d/8e)

- Recompute `node_vals = [o0, o0@Pr, (o0@Pr)@Pr, ...]` for hops `0..N_HOPS`.
- Reuse the **hop-0 degenerate fix** already applied in 8e: when a hop's log-range span is ~0
  (uniform `o0`), map present cells to a single bright colour instead of the black bottom of
  magma.

## 4. PyVista renderer — interactive, per-hop

- Function `render_hops_pyvista(hops=range(N_HOPS+1), lod=MESH_LOD, opacity=..., cmap="magma")`.
- Layout: a `pyvista.Plotter(shape=(2, 3))` (or `(3, 2)`) small-multiple grid, one subplot per hop.
- For each hop:
  - Compute per-cell `norm01` from `log10(OC)` (with the degenerate-hop guard from §3).
  - Add each cell mesh with `plotter.add_mesh(pd, color=magma(norm01), opacity=<by level>,
    smooth_shading=True)`.
  - Optimization: **combine all cells of a hop into one `PolyData`** via `pv.MultiBlock(...).combine()`
    or `blk.append(...)` + a per-vertex scalar array, then a single `add_mesh(..., scalars="oc",
    cmap="magma")`. One actor per hop instead of one-per-cell is dramatically faster than Plotly's
    30 traces and avoids the per-mesh-opacity limitation.
  - Link camera across subplots (`plotter.link_views()`), set a sensible azimuth/elevation.
- `plotter.show()` for interactive use inside VS Code / a window.

## 5. Optional still export (PNG per hop)

- `render_hops_pyvista(..., screenshot_dir="mesh_renders")` with `pv.OFF_SCREEN = True`:
  - One PNG per hop (`hop0.png` … `hop5.png`) plus an optional combined contact sheet.
- These are the publication-style stills; no Blender needed for a first pass.

## 6. Notebook integration

- Add a new **Section 8f** code cell implementing §1–§5, placed right after the 8e cell.
- Leave 8d/8e in place (commented note that 8f supersedes 8d for speed); do not delete, so the
  Plotly/neuroglancer versions remain for reference.
- Add a short markdown cell above 8f describing the approach and the `pip install pyvista` step.

## 7. Follow-ups (out of scope for the first pass)

- **Neuroglancer glitch (8e):** likely the segmentation-layer alpha (`objectAlpha` with many
  overlapping transparent meshes → z-fighting/flicker) or too many segments per scene. Fix
  independently by raising alpha, reducing segment count, or disabling 2D cross-section alpha.
- **Blender path:** if a final animation is wanted later, the `.ply` files from §2 feed a headless
  `blender --background --python render.py` script (emission = magma(log OC), alpha by level,
  one frame per hop). Deferred until the PyVista output is validated.

---

## Deliverables

1. `pip install pyvista` (one-time).
2. New **Section 8f** cell: cached-mesh load → optional `.ply` export → per-hop PyVista grid
   (interactive) with optional PNG export.
3. Reused hop-0 degenerate-colour fix so hop 0 isn't uniformly black.
4. (Later) neuroglancer alpha fix and optional Blender render script.
