# Synaptic Fields Pipeline

A synaptic field describes, for a given neuron of interest (e.g. LC11), how much each retinal column contributes to that neuron's input. The pipeline runs in two phases: **graph construction** (shared by both methods) and **field computation** (method-specific).

---

## Key Packages

| Package | Role |
|---------|------|
| **`fafbseg.flywire`** | FlyWire connectome interface — root ID lookup (`search_annotations`), Neuroglancer URL generation, skeleton retrieval |
| **`navis`** | Morphology processing — `make_dotprops` + `nblast_allbyall` for sorting connectivity matrices |
| **`networkx`** | Directed graph storage (`Paths.graph`); neighbour sampling in NetworkX MC; graph intersection |
| **`scipy.sparse`** | CSR transition matrix construction and sparse matrix multiplication in analytic / matrix-MC propagation |
| **`h5py`** | Chunked HDF5 on-disk storage for MC paths `(S, H, R)` and streaming reads during field extraction |
| **`pandas`** / **`numpy`** | Edge lists, node metadata, column assignments, retinal-grid binning, SVD post-processing |

### Data files used

| File | Source | Purpose |
|------|--------|---------|
| `connections_no_threshold.csv` | FlyWire public release (mat. 783) | Full synapse edge list: `pre`, `post`, `neuropil`, `weight`, `transmitter` |
| `column_assignment.csv` | FlyWire / Codex (downloaded automatically) | Connectome-wide lookup table: maps each visual neuron root ID to its integer retinal lattice column `(p, q)`, cell type, and hemisphere. Covers all visual cell types (Mi1, L1–L5, Tm, LC, etc.) on both sides. Used during field *computation* to bin absorbed/sampled mass onto the retinal grid. |
| `mi1_visual_data.csv` | Local / pre-computed from Mi1 skeletons | Mi1-only reference grid with full optical geometry: 3D soma positions, optical axis unit vectors, azimuth/elevation angles, corrected lattice coordinates, and binocularity flags (38 columns). Used *after* computation, by `SynapticField.project()` and `SynapticProjection.spherical_projection()`, to project the discrete `(p, q)` grid onto physical or visual-sphere coordinates. |

> `column_assignment.csv` is the computation-time lookup (root ID → column); `mi1_visual_data.csv` is the post-computation geometry reference (column → 3D position / optical axis), used only by `SynapticField.project()`.

---

## 1. Graph Construction — `Connectome.get_paths()`

```python
paths = connectome.get_paths('LC11', max_hops=3, direction='downstream')
```

This builds a directed, synapse-weighted graph from the cells of interest outward. Each node is assigned a **level**:

| Level | Meaning |
|-------|---------|
| `0` | **Target cells** — the queried cell type (e.g. LC11 on one side). These are the *starting cells* for all walks. |
| `1`, `2`, … | Downstream cells at each hop (medulla interneurons, etc.) |
| `-1`, `-2`, … | Upstream cells (if `direction='upstream'`) |

The `max_hops` parameter controls how deep the graph extends. For LC11 → lamina inputs, `max_hops=3` is typical (LC11 → medulla → lamina).

### Stopping points vs. starting cells

- **Starting cells**: always the level-0 target neurons.
- **Stopping points**: input cell types — default `['L1', 'L2', 'L3', 'L4', 'L5', 'R7', 'R8']` — that anchor the field to retinal (p, q) coordinates. A walk terminates on first contact with any stopping point.

---

## 2. Method A — Analytic Propagation

```python
fields, effects = paths.get_synaptic_fields(method='analytic', side='left')
```

### How it works

1. **Transition matrix** `P` (N × N, row-stochastic) is built from the edge weights: $P_{ij} = w_{ij} / \sum_k w_{ik}$.
2. **Absorbing matrix** `P_eff` is formed by zeroing the outgoing rows of all stopping-point cells, so mass stops on first arrival.
3. A probability distribution is initialized with **unit mass at each level-0 (target) cell**.
4. At each hop the distribution is multiplied by `P_eff`. Mass that lands on a stopping-point cell is accumulated into `absorbed`.
5. After `num_hops` steps, `absorbed[t, c]` gives the exact first-passage probability from target `t` to stopping-point cell `c`.
6. The absorbed mass is binned onto the (p, q) retinal grid to produce one `SynapticField` per stopping-point type plus an `'all'` aggregate.

**Key property:** exact — no sampling noise, no reps parameter.

---

## 3. Method B — Monte Carlo Simulation

### Step 1: run the simulation

```python
# NetworkX-based (original)
paths.monte_carlo(reps=1e5, direction='downstream', save_fn='lc11_mc.h5')

# Matrix-based (faster drop-in)
paths.monte_carlo_matrix(reps=1e5, direction='downstream', save_fn='lc11_mc.h5')
```

One independent random walker is launched per starting cell × per rep. At each hop, the walker samples a neighbor with probability proportional to synapse weight. Paths are stored in an HDF5 dataset of shape `(S, H, R)`:

| Dimension | Meaning |
|-----------|---------|
| `S` | Starting cells (level-0 targets, one per unique target neuron) |
| `H` | Hops (graph depth) |
| `R` | Reps (independent walks per starting cell) |

### Step 2: extract fields

```python
fields, effects = paths.get_synaptic_fields(
    stopping_points=['L1', 'L2', 'L3', 'L4', 'L5', 'R7', 'R8'],
    side='left'
)
```

The simulation paths are scanned chunk-by-chunk. For each pathway, the method finds which stopping-point cell a path first encounters, looks up that cell's retinal (p, q) coordinates, and increments a count in the probability grid. Counts are normalized by the total number of reps to give probabilities.

**Key property:** stochastic — results converge as `reps` increases. Pathway detail (full trajectory) is preserved in the HDF5 file.

---

## 4. Output — `SynapticFields`

Both methods return a `SynapticFields` dict:

```python
fields['L1']   # SynapticField, shape (num_targets, width, height)
fields['L5']   # one entry per stopping-point type found on the chosen side
fields['all']  # sum across all types
```

Each `SynapticField` is a NumPy array subclass with `.ps` and `.qs` attributes giving the retinal coordinates of each grid position. Use `.plot()` or `fp.draw_hex()` to visualize.

### Saving / loading

```python
fields.save('fields_ds_left/LC11_analytic.npz')
fields = load_SynapticFields('fields_ds_left/LC11_analytic.npz')
```

---

## 5. Method Comparison

| | Analytic | Monte Carlo |
|---|---|---|
| Starting cells | Level-0 targets (e.g. LC11 neurons) | Same |
| Stopping cells | L1–L5, R7, R8 (absorbing boundary) | Same |
| Result | Exact first-passage probability | Sampled histogram (converges with reps) |
| Pathway storage | None | Full trajectories in HDF5 |
| Speed | $O(N^2 \cdot H)$ matrix ops | $O(S \cdot H \cdot R \cdot \bar{d})$ walks |
| API | `method='analytic'` | `monte_carlo()` / `monte_carlo_matrix()` |
