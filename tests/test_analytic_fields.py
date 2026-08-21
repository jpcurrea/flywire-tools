"""Offline unit tests for the analytic synaptic-field core.

These validate the real ``Paths._build_transition_matrix`` and the first-passage
absorbing-propagation math against small hand-computed examples. FlyWire network
dependencies (fafbseg, navis) are stubbed so the test runs anywhere without a token.

Run:  python tests/test_analytic_fields.py
"""
import os
import sys
import types

import numpy as np

# --- stub non-essential / network deps that connectome.py imports at module load,
#     so the analytic core can be tested without FlyWire or plotting libraries ---
def _stub(name, **attrs):
    module = sys.modules.get(name) or types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _stub_submodule(parent, child, **attrs):
    full = f"{parent}.{child}"
    module = _stub(full, **attrs)
    setattr(sys.modules[parent], child, module)
    return module


_stub("navis", __version__="stub")
_stub_submodule("navis", "models")
_stub("fafbseg")
_stub_submodule("fafbseg", "flywire")
_stub("seaborn", set_style=lambda *a, **k: None)
_stub("svgutils")
_stub_submodule("svgutils", "transform")
_stub_submodule("svgutils", "compose", Unit=object)
_stub("plotly")
_stub_submodule("plotly", "io")
_stub_submodule("plotly", "graph_objects")
_stub("h5py")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "src", "flywire_tools")
sys.path.insert(0, PKG)

import networkx as nx  # noqa: E402
import connectome as C  # noqa: E402


def make_paths(graph, direction):
    """Build a Paths instance without running __init__ (which needs FlyWire)."""
    p = C.Paths.__new__(C.Paths)
    p.graph = graph
    p.direction = direction
    p.node_ids = np.array(sorted(graph.nodes()))
    return p


def first_passage(P, node_ids, target_ids, absorbing_ids, num_hops):
    """Replicate the analytic propagation to compare against the matrix builder."""
    import scipy.sparse as sp

    N = len(node_ids)
    absorbing_idx = np.searchsorted(node_ids, absorbing_ids)
    is_abs = np.zeros(N, dtype=bool)
    is_abs[absorbing_idx] = True
    P_eff = (sp.diags((~is_abs).astype(float)) @ P).tocsr()
    target_idx = np.searchsorted(node_ids, target_ids)
    dist = sp.csr_matrix(
        (np.ones(len(target_idx)), (np.arange(len(target_idx)), target_idx)),
        shape=(len(target_idx), N),
    )
    absorbed = np.zeros((len(target_idx), N))
    for _ in range(num_hops):
        absorbed[:, absorbing_idx] += dist[:, absorbing_idx].toarray()
        dist = dist @ P_eff
    absorbed[:, absorbing_idx] += dist[:, absorbing_idx].toarray()
    return absorbed


def test_downstream_transition_matrix():
    g = nx.DiGraph()
    g.add_edge(10, 20, weight=3)
    g.add_edge(10, 30, weight=1)
    g.add_edge(20, 30, weight=1)
    p = make_paths(g, "downstream")
    P = p._build_transition_matrix().toarray()
    expected = np.array([[0.0, 0.75, 0.25], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    assert np.allclose(P, expected), f"downstream P wrong:\n{P}"
    print("ok: downstream transition matrix")


def test_upstream_transition_matrix():
    # same edges; upstream walker moves post -> pre, normalized over in-edges
    g = nx.DiGraph()
    g.add_edge(10, 30, weight=3)  # 30's in-edge from 10 (w3)
    g.add_edge(20, 30, weight=1)  # 30's in-edge from 20 (w1)
    p = make_paths(g, "upstream")
    P = p._build_transition_matrix().toarray()
    # rows/cols ordered by node_ids [10, 20, 30]; only node 30 has predecessors
    expected = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.75, 0.25, 0.0]])
    assert np.allclose(P, expected), f"upstream P wrong:\n{P}"
    print("ok: upstream transition matrix")


def test_first_passage_single_absorber():
    g = nx.DiGraph()
    g.add_edge(10, 20, weight=3)
    g.add_edge(10, 30, weight=1)
    g.add_edge(20, 30, weight=1)
    p = make_paths(g, "downstream")
    P = p._build_transition_matrix()
    absorbed = first_passage(P, p.node_ids, target_ids=[10], absorbing_ids=[30], num_hops=2)
    # every path from A eventually reaches C within 2 hops -> prob 1
    assert np.isclose(absorbed[0, np.searchsorted(p.node_ids, 30)], 1.0), absorbed
    print("ok: first-passage single absorber (reach prob = 1.0)")


def test_first_passage_two_absorbers():
    g = nx.DiGraph()
    g.add_edge(10, 20, weight=3)
    g.add_edge(10, 30, weight=1)
    g.add_edge(20, 30, weight=1)
    p = make_paths(g, "downstream")
    P = p._build_transition_matrix()
    absorbed = first_passage(P, p.node_ids, target_ids=[10], absorbing_ids=[20, 30], num_hops=2)
    b = absorbed[0, np.searchsorted(p.node_ids, 20)]
    c = absorbed[0, np.searchsorted(p.node_ids, 30)]
    # first arrival: B with 0.75 (direct), C with 0.25 (direct)
    assert np.isclose(b, 0.75) and np.isclose(c, 0.25), (b, c)
    print("ok: first-passage two absorbers (0.75 / 0.25 split)")


if __name__ == "__main__":
    test_downstream_transition_matrix()
    test_upstream_transition_matrix()
    test_first_passage_single_absorber()
    test_first_passage_two_absorbers()
    print("\nAll analytic-core tests passed.")
