"""test_ariad_single_simplified.py

Deterministic unit tests for the simplified single-graph Ariad
(ariad_single.py in this folder): no replicas, no reset, no elite-memory
carry -- see simplified_ariad.md for the design rationale.

Covers:
  1. `_sample_forward_paths` (the 2-hop random-walk primitive shared by both
     the O(O(i)) and O(I(i)) candidate families) lands exactly where it
     should, not in some other node's neighborhood.
  2. `_build_reverse` correctly identifies in-neighbors ("who points to me").
  3. A 2-hop path through a node's CURRENT neighbor can introduce a strictly
     better candidate and have it win the top-k re-selection (proves the O2
     expansion actually does something, not just re-sorting O(i)).
  4. With every expansion budget at 0 (pure O(i) rescoring, no new
     candidates ever enter the pool), the graph's neighbor SET is exactly
     stable across repeated refreshes -- this is the "O(i) alone preserves
     true top-k members" property the removed elite-memory-carry step used
     to provide redundantly once there was no reset to protect against.
  5. The very first forward() call returns a graph that has actually been
     scored against Z (not raw unscored random indices).

Run directly: `python test_ariad_single_simplified.py`
Or via pytest: `pytest test_ariad_single_simplified.py -v`
"""

import torch

from ariad_single import AriadSingleConfig, AriadSeekerSingle


def _make_seeker(k=4, k_rev=8, c_o2=0, c_i1=0, c_oi=0, c_rand=0, score_chunk=64):
    cfg = AriadSingleConfig(
        k=k, k_rev=k_rev, c_o2=c_o2, c_i1=c_i1, c_oi=c_oi, c_rand=c_rand,
        score_chunk=score_chunk, use_tiebreak=False, compute_churn_stats=False,
    )
    return AriadSeekerSingle(cfg)


# ============================================================
# 1. _sample_forward_paths primitive
# ============================================================

def test_sample_forward_paths_semantics():
    """O(O(0)) with O(0) pinned to {2} and graph row 2 pinned to {4,5} must
    land exactly in {4,5} (2-hop random-walk semantics)."""
    n, k_anchor, k_graph = 8, 4, 8
    seeker = _make_seeker()
    torch.manual_seed(0)

    anchors = torch.arange(n, dtype=torch.long).view(n, 1).expand(n, k_anchor).clone()
    anchors[0, :] = 2   # O(0) = {2} (every anchor slot points to node 2)

    graph = torch.zeros(n, k_graph, dtype=torch.long)
    graph[2, :] = torch.tensor([4, 5, 4, 5, 4, 5, 4, 5])

    out = seeker._sample_forward_paths(anchors, graph, c=20)
    produced = set(out[0, :].tolist())
    assert produced <= {4, 5}, f"leaked outside {{4,5}}: {produced}"
    assert produced == {4, 5}, f"didn't cover both targets: {produced}"

    graph2 = torch.zeros(n, k_graph, dtype=torch.long)
    graph2[2, :] = torch.tensor([6, 7, 6, 7, 6, 7, 6, 7])
    out2 = seeker._sample_forward_paths(anchors, graph2, c=20)
    produced2 = set(out2[0, :].tolist())
    assert produced2 == {6, 7}, f"didn't follow the updated graph: {produced2}"


def test_sample_forward_paths_zero_budget_is_empty():
    seeker = _make_seeker()
    anchors = torch.zeros(5, 3, dtype=torch.long)
    graph = torch.zeros(5, 3, dtype=torch.long)
    out = seeker._sample_forward_paths(anchors, graph, c=0)
    assert out.shape == (5, 0)


# ============================================================
# 2. _build_reverse primitive
# ============================================================

def test_build_reverse_identifies_in_neighbors():
    """Only node 2 points to node 0 (row 2 -> all 0's); reverse[0] must be
    exactly {2} (padded with random filler beyond that one real hit)."""
    n, k_rev = 8, 4
    seeker = _make_seeker(k_rev=k_rev)

    neighbors = torch.arange(n, dtype=torch.long).view(n, 1).expand(n, 1).clone()
    neighbors[0, 0] = 1   # avoid a stray self-loop (0 -> 0) polluting reverse[0]
    neighbors[2, 0] = 0   # node 2 -> node 0: the only real in-edge into node 0

    rev = seeker._build_reverse(neighbors, n)
    assert rev.shape == (n, k_rev)
    # Real in-neighbor 2 must appear at least once in reverse[0]; the rest of
    # the row is random padding (k_rev=4 > the single real in-edge).
    assert 2 in rev[0].tolist(), f"missed the real in-neighbor: {rev[0].tolist()}"


# ============================================================
# 3. O2 expansion can introduce and win with a strictly better candidate
# ============================================================

def test_o2_expansion_improves_neighbor():
    """Node 0's current (bad) neighbor is node 1. Node 1's own neighbor is
    node 2, and z2 is deliberately the best possible match for z0. With
    c_o2>0 the O(O(0)) expansion must discover node 2 via the 1 -> 2 hop,
    and it must win the top-k=1 re-selection over the original bad
    candidate 1."""
    n, d = 6, 4
    seeker = _make_seeker(k=1, c_o2=8, c_i1=0, c_oi=0, c_rand=0)
    torch.manual_seed(0)

    Z = torch.randn(n, d)
    # Make node 2 the unambiguous best match for node 0 (largest dot product),
    # and node 1 (0's current neighbor) a clearly worse match.
    Z[2] = Z[0] * 10.0 + 0.01 * torch.randn(d)
    Z[1] = -Z[0]   # deliberately bad match (near-minimal dot product)

    seeker.init_random(n, Z.device)
    seeker.G[0, 0] = 1   # 0's current (bad) neighbor
    seeker.G[1, 0] = 2   # 1's neighbor is 2 -- the 2-hop path 0 -> 1 -> 2

    seeker.refresh(Z)

    assert seeker.G[0, 0].item() == 2, (
        f"O2 expansion should have replaced bad neighbor 1 with the true "
        f"best match 2 (found via 0->1->2), got {seeker.G[0, 0].item()}"
    )


# ============================================================
# 4. Zero-budget refresh never changes the neighbor SET (O(i) alone is
#    stable -- the property that made elite-memory-carry redundant here)
# ============================================================

def test_zero_budget_refresh_is_a_pure_rescore_no_set_change():
    n, d, k = 30, 8, 5
    seeker = _make_seeker(k=k, c_o2=0, c_i1=0, c_oi=0, c_rand=0)
    torch.manual_seed(1)

    Z = torch.randn(n, d)
    seeker.init_random(n, Z.device)
    before = seeker.G.clone()

    for _ in range(5):
        seeker.refresh(Z)
        after_sets = [set(row.tolist()) for row in seeker.G]
        before_sets = [set(row.tolist()) for row in before]
        assert after_sets == before_sets, (
            "neighbor SET drifted under zero expansion budget -- O(i) alone "
            "should be a fixed point once nothing new ever enters the pool"
        )


# ============================================================
# 5. First forward() call is scored against Z, not raw random
# ============================================================

def test_first_forward_call_is_scored():
    """forward() on a fresh (uninitialized) seeker must actually run a
    scoring refresh() before returning -- not just init_random() followed by
    handing back raw, unscored random indices. Proxy: refresh() is the only
    thing that advances step_counter / populates last_refresh_stats;
    init_random() alone does neither."""
    n, d = 20, 8
    seeker = _make_seeker(k=4, c_o2=8, c_i1=4, c_oi=4, c_rand=0)
    Z = torch.randn(n, d)

    assert not seeker.initialized
    out = seeker.forward(Z)

    assert seeker.initialized
    assert seeker.step_counter == 1, "forward() must call refresh() once after init_random(), not skip it"
    assert seeker.last_refresh_stats is not None
    assert torch.equal(out, seeker.G)

    # And with a genuine 2-hop improvement planted right after init, a
    # SECOND forward() call (now going through the elif self.training
    # branch) must pick it up too -- reusing the same mechanism as
    # test_o2_expansion_improves_neighbor, just entered via forward().
    Z2 = torch.randn(n, d)
    Z2[2] = Z2[0] * 10.0 + 0.01 * torch.randn(d)
    Z2[1] = -Z2[0]
    seeker.G[0, 0] = 1
    seeker.G[1, 0] = 2
    seeker.train()
    seeker.forward(Z2)
    assert seeker.G[0, 0].item() == 2


ALL_TESTS = [
    test_sample_forward_paths_semantics,
    test_sample_forward_paths_zero_budget_is_empty,
    test_build_reverse_identifies_in_neighbors,
    test_o2_expansion_improves_neighbor,
    test_zero_budget_refresh_is_a_pure_rescore_no_set_change,
    test_first_forward_call_is_scored,
]


if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
        print(f"PASS {t.__name__}")
    print("ALL SIMPLIFIED ARIAD TESTS PASSED")
