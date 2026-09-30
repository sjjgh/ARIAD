# Method and implementation

ARIAD maintains graph state across optimizer steps for a fixed node set.
The released algorithms are the simplified, single-graph variants described
in the author's [single-space notes](../ariad_single_space/simplified_ariad.md)
and [QK notes](../ariad_qk_space/simplified_ariad_qk.md) (Chinese).

## Single space

For embeddings Z, maintain G with shape N × k. At each training forward pass,
form a candidate pool for every row i:

| Family | Construction | Experiment slots |
|---|---|---:|
| O(i) | Retain the current outgoing neighbors | 32 |
| O(O(i)) | Sample an outgoing anchor, then one of its outgoing neighbors | 80 |
| I(i) | Sampled incoming-edge buffer | 32 |
| O(I(i)) | Outgoing neighbors of an incoming anchor | 0 |
| Random | Uniform node proposals | 0 |

Score candidates with z_i · z_j using detached embeddings, mask duplicate IDs
and self-links, and select top-k. Reverse buffers use real incoming edges first
and random nonself padding for unfilled slots, so some random exploration remains
even with the explicit random-candidate budget set to zero.

Keeping O(i) makes a separate copy of the previous final graph redundant when
there is no reset. A retained neighbor that remains in the global top-k under
the current scores cannot be displaced by worse candidates (assuming sufficient
distinct valid candidates and consistent tie handling). This is not a proof of
global convergence or monotonically improving recall under changing embeddings.

## Independent Q/K spaces

Refresh QQ using Q, then KK using K. Refresh QK using the new within-space graphs
and the previous QK graph. All candidate paths below end in **key IDs**:

| Family | Path | Experiment slots |
|---|---|---:|
| Retained | QK(i) | 32 |
| Key forward | KK(QK(i)) | 32 |
| Query forward | QK_previous(QQ(i)) | 32 |
| Key reverse | Reverse_KK(QK(i)) | 32 |
| Query reverse | QK_previous(Reverse_QQ(i)) | 0 |

Rerank with q_i · k_j, remove duplicate and self candidates, and select 32 keys.
All chunks read the same previous QK graph before its replacement. Attention
uses softmax(q_i · k_j / √d) over the selected keys and aggregates their values.
Gradients flow through Q, K, values, and selected attention scores, but not
through graph search or discrete neighbor selection.

QQ and KK have width 64 and each score 64 + 80 + 32 = 176 slots. QK scores
128, totaling 480 per node per step. Unlike the single-space runtime counter,
480 is derived from configuration, not a measured `n_scored` counter.

## Cost model

For single-space training, the supplied arithmetic model is:

```text
F = 6 N d_in (d_z + d_v)
  + 6 P_attn (d_z + d_v)
  + 2 P_search d_z
```

QK replaces the projection term with `6 N d_in (2 d_z + d_v)`.
`(P_attn, P_search)` is `(N², 0)` for Dense, `(Nk, N²)` for Exact, and
`(Nk, NC)` for ARIAD. One multiply-add counts as two FLOPs; differentiable
forward/backward work is approximated as three forward computations.
For the fixed-value QK experiment this convention overcounts backward work.

The model excludes softmax, top-k, sorting/deduplication, reverse-buffer
construction, gathers/scatters, memory traffic, and evaluation audits.
It cannot be converted into a measured speedup. Fixed-budget graph storage is
O(Nk); the gather-based implementation also uses candidate and chunk intermediates.

## Implementation boundaries

- **Historical QK auxiliary graphs:** the QK folder retains an older
  `ariad_single.py` that can retain self-links in QQ/KK and lacks `n_scored`.
  QK attention excludes self-links. This can waste auxiliary candidate slots.
  Synchronizing it with the single-space version would change the experiment
  and requires rerunning the QK results.
- **Fixed identity/order:** graph state is tied to a fixed node count, device,
  and row ordering. Instantiate a new model/seeker for a different node set.
  Do not treat independent minibatches as consecutive updates of the same graph.
- **Evaluation:** after initialization, `.eval()` reuses graph state. It does
  not search new inputs. A first call initializes and refreshes even in eval mode.
- **Checkpointing:** the graph tensors are ordinary attributes rather than
  registered buffers. `state_dict()` does not provide complete graph/RNG state
  for an exact resumed run; device moves should happen before initialization.
- **Candidate sufficiency:** undersized or highly degenerate candidate pools can
  contain fewer than k distinct valid nodes. The historical fallback is not a
  general guarantee of k unique nonself neighbors. Use the supplied settings
  with adequate N; the demos test valid output at N=256.
- **Task scope:** QK here is noncausal self-attention over the same node set,
  not cross-attention with different query/key lengths or a batched language model.

The core numerical implementations are retained from the experiment sources
to preserve provenance. The release adds documentation, entry points, and
validation rather than silently changing the reported algorithm.
