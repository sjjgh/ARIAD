from dataclasses import dataclass

import torch
import torch.nn as nn

"""
Simplified single-space Ariad: single graph + local neighborhood expansion.

This drops the async multi-replica machinery of the original ariad_single.py
entirely: no M replicas, no legacy fixed-period reset, no maturity-triggered
reset, and no separate "elite memory carry" step. See simplified_ariad.md for
the full rationale and the empirical observation that motivated it (match_true
still converges to 1 across several N with M=1 and reset effectively disabled).

Why elite-memory-carry is ALSO dropped here (not just reset): the original
carry step existed to survive REPLICA RESET -- when a replica got randomly
wiped, its own local neighbors were gone, so the separately-tracked "final"
graph (cached_neighbors) was needed as a backup that didn't get wiped by that
reset. With a single graph and no reset event ever wiping it, that backup role
is already served by the graph's own current neighbor set O(i), which is
unconditionally included in every refresh step's candidate pool. Under exact
(fp32) scoring, a node that is still genuinely among i's current top-k
neighbors can never be crowded out of a top-k re-selection over any candidate
pool that contains it (at most k-1 OTHER candidates can outscore it, by
definition of top-k) -- so O(i) alone already guarantees "no true neighbor is
silently lost between steps." Re-appending a separately-tracked copy of the
same graph on top of that is a provable no-op once there's no reset to
protect against, so it's removed as unnecessary complexity along with reset
itself.
"""


DEFAULT_ARIAD_SINGLE_CONFIG = {
    "k": 16,               # 近邻数（唯一一张图的宽度，合并了原来的 k_replica/k_final）
    "k_rev": 32,            # 反向缓冲宽度：每个节点记录多少"谁指向我"
    "c_o2": 64,             # 局部扩张候选预算：二跳 outward 候选数量
    "c_i1": 32,             # 局部扩张候选预算：一跳 inbound 候选数量
    "c_oi": 16,             # 局部扩张候选预算：OI 混合候选（outbound 的 inbound）数量
    "c_rand": 0,            # 局部扩张候选预算：随机候选数量
    "score_chunk": 4096,    # 打分分块大小：越大通常越快但显存/内存更高
    "use_tiebreak": False,
    "compute_churn_stats": False,
}


@dataclass
class AriadSingleConfig:
    k: int
    k_rev: int
    c_o2: int
    c_i1: int
    c_oi: int
    c_rand: int
    score_chunk: int
    use_tiebreak: bool = False
    compute_churn_stats: bool = False


class AriadSeekerSingle(nn.Module):
    """Single graph, no replicas, no reset.

    Each refresh() call (once per TRAINING forward):
      1. Build a reverse (in-neighbor) buffer from the current graph.
      2. For every node i, build a candidate pool:
             O(i)    -- current neighbors, unconditionally kept
             O(O(i)) -- 2-hop forward (random anchor in O(i), random hop from it)
             I(i)    -- reverse buffer (who points to i)
             O(I(i)) -- 2-hop via inbound (random anchor in I(i), random hop from it)
             random  -- optional, budget c_rand (default 0)
      3. Rescore all candidates against the CURRENT embedding Z and keep the
         top-k -- this becomes the new graph, consumed directly by this
         step's attention aggregation (no separate "final" stage; the graph
         IS the output).
    """

    def __init__(self, cfg: AriadSingleConfig):
        super().__init__()
        self.k = int(cfg.k)
        self.k_rev = max(1, int(cfg.k_rev))
        self.c_o2 = max(0, int(cfg.c_o2))
        self.c_i1 = max(0, int(cfg.c_i1))
        self.c_oi = max(0, int(cfg.c_oi))
        self.c_rand = max(0, int(cfg.c_rand))
        self.score_chunk = int(cfg.score_chunk)
        self.use_tiebreak = bool(cfg.use_tiebreak)
        self.compute_churn_stats = bool(cfg.compute_churn_stats)

        self.initialized = False
        self.step_counter = 0  # number of refresh() calls so far

        self.G = None                 # LongTensor [n, k] -- the single graph
        self.last_refresh_stats = None

    # ---------- initialization ----------

    @staticmethod
    def _random_neighbors(n, k, device):
        return torch.randint(0, n, (n, k), device=device, dtype=torch.long)

    def init_random(self, n, device):
        self.G = self._random_neighbors(n, self.k, device)
        self.step_counter = 0
        self.initialized = True

    # ---------- scoring helpers ----------

    def _score_gather_chunk(self, Z_src_chunk, Z_tgt_all, cand_chunk):
        """Z_src_chunk: [b, d]. Z_tgt_all: [n, d]. cand_chunk: [b, C]."""
        gathered = Z_tgt_all[cand_chunk]             # [b, C, d]
        Zs_local = Z_src_chunk.unsqueeze(1)           # [b, 1, d]
        return (Zs_local * gathered).sum(-1)          # [b, C]

    @staticmethod
    def _mask_row_duplicates(cand, scores, valid):
        sorted_cand, sort_idx = cand.sort(dim=-1)
        dup_sorted = torch.zeros_like(sorted_cand, dtype=torch.bool)
        dup_sorted[..., 1:] = sorted_cand[..., 1:] == sorted_cand[..., :-1]
        dup_orig = torch.zeros_like(dup_sorted)
        dup_orig.scatter_(-1, sort_idx, dup_sorted)
        return scores.masked_fill(dup_orig & valid, float("-inf"))

    def _apply_id_tiebreak(self, scores, cand, eps=1e-12):
        if not self.use_tiebreak:
            return scores
        return scores.to(torch.float64) - cand.to(torch.float64) * float(eps)

    @staticmethod
    def _sample_forward_paths(anchors, graph, c):
        """anchors: [n, k_anchor]; graph: [n_graph, k_graph]. Returns [n, c]:
        for each row, randomly pick an anchor id from `anchors`, then
        randomly pick one of that anchor's own neighbors from `graph` -- a
        single 2-hop random-walk step."""
        n, k_anchor = anchors.shape
        if c <= 0:
            return anchors.new_empty((n, 0))
        k_graph = graph.size(-1)
        if k_anchor <= 0 or k_graph <= 0:
            return anchors.new_empty((n, 0))

        device = anchors.device
        anchor_pos = torch.randint(0, k_anchor, (n, c), device=device)
        graph_pos = torch.randint(0, k_graph, (n, c), device=device)
        sampled_anchors = anchors.gather(-1, anchor_pos)   # [n, c] node ids
        return graph[sampled_anchors, graph_pos]            # [n, c]

    def _build_reverse(self, neighbors, n):
        """Fixed-size reverse buffer [n, k_rev] (v3-style): who points to me."""
        k = neighbors.size(1)
        device = neighbors.device
        src = torch.arange(n, device=device, dtype=torch.long).unsqueeze(1).expand(n, k).reshape(-1)
        tgt = neighbors.reshape(-1)
        mask_t = (tgt >= 0) & (tgt < n)
        src = src[mask_t]
        tgt = tgt[mask_t]

        if tgt.numel() == 0:
            rand_part = torch.randint(0, n, (n, self.k_rev), device=device, dtype=torch.long)
            rows_idx = torch.arange(n, device=device, dtype=torch.long).unsqueeze(1)
            collision = rand_part == rows_idx
            if collision.any():
                rand_part = torch.where(collision, (rand_part + 1) % n, rand_part)
            return rand_part

        perm = torch.randperm(tgt.numel(), device=device)
        tgt = tgt[perm]
        src = src[perm]
        order = tgt.argsort()
        tgt_sorted = tgt[order]
        src_sorted = src[order]
        counts = torch.bincount(tgt_sorted, minlength=n)
        offsets = torch.cumsum(counts, dim=0) - counts
        slot_idx = torch.arange(self.k_rev, device=device, dtype=torch.long).unsqueeze(0)
        abs_pos = offsets.unsqueeze(1) + slot_idx
        abs_pos_clip = torch.clamp(abs_pos, max=max(0, src_sorted.numel() - 1))
        real_part = src_sorted[abs_pos_clip]
        need_rand = slot_idx >= counts.unsqueeze(1)

        rand_part = torch.randint(0, n, (n, self.k_rev), device=device, dtype=torch.long)
        rows_idx = torch.arange(n, device=device, dtype=torch.long).unsqueeze(1)
        collision = rand_part == rows_idx
        if collision.any():
            rand_part = torch.where(collision, (rand_part + 1) % n, rand_part)
        return torch.where(need_rand, rand_part, real_part)

    # ---------- graph refresh ----------

    def refresh(self, Z):
        """One training step. Assumes Z is detached."""
        assert self.initialized
        n, k = self.G.shape
        device = Z.device

        with torch.no_grad():
            rev = self._build_reverse(self.G, n)   # [n, k_rev]

            new_G = torch.empty(n, k, device=device, dtype=torch.long)
            n_scored = 0   # dot products actually evaluated this refresh (incl. dup/self, masked afterwards)
            chunk = max(1, self.score_chunk)
            for s in range(0, n, chunk):
                e = min(s + chunk, n)

                local = self.G[s:e, :]   # O(i) -- unconditionally kept
                cand_parts = [local]

                if self.c_o2 > 0:
                    cand_parts.append(self._sample_forward_paths(local, self.G, self.c_o2))
                if self.c_i1 > 0:
                    c_i1 = min(self.c_i1, rev.size(-1))
                    cand_parts.append(rev[s:e, :c_i1])
                if self.c_oi > 0:
                    cand_parts.append(self._sample_forward_paths(rev[s:e, :], self.G, self.c_oi))
                if self.c_rand > 0:
                    cand_parts.append(torch.randint(0, n, (e - s, self.c_rand), device=device, dtype=torch.long))

                cand_chunk = torch.cat(cand_parts, dim=-1)   # [b, C]
                n_scored += cand_chunk.numel()
                scores_chunk = self._score_gather_chunk(Z[s:e], Z, cand_chunk)
                valid_chunk = torch.ones_like(cand_chunk, dtype=torch.bool)
                scores_chunk = self._mask_row_duplicates(cand_chunk, scores_chunk, valid_chunk)
                # exclude self-loops at scoring time so i never occupies one of its own k slots
                row_ids = torch.arange(s, e, device=device, dtype=torch.long).unsqueeze(1)
                scores_chunk = scores_chunk.masked_fill(cand_chunk == row_ids, float("-inf"))
                scores_chunk = self._apply_id_tiebreak(scores_chunk, cand_chunk)

                k_eff = min(k, cand_chunk.size(-1))
                topk = scores_chunk.topk(k_eff, dim=-1)
                new_chunk = cand_chunk.gather(-1, topk.indices)
                if k_eff < k:
                    pad = cand_chunk[:, : k - k_eff]
                    new_chunk = torch.cat([new_chunk, pad], dim=-1)
                new_G[s:e, :] = new_chunk

            prev_G = self.G
            self.G = new_G

            churn = None
            if self.compute_churn_stats:
                churn = self._rowwise_topk_churn(prev_G, self.G)

            self.last_refresh_stats = {
                "step": self.step_counter,
                "churn": None if churn is None else float(churn),
                "n_scored": n_scored,
            }

        self.step_counter += 1

    @staticmethod
    def _rowwise_topk_churn(prev, curr):
        """Jaccard-style churn across the top-k neighbor set per node, averaged."""
        if prev is None or curr is None or prev.shape != curr.shape:
            return None
        n, k = prev.shape
        if n == 0 or k == 0:
            return None
        prev_sorted, _ = prev.sort(dim=1)
        curr_sorted, _ = curr.sort(dim=1)
        a = prev_sorted.unsqueeze(2)
        b = curr_sorted.unsqueeze(1)
        inter = (a == b).any(dim=2).sum(dim=1).float()
        union = (2 * k) - inter
        jacc = (inter / union.clamp(min=1)).mean().item()
        return 1.0 - jacc

    def forward(self, Z):
        Z = Z.detach()
        n = Z.size(0)
        if not self.initialized:
            self.init_random(n, Z.device)
            self.refresh(Z)   # score the random init against Z once before ever returning it
        elif self.training:
            self.refresh(Z)
        return self.G
