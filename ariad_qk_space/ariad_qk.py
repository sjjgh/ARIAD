from dataclasses import dataclass

import torch
import torch.nn as nn

from ariad_single import AriadSeekerSingle

"""
QK dual-space Ariad: separate Q-space and K-space, self-attention over the
SAME n nodes (node i has both a query role and a key role, like standard
self-attention -- so "self" means key_id == query_id, both being the same
underlying node index, and is excluded the same way the single-space
version excludes it).

Three graphs are maintained:
  - QQ graph (query-query): maintained by a plain AriadSeekerSingle fed Q.
  - KK graph (key-key):     maintained by a plain AriadSeekerSingle fed K.
  - QK graph (query->key):  THIS module. A(i) = QK[i] is query i's current
    set of attended keys. Candidates for A(i)'s next refresh come from four
    families, all producing KEY ids:

        A(i)     = QK[i]                     current keys, kept in full
        kF(i)    = O_KK( A(i) )               key-side forward
        qF(i)    = O_QK( O_QQ(i) )            query-side forward (borrow
                                               similar queries' keys)
        kR(i)    = I_KK( A(i) )               key-side reverse (keys that
                                               point to my current keys)
        qR(i)    = O_QK( I_QQ(i) )            query-side reverse (borrow
                                               keys found by queries that
                                               point to me)

  qF/kF/kR/qR are all built with `AriadSeekerSingle._sample_forward_paths`
  (reused as-is, unbound static call -- no need to duplicate it), so they
  follow EXACTLY the same "random anchor, random hop" semantics the
  single-space QQ/KK graphs already use internally for their own O(O(i))
  family.

  NOT implemented (deliberately, out of scope for this round): I_QK(k) --
  that returns QUERY ids (the queries currently attending to key k), not key
  ids, so it can't be used as a key candidate directly. A type-correct 3-hop
  route O_QK(I_QK(A(i))) exists on paper but is not introduced here.
"""


DEFAULT_ARIAD_QK_CONFIG = {
    "k_final": 16,      # 每个 query 最终保留的 key 数量
    "k_rev": 32,         # KK / QQ 反向缓冲宽度（构造 kR / qR 用）
    "b_kF": 16,          # key 侧前向 kF 预算
    "b_qF": 16,          # query 侧前向 qF 预算
    "b_kR": 16,          # key 侧反向 kR 预算（0 = 关闭）
    "b_qR": 0,           # query 侧反向 qR 预算（0 = 关闭；默认关，诊断开关，见 candidate_path_ablation.py 起点建议）
    "score_chunk": 4096,
    "use_tiebreak": False,
}


@dataclass
class AriadQKConfig:
    k_final: int
    k_rev: int
    b_kF: int
    b_qF: int
    b_kR: int
    b_qR: int
    score_chunk: int = 4096
    use_tiebreak: bool = False


def build_reverse_with_mask(neighbors: torch.Tensor, n: int, k_rev: int):
    """Same construction as AriadSeekerSingle._build_reverse (real in-edges
    first, random-id padding when a row's in-degree < k_rev), but ALSO
    returns a [n, k_rev] bool mask: True where that slot is a genuine
    in-edge, False where it's random padding.

    Kept standalone (not a method on AriadSeekerSingle, which is reused
    UNMODIFIED from ariad_single_space) specifically so kR/qR attribution
    can separate "found via a real reverse edge" from "found via the random
    filler that's silently mixed into the buffer when in-degree is thin" --
    without this, kR/qR's apparent value could just be disguised random
    exploration.
    """
    k = neighbors.size(1)
    device = neighbors.device
    src = torch.arange(n, device=device, dtype=torch.long).unsqueeze(1).expand(n, k).reshape(-1)
    tgt = neighbors.reshape(-1)
    mask_t = (tgt >= 0) & (tgt < n)
    src = src[mask_t]
    tgt = tgt[mask_t]

    if tgt.numel() == 0:
        rand_part = torch.randint(0, n, (n, k_rev), device=device, dtype=torch.long)
        rows_idx = torch.arange(n, device=device, dtype=torch.long).unsqueeze(1)
        collision = rand_part == rows_idx
        if collision.any():
            rand_part = torch.where(collision, (rand_part + 1) % n, rand_part)
        is_real = torch.zeros(n, k_rev, device=device, dtype=torch.bool)
        return rand_part, is_real

    perm = torch.randperm(tgt.numel(), device=device)
    tgt = tgt[perm]
    src = src[perm]
    order = tgt.argsort()
    tgt_sorted = tgt[order]
    src_sorted = src[order]
    counts = torch.bincount(tgt_sorted, minlength=n)
    offsets = torch.cumsum(counts, dim=0) - counts
    slot_idx = torch.arange(k_rev, device=device, dtype=torch.long).unsqueeze(0)
    abs_pos = offsets.unsqueeze(1) + slot_idx
    abs_pos_clip = torch.clamp(abs_pos, max=max(0, src_sorted.numel() - 1))
    real_part = src_sorted[abs_pos_clip]
    need_rand = slot_idx >= counts.unsqueeze(1)

    rand_part = torch.randint(0, n, (n, k_rev), device=device, dtype=torch.long)
    rows_idx = torch.arange(n, device=device, dtype=torch.long).unsqueeze(1)
    collision = rand_part == rows_idx
    if collision.any():
        rand_part = torch.where(collision, (rand_part + 1) % n, rand_part)

    buffer = torch.where(need_rand, rand_part, real_part)
    is_real = ~need_rand
    return buffer, is_real


class AriadSeekerQK(nn.Module):
    """Maintains QK[i] = A(i), query i's current key set. Requires the
    CURRENT KK graph and QQ graph (each an AriadSeekerSingle's `.G`) to be
    passed in every refresh -- this module does not own or refresh them."""

    def __init__(self, cfg: AriadQKConfig):
        super().__init__()
        self.k_final = int(cfg.k_final)
        self.k_rev = max(1, int(cfg.k_rev))
        self.b_kF = max(0, int(cfg.b_kF))
        self.b_qF = max(0, int(cfg.b_qF))
        self.b_kR = max(0, int(cfg.b_kR))
        self.b_qR = max(0, int(cfg.b_qR))
        self.score_chunk = int(cfg.score_chunk)
        self.use_tiebreak = bool(cfg.use_tiebreak)

        self.initialized = False
        self.step_counter = 0
        self.QK = None   # LongTensor [n_q, k_final] -- A(i)
        self.last_refresh_stats = None

    def init_random(self, n_q: int, n_k: int, device):
        self.QK = torch.randint(0, n_k, (n_q, self.k_final), device=device, dtype=torch.long)
        self.initialized = True
        self.step_counter = 0

    @staticmethod
    def _score_gather_chunk(Q_src_chunk, K_tgt_all, cand_chunk):
        gathered = K_tgt_all[cand_chunk]              # [b, C, d]
        Qs_local = Q_src_chunk.unsqueeze(1)             # [b, 1, d]
        return (Qs_local * gathered).sum(-1)             # [b, C]

    def refresh(self, Q: torch.Tensor, K: torch.Tensor, kk_graph: torch.Tensor, qq_graph: torch.Tensor):
        """Q: [n_q,d] detached. K: [n_k,d] detached. kk_graph: KK's current
        [n_k,k_KK] (= O_KK). qq_graph: QQ's current [n_q,k_QQ] (= O_QQ)."""
        assert self.initialized
        n_q, k = self.QK.shape
        n_k = K.size(0)
        device = Q.device

        with torch.no_grad():
            kk_rev = kk_real = None
            if self.b_kR > 0:
                kk_rev, kk_real = build_reverse_with_mask(kk_graph, n_k, self.k_rev)
            qq_rev = qq_real = None
            if self.b_qR > 0:
                qq_rev, qq_real = build_reverse_with_mask(qq_graph, n_q, self.k_rev)

            new_QK = torch.empty(n_q, k, device=device, dtype=torch.long)
            row_idx_full = torch.arange(n_q, device=device)
            chunk = max(1, self.score_chunk)
            for s in range(0, n_q, chunk):
                e = min(s + chunk, n_q)
                A = self.QK[s:e, :]   # current keys -- unconditionally kept
                cand_parts = [A]

                if self.b_kF > 0:
                    cand_parts.append(AriadSeekerSingle._sample_forward_paths(A, kk_graph, self.b_kF))
                if self.b_qF > 0:
                    qF_anchors = qq_graph[s:e, :]
                    cand_parts.append(AriadSeekerSingle._sample_forward_paths(qF_anchors, self.QK, self.b_qF))
                if self.b_kR > 0:
                    cand_parts.append(AriadSeekerSingle._sample_forward_paths(A, kk_rev, self.b_kR))
                if self.b_qR > 0:
                    qR_anchors = qq_rev[s:e, :]
                    cand_parts.append(AriadSeekerSingle._sample_forward_paths(qR_anchors, self.QK, self.b_qR))

                cand_chunk = torch.cat(cand_parts, dim=-1)
                scores_chunk = self._score_gather_chunk(Q[s:e], K, cand_chunk)
                valid_chunk = torch.ones_like(cand_chunk, dtype=torch.bool)
                scores_chunk = AriadSeekerSingle._mask_row_duplicates(cand_chunk, scores_chunk, valid_chunk)
                # self-attention over the same node set: key_id == query_id is "self"
                scores_chunk = scores_chunk.masked_fill(
                    cand_chunk == row_idx_full[s:e].unsqueeze(1), float("-inf")
                )
                if self.use_tiebreak:
                    scores_chunk = scores_chunk.to(torch.float64) - cand_chunk.to(torch.float64) * 1e-12

                k_eff = min(k, cand_chunk.size(-1))
                topk = scores_chunk.topk(k_eff, dim=-1)
                new_chunk = cand_chunk.gather(-1, topk.indices)
                if k_eff < k:
                    pad = cand_chunk[:, : k - k_eff]
                    new_chunk = torch.cat([new_chunk, pad], dim=-1)
                new_QK[s:e, :] = new_chunk

            self.QK = new_QK

            stats = {"step": self.step_counter}
            if kk_real is not None:
                stats["kk_rev_real_frac"] = float(kk_real.float().mean().item())
            if qq_real is not None:
                stats["qq_rev_real_frac"] = float(qq_real.float().mean().item())
            self.last_refresh_stats = stats

        self.step_counter += 1

    def forward(self, Q: torch.Tensor, K: torch.Tensor, kk_graph: torch.Tensor, qq_graph: torch.Tensor):
        Q = Q.detach()
        K = K.detach()
        n_q = Q.size(0)
        n_k = K.size(0)
        if not self.initialized:
            self.init_random(n_q, n_k, Q.device)
            self.refresh(Q, K, kk_graph, qq_graph)   # score the random init before ever returning it
        elif self.training:
            self.refresh(Q, K, kk_graph, qq_graph)
        return self.QK
