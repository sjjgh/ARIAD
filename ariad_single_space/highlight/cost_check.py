import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compare_dense_exact_ariad import est_flops   # the real, already-used cost model
sys.path.insert(0, str(Path(__file__).resolve().parent))
import highlight_kNN_convergence as m

N = m.N_CLUSTERS * m.CLUSTER_SIZE
d_in = m.D_G + m.D_V
d_z = m.D_Z
d_v = m.D_V
k = m.K
# ARIAD's own candidate pool per row per refresh (see AriadSeekerSingle.refresh):
# O(i) kept unconditionally (k) + c_o2 (2-hop forward) + c_i1 (reverse buffer), c_oi=c_rand=0
c_scored = m.ARIAD_CFG.k + m.ARIAD_CFG.c_o2 + m.ARIAD_CFG.c_i1

f_exact = est_flops(N, d_in, d_z, d_v, p_attn=N * k, p_search=N * N)
f_ariad = est_flops(N, d_in, d_z, d_v, p_attn=N * k, p_search=N * c_scored)
f_dense = est_flops(N, d_in, d_z, d_v, p_attn=N * N, p_search=0)

print(f"N={N}, d_in={d_in}, d_z={d_z}, d_v={d_v}, k={k}, ARIAD C_scored={c_scored}")
print(f"Dense      : {f_dense/1e9:.3f} GFLOPs/step")
print(f"Exact top-k: {f_exact/1e9:.3f} GFLOPs/step  (search: N^2 = {N*N:,} scores/row-total)")
print(f"ARIAD      : {f_ariad/1e9:.3f} GFLOPs/step  (search: N*C = {N*c_scored:,} scores/row-total)")
print()
print(f"ARIAD / Exact total FLOPs   = {100*f_ariad/f_exact:.2f}%  -> {100*(1-f_ariad/f_exact):.2f}% fewer total FLOPs")
print(f"ARIAD / Exact search scores = {100*(N*c_scored)/(N*N):.2f}%  -> {100*(1-(N*c_scored)/(N*N)):.2f}% fewer search-score evals")
