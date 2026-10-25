"""比分模型事件機率的校準：逐季留一驗證（淨勝 ≥k、總進球 ≥k 兩類，各自 logit 截距+斜率）。"""
import sys, math
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import data_football, analyze, calibration as cal
from strategies import lam_from_over
RHO = -0.08
fb = data_football.load_history().dropna(subset=["c_avg_h", "c_avg_d", "c_avg_a", "c_avg_over", "c_avg_under"]).reset_index(drop=True)
E = {"m": [], "t": []}
for r in fb.itertuples():
    pc = cal.power_devig([r.c_avg_h, r.c_avg_d, r.c_avg_a]); po = cal.power_devig([r.c_avg_over, r.c_avg_under])
    lh, la = analyze.team_lambdas(pc, lam_from_over(po[0], 2.5))
    m = np.array(analyze._score_matrix(lh, la))
    m[0, 0] *= 1 - lh * la * RHO; m[0, 1] *= 1 + lh * RHO; m[1, 0] *= 1 + la * RHO; m[1, 1] *= 1 - RHO; m /= m.sum()
    i, j = np.indices(m.shape); d, t = r.hg - r.ag, r.hg + r.ag
    for k in (1, 2, 3):
        E["m"].append((r.season, m[(i - j) >= k].sum(), d >= k)); E["m"].append((r.season, m[(j - i) >= k].sum(), -d >= k))
    for k in (2, 3, 4, 5):
        E["t"].append((r.season, m[(i + j) >= k].sum(), t >= k))
for fam, ev in E.items():
    s = np.array([e[0] for e in ev]); p = np.clip(np.array([e[1] for e in ev]), 1e-4, 1 - 1e-4); y = np.array([e[2] for e in ev], float)
    lp = np.log(p / (1 - p)); out = []
    for S in sorted(set(s)):
        te = s == S
        w = cal._fit(lp[~te][:, None], y[~te])
        q = 1 / (1 + np.exp(-(w[0] + w[1] * lp[te])))
        ll = lambda x: -(y[te] * np.log(x) + (1 - y[te]) * np.log(1 - x))
        d = ll(q) - ll(p[te])
        out.append(f"{S} {d.mean()*1000:+.2f}‰ t={d.mean()/(d.std()/math.sqrt(len(d))):+.1f}")
    w = cal._fit(lp[:, None], y)
    print(f"{'淨勝' if fam=='m' else '總進球'}：係數 a={w[0]:+.3f} b={w[1]:.3f}｜" + "　".join(out))
for rho in (0.0, -0.05, -0.08, -0.12):
    pass  # rho 的選擇在本機 3 季看過：-0.05 ~ -0.10 比分 log loss 最低
