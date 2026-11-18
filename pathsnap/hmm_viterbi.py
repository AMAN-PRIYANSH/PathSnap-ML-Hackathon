"""STEP 6 - HMM + VITERBI route reconstruction.

Hidden states   : at GPS point i, the candidate arcs of that point.
Emission        : log of the classifier's probability that the candidate is
                  the true arc (re-normalised over the point's candidates),
                  raised to a weight `w_emit`.
Transition a->b : how plausible it is to drive from candidate a (at point i-1)
                  to candidate b (at point i). We compare the driving distance
                  along the network with the straight-line distance between the
                  two GPS points (Newson & Krumm style):
                      log p = -|d_network - d_straight| / beta
                  If b cannot be reached from a within the search limit, the
                  transition gets a large fixed penalty instead of -inf, so the
                  decoder can still continue (handles disconnected segments).
GPS gaps        : beta and the search limit grow with the time gap, because a
                  long gap allows more driving.
Missing cands   : points with no candidate are skipped by the decoder and later
                  given the arc decoded for the nearest previous point.
"""
import heapq
import numpy as np
from collections import defaultdict


class RoadGraph:
    def __init__(self, arcs_uv, arc_len):
        self.uv = arcs_uv
        self.len = arc_len
        self.out = defaultdict(list)          # node -> [(next_node, length)]
        for a, (u, v) in enumerate(arcs_uv):
            self.out[u].append((v, arc_len[a]))
        self._cache = {}                      # node -> (limit, {node: dist})

    def dists_from(self, src, limit):
        """Shortest driving distance from node `src` to every node within `limit` m."""
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, n = heapq.heappop(pq)
            if d > dist.get(n, np.inf) or d > limit:
                continue
            for m, w in self.out[n]:
                nd = d + w
                if nd <= limit and nd < dist.get(m, np.inf):
                    dist[m] = nd
                    heapq.heappush(pq, (nd, m))
        self._cache[src] = (limit, dist)
        return dist

    def network_distance(self, a, ta, b, tb, limit):
        """Driving distance from position ta on arc a to position tb on arc b."""
        La, Lb = self.len[a], self.len[b]
        if a == b:
            fwd = (tb - ta) * La
            # small backward moves are GPS noise (e.g. a stopped car); larger ones mean the
            # car is driving against this arc's direction, which is impossible.
            # (A 15 m slack let the opposite-direction arc win at ~10 m/s, so we keep it at 5 m.)
            return abs(fwd) if fwd >= -5.0 else np.inf
        start = (1 - ta) * La                           # rest of arc a
        d = self.dists_from(self.uv[a, 1], limit).get(self.uv[b, 0], np.inf)
        return start + d + tb * Lb


def viterbi_trip(cand, P, t, graph, beta=30.0, w_emit=1.0, unreachable=-30.0,
                 base_limit=300.0, score_col="score"):
    """Decode one trip.

    cand : DataFrame with point_idx, arc_id, t_along, score (classifier probability)
    P    : GPS points in local metres, t: timestamps
    Returns array (len(P),) of decoded arc ids (-1 if nothing could be decoded).
    """
    n = len(P)
    groups = {i: g for i, g in cand.groupby("point_idx")}
    pts = sorted(groups)
    if not pts:
        return np.full(n, -1)
    states, emis = [], []
    for i in pts:
        g = groups[i]
        s = g[score_col].values.astype(float)
        s = np.clip(s / max(s.sum(), 1e-12), 1e-9, 1.0)
        states.append((g["arc_id"].values, g["t_along"].values))
        emis.append(w_emit * np.log(s))

    delta = emis[0].copy()
    back = [None]
    for k in range(1, len(pts)):
        i0, i1 = pts[k - 1], pts[k]
        straight = float(np.hypot(*(P[i1] - P[i0])))
        dt = max(float(t[i1] - t[i0]), 1.0)
        b_k = beta * np.sqrt(dt)                       # more slack after long gaps
        limit = max(base_limit, 3 * straight + 100, 40.0 * dt)   # 40 m/s = 144 km/h
        A0, T0 = states[k - 1]; A1, T1 = states[k]
        trans = np.empty((len(A0), len(A1)))
        for x in range(len(A0)):
            for y in range(len(A1)):
                d = graph.network_distance(A0[x], T0[x], A1[y], T1[y], limit)
                trans[x, y] = unreachable if not np.isfinite(d) else -abs(d - straight) / b_k
        tot = delta[:, None] + trans
        bp = tot.argmax(0)
        delta = tot[bp, np.arange(len(A1))] + emis[k]
        back.append(bp)

    out = np.full(n, -1, dtype=np.int64)
    j = int(delta.argmax())
    for k in range(len(pts) - 1, -1, -1):
        out[pts[k]] = states[k][0][j]
        if k > 0:
            j = int(back[k][j])
    # fill points without candidates with the last decoded arc
    last = -1
    for i in range(n):
        if out[i] == -1:
            out[i] = last
        else:
            last = out[i]
    return out
