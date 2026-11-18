"""STEP 2 - PREPROCESSING: projection, candidate generation and per-point labels.

For one trip:
  2.1 project_trip          - GPS points and road nodes from lon/lat to local metres
  2.2 generate_candidates   - for every GPS point, the road segments (arcs) within
                              `radius` metres, keeping the `k_max` nearest
  2.3 align_route_to_points - the reference route is a list of arcs, NOT one arc per
                              GPS point, so each point is matched to one route arc
                              with a monotonic dynamic programme
  2.4 label_candidates      - target = 1 if a candidate is the point's reference arc;
                              points > radius m from every route arc are 'invalid'
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometry import LocalProjection, point_segment, bearing_xy

RADIUS_M = 50.0   # chosen from the radius sweep in notebook 02
K_MAX = 10


# ------------------------------------------------------------- 2.1 projection
def project_trip(trip):
    """Convert one trip to metres, centred on the mean GPS position."""
    tr = trip["track"]
    proj = LocalProjection(tr["lon"].mean(), tr["lat"].mean())
    P = proj.to_xy(tr["lon"].values, tr["lat"].values)          # GPS points (n, 2)
    N = proj.to_xy(trip["nodes"]["lon"].values, trip["nodes"]["lat"].values)
    u = trip["arcs"]["u"].values; v = trip["arcs"]["v"].values
    A, B = N[u], N[v]                                            # start / end of every arc
    arc_len = np.sqrt(((B - A) ** 2).sum(1))
    arc_bear = bearing_xy(B[:, 0] - A[:, 0], B[:, 1] - A[:, 1])  # direction of travel
    return {"proj": proj, "P": P, "N": N, "A": A, "B": B,
            "arc_len": arc_len, "arc_bear": arc_bear}


# ---------------------------------------------------- 2.2 candidate generation
def generate_candidates(geo, radius=RADIUS_M, k_max=K_MAX, densify=10.0):
    """One row per (point_idx, arc_id) candidate, with distance and position on the arc."""
    P, A, B, L = geo["P"], geo["A"], geo["B"], geo["arc_len"]
    # keep only arcs near the trip's bounding box (speed)
    lo = P.min(0) - radius - 50; hi = P.max(0) + radius + 50
    near = ((np.minimum(A, B) <= hi).all(1) & (np.maximum(A, B) >= lo).all(1))
    arc_ids = np.where(near)[0]
    # sample each arc every `densify` metres and index the samples in a KD-tree
    n_samp = np.maximum(2, np.ceil(L[arc_ids] / densify).astype(int) + 1)
    rep = np.repeat(arc_ids, n_samp)
    frac = np.concatenate([np.linspace(0, 1, k) for k in n_samp])
    S = A[rep] + (B[rep] - A[rep]) * frac[:, None]
    tree = cKDTree(S)
    # a point within `radius` of an arc has a sample within radius + densify/2
    hits = tree.query_ball_point(P, r=radius + densify / 2)
    pi, ai = [], []
    for i, h in enumerate(hits):
        if h:
            a = np.unique(rep[h]); pi.append(np.full(len(a), i)); ai.append(a)
    if not pi:
        return pd.DataFrame(columns=["point_idx", "arc_id", "dist", "t_along", "dist_rank"])
    pi = np.concatenate(pi); ai = np.concatenate(ai)
    d, tt = point_segment(P[pi], A[ai], B[ai])                    # exact distance to the arc
    c = pd.DataFrame({"point_idx": pi, "arc_id": ai, "dist": d, "t_along": tt})
    c = c[c["dist"] <= radius].sort_values(["point_idx", "dist"])
    c["dist_rank"] = c.groupby("point_idx").cumcount()
    return c[c["dist_rank"] < k_max].reset_index(drop=True)


# ------------------------------------------------- 2.3 label alignment (DP)
def align_route_to_points(geo, route):
    """Assign each GPS point to one arc of the reference route.

    Monotonic DP: point i -> route position j(i) with j never decreasing,
    minimising the total point-to-arc distance.  O(n_points * n_route).
    Returns (reference arc per point, distance of the point to that arc).
    """
    P, A, B = geo["P"], geo["A"], geo["B"]
    n, m = len(P), len(route)
    Ar, Br = A[route], B[route]
    D = np.empty((n, m))
    for j in range(m):
        D[:, j], _ = point_segment(P, np.repeat(Ar[j:j + 1], n, 0), np.repeat(Br[j:j + 1], n, 0))
    cost = D[0].copy()
    back = np.zeros((n, m), dtype=np.int32)
    for i in range(1, n):
        pm = np.minimum.accumulate(cost)                          # best cost at any k <= j
        pm_idx = np.maximum.accumulate(np.where(cost <= pm, np.arange(m), 0)).astype(np.int32)
        cost = D[i] + pm
        back[i] = pm_idx
    j = int(np.argmin(cost))
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):                                # trace back
        path[i] = j
        j = back[i, j]
    return route[path], D[np.arange(n), path]


# ---------------------------------------------------------- 2.4 labelling
def label_candidates(geo, cand, route, radius=RADIUS_M):
    """Add `label` and `valid` columns. Returns (labelled candidates, info dict)."""
    gt, gt_dist = align_route_to_points(geo, route)
    valid = gt_dist <= radius     # the reference does not cover points further away
    c = cand.copy()
    c["label"] = (c["arc_id"].values == gt[c["point_idx"].values]).astype(int)
    c["valid"] = valid[c["point_idx"].values].astype(int)
    return c, {"gt_arc": gt, "gt_dist": gt_dist, "valid": valid}
