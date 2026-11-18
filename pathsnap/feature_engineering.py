"""STEP 3 - FEATURE ENGINEERING: one feature row per (GPS point, candidate arc).

Only relative or metric features are used (no arc ids, no coordinates),
because every trip has its own road network in a different city.

| feature          | meaning                                                         |
|------------------|-----------------------------------------------------------------|
| dist             | metres from the GPS point to the arc                            |
| dist_rank        | 0 = nearest candidate, 1 = second nearest, ...                  |
| dist_minus_min   | extra metres compared with the nearest candidate                |
| dist_ratio       | dist / (nearest dist + 1)                                       |
| n_cand           | number of candidates of this point (road density)               |
| heading_diff     | angle 0-180 deg between the car's direction and the arc's       |
| speed            | m/s, from neighbouring GPS points                               |
| arc_len          | arc length in metres                                            |
| t_along          | where along the arc the closest point lies (0 start, 1 end)     |
| dist_prev/next   | distance of the previous / next GPS point to the same arc       |
| gap              | seconds since the previous GPS point                            |
"""
import numpy as np

from .geometry import point_segment, bearing_xy, angle_diff
from .preprocessing import project_trip, generate_candidates, label_candidates, RADIUS_M, K_MAX

FEATURES = ["dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand",
            "heading_diff", "speed", "arc_len", "t_along",
            "dist_prev", "dist_next", "gap"]


def gps_kinematics(P, t, smooth=2):
    """Speed (m/s) and heading (deg) of each GPS point, from points `smooth` steps before/after."""
    n = len(P)
    i0 = np.clip(np.arange(n) - smooth, 0, n - 1)
    i1 = np.clip(np.arange(n) + smooth, 0, n - 1)
    d = P[i1] - P[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    speed = np.sqrt((d ** 2).sum(1)) / dt
    heading = bearing_xy(d[:, 0], d[:, 1])
    step = np.r_[0.0, np.sqrt((np.diff(P, axis=0) ** 2).sum(1))]
    gap = np.r_[0.0, np.diff(t)]
    return speed, heading, step, gap


def build_features(geo, cand, track_t):
    """Add the FEATURES columns to the candidate table."""
    P, A, B = geo["P"], geo["A"], geo["B"]
    speed, heading, step, gap = gps_kinematics(P, track_t)
    c = cand.copy()
    i = c["point_idx"].values; a = c["arc_id"].values
    grp = c.groupby("point_idx")["dist"]
    c["dist_minus_min"] = c["dist"] - grp.transform("min")
    c["dist_ratio"] = c["dist"] / (grp.transform("min") + 1.0)
    c["n_cand"] = grp.transform("size")
    c["heading_diff"] = angle_diff(heading[i], geo["arc_bear"][a])
    c["speed"] = speed[i]
    c["arc_len"] = geo["arc_len"][a]
    n = len(P)
    ip = np.clip(i - 1, 0, n - 1); inx = np.clip(i + 1, 0, n - 1)
    c["dist_prev"], _ = point_segment(P[ip], A[a], B[a])
    c["dist_next"], _ = point_segment(P[inx], A[a], B[a])
    c["gap"] = gap[i]
    return c


def build_training_table(trip, radius=RADIUS_M, k_max=K_MAX):
    """Steps 2 + 3 for one trip: candidates -> features -> labels (if a route exists).
    Returns (table, info)."""
    geo = project_trip(trip)
    cand = generate_candidates(geo, radius=radius, k_max=k_max)
    feats = build_features(geo, cand, trip["track"]["t"].values)
    feats["trip_id"] = trip["trip_id"]
    info = {"geo": geo, "n_points": len(geo["P"])}
    if trip.get("route") is not None:
        feats, lab = label_candidates(geo, feats, trip["route"], radius=radius)
        info.update(lab)
    return feats, info
