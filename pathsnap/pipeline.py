"""STEP 7 - END-TO-END PREDICTION for one trip (used by the API and notebook 05).

predict_trip : raw trip (loaded WITHOUT its route) -> candidates -> features ->
               classifier probabilities -> classifier-only arcs + HMM/Viterbi arcs
score_trip   : afterwards, on the server only, compares the prediction with the
               reference route. If a trip has no route file, scores are None.
"""
import time
import numpy as np

from .data_loading import load_trip, has_route
from .preprocessing import project_trip, generate_candidates, align_route_to_points, RADIUS_M, K_MAX
from .feature_engineering import build_features, FEATURES
from .hmm_viterbi import RoadGraph, viterbi_trip
from .metrics import route_continuity, route_overlap


def predict_trip(trip, model, hmm_params, radius=RADIUS_M, k_max=K_MAX):
    t0 = time.time()
    geo = project_trip(trip)
    cand = generate_candidates(geo, radius=radius, k_max=k_max)
    feats = build_features(geo, cand, trip["track"]["t"].values)
    feats["score"] = model.predict_proba(feats[FEATURES].values)[:, 1] if len(feats) else []
    n = len(geo["P"])
    clf = np.full(n, -1, dtype=np.int64)          # classifier only: best candidate per point
    if len(feats):
        top = feats.loc[feats.groupby("point_idx")["score"].idxmax()]
        clf[top["point_idx"].values] = top["arc_id"].values
    t1 = time.time()
    graph = RoadGraph(trip["arcs"].values, geo["arc_len"])
    hmm = viterbi_trip(feats, geo["P"], trip["track"]["t"].values, graph, **hmm_params)
    t2 = time.time()
    return {"geo": geo, "clf_arcs": clf, "hmm_arcs": hmm, "n_candidates": int(len(feats)),
            "timing_s": {"features_and_classifier": t1 - t0, "hmm_viterbi": t2 - t1}}


def score_trip(partition, trip_id, pred, radius=RADIUS_M):
    """Server-side scoring against the reference route (never sent to the model or UI)."""
    n = len(pred["geo"]["P"])
    if not has_route(partition, trip_id):
        return {"n_points": n, "n_valid_points": None, "reference_available": False}
    ref = load_trip(partition, trip_id, with_route=True)
    gt, gt_dist = align_route_to_points(pred["geo"], ref["route"])
    valid = gt_dist <= radius
    arcs = ref["arcs"].values
    out = {"n_points": int(n), "n_valid_points": int(valid.sum()), "reference_available": True}
    for name in ["clf", "hmm"]:
        a = pred[f"{name}_arcs"]
        out[f"{name}_top1_accuracy"] = float((a[valid] == gt[valid]).mean()) if valid.any() else None
        out[f"{name}_route_continuity"] = route_continuity(a[a >= 0], arcs)
        for k, v in route_overlap(a, ref["route"], pred["geo"]["arc_len"]).items():
            out[f"{name}_{k}"] = v
    # JSON cannot carry NaN -> use None (shown as n/a in the UI)
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in out.items()}
