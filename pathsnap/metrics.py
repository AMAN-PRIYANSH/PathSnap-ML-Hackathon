"""STEP 4 - EVALUATION METRICS, shared by every model and by the HMM decoder.

Point / candidate level
  candidate_metrics   macro F1, precision, recall, log loss of the 0/1 candidate labels
  observation_top1    share of valid GPS points whose highest-scoring candidate is the
                      reference arc (points with no candidate count as errors)
Sequence / route level
  route_continuity    share of consecutive points whose predicted arcs are the same arc
                      or directly connected in the road network
  route_overlap       length-weighted precision / recall / F1 between the set of arcs the
                      prediction drives on and the reference route
"""
import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {
        "macro_f1": f1_score(y, yhat, average="macro"),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def top1_per_point(df, score_col="score"):
    """For each (trip, point) keep the candidate with the highest score."""
    idx = df.groupby(["trip_id", "point_idx"])[score_col].idxmax()
    return df.loc[idx, ["trip_id", "point_idx", "arc_id", "label"]]


def observation_top1(df, n_valid_points, score_col="score"):
    top = top1_per_point(df, score_col)
    return top["label"].sum() / n_valid_points


def route_continuity(pred_arcs, arcs_uv):
    a = np.asarray(pred_arcs)
    if len(a) < 2:
        return np.nan
    a1, a2 = a[:-1], a[1:]
    ok = (a1 == a2) | (arcs_uv[a1, 1] == arcs_uv[a2, 0])
    return float(ok.mean())


def route_overlap(pred_arcs, ref_route, arc_len):
    """Sequence-level score: how much of the driven route is right, by length.

    precision = length of predicted arcs that are on the reference route / predicted length
    recall    = length of reference arcs that were predicted            / reference length
    """
    pred = set(int(a) for a in np.asarray(pred_arcs) if a >= 0)
    ref = set(int(a) for a in ref_route)
    if not pred or not ref:
        return {"route_precision": np.nan, "route_recall": np.nan, "route_f1": np.nan}
    common = pred & ref
    L = lambda s: float(sum(arc_len[a] for a in s))
    p = L(common) / L(pred) if L(pred) > 0 else 0.0
    r = L(common) / L(ref) if L(ref) > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"route_precision": p, "route_recall": r, "route_f1": f}
