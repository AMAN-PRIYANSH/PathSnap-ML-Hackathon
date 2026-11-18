"""STEP 8 - PathSnap API (runs on Laptop 1).

Start:   deployment\\laptop1_START_API.bat
         (= py -m uvicorn api.app:app --host 0.0.0.0 --port 8000)
Env:     PATHSNAP_DATA       folder holding the partitions (train/, student_test/, ...)
         PATHSNAP_PARTITION  partition whose trips the UI may choose (default student_test;
                             on evaluation day: the evaluator partition folder)
         PATHSNAP_MODEL      trained classifier (default models/final_model.joblib)

Endpoints
  GET  /health   -> model and partition in use
  GET  /trips    -> the trip ids the UI can offer
  POST /predict  {"trip_id": "..."} -> predicted route + scores
Every request is logged in this laptop's command window.
"""
import os
import sys
import json
import time
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.data_loading import list_trips, load_trip, DATA_ROOT
from pathsnap.pipeline import predict_trip, score_trip

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("pathsnap")

PARTITION = os.environ.get("PATHSNAP_PARTITION", "student_test")
MODEL_PATH = os.environ.get("PATHSNAP_MODEL", os.path.join(ROOT, "models", "final_model.joblib"))
HMM_PATH = os.path.join(ROOT, "models", "hmm_config.json")

# ---- model-loading logic: classifier + HMM parameters, loaded once at start-up
model = joblib.load(MODEL_PATH)
hmm_params = json.load(open(HMM_PATH)) if os.path.exists(HMM_PATH) else {}
log.info(f"Loaded model {os.path.basename(MODEL_PATH)} | HMM params {hmm_params}")
log.info(f"Data folder {DATA_ROOT} | partition '{PARTITION}' | {len(list_trips(PARTITION))} trips")

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH), "partition": PARTITION}


@app.get("/trips")
def trips():
    ids = list_trips(PARTITION)
    log.info(f"GET /trips -> {len(ids)} trips")
    return {"partition": PARTITION, "trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in list_trips(PARTITION):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.time()
    log.info(f"POST /predict trip={req.trip_id} -> running pipeline")
    trip = load_trip(PARTITION, req.trip_id, with_route=False)   # prediction never sees the route
    pred = predict_trip(trip, model, hmm_params)
    scores = score_trip(PARTITION, req.trip_id, pred)            # server-side only, after prediction
    nodes = trip["nodes"].values; arcs = trip["arcs"].values
    h = pred["hmm_arcs"]
    route_arcs = [int(a) for i, a in enumerate(h) if a >= 0 and (i == 0 or a != h[i - 1])]
    segments = [[[float(nodes[arcs[a, 0], 1]), float(nodes[arcs[a, 0], 0])],
                 [float(nodes[arcs[a, 1], 1]), float(nodes[arcs[a, 1], 0])]] for a in route_arcs]
    tr = trip["track"]
    elapsed = time.time() - t0
    log.info(f"trip={req.trip_id} points={scores['n_points']} candidates={pred['n_candidates']} "
             f"top1 classifier={_fmt(scores.get('clf_top1_accuracy'))} hmm={_fmt(scores.get('hmm_top1_accuracy'))} "
             f"route_f1 hmm={_fmt(scores.get('hmm_route_f1'))} continuity hmm={_fmt(scores.get('hmm_route_continuity'))} "
             f"time={elapsed:.2f}s")
    return {
        "trip_id": req.trip_id,
        "gps": [[float(a), float(b)] for a, b in zip(tr["lat"].values, tr["lon"].values)],
        "predicted_route": segments,
        "predicted_arc_per_point": [int(a) for a in h],
        "metrics": scores,
        "timing_s": {**pred["timing_s"], "total": elapsed},
    }
