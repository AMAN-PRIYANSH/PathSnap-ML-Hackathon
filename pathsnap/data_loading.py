"""STEP 1 - DATA LOADING: read the raw PathSnap files.

Each trip folder holds four header-less, whitespace-separated text files:
  .track  -> lon, lat, t (seconds)          one row per GPS observation
  .nodes  -> lon, lat                       row index = node_id (0-based)
  .arcs   -> from_node_id, to_node_id       row index = arc_id  (0-based)
  .route  -> arc_id                         reference route, in travel order

Where is the data?  The first folder that exists is used:
  1. the environment variable PATHSNAP_DATA
  2. <repo>/data
  3. <repo>/../StudentFolder/Student folder
  4. <repo>/../ML_Hackathon/StudentFolder/Student folder
Each of these must contain the partition folders (train/, student_test/, ...).
"""
import os
import glob
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_data_root():
    candidates = [
        os.environ.get("PATHSNAP_DATA", ""),
        os.path.join(REPO_ROOT, "data"),
        os.path.join(REPO_ROOT, "..", "StudentFolder", "Student folder"),
        os.path.join(REPO_ROOT, "..", "ML_Hackathon", "StudentFolder", "Student folder"),
    ]
    if candidates[0] and os.path.isdir(candidates[0]):   # an explicit setting always wins
        return os.path.abspath(candidates[0])
    for c in candidates[1:]:
        if os.path.isdir(os.path.join(c, "train")):
            return os.path.abspath(c)
    return os.path.abspath(candidates[1])


DATA_ROOT = _find_data_root()


def list_trips(partition):
    """Trip ids (folder names) inside one partition, e.g. 'train' or 'student_test'."""
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_ROOT, partition, "*"))
                  if os.path.isdir(p))


def load_trip(partition, trip_id, with_route=True):
    """Load one trip. with_route=False is used by the prediction path, so the
    reference route can never leak into a prediction."""
    folder = os.path.join(DATA_ROOT, partition, trip_id)
    f = lambda ext: os.path.join(folder, f"{trip_id}.{ext}")
    track = pd.read_csv(f("track"), sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(f("nodes"), sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(f("arcs"), sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64)
    route = None
    if with_route and os.path.exists(f("route")):
        route = pd.read_csv(f("route"), header=None, names=["arc_id"], dtype=np.int64)["arc_id"].to_numpy()
    return {"trip_id": trip_id, "partition": partition, "track": track,
            "nodes": nodes, "arcs": arcs, "route": route}


def has_route(partition, trip_id):
    return os.path.exists(os.path.join(DATA_ROOT, partition, trip_id, f"{trip_id}.route"))
