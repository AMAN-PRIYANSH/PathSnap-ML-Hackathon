"""STEP 2a - GEOMETRY: distances in metres.

Longitude/latitude are angles, so we never measure distance in degrees.
Each trip covers at most a few tens of km, so we use a local
equirectangular projection centred on the trip:
    x = R * cos(lat0) * (lon - lon0)      (metres east)
    y = R * (lat - lat0)                  (metres north)
At this scale the error versus the true geodesic distance is well under 0.5 %,
and it lets us use fast Euclidean geometry (KD-trees, point-to-segment distance).
"""
import numpy as np

R_EARTH = 6_371_000.0


class LocalProjection:
    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = R_EARTH * np.cos(np.radians(self.lat0)) * np.pi / 180.0
        self.ky = R_EARTH * np.pi / 180.0

    def to_xy(self, lon, lat):
        lon = np.asarray(lon, float); lat = np.asarray(lat, float)
        return np.column_stack([(lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky])


def haversine(lon1, lat1, lon2, lat2):
    """True great-circle distance in metres (used to check the projection)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def point_segment(p, a, b):
    """Distance from points p to segments a->b (all arrays of shape (n, 2)).

    Returns (distance, t) where t in [0, 1] is how far along the segment the
    closest point lies (0 = start node, 1 = end node).
    """
    ab = b - a
    L2 = (ab ** 2).sum(1)
    t = np.where(L2 > 0, ((p - a) * ab).sum(1) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0, 1)
    proj = a + ab * t[:, None]
    return np.sqrt(((p - proj) ** 2).sum(1)), t


def bearing_xy(dx, dy):
    """Compass bearing in degrees (0 = north, 90 = east) from a displacement."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
