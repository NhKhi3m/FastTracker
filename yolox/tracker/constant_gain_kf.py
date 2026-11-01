"""Steady-State (Constant-Gain) Kalman Filter for bounding-box tracking.

Drop-in replacement for ``BboxKalmanFilter`` in ``fast_tracker.py``
(same state layout [cx, cy, w, h, vx, vy, vw, vh] stored as an (8, 1) array,
same ``predict()`` / ``update(bbox)`` interface), but with NO covariance P,
NO innovation covariance S and NO matrix inversion at runtime.

Why this is exact enough for our model
--------------------------------------
With F = constant velocity, H = [I4 0], diagonal Q and isotropic R, the 8-D
filter decouples into 4 identical, independent 2-state (position, velocity)
filters. The Kalman gain therefore has the structure

    K = [[kp * I4],
         [kv * I4]]

so an update is just two scalar-vector ops:

    y      = z - x[:4]
    x[:4] += kp * y
    x[4:] += kv * y

The Riccati recursion for P does not depend on the measured data, only on
(Q, R, P0) and the number of steps, so the gains can be precomputed offline:

* Transient schedule: gains for the first N consecutive updates of a new
  track, starting from the same P0 as the standard filter. This reproduces the
  fast "snap-to-measurement" behaviour of a freshly spawned track.
* Steady state: the gain the recursion converges to (solution of the DARE).

Height buckets
--------------
Measurement noise is modelled as  R(h) = sigma0^2 + (alpha * h)^2  (constant
pixel quantisation floor + error proportional to box height). Q is absolute.
Because the ratio Q/R changes with h, each height bucket gets its own gain
table (K_small ... K_xlarge), as recommended in
``Steady_State_Constant_Gain_Kalman_Filter_Notes.md``.
Note: if Q and R were BOTH scaled by h^2 the steady-state gain would be
height-invariant and buckets would be pointless.
"""

import numpy as np

# ---- Noise model (matches the standard BboxKalmanFilter defaults) ----------
P0_POS = 10.0          # initial position variance  (P = 10 * I, vel * 100)
P0_VEL = 1000.0        # initial velocity variance
Q_POS = 0.1            # process noise variance, position
Q_VEL = 0.01           # process noise variance, velocity
SIGMA0 = 1.0           # measurement noise floor (std, px) -> R = 1.0 at h = 0
ALPHA = 0.01           # measurement noise proportional to box height

# Bucket edges are in tracker-coordinate pixels (the frame fed to the tracker).
# Defaults suit 224-320 px inputs; scale them for full-resolution frames.
HEIGHT_EDGES = (32.0, 64.0, 128.0)              # -> 4 buckets
BUCKET_REP_HEIGHTS = (24.0, 48.0, 96.0, 160.0)  # representative h per bucket
BUCKET_NAMES = ("small", "medium", "large", "xlarge")

SCHEDULE_LEN = 15      # transient gains for the first N updates, then steady


def _scalar_gain_schedule(r, n_steps=SCHEDULE_LEN, iters=2000, tol=1e-12):
    """Run the 2-state Riccati recursion (predict + update every frame).

    Returns (schedule[n_steps, 2], steady[2]) with rows = (kp, kv).
    """
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([Q_POS, Q_VEL])
    P = np.diag([P0_POS, P0_VEL])

    sched = []
    k_prev = None
    k = None
    for i in range(iters):
        P = F @ P @ F.T + Q                       # predict
        S = (H @ P @ H.T).item() + r              # scalar innovation variance
        k = (P @ H.T).ravel() / S                 # (kp, kv)
        P = (np.eye(2) - np.outer(k, H)) @ P      # update
        P = 0.5 * (P + P.T)
        if i < n_steps:
            sched.append(k.copy())
        if k_prev is not None and i >= n_steps and np.max(np.abs(k - k_prev)) < tol:
            break
        k_prev = k.copy()
    return np.array(sched), k.copy()


def _build_tables():
    tables = []
    for h in BUCKET_REP_HEIGHTS:
        r = SIGMA0 ** 2 + (ALPHA * h) ** 2
        sched, steady = _scalar_gain_schedule(r)
        tables.append({"R": r, "schedule": sched, "steady": steady})
    return tables


# Precomputed once at import (a few ms, data-independent).
GAIN_TABLES = _build_tables()


def bucket_index(h):
    if h < HEIGHT_EDGES[0]:
        return 0
    if h < HEIGHT_EDGES[1]:
        return 1
    if h < HEIGHT_EDGES[2]:
        return 2
    return 3


def describe_gains():
    """Human-readable dump of the precomputed gains."""
    lines = []
    for name, h, t in zip(BUCKET_NAMES, BUCKET_REP_HEIGHTS, GAIN_TABLES):
        kp, kv = t["steady"]
        k1p, k1v = t["schedule"][0]
        lines.append(
            f"K_{name:<6s} (h~{h:5.0f}px, R={t['R']:.3f}): "
            f"steady kp={kp:.4f} kv={kv:.4f} | first-update kp={k1p:.4f} kv={k1v:.4f}"
        )
    return "\n".join(lines)


class ConstantGainKalmanFilter:
    """Constant-gain KF with the same interface as ``BboxKalmanFilter``."""

    __slots__ = ("x", "n_updates")

    def __init__(self, bbox):
        self.x = np.zeros((8, 1), dtype=np.float64)
        self.x[:4, 0] = bbox
        self.n_updates = 0

    def predict(self):
        # Sparse F: position += velocity (4 additions instead of 8x8 matmul + P).
        self.x[:4] += self.x[4:]
        return self.x[:4, 0]

    def update(self, bbox):
        table = GAIN_TABLES[bucket_index(self.x[3, 0])]
        if self.n_updates < SCHEDULE_LEN:
            kp, kv = table["schedule"][self.n_updates]
        else:
            kp, kv = table["steady"]
        self.n_updates += 1

        y = np.asarray(bbox, dtype=np.float64) - self.x[:4, 0]
        self.x[:4, 0] += kp * y
        self.x[4:, 0] += kv * y


if __name__ == "__main__":
    print(describe_gains())
