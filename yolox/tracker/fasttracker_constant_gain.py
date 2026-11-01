"""FastTracker implementation for Automated Camera & Multi-Object Tracking.
Paper: "FastTracker: Real-Time and Accurate Visual Tracking" (arXiv:2508.14370)
GitHub Reference: https://github.com/Hamidreza-Hashempoor/FastTracker

Key Architectural Innovations over ByteTrack:
1. Occlusion-Aware Tracking:
   - Center-proximity & IoU overlap detection between tracklets (is_occluded_by).
   - Pre-occlusion velocity and position restoration to eliminate Kalman filter divergence.
   - Bounding box search area enlargement (enlarge_bbox_occ = 1.1x) to facilitate re-acquisition.
   - Motion dampening (dampen_motion_occ = 0.89) to prevent tracklet drift while obscured.
2. Duplicate Suppression via IoU-Suppressed Track Initialization (init_iou_suppress):
   - Prevents duplicate/ghost tracks when detectors produce overlapping multi-scale bounding boxes.
3. Extended Occlusion Memory Grace Period:
   - Retains tracks during prolonged podium, pillar, or presenter occlusions.
"""

import numpy as np
from collections import deque

from constant_gain_kf import ConstantGainKalmanFilter


class BboxKalmanFilter:
    """Constant velocity Kalman Filter for bounding box [cx, cy, w, h, vx, vy, vw, vh]."""

    def __init__(self, bbox):
        self.x = np.zeros((8, 1), dtype=np.float64)
        self.x[:4, 0] = bbox

        self.P = np.eye(8, dtype=np.float64)
        self.P[4:, 4:] *= 100.0
        self.P *= 10.0

        self.Q = np.eye(8, dtype=np.float64)
        self.Q[:4, :4] *= 0.1
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(4, dtype=np.float64) * 1.0

        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            self.H[i, i] = 1.0

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:4, 0]

    def update(self, bbox):
        z = np.expand_dims(bbox, axis=1)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(8) - np.dot(K, self.H)), self.P)


def get_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-7

    return inter / union


def get_iou_matrix_vec(boxes_a, boxes_b):
    """Vectorized IoU matrix between boxes_a (N, 4) and boxes_b (M, 4)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    a = np.asarray(boxes_a, dtype=np.float64)[:, None, :]  # (N, 1, 4)
    b = np.asarray(boxes_b, dtype=np.float64)[None, :, :]  # (1, M, 4)
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[..., 2] - a[..., 0]) * np.maximum(0.0, a[..., 3] - a[..., 1])
    area_b = np.maximum(0.0, b[..., 2] - b[..., 0]) * np.maximum(0.0, b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-7)


def is_occluded_by(box_a, box_b, iou_thresh=0.5):
    """Returns True if box_a is substantially occluded/overlapped by box_b."""
    inter = (
        max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) *
        max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    )
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    if area_a <= 0:
        return False
    return (inter / area_a) > iou_thresh


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1)])


def xywh2xyxy(bbox):
    cx, cy, w, h = bbox
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


class FastTrack:
    """Tracklet representation with FastTracker occlusion memory."""

    def __init__(self, track_id, bbox, score, kf_mode="standard"):
        self.track_id = track_id
        if kf_mode == "constant_gain":
            self.kf = ConstantGainKalmanFilter(xyxy2xywh(bbox))
        else:
            self.kf = BboxKalmanFilter(xyxy2xywh(bbox))
        self.score = score
        self.is_lost = False
        self.time_since_update = 0
        self.hits = 1

        # FastTracker-specific occlusion states
        self.is_occluded = False
        self.occluded_len = 0
        self.last_occluded_frame = -1
        self.was_recently_occluded = False
        self.not_matched = 0

        # State history for anti-drift velocity/position resets
        self.state_history = deque(maxlen=60)
        self.state_history.append(self.kf.x.copy())

    @property
    def bbox(self):
        return xywh2xyxy(self.kf.x[:4, 0])

    def predict(self):
        self.time_since_update += 1
        predicted_box = self.kf.predict()
        self.state_history.append(self.kf.x.copy())
        return xywh2xyxy(predicted_box)

    def update(self, bbox, score):
        self.kf.update(xyxy2xywh(bbox))
        self.score = score
        self.time_since_update = 0
        self.is_lost = False
        self.hits += 1

        # Reset occlusion flags upon successful re-association
        self.is_occluded = False
        self.occluded_len = 0
        self.not_matched = 0
        self.state_history.append(self.kf.x.copy())

    def apply_occlusion(self, frame_id, reset_vel_offset=5, reset_pos_offset=3, enlarge_factor=1.1, dampen_factor=0.89):
        """FastTracker Occlusion-Aware motion dampening and state reset."""
        self.is_occluded = True
        self.occluded_len += 1
        self.last_occluded_frame = frame_id
        self.was_recently_occluded = True

        # 1. Reset velocity to pre-occlusion estimate to prevent velocity drift
        if len(self.state_history) >= reset_vel_offset:
            old_vel = self.state_history[-reset_vel_offset][4:8]
            self.kf.x[4:8] = old_vel.copy()

        # 2. Reset position to stable anchor
        if len(self.state_history) >= reset_pos_offset:
            old_pos = self.state_history[-reset_pos_offset][:4]
            self.kf.x[:4] = old_pos.copy()

        # 3. Enlarge search bounding box on initial occlusion frame
        if self.occluded_len == 1:
            self.kf.x[2, 0] *= enlarge_factor  # enlarge width
            self.kf.x[3, 0] *= enlarge_factor  # enlarge height

        # 4. Dampen remaining motion velocity
        self.kf.x[4:8] *= dampen_factor


class SpeakerFastTracker:
    """FastTracker (arXiv:2508.14370) optimized for Human/Speaker Visual Camera Tracking.
    
    Tuning for Non-Rigid Human Targets:
    - init_iou_suppress = 0.55: Suppresses split torso/full-body duplicate tracks.
    - enlarge_bbox_occ = 1.0: Disables vehicle-specific box expansion which can cause ghost tracks
      to hijack the primary speaker.
    - dampen_motion_occ = 0.89: Preserves anti-drift motion damping.
    """

    def __init__(
        self,
        det_thresh=0.35,
        track_buffer=60,
        min_box_area=100,
        new_track_thresh=0.30,
        min_hits=2,
        init_iou_suppress=0.55,  # Lowered from 0.75 to prevent torso/full-body duplicate tracks
        dampen_motion_occ=0.89,
        enlarge_bbox_occ=1.0,   # Set to 1.0 for articulated human body (1.1 is for rigid vehicles)
        active_occ_to_lost_thresh=10,
        recent_occ_grace_period=40,
        frame_center=None,
        kf_mode="standard",     # "standard" (full KF) | "constant_gain" (steady-state KF)
    ):
        self.kf_mode = kf_mode
        self.det_thresh = det_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.new_track_thresh = new_track_thresh
        self.min_hits = min_hits
        self.frame_center = frame_center

        # FastTracker Hyperparameters
        self.init_iou_suppress = init_iou_suppress
        self.dampen_motion_occ = dampen_motion_occ
        self.enlarge_bbox_occ = enlarge_bbox_occ
        self.active_occ_to_lost_thresh = active_occ_to_lost_thresh
        self.recent_occ_grace_period = recent_occ_grace_period

        self.frame_id = 0
        self.next_id = 1
        self.tracks = []
        self.locked_id = None

    @classmethod
    def from_yaml(cls, yaml_path):
        """Load FastTracker configuration from a YAML file (Ultralytics-style config)."""
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        valid_keys = {
            "det_thresh", "track_buffer", "min_box_area", "new_track_thresh",
            "min_hits", "init_iou_suppress", "dampen_motion_occ",
            "enlarge_bbox_occ", "active_occ_to_lost_thresh", "recent_occ_grace_period",
            "kf_mode",
        }
        kwargs = {k: v for k, v in cfg.items() if k in valid_keys}
        return cls(**kwargs)

    def update(self, detections):
        """Update FastTracker with detections [N, 6] -> [x1, y1, x2, y2, score, class_id]."""
        self.frame_id += 1

        # If no speaker is locked yet, sort detections by distance to frame center so center person gets ID 1
        if self.locked_id is None and len(detections) > 0:
            cx_ref, cy_ref = self.frame_center if self.frame_center is not None else (160.0, 160.0)
            cxs = (detections[:, 0] + detections[:, 2]) * 0.5
            cys = (detections[:, 1] + detections[:, 3]) * 0.5
            dists = np.hypot(cxs - cx_ref, cys - cy_ref)
            detections = detections[np.argsort(dists)]

        # 1. Filter detections by size and partition into high / low score sets
        raw_high = []
        low_dets = []

        for det in detections:
            x1, y1, x2, y2, score, cls_id = det
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_box_area:
                continue

            det_dict = {"bbox": det[:4], "score": float(score), "cls": int(cls_id)}
            if score >= self.det_thresh:
                raw_high.append(det_dict)
            elif score >= 0.14:
                low_dets.append(det_dict)

        # Suppress duplicate multi-scale boxes (torso vs full-body) while preserving center-sorted order
        high_dets = []
        for i, d in enumerate(raw_high):
            is_dup = any(
                i != j and other["score"] > d["score"] and get_iou(d["bbox"], other["bbox"]) >= self.init_iou_suppress
                for j, other in enumerate(raw_high)
            )
            if not is_dup:
                high_dets.append(d)

        # 2. Kalman Filter Predict for all active/lost tracks
        for track in self.tracks:
            track.predict()

        # 3. First Association: High-score detections
        matched_1, unmatched_tracks_1, unmatched_high = self._associate(
            self.tracks, high_dets, iou_threshold=0.4
        )

        matched_tracks = []
        for trk, det in matched_1:
            trk.update(det["bbox"], det["score"])
            matched_tracks.append(trk)

        # 4. Second Association: Low-score detections with remaining unmatched tracks
        matched_2, unmatched_tracks_2, _ = self._associate(
            unmatched_tracks_1, low_dets, iou_threshold=0.35
        )

        for trk, det in matched_2:
            trk.update(det["bbox"], det["score"])
            matched_tracks.append(trk)

        # 5. FastTracker Occlusion Detection & Anti-Drift Dampening
        for trk in unmatched_tracks_2:
            trk.not_matched += 1

            # Check if this unmatched track is occluded by any matched/active track
            occluded = False
            for other in matched_tracks:
                if trk.track_id == other.track_id or other.is_lost:
                    continue
                if is_occluded_by(trk.bbox, other.bbox, iou_thresh=0.5):
                    trk.apply_occlusion(
                        frame_id=self.frame_id,
                        dampen_factor=self.dampen_motion_occ,
                        enlarge_factor=self.enlarge_bbox_occ,
                    )
                    occluded = True
                    break

            if not occluded:
                trk.occluded_len = 0

            # Expire recent occlusion flag after grace period
            if trk.was_recently_occluded and (self.frame_id - trk.last_occluded_frame > self.recent_occ_grace_period):
                trk.was_recently_occluded = False

            # Mark as lost if missing for >2 frames and not sheltered by occlusion
            if trk.not_matched > 2 and (not trk.is_occluded or trk.occluded_len > self.active_occ_to_lost_thresh):
                trk.is_lost = True

        # 6. FastTracker IoU-Suppressed Initialization
        # Prevents duplicate track initialization when detector outputs overlapping boxes
        active_now = [t for t in self.tracks if not t.is_lost]

        for det in unmatched_high:
            if det["score"] < self.new_track_thresh:
                continue

            # Compute maximum IoU with any existing active track
            max_iou = 0.0
            for act in active_now:
                max_iou = max(max_iou, get_iou(det["bbox"], act.bbox))
                if max_iou >= self.init_iou_suppress:
                    break

            # Only spawn new track if it does NOT duplicate an existing active track
            if max_iou < self.init_iou_suppress:
                new_track = FastTrack(self.next_id, det["bbox"], det["score"], kf_mode=self.kf_mode)
                self.tracks.append(new_track)
                active_now.append(new_track)
                self.next_id += 1

        # 7. Clean up expired tracks (with FastTracker extended occlusion memory)
        alive_tracks = []
        for t in self.tracks:
            # Grant grace period if recently occluded
            is_in_grace = t.was_recently_occluded and (self.frame_id - t.last_occluded_frame <= self.recent_occ_grace_period)
            if not is_in_grace and t.time_since_update > self.track_buffer:
                continue
            alive_tracks.append(t)
        self.tracks = alive_tracks

        # 8. Speaker Lock Management
        self._manage_speaker_lock()

        return self._get_tracking_outputs()

    def predict_only(self):
        """Lightweight Kalman-only step for Frame-Skipping (Detect 1, Predict N-1).
        Advances active tracks along their constant-velocity trajectory without running the detector.
        """
        self.frame_id += 1
        for track in self.tracks:
            if not track.is_lost:
                # Advance state using sparse velocity step without marking track as lost
                track.kf.predict()
                track.state_history.append(track.kf.x.copy())
        return self._get_tracking_outputs()

    def _associate(self, tracks, detections, iou_threshold):
        """Greedy matching based on IoU distance."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(tracks), list(detections)

        if self.kf_mode == "constant_gain":
            t_boxes = [t.bbox for t in tracks]
            d_boxes = [d["bbox"] for d in detections]
            iou_matrix = get_iou_matrix_vec(t_boxes, d_boxes)
        else:
            iou_matrix = np.zeros((len(tracks), len(detections)))
            for i, track in enumerate(tracks):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = get_iou(track.bbox, det["bbox"])

        matched_pairs = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        while True:
            if len(unmatched_tracks) == 0 or len(unmatched_dets) == 0:
                break
            max_iou = -1.0
            best_i, best_j = -1, -1
            for i in unmatched_tracks:
                for j in unmatched_dets:
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        best_i, best_j = i, j

            if max_iou < iou_threshold:
                break

            matched_pairs.append((tracks[best_i], detections[best_j]))
            unmatched_tracks.remove(best_i)
            unmatched_dets.remove(best_j)

        return (
            matched_pairs,
            [tracks[i] for i in unmatched_tracks],
            [detections[j] for j in unmatched_dets],
        )

    def _manage_speaker_lock(self):
        """Maintain single speaker lock with occlusion resilience."""
        active_tracks = [t for t in self.tracks if not t.is_lost]

        if self.locked_id is None:
            confirmed = [t for t in active_tracks if t.hits >= self.min_hits]
            if len(confirmed) > 0:
                cx_ref, cy_ref = self.frame_center if self.frame_center is not None else (160.0, 160.0)
                best_track = min(confirmed, key=lambda t: np.hypot((t.bbox[0] + t.bbox[2]) * 0.5 - cx_ref, (t.bbox[1] + t.bbox[3]) * 0.5 - cy_ref))
                self.locked_id = best_track.track_id
                print(f"[FastTracker] Speaker locked onto Target ID {self.locked_id}")
            return

        locked_track = self._get_track_by_id(self.locked_id)
        if locked_track is None:
            self.locked_id = None
            print("[FastTracker] Lost track expired. Resetting speaker lock.")
            return

    def _get_track_by_id(self, track_id):
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None

    def _get_tracking_outputs(self):
        outputs = []
        if self.locked_id is not None:
            track = self._get_track_by_id(self.locked_id)
            if track is not None:
                outputs.append(
                    (
                        track.bbox[0],
                        track.bbox[1],
                        track.bbox[2],
                        track.bbox[3],
                        track.track_id,
                        track.is_lost or track.is_occluded,
                        track.score,
                    )
                )
        else:
            for track in self.tracks:
                if not track.is_lost:
                    outputs.append(
                        (
                            track.bbox[0],
                            track.bbox[1],
                            track.bbox[2],
                            track.bbox[3],
                            track.track_id,
                            track.is_occluded,
                            track.score,
                        )
                    )
        return outputs
