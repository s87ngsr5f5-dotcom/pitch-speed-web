#!/usr/bin/env python3
"""
Pitcher Pose Release-Gate (YOLO11n-pose 17-Keypoint Biomechanics Engine)
------------------------------------------------------------------------
Analyzes a pitching video frame-by-frame using YOLO11n-pose to lock onto the
primary pitcher (largest central person), track their 17 COCO keypoints
(shoulders, elbows, wrists, hips, knees, ankles), classify the 4 delivery phases:
  - IDLE:           standing / taking sign / hands together
  - WINDUP:         leg lift / arm cocking (wrist pulled back/low)
  - RELEASE:        peak throwing-wrist whip velocity + high extension
  - FOLLOW_THROUGH: arm decel / follow-through after release

Outputs exact `release_frame` windows so the ball speed engine NEVER starts
measuring before the pitcher actually releases the baseball.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# COCO 17-keypoint indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

SKELETON_EDGES = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]

DEFAULT_POSE_WEIGHTS = Path(
    r"C:\Users\user\.gemini\antigravity\scratch\pitch-private\weights\yolo11n-pose.pt"
)


@dataclass
class FramePoseState:
    frame_idx: int
    timestamp_ms: float
    phase: str  # IDLE | WINDUP | RELEASE | FOLLOW_THROUGH
    keypoints: Optional[np.ndarray]  # shape (17, 3) -> x, y, conf
    throwing_hand: str  # "R" | "L" | "?"
    wrist_speed_norm: float  # normalized by torso height per second
    wrist_above_shoulder: float  # positive when wrist is above shoulder (in torso units)
    knee_lift_norm: float  # positive when front knee lifts toward hip
    release_score: float


@dataclass
class ReleaseEvent:
    release_frame: int
    release_ms: float
    window_start_frame: int
    window_end_frame: int
    throwing_hand: str
    wrist_xy: Tuple[float, float]
    peak_score: float


class PitcherPoseGate:
    def __init__(self, weights_path: Path = DEFAULT_POSE_WEIGHTS):
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        # Ultralytics auto-downloads yolo11n-pose.pt (~6 MB) if not present
        self.model = YOLO(str(weights_path) if weights_path.exists() else "yolo11n-pose.pt")
        if not weights_path.exists() and Path("yolo11n-pose.pt").exists():
            Path("yolo11n-pose.pt").replace(weights_path)

    @staticmethod
    def _pick_pitcher(boxes_xyxy: np.ndarray, kpts: np.ndarray, w: int, h: int) -> Optional[np.ndarray]:
        """Select the main pitcher: large foreground person closest to horizontal center."""
        if len(boxes_xyxy) == 0:
            return None
        best_idx = -1
        best_score = -1.0
        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            area_norm = (bw * bh) / max(1.0, w * h)
            cx = (x1 + x2) * 0.5 / w
            cy = (y1 + y2) * 0.5 / h
            # Pitcher is typically large in foreground (not tiny distant catcher) and near center
            if bh / h < 0.16 or cy < 0.15 or cy > 0.90:
                continue
            center_bias = max(0.2, 1.0 - abs(cx - 0.48) * 1.4)
            score = area_norm * center_bias
            if score > best_score:
                best_score = score
                best_idx = i
        return kpts[best_idx] if best_idx >= 0 else None

    def analyze_frames(
        self, frames: List[np.ndarray], fps: float, stride: int = 1
    ) -> Tuple[List[FramePoseState], List[ReleaseEvent]]:
        if not frames:
            return [], []
        h, w = frames[0].shape[:2]
        raw_kpts: List[Optional[np.ndarray]] = [None] * len(frames)

        for idx in range(0, len(frames), stride):
            res = self.model.predict(frames[idx], imgsz=320, conf=0.25, verbose=False)[0]
            if res.keypoints is not None and res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                kpts = res.keypoints.data.cpu().numpy()  # (N, 17, 3)
                kp = self._pick_pitcher(boxes, kpts, w, h)
                raw_kpts[idx] = kp
                # Fill skipped frames if stride > 1
                for s in range(1, stride):
                    if idx + s < len(frames):
                        raw_kpts[idx + s] = kp

        # Compute per-frame biomechanics for both R and L wrists to auto-detect throwing hand
        r_speeds, l_speeds = [], []
        r_above, l_above = [], []
        knee_lifts = []
        torso_heights = []

        for idx, kp in enumerate(raw_kpts):
            if kp is None:
                r_speeds.append(0.0); l_speeds.append(0.0)
                r_above.append(-1.0); l_above.append(-1.0)
                knee_lifts.append(0.0); torso_heights.append(h * 0.25)
                continue

            # Torso height reference (shoulder midpoint to hip midpoint)
            sh_y = self._mid_y(kp, L_SHOULDER, R_SHOULDER)
            hp_y = self._mid_y(kp, L_HIP, R_HIP)
            torso = max(24.0, abs(hp_y - sh_y)) if (sh_y > 0 and hp_y > 0) else h * 0.22
            torso_heights.append(torso)

            # Wrist height above shoulder (positive = wrist higher than shoulder in image, i.e. smaller y)
            ra = (kp[R_SHOULDER, 1] - kp[R_WRIST, 1]) / torso if (kp[R_WRIST, 2] > 0.25 and kp[R_SHOULDER, 2] > 0.25) else -1.0
            la = (kp[L_SHOULDER, 1] - kp[L_WRIST, 1]) / torso if (kp[L_WRIST, 2] > 0.25 and kp[L_SHOULDER, 2] > 0.25) else -1.0
            r_above.append(float(ra))
            l_above.append(float(la))

            # Knee lift (how close highest knee is to hip height)
            kl = 0.0
            if hp_y > 0:
                valid_knees = [kp[k, 1] for k in (L_KNEE, R_KNEE) if kp[k, 2] > 0.25]
                if valid_knees:
                    highest_knee_y = min(valid_knees)
                    kl = max(0.0, 1.0 - (highest_knee_y - hp_y) / max(24.0, torso * 1.1))
            knee_lifts.append(float(kl))

            # Wrist speed (torso-heights per second)
            if idx > 0 and raw_kpts[idx - 1] is not None:
                prev = raw_kpts[idx - 1]
                dt = 1.0 / max(1.0, fps)
                rs = (
                    math.hypot(kp[R_WRIST, 0] - prev[R_WRIST, 0], kp[R_WRIST, 1] - prev[R_WRIST, 1]) / torso / dt
                    if (kp[R_WRIST, 2] > 0.25 and prev[R_WRIST, 2] > 0.25)
                    else 0.0
                )
                ls = (
                    math.hypot(kp[L_WRIST, 0] - prev[L_WRIST, 0], kp[L_WRIST, 1] - prev[L_WRIST, 1]) / torso / dt
                    if (kp[L_WRIST, 2] > 0.25 and prev[L_WRIST, 2] > 0.25)
                    else 0.0
                )
                r_speeds.append(min(18.0, float(rs)))
                l_speeds.append(min(18.0, float(ls)))
            else:
                r_speeds.append(0.0)
                l_speeds.append(0.0)

        # Pick throwing hand = whichever wrist achieves higher high-extension whip energy
        r_energy = sum(max(0.0, s) * (1.2 if a > 0.05 else 0.3) for s, a in zip(r_speeds, r_above))
        l_energy = sum(max(0.0, s) * (1.2 if a > 0.05 else 0.3) for s, a in zip(l_speeds, l_above))
        hand = "R" if r_energy >= l_energy else "L"
        w_idx = R_WRIST if hand == "R" else L_WRIST
        speeds = r_speeds if hand == "R" else l_speeds
        aboves = r_above if hand == "R" else l_above

        # Smooth wrist speed with 3-frame moving average
        smooth_spd = [
            float(np.mean(speeds[max(0, i - 1) : min(len(speeds), i + 2)]))
            for i in range(len(speeds))
        ]

        # Compute release score per frame: high whip speed + wrist at/above shoulder
        rel_scores = []
        for i in range(len(frames)):
            height_bonus = 1.0 + max(-0.4, min(1.0, aboves[i] + 0.15))
            rel_scores.append(smooth_spd[i] * height_bonus if aboves[i] > -0.25 else smooth_spd[i] * 0.2)

        # Find distinct release peaks (separated by at least 1.5 seconds).
        # Rank candidates with a windup leg-lift bonus so a true post-leg-lift release
        # beats pre-release arm-raise or post-pitch follow-through spikes.
        min_gap_frames = max(15, int(round(fps * 1.5)))
        knee_lookback = max(6, int(round(fps * 0.9)))

        def _rank_score(i: int) -> float:
            prev_knee = max(knee_lifts[max(0, i - knee_lookback) : i], default=0.0)
            return rel_scores[i] * (1.6 if 0.40 <= prev_knee <= 0.95 else 1.0)

        events: List[ReleaseEvent] = []
        candidates = sorted(
            [i for i in range(len(rel_scores)) if rel_scores[i] >= 1.6],
            key=_rank_score,
            reverse=True,
        )
        # Windup guard: a genuine release must be preceded by a visible windup
        # (leg lift or arm cocking motion) and cannot sit in the first ~0.2 s of the clip.
        min_pre_frames = max(4, int(round(fps * 0.2)))
        windup_lookback = max(8, int(round(fps * 1.2)))
        min_windup_frames = max(3, int(round(fps * 0.1)))

        def _has_windup(peak: int) -> bool:
            if peak < min_pre_frames:
                return False
            lo = max(0, peak - windup_lookback)
            hi = max(lo, peak - 2)
            count = 0
            for j in range(lo, hi):
                if raw_kpts[j] is None:
                    continue
                if knee_lifts[j] > 0.35 or smooth_spd[j] > 0.8:
                    count += 1
            return count >= min_windup_frames

        self.rejected_no_windup: List[int] = []
        chosen_peaks: List[int] = []
        for idx in candidates:
            if not all(abs(idx - p) >= min_gap_frames for p in chosen_peaks):
                continue
            if not _has_windup(idx):
                self.rejected_no_windup.append(idx)
                continue
            chosen_peaks.append(idx)
        chosen_peaks.sort()

        for peak in chosen_peaks:
            # Fine-tune release frame: right at peak or 1 frame after peak wrist height/whip
            kp = raw_kpts[peak]
            wxy = (float(kp[w_idx, 0]), float(kp[w_idx, 1])) if kp is not None else (w * 0.5, h * 0.35)
            # Flight measurement gate: allow ball start from [release - 2 frames] to [release + 1.05s]
            win_start = max(0, peak - max(2, int(round(fps * 0.07))))
            win_end = min(len(frames) - 1, peak + int(round(fps * 1.15)))
            events.append(
                ReleaseEvent(
                    release_frame=peak,
                    release_ms=round(peak * 1000.0 / fps, 1),
                    window_start_frame=win_start,
                    window_end_frame=win_end,
                    throwing_hand=hand,
                    wrist_xy=wxy,
                    peak_score=round(rel_scores[peak], 2),
                )
            )

        # Label 4 phases per frame relative to nearest release event
        states: List[FramePoseState] = []
        for i in range(len(frames)):
            phase = "IDLE"
            for ev in events:
                if abs(i - ev.release_frame) <= max(2, int(round(fps * 0.07))):
                    phase = "RELEASE"
                    break
                elif ev.release_frame - int(round(fps * 0.85)) <= i < ev.release_frame:
                    phase = "WINDUP"
                    break
                elif ev.release_frame < i <= ev.release_frame + int(round(fps * 0.65)):
                    phase = "FOLLOW_THROUGH"
                    break
            if phase == "IDLE" and knee_lifts[i] > 0.45:
                phase = "WINDUP"
            states.append(
                FramePoseState(
                    frame_idx=i,
                    timestamp_ms=round(i * 1000.0 / fps, 1),
                    phase=phase,
                    keypoints=raw_kpts[i],
                    throwing_hand=hand,
                    wrist_speed_norm=round(smooth_spd[i], 2),
                    wrist_above_shoulder=round(aboves[i], 2),
                    knee_lift_norm=round(knee_lifts[i], 2),
                    release_score=round(rel_scores[i], 2),
                )
            )
        return states, events

    @staticmethod
    def _mid_y(kp: np.ndarray, a: int, b: int) -> float:
        vals = [kp[k, 1] for k in (a, b) if kp[k, 2] > 0.25]
        return float(np.mean(vals)) if vals else -1.0

    @staticmethod
    def is_inside_pitcher_pants(
        state: Optional[FramePoseState], x: float, y: float, box_size: float, track_len: int
    ) -> bool:
        """Reject bright white-pants/knee reflections on foreground pitcher before a flight starts."""
        if state is None or state.keypoints is None:
            return False
        kp = state.keypoints
        sh_y = PitcherPoseGate._mid_y(kp, L_SHOULDER, R_SHOULDER)
        hp_y = PitcherPoseGate._mid_y(kp, L_HIP, R_HIP)
        if sh_y <= 0 or hp_y <= 0:
            return False
        torso = abs(hp_y - sh_y)
        if torso < 48.0:
            return False
        lower = [k for k in (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE) if kp[k, 2] > 0.25]
        if len(lower) < 2:
            return False
        xs = [float(kp[k, 0]) for k in lower]
        ys = [float(kp[k, 1]) for k in lower]
        pad_x = torso * 0.22
        if not (min(xs) - pad_x <= x <= max(xs) + pad_x and hp_y - torso * 0.05 <= y <= max(ys) + torso * 0.12):
            return False
        return track_len < 3 or box_size >= torso * 0.14


def draw_pose_overlay(frame_bgr: np.ndarray, state: FramePoseState) -> np.ndarray:
    """Draw skeleton, throwing wrist highlight, and 4-phase badge for visual verification."""
    out = frame_bgr.copy()
    phase_colors = {
        "IDLE": (148, 163, 184),
        "WINDUP": (56, 189, 248),
        "RELEASE": (0, 68, 255),
        "FOLLOW_THROUGH": (16, 185, 129),
    }
    col = phase_colors.get(state.phase, (255, 255, 255))

    if state.keypoints is not None:
        kp = state.keypoints
        for a, b in SKELETON_EDGES:
            if kp[a, 2] > 0.25 and kp[b, 2] > 0.25:
                p1 = (int(kp[a, 0]), int(kp[a, 1]))
                p2 = (int(kp[b, 0]), int(kp[b, 1]))
                cv2.line(out, p1, p2, col, 2, cv2.LINE_AA)
        for k in range(5, 17):
            if kp[k, 2] > 0.25:
                cv2.circle(out, (int(kp[k, 0]), int(kp[k, 1])), 4, (255, 255, 255), -1, cv2.LINE_AA)
        w_idx = R_WRIST if state.throwing_hand == "R" else L_WRIST
        if kp[w_idx, 2] > 0.25:
            wx, wy = int(kp[w_idx, 0]), int(kp[w_idx, 1])
            r = 16 if state.phase == "RELEASE" else 8
            cv2.circle(out, (wx, wy), r, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.rectangle(out, (12, 12), (340, 64), (15, 20, 30), -1)
    cv2.putText(
        out,
        f"f{state.frame_idx:03d} | {state.phase} ({state.throwing_hand}HP) sc={state.release_score:.1f}",
        (22, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        col,
        2,
        cv2.LINE_AA,
    )
    return out
