#!/usr/bin/env python3
"""
Hybrid Pose-Gated 3-Frame Temporal-Difference Ballistic Ball Tracker
--------------------------------------------------------------------
Recovers tiny (2-14px) flying baseballs in unseen outdoor stadiums, portrait IG
screen recordings (e.g. wba_shirakawa.mp4), and high-motion-blur fastballs where
single-frame 320px/640px YOLO misses the ball.

Core guarantees:
1. Camera-Pan & Shake Rejection: computes fast global frame MAD + changed-pixel
   ratio; skips frames where the camera is panning/zooming (e.g. panning to a
   tripod after the pitch).
2. Pose Release-Window & Pitcher-Body/Pants Exclusion: masks out the pitcher's
   torso/legs/pants using 17-keypoint skeleton boxes and anchors track start to
   the verified throwing-wrist release window.
3. Perspective-Aware Ballistic Linking: handles 3D behind-pitcher perspective
   deceleration (step size shrinking smoothly as 1/Z^2) AND behind-catcher /
   side-view flights, across 30/60/120fps via fps-scaled temporal stride.
4. Strict Quadratic Smoothness + Glove-Stop Finalization: fits a 2D parabola
   x(t), y(t) and trims any pre-release arm artifacts or post-catch dead frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class TemporalBlob:
    x: float
    y: float
    w: float
    h: float
    area: float
    brightness: float
    polarity: int  # +1 bright ball, -1 dark ball against bright sky


@dataclass
class TemporalDetection:
    frame_idx: int
    timestamp_ms: float
    x: float
    y: float
    width: float
    height: float
    confidence: float


def _pitcher_exclusion_boxes(
    pose_state: Any, proc_w: int, proc_h: int, orig_w: int, orig_h: int
) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[Tuple[float, float]], Optional[Tuple[float, float]], float]:
    """Return (body_pants_box, throwing_wrist_xy, glove_wrist_xy, torso_proc) in proc coordinates."""
    if pose_state is None or getattr(pose_state, "keypoints", None) is None:
        return None, None, None, 0.0
    kp = pose_state.keypoints
    sx = proc_w / max(1.0, float(orig_w))
    sy = proc_h / max(1.0, float(orig_h))

    # COCO indices: 5,6 shoulders; 9,10 wrists; 11,12 hips; 13,14 knees; 15,16 ankles
    sh_ys = [float(kp[k, 1]) * sy for k in (5, 6) if kp[k, 2] > 0.25]
    hp_ys = [float(kp[k, 1]) * sy for k in (11, 12) if kp[k, 2] > 0.25]
    torso = abs(float(np.mean(hp_ys)) - float(np.mean(sh_ys))) if (sh_ys and hp_ys) else proc_h * 0.14
    torso = max(14.0, torso)

    hand = getattr(pose_state, "throwing_hand", "R")
    w_idx = 10 if hand == "R" else 9
    g_idx = 9 if hand == "R" else 10
    wrist_xy = (float(kp[w_idx, 0]) * sx, float(kp[w_idx, 1]) * sy) if kp[w_idx, 2] > 0.25 else None
    glove_xy = (float(kp[g_idx, 0]) * sx, float(kp[g_idx, 1]) * sy) if kp[g_idx, 2] > 0.25 else None

    # Lower-body + torso core box (hips/knees/ankles + chest) to reject pants, cleats, jersey
    body_pts = [(float(kp[k, 0]) * sx, float(kp[k, 1]) * sy) for k in (5, 6, 11, 12, 13, 14, 15, 16) if kp[k, 2] > 0.25]
    if len(body_pts) < 3:
        return None, wrist_xy, glove_xy, torso

    xs = [p[0] for p in body_pts]
    ys = [p[1] for p in body_pts]
    pad_x = torso * 0.24
    pad_y_top = torso * 0.10
    pad_y_bot = torso * 0.16
    box = (min(xs) - pad_x, min(ys) - pad_y_top, max(xs) + pad_x, max(ys) + pad_y_bot)
    return box, wrist_xy, glove_xy, torso


def _fit_quadratic_residual(pts: Sequence[Tuple[float, float, float]]) -> Tuple[float, float]:
    """Fit x(u), y(u) = a*u^2 + b*u + c for u in [0, 1]; return (mean_err_px, max_err_px)."""
    n = len(pts)
    if n < 4:
        return 999.0, 999.0
    t0 = pts[0][2]
    dt_span = max(1.0, pts[-1][2] - t0)
    u = np.array([(p[2] - t0) / dt_span for p in pts], dtype=np.float64)
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    A = np.vstack([u * u, u, np.ones_like(u)]).T
    cx, *_ = np.linalg.lstsq(A, xs, rcond=None)
    cy, *_ = np.linalg.lstsq(A, ys, rcond=None)
    ex = A @ cx - xs
    ey = A @ cy - ys
    errs = np.hypot(ex, ey)
    return float(np.mean(errs)), float(np.max(errs))


def filter_dominant_releases(release_events: Optional[Sequence[Any]], fps: float) -> List[Any]:
    """Suppress weak pre-windup hand-raise peaks sitting within 3.5s of a dominant release peak."""
    if not release_events:
        return []
    strong = [e for e in release_events if getattr(e, "peak_score", 0.0) >= 3.5] or list(release_events)
    if len(strong) <= 1:
        return strong
    win_frames = int(round(fps * 3.5))
    kept: List[Any] = []
    for ev in strong:
        sc = float(getattr(ev, "peak_score", 0.0))
        rf = int(getattr(ev, "release_frame", 0))
        dominated = any(
            o is not ev
            and abs(int(getattr(o, "release_frame", 0)) - rf) <= win_frames
            and float(getattr(o, "peak_score", 0.0)) >= max(10.0, sc * 3.2)
            for o in strong
        )
        if not dominated:
            kept.append(ev)
    return kept or strong


def extract_temporal_ball_tracks(
    frames: List[np.ndarray],
    fps: float,
    pose_states: Optional[Sequence[Any]] = None,
    release_events: Optional[Sequence[Any]] = None,
    allowed_frames: Optional[set[int]] = None,
    target_short_side: int = 540,
) -> List[List[TemporalDetection]]:
    """
    Extract verified flying-baseball tracks from BGR video frames using
    camera-pan-safe, pose-gated 3-frame temporal difference + perspective ballistic linking.
    Returns a list of tracks (each sorted chronologically) in original frame pixel coordinates.
    """
    if len(frames) < 7:
        return []

    orig_h, orig_w = frames[0].shape[:2]
    # Scale by short side so portrait 590x1280 stays crisp (540x1172) and tiny 4px balls don't vanish!
    scale = min(1.0, target_short_side / max(1.0, float(min(orig_w, orig_h))))
    proc_w = max(64, int(round(orig_w * scale)))
    proc_h = max(64, int(round(orig_h * scale)))
    inv_sx = orig_w / float(proc_w)
    inv_sy = orig_h / float(proc_h)

    # Frame stride so temporal difference spans ~33ms regardless of 30/60/120fps
    stride = max(1, int(round(fps / 32.0)))

    # Build strong release anchors if pose events exist (suppressing weak windup spikes)
    strong_releases: List[Any] = filter_dominant_releases(release_events, fps)

    # Determine frame indices to inspect
    if strong_releases:
        dom_frames: set[int] = set()
        for ev in strong_releases:
            dom_frames.update(range(ev.window_start_frame, ev.window_end_frame + 1))
        if allowed_frames:
            dom_frames &= allowed_frames
        active_indices = sorted(i for i in dom_frames if stride <= i < len(frames) - stride)
    elif allowed_frames:
        min_idx = max(stride, min(allowed_frames))
        max_idx = min(len(frames) - 1 - stride, max(allowed_frames))
        active_indices = [i for i in range(min_idx, max_idx + 1) if i in allowed_frames]
    else:
        active_indices = list(range(stride, len(frames) - stride))

    if len(active_indices) < 6:
        return []

    # Pre-convert needed grayscale frames (lazily only around active_indices)
    needed = set()
    for i in active_indices:
        needed.add(i - stride)
        needed.add(i)
        needed.add(i + stride)

    grays: dict[int, np.ndarray] = {}
    for i in sorted(needed):
        if 0 <= i < len(frames):
            fr = frames[i]
            if scale < 0.999:
                fr = cv2.resize(fr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            grays[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.int16)

    # Vertical search bounds (exclude extreme top status bar & bottom UI/shoes)
    y_min = int(proc_h * 0.06)
    y_max = int(proc_h * 0.80)
    x_min = int(proc_w * 0.02)
    x_max = int(proc_w * 0.98)

    # Adaptive blob size limits based on processing resolution (allows motion-blurred fastball right at release)
    dim_scale = min(proc_w, proc_h) / 540.0
    max_dim = max(16, int(round(34 * dim_scale)))
    max_area = max(130, int(round(420 * dim_scale * dim_scale)))

    cands_by_frame: List[Tuple[int, float, List[TemporalBlob]]] = []

    for i in active_indices:
        if (i - stride) not in grays or i not in grays or (i + stride) not in grays:
            continue
        prev_g = grays[i - stride]
        curr_g = grays[i]
        next_g = grays[i + stride]

        # 1) Camera-pan / shake rejection (fast strided subsample)
        abs_prev = np.abs(curr_g[::4, ::4] - prev_g[::4, ::4])
        abs_next = np.abs(curr_g[::4, ::4] - next_g[::4, ::4])
        mad = float(np.median(abs_prev))
        changed_frac = float(np.mean(abs_prev >= 14))
        changed_frac2 = float(np.mean(abs_next >= 14))
        if mad > 4.0 or changed_frac > 0.14 or changed_frac2 > 0.14:
            continue

        # 2) 3-frame temporal difference (bright ball OR dark ball against bright sky)
        sub_p = prev_g[y_min:y_max, x_min:x_max]
        sub_c = curr_g[y_min:y_max, x_min:x_max]
        sub_n = next_g[y_min:y_max, x_min:x_max]
        d1 = sub_c - sub_p
        d2 = sub_c - sub_n

        bright_mask = (sub_c >= 105) & (d1 >= 16) & (d2 >= 16)
        dark_mask = (sub_p >= 170) & (sub_n >= 170) & (d1 <= -18) & (d2 <= -18)

        # Pitcher body/pants/glove exclusion (for this frame)
        st = pose_states[i] if (pose_states is not None and i < len(pose_states)) else None
        body_box, wrist_xy, glove_xy, torso = _pitcher_exclusion_boxes(st, proc_w, proc_h, orig_w, orig_h)

        blobs: List[TemporalBlob] = []
        for mask_bool, pol in ((bright_mask, 1), (dark_mask, -1)):
            if not np.any(mask_bool):
                continue
            num, _, stats, centroids = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
            for k in range(1, num):
                area = int(stats[k, cv2.CC_STAT_AREA])
                bw = int(stats[k, cv2.CC_STAT_WIDTH])
                bh = int(stats[k, cv2.CC_STAT_HEIGHT])
                if area < 3 or area > max_area:
                    continue
                if max(bw, bh) > max_dim:
                    continue
                aspect = max(bw, bh) / max(1.0, float(min(bw, bh)))
                if aspect > 3.6:
                    continue
                cx = float(centroids[k][0]) + x_min
                cy = float(centroids[k][1]) + y_min
                # Reject blobs inside pitcher's torso/pants/legs unless right next to throwing wrist
                if body_box is not None:
                    bx0, by0, bx1, by1 = body_box
                    if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                        near_wrist = wrist_xy is not None and math.hypot(cx - wrist_xy[0], cy - wrist_xy[1]) <= torso * 0.45
                        if not near_wrist:
                            continue
                # Reject blobs sitting directly on pitcher's non-throwing glove hand
                if glove_xy is not None and torso > 0:
                    if math.hypot(cx - glove_xy[0], cy - glove_xy[1]) <= torso * 0.28:
                        continue
                br = float(curr_g[min(proc_h - 1, max(0, int(round(cy)))), min(proc_w - 1, max(0, int(round(cx))))])
                blobs.append(TemporalBlob(x=cx, y=cy, w=float(bw), h=float(bh), area=float(area), brightness=br, polarity=pol))

        if 0 < len(blobs) <= 45:
            cands_by_frame.append((i, round(i * 1000.0 / fps, 2), blobs))

    if len(cands_by_frame) < 6:
        return []

    # 3) Perspective-aware ballistic linking
    max_step_base = max(48.0, proc_w * 0.16)
    min_step_base = max(1.2, proc_w * 0.0028)
    max_gap_frames = max(2, int(round(fps * 0.08)))  # allow 1-2 dropped frames

    # Group candidates by time windows (one pitch per release window if available, else global)
    # Track format: list of (frame_idx, ts_ms, blob)
    scored_tracks: List[Tuple[float, List[Tuple[int, float, TemporalBlob]]]] = []

    for a in range(len(cands_by_frame) - 5):
        f_a, ts_a, blobs_a = cands_by_frame[a]
        # If strong release events exist, prioritize starting near one
        matched_rel = None
        if strong_releases:
            for ev in strong_releases:
                if ev.window_start_frame - 2 <= f_a <= ev.release_frame + max(8, int(round(fps * 0.25))):
                    matched_rel = ev
                    break
            if matched_rel is None:
                continue

        for b0 in blobs_a:
            # Try each valid 2nd-step seed within next 1..2 frames to establish initial velocity vector
            seeds: List[Tuple[int, float, TemporalBlob]] = []
            for b in range(a + 1, min(len(cands_by_frame), a + 4)):
                f_b, ts_b, blobs_b = cands_by_frame[b]
                df = f_b - f_a
                if df <= 0 or df > max_gap_frames:
                    continue
                for cand in blobs_b:
                    if cand.polarity != b0.polarity:
                        continue
                    dist = math.hypot(cand.x - b0.x, cand.y - b0.y)
                    if min_step_base * df <= dist <= max_step_base * df:
                        seeds.append((b, dist, cand))

            # Keep top 4 closest/plausible initial seeds per start blob to stay fast
            seeds.sort(key=lambda item: item[1])
            for seed_entry_idx, _, b1 in seeds[:4]:
                f_b, ts_b, _ = cands_by_frame[seed_entry_idx]
                track: List[Tuple[int, float, TemporalBlob]] = [(f_a, ts_a, b0), (f_b, ts_b, b1)]
                df0 = max(1, f_b - f_a)
                vx = (b1.x - b0.x) / df0
                vy = (b1.y - b0.y) / df0
                prev_speed = math.hypot(vx, vy)
                speed_ratio = 0.85  # smooth ratio tracker for 1/Z^2 perspective deceleration

                for c in range(seed_entry_idx + 1, len(cands_by_frame)):
                    f_c, ts_c, blobs_c = cands_by_frame[c]
                    last_f, last_ts, last_b = track[-1]
                    df = f_c - last_f
                    if df > max_gap_frames:
                        break
                    if ts_c - ts_a > 920.0:
                        break

                    # Perspective-aware predicted velocity & position
                    pred_vx = vx * speed_ratio
                    pred_vy = vy * speed_ratio
                    pred_x = last_b.x + pred_vx * df
                    pred_y = last_b.y + pred_vy * df
                    pred_spd = math.hypot(pred_vx, pred_vy)

                    # Search radius scales with predicted step (tight when slow near plate, wider near hand)
                    tol = max(7.5, min(28.0, pred_spd * df * 0.55 + 5.0))

                    best_cand = None
                    best_err = 1e9
                    for cand in blobs_c:
                        if cand.polarity != b0.polarity:
                            continue
                        dx = cand.x - last_b.x
                        dy = cand.y - last_b.y
                        step_d = math.hypot(dx, dy)
                        if step_d < min_step_base * 0.65 * df or step_d > max_step_base * df:
                            continue
                        step_spd = step_d / df
                        # Smooth speed evolution (prevents wild jumps to random background blobs)
                        if prev_speed > 2.0 and not (0.38 * prev_speed <= step_spd <= 1.65 * prev_speed):
                            continue
                        # Forward direction consistency (dot product with current velocity)
                        v_norm = math.hypot(vx, vy) + 1e-6
                        cos_sim = (dx * vx + dy * vy) / (step_d * v_norm + 1e-6)
                        if cos_sim < 0.68:
                            continue
                        err = math.hypot(cand.x - pred_x, cand.y - pred_y)
                        if err <= tol and err < best_err:
                            best_err = err
                            best_cand = cand

                    if best_cand is not None:
                        new_vx = (best_cand.x - last_b.x) / df
                        new_vy = (best_cand.y - last_b.y) / df
                        new_spd = math.hypot(new_vx, new_vy)
                        if prev_speed > 1.5:
                            inst_ratio = max(0.55, min(1.35, new_spd / prev_speed))
                            speed_ratio = 0.5 * speed_ratio + 0.5 * inst_ratio
                        vx, vy = new_vx, new_vy
                        prev_speed = new_spd
                        track.append((f_c, ts_c, best_cand))

                if len(track) < 6:
                    continue

                # Trim post-catch stagnation / sudden reversal tail
                core_a = max(0, len(track) // 6)
                core_b = min(len(track) - 1, (len(track) * 3) // 4)
                dir_x = track[core_b][2].x - track[core_a][2].x
                dir_y = track[core_b][2].y - track[core_a][2].y
                dir_len = math.hypot(dir_x, dir_y)
                if dir_len < proc_w * 0.04:
                    continue
                ux, uy = dir_x / dir_len, dir_y / dir_len

                # Trim post-catch stagnation, recoil, or late-gap catcher-glove artifacts
                end_i = len(track) - 1
                while end_i >= 6:
                    dx = track[end_i][2].x - track[end_i - 1][2].x
                    dy = track[end_i][2].y - track[end_i - 1][2].y
                    st = math.hypot(dx, dy)
                    df_tail = track[end_i][0] - track[end_i - 1][0]
                    align = (dx * ux + dy * uy) / max(1e-6, st)
                    # Cut off post-catch glove movement after a frame gap (ball vanished into mitt)
                    prev_st = math.hypot(track[end_i - 1][2].x - track[end_i - 2][2].x, track[end_i - 1][2].y - track[end_i - 2][2].y)
                    is_post_catch_gap = end_i >= 8 and df_tail >= 2 and (st > prev_st * 1.25 or st < min_step_base * 1.4)
                    if st < min_step_base * 0.85 or align < 0.25 or is_post_catch_gap:
                        end_i -= 1
                    else:
                        break
                track = track[: end_i + 1]
                if len(track) < 6:
                    continue

                # Trim pre-release backward-cocking frames if any
                start_i = 0
                while len(track) - start_i >= 6:
                    dx = track[start_i + 1][2].x - track[start_i][2].x
                    dy = track[start_i + 1][2].y - track[start_i][2].y
                    st = math.hypot(dx, dy)
                    align = (dx * ux + dy * uy) / max(1e-6, st)
                    if align < 0.25:
                        start_i += 1
                    else:
                        break
                track = track[start_i:]
                if len(track) < 6:
                    continue

                dt_ms = track[-1][1] - track[0][1]
                span = math.hypot(track[-1][2].x - track[0][2].x, track[-1][2].y - track[0][2].y)
                if not (300.0 <= dt_ms <= 900.0):
                    continue
                if span < proc_w * 0.055:
                    continue

                # Reject steep upward windup hand-raise trajectories (pitches fly forward/downward under gravity)
                upward_rise = track[0][2].y - track[-1][2].y
                horiz_span = abs(track[-1][2].x - track[0][2].x)
                if upward_rise > proc_h * 0.04 and upward_rise > horiz_span * 0.45:
                    continue

                # Check quadratic ballistic parabola smoothness
                pts_for_fit = [(p[2].x, p[2].y, p[1]) for p in track]
                mean_err, max_err = _fit_quadratic_residual(pts_for_fit)
                err_limit = max(3.5, proc_w * 0.009)
                if mean_err > err_limit or max_err > err_limit * 2.6:
                    continue

                # Check monotonic projection along flight direction (no zig-zag noise)
                projs = [(p[2].x - track[0][2].x) * ux + (p[2].y - track[0][2].y) * uy for p in track]
                if any(projs[k] <= projs[k - 1] for k in range(1, len(projs))):
                    continue

                # Perspective size consistency (ball shrinks behind pitcher or grows behind catcher, or stays small)
                start_sz = max(track[0][2].w, track[0][2].h)
                end_sz = max(track[-1][2].w, track[-1][2].h)
                size_ratio = max(start_sz / max(1.0, end_sz), end_sz / max(1.0, start_sz))

                # Score track quality (favor capturing the full release-to-glove span over dropping frame 0)
                score = (
                    len(track) * 22.0
                    + (span / proc_w) * 140.0
                    - mean_err * 6.0
                    + (18.0 if size_ratio >= 1.35 else 0.0)
                    + (10.0 if b0.polarity == 1 else 0.0)
                )

                # Bonus if start aligns tightly with detected throwing wrist & release frame
                if matched_rel is not None:
                    f_diff = abs(track[0][0] - matched_rel.release_frame)
                    if f_diff <= max(3, int(round(fps * 0.10))):
                        score += 45.0 - f_diff * 8.0
                    wx = matched_rel.wrist_xy[0] * scale
                    wy = matched_rel.wrist_xy[1] * scale
                    w_dist = math.hypot(track[0][2].x - wx, track[0][2].y - wy)
                    if w_dist <= proc_w * 0.25:
                        score += 30.0
                    elif w_dist > proc_w * 0.55:
                        score -= 45.0  # far from pitcher's throwing hand -> penalize heavily

                scored_tracks.append((score, track))

    if not scored_tracks:
        return []

    # Deduplicate overlapping tracks (keep highest-scoring non-overlapping pitch tracks)
    scored_tracks.sort(key=lambda item: item[0], reverse=True)
    chosen_tracks: List[List[TemporalDetection]] = []
    used_windows: List[Tuple[float, float]] = []

    for score, tr in scored_tracks:
        if score < 75.0:
            continue
        t0, t1 = tr[0][1], tr[-1][1]
        if any(not (t1 < u0 - 350.0 or t0 > u1 + 350.0) for u0, u1 in used_windows):
            continue
        used_windows.append((t0, t1))

        # Probe +1 frame after last blob for tiny catcher-mitt impact (1-3px final arrival frame)
        ext_tr = list(tr)
        if len(ext_tr) >= 6:
            f_last, ts_last, b_last = ext_tr[-1]
            f_prev, _, b_prev = ext_tr[-2]
            df_last = max(1, f_last - f_prev)
            step_x = (b_last.x - b_prev.x) / df_last
            step_y = (b_last.y - b_prev.y) / df_last
            f_next = f_last + 1
            if (f_next - stride) in grays and f_next in grays and (f_next + stride) in grays:
                pred_nx = b_last.x + step_x * 0.85
                pred_ny = b_last.y + step_y * 0.85
                px0 = max(0, int(round(pred_nx - 10)))
                px1 = min(proc_w, int(round(pred_nx + 11)))
                py0 = max(0, int(round(pred_ny - 10)))
                py1 = min(proc_h, int(round(pred_ny + 11)))
                if px1 > px0 + 2 and py1 > py0 + 2:
                    sc_patch = grays[f_next][py0:py1, px0:px1]
                    sp_patch = grays[f_next - stride][py0:py1, px0:px1]
                    sn_patch = grays[f_next + stride][py0:py1, px0:px1]
                    if b_last.polarity == 1:
                        m_patch = (sc_patch >= 100) & ((sc_patch - sp_patch) >= 14) & ((sc_patch - sn_patch) >= 14)
                    else:
                        m_patch = (sp_patch >= 165) & (sn_patch >= 165) & ((sc_patch - sp_patch) <= -14) & ((sc_patch - sn_patch) <= -14)
                    ys_p, xs_p = np.where(m_patch)
                    if 2 <= len(xs_p) <= 36:
                        nx = float(np.mean(xs_p)) + px0
                        ny = float(np.mean(ys_p)) + py0
                        nbw = float(np.ptp(xs_p) + 1)
                        nbh = float(np.ptp(ys_p) + 1)
                        dot_fwd = (nx - b_last.x) * step_x + (ny - b_last.y) * step_y
                        not_glove_recoil = max(nbw, nbh) <= max(b_last.w, b_last.h) * 1.8 + 2.0
                        if dot_fwd > 0 and not_glove_recoil and math.hypot(nx - pred_nx, ny - pred_ny) <= 7.0:
                            ext_tr.append((
                                f_next,
                                round(f_next * 1000.0 / fps, 2),
                                TemporalBlob(nx, ny, nbw, nbh, float(len(xs_p)), float(np.mean(sc_patch[ys_p, xs_p])), b_last.polarity),
                            ))

        chosen_tracks.append(
            [
                TemporalDetection(
                    frame_idx=f_idx,
                    timestamp_ms=ts_ms,
                    x=b.x * inv_sx,
                    y=b.y * inv_sy,
                    width=max(4.0, b.w * inv_sx),
                    height=max(4.0, b.h * inv_sy),
                    confidence=0.82,
                )
                for f_idx, ts_ms, b in ext_tr
            ]
        )

    chosen_tracks.sort(key=lambda tr: tr[0].timestamp_ms)
    return chosen_tracks
