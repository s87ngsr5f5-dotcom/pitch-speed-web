#!/usr/bin/env python3
"""
SmartScout AI Desktop Batch Studio (Windows High-Precision Edition)
-------------------------------------------------------------------
Processes single or multi-pitch videos (30/60/120/240fps MP4/MOV), detects the
baseball using ONNX YOLO11 (320 fast + automatic 640 high-precision fallback),
computes SmartScout physics (release speed, pitch type, vertical/horizontal break,
spin rate, efficiency, spin clock, gyro angle), and exports broadcast-grade MP4
clips with progressive 3D tapered red trajectory ribbons + 8-metric HUD cards
compatible with iPhone Photos / LINE / Instagram Reels (H.264 yuv420p faststart).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

try:
    from pitcher_pose_gate import PitcherPoseGate, ReleaseEvent
except ImportError:
    PitcherPoseGate = None  # type: ignore
    ReleaseEvent = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_320 = REPO_ROOT / "public" / "models" / "ball_yolo11n.onnx"
DEFAULT_MODEL_640 = Path(
    r"C:\Users\user\.gemini\antigravity\scratch\pitch-private\train_640_work\runs\y11s_640_r17\weights\best.onnx"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

MAX_PITCH_SAMPLE_GAP_MS = 145.0

PITCH_COLORS = {
    "四縫線快速球": (239, 68, 68),
    "伸卡球": (249, 115, 22),
    "卡特球": (168, 85, 247),
    "滑球": (234, 179, 8),
    "曲球": (56, 189, 248),
    "指叉球": (236, 72, 153),
    "變速球": (16, 185, 129),
}


@dataclass
class Detection:
    frame_idx: int
    timestamp_ms: float
    x: float
    y: float
    width: float
    height: float
    confidence: float


@dataclass
class PitchResult:
    source_video: str
    pitch_index: int
    output_video: str
    speed_kmh: float
    plate_speed_kmh: float
    pitch_type: str
    vertical_break_cm: float
    horizontal_break_cm: float
    total_spin_rpm: int
    active_spin_rpm: int
    efficiency_pct: int
    spin_direction: str
    gyro_degree: int
    is_strike: bool
    flight_time_ms: float
    detection_count: int
    start_frame: int
    end_frame: int
    model_used: str
    ribbon_start: Tuple[float, float]
    ribbon_ctrl: Tuple[float, float]
    ribbon_end: Tuple[float, float]


class OnnxBallDetector:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(8, (cv2.getNumberOfCPUs() or 4)))
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.input_size = int(shape[2]) if isinstance(shape[2], int) else 320

    def detect_topk(
        self, frame_bgr: np.ndarray, conf_thresh: float = 0.20, max_k: int = 8
    ) -> List[Tuple[float, float, float, float, float]]:
        h, w = frame_bgr.shape[:2]
        size = self.input_size
        scale = min(size / w, size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        dx, dy = (size - nw) // 2, (size - nh) // 2
        canvas[dy : dy + nh, dx : dx + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
        out = self.session.run(None, {self.input_name: tensor})[0][0]
        scores = out[4]
        idxs = np.where(scores >= conf_thresh)[0]
        if len(idxs) == 0:
            return []
        order = idxs[np.argsort(scores[idxs])[::-1]]
        sx, sy = w / nw, h / nh
        picked: List[Tuple[float, float, float, float, float]] = []
        for i in order:
            sc = float(scores[i])
            cx = float((out[0, i] - dx) * sx)
            cy = float((out[1, i] - dy) * sy)
            bw = float(out[2, i] * sx)
            bh = float(out[3, i] * sy)
            if not (0.08 * h <= cy <= 0.78 * h and bw <= w * 0.14 and bh <= h * 0.10):
                continue
            if any(math.hypot(cx - px, cy - py) < max(18.0, w * 0.03) for px, py, *_ in picked):
                continue
            picked.append((cx, cy, bw, bh, sc))
            if len(picked) >= max_k:
                break
        return picked


def clean_and_smooth_track(track: List[Detection], video_width: int) -> List[Detection]:
    if len(track) < 4:
        return track
    max_step = max(24.0, video_width * 0.16)
    cleaned = [track[0]]
    for i in range(1, len(track) - 1):
        prev = cleaned[-1]
        curr = track[i]
        nxt = track[i + 1]
        dt_total = max(1.0, nxt.timestamp_ms - prev.timestamp_ms)
        alpha = (curr.timestamp_ms - prev.timestamp_ms) / dt_total
        ix = prev.x + (nxt.x - prev.x) * alpha
        iy = prev.y + (nxt.y - prev.y) * alpha
        dev = math.hypot(curr.x - ix, curr.y - iy)
        step = math.hypot(curr.x - prev.x, curr.y - prev.y)
        if dev <= max_step * 0.65 and step <= max_step:
            cleaned.append(curr)
    last = track[-1]
    if math.hypot(last.x - cleaned[-1].x, last.y - cleaned[-1].y) <= max_step:
        cleaned.append(last)
    return cleaned if len(cleaned) >= 4 else track


def estimate_pitch_segments(
    dets: List[Detection],
    w: int,
    h: int,
    distance_m: float = 18.44,
    extension_m: float = 1.75,
) -> List[Tuple[List[Detection], float, float]]:
    """Mirror of TypeScript estimateAutomaticPitchFlight (finalized segment-aware)."""
    if len(dets) < 6:
        return []
    # Drop static repeating background false positives (same pixel cluster >= 6 frames)
    filtered: List[Detection] = []
    for d in dets:
        near = sum(1 for o in dets if abs(o.x - d.x) <= 6.0 and abs(o.y - d.y) <= 6.0)
        if near >= 6:
            continue
        filtered.append(d)
    dets = filtered
    if len(dets) < 6:
        return []

    flight_dist = distance_m - extension_m
    pts = [(d.x / w, d.y / h) for d in dets]
    steps = []
    for i in range(len(dets) - 1):
        dt = dets[i + 1].timestamp_ms - dets[i].timestamp_ms
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        step_px = math.hypot(dets[i + 1].x - dets[i].x, dets[i + 1].y - dets[i].y)
        valid = (
            0 < dt <= MAX_PITCH_SAMPLE_GAP_MS
            and step_px <= w * 0.22
            and dets[i].confidence >= 0.12
            and dets[i + 1].confidence >= 0.12
        )
        rate = math.hypot(dx, dy) / dt if dt > 0 else 0.0
        steps.append((dx, dy, dt, valid, rate))

    def depth_stat(idx: int) -> bool:
        return 0 <= idx < len(steps) and steps[idx][3] and steps[idx][4] < 0.000038

    found: List[Tuple[List[Detection], float, float]] = []
    seg_start = 0
    for i in range(len(steps) + 1):
        is_break = i == len(steps) or not steps[i][3]
        if not is_break:
            continue
        seg_end = i
        if seg_end - seg_start >= 5:
            s, e = seg_start, seg_end
            while s < e and depth_stat(s):
                s += 1
            while e > s and depth_stat(e - 1):
                e -= 1
            # 濾除出手前拉臂／蓄力回擺幀，對齊真正出手瞬間
            if e - s >= 5:
                dir_x = pts[e][0] - pts[s][0]
                dir_y = pts[e][1] - pts[s][1]
                dir_len = math.hypot(dir_x, dir_y)
                if dir_len >= 0.012:
                    while e - s >= 5 and (steps[s][0] * dir_x + steps[s][1] * dir_y) / dir_len < -0.004:
                        s += 1
            if e - s >= 5:
                d_first, d_last = dets[s], dets[e]
                s_sz = max(d_first.width, d_first.height)
                e_sz = max(d_last.width, d_last.height)
                size_ratio = max(s_sz / max(1.0, e_sz), e_sz / max(1.0, s_sz))
                span = math.hypot(pts[e][0] - pts[s][0], pts[e][1] - pts[s][1])
                has_end_stop = e < seg_end or size_ratio >= 1.30 or (span >= 0.14 and (e - s) >= 7)
                is_depth = size_ratio >= 1.30 and span >= 0.014
                is_oblique = span >= 0.14 and (size_ratio >= 1.12 or (e - s) >= 7)
                rel_pad = 33.37 if (s == seg_start and steps[s][2] > 45) else 0.0
                cat_pad = 33.37 if (e == seg_end and steps[e - 1][2] > 45 and size_ratio >= 1.50) else 0.0
                # 亞幀進手套端點修正：若最後一步位移明顯小於前一步，扣除球已停在捕手手套內的死時間
                sub_catch_trim = 0.0
                if e - s >= 3:
                    d_last_step = math.hypot(dets[e].x - dets[e - 1].x, dets[e].y - dets[e - 1].y)
                    d_prev_step = math.hypot(dets[e - 1].x - dets[e - 2].x, dets[e - 1].y - dets[e - 2].y)
                    dt_last = dets[e].timestamp_ms - dets[e - 1].timestamp_ms
                    if d_prev_step > 1.0 and d_last_step < d_prev_step * 0.85 and 0 < dt_last <= 50.0:
                        sub_catch_trim = dt_last * max(0.0, 1.0 - d_last_step / d_prev_step) * 0.65
                flight_ms = d_last.timestamp_ms - d_first.timestamp_ms - rel_pad - cat_pad - sub_catch_trim
                if 320.0 <= flight_ms <= 920.0:
                    spd = (flight_dist * 3600.0) / flight_ms / 0.955
                    seg_dets = dets[s : e + 1]
                    mean_conf = sum(d.confidence for d in seg_dets) / max(1, len(seg_dets))
                    if (is_depth or is_oblique) and has_end_stop and mean_conf >= 0.24 and 45.0 <= spd <= 165.0:
                        found.append((seg_dets, round(flight_ms, 1), round(spd, 1)))
        seg_start = i + 1
    return found


def classify_pitch_type(speed_kmh: float, hb_cm: float, vb_cm: float) -> str:
    abs_hb = abs(hb_cm)
    if speed_kmh < 112 and vb_cm <= -12:
        return "曲球"
    if abs_hb >= 16 and vb_cm <= 14 and speed_kmh < 132:
        return "滑球"
    if vb_cm <= -10 and abs_hb < 12 and speed_kmh >= 112:
        return "指叉球"
    if speed_kmh < 118 and hb_cm >= 8 and vb_cm < 16:
        return "變速球"
    if speed_kmh >= 118 and hb_cm >= 14 and vb_cm < 24:
        return "伸卡球"
    if speed_kmh >= 118 and hb_cm <= -10 and vb_cm >= 8:
        return "卡特球"
    return "四縫線快速球"


def spin_axis_to_clock(deg: float) -> str:
    norm = deg % 360.0
    total_mins = int(round((norm / 360.0) * 720.0)) % 720
    hh = total_mins // 60
    mm = (total_mins % 60 // 15) * 15
    if hh == 0:
        hh = 12
    return f"{hh}:{mm:02d}"


def compute_metrics(
    raw_track: List[Detection],
    flight_ms: float,
    speed_kmh: float,
    w: int,
    h: int,
    camera_angle: str,
) -> dict:
    track = clean_and_smooth_track(raw_track, w)
    first, last = track[0], track[-1]
    mid_idx = len(track) // 2
    mid_win = track[max(0, mid_idx - 1) : min(len(track), mid_idx + 2)]
    mid_x = sum(p.x for p in mid_win) / len(mid_win)
    mid_y = sum(p.y for p in mid_win) / len(mid_win)

    sx, sy = first.x / w, first.y / h
    ex, ey = last.x / w, last.y / h
    cx = max(0.0, min(1.0, 2 * mid_x / w - (sx + ex) / 2))
    cy = max(0.0, min(1.0, 2 * mid_y / h - (sy + ey) / 2))

    behind_pitcher = camera_angle == "behind_pitcher"
    line_mid_x = (first.x + last.x) * 0.5
    line_mid_y = (first.y + last.y) * 0.5
    px_to_cm = 86.0 / max(110.0, w * 0.2)
    sign_x = -1.0 if behind_pitcher else 1.0
    raw_hb = (mid_x - line_mid_x) * px_to_cm * sign_x * 1.8
    flight_sec = flight_ms / 1000.0
    gravity_sag_cm = 0.125 * 981.0 * flight_sec * flight_sec
    observed_arc_cm = (line_mid_y - mid_y) * px_to_cm * 1.6
    raw_vb = (gravity_sag_cm - observed_arc_cm) * 1.35 + 18.0

    hb_cm = round(max(-58.0, min(58.0, raw_hb)), 1)
    vb_cm = round(max(-58.0, min(68.0, raw_vb)), 1)
    plate_x = round(max(-42.0, min(42.0, (ex - 0.48) * 140.0)), 1)
    plate_y = round(max(25.0, min(130.0, 76.0 - (ey - 0.50) * 120.0)), 1)
    is_strike = abs(plate_x) <= 21.6 and abs(plate_y - 76.0) <= 30.0

    total_break = math.hypot(hb_cm, vb_cm)
    active_spin = int(max(900, min(3100, round(620 + total_break * 32.0 + speed_kmh * 4.2))))
    eff_pct = int(max(32, min(98, round(54 + min(42, total_break * 0.85)))))
    total_spin = int(round(active_spin / (eff_pct / 100.0)))
    gyro_deg = int(round(math.degrees(math.acos(max(-1.0, min(1.0, eff_pct / 100.0))))))
    axis_deg = (math.degrees(math.atan2(hb_cm, vb_cm)) + 360.0) % 360.0
    spin_clock = spin_axis_to_clock(axis_deg)
    plate_speed = round(speed_kmh * 0.916, 1)
    pitch_type = classify_pitch_type(speed_kmh, hb_cm, vb_cm)

    return {
        "track": track,
        "plate_speed_kmh": plate_speed,
        "vertical_break_cm": vb_cm,
        "horizontal_break_cm": hb_cm,
        "total_spin_rpm": total_spin,
        "active_spin_rpm": active_spin,
        "efficiency_pct": eff_pct,
        "spin_direction": spin_clock,
        "gyro_degree": gyro_deg,
        "is_strike": is_strike,
        "pitch_type": pitch_type,
        "ribbon_start": (sx, sy),
        "ribbon_ctrl": (cx, cy),
        "ribbon_end": (ex, ey),
    }


def load_cjk_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        [r"C:\Windows\Fonts\msjhbd.ttc", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msjh.ttc"]
        if bold
        else [r"C:\Windows\Fonts\msjh.ttc", r"C:\Windows\Fonts\msyh.ttc"]
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_tapered_ribbon(
    frame_bgr: np.ndarray,
    start_norm: Tuple[float, float],
    ctrl_norm: Tuple[float, float],
    end_norm: Tuple[float, float],
    progress: float,
    color_rgb: Tuple[int, int, int],
    camera_angle: str,
) -> np.ndarray:
    if progress <= 0.02:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    steps = max(8, int(round(36 * progress)))
    p0 = np.array([start_norm[0] * w, start_norm[1] * h], dtype=np.float32)
    p1 = np.array([ctrl_norm[0] * w, ctrl_norm[1] * h], dtype=np.float32)
    p2 = np.array([end_norm[0] * w, end_norm[1] * h], dtype=np.float32)

    scale_dim = min(w, h) / 720.0
    behind_pitcher = camera_angle == "behind_pitcher"
    start_r = (22.0 if behind_pitcher else 5.0) * scale_dim
    end_r = (4.5 if behind_pitcher else 22.0) * scale_dim

    pts = []
    radii = []
    for i in range(steps + 1):
        t = (i / steps) * min(1.0, progress)
        inv = 1.0 - t
        pt = inv * inv * p0 + 2 * inv * t * p1 + t * t * p2
        pts.append(pt)
        radii.append(start_r + (end_r - start_r) * t)

    def build_poly(radius_mul: float) -> np.ndarray:
        left_pts = []
        right_pts = []
        for idx, pt in enumerate(pts):
            if idx == 0:
                tang = pts[1] - pts[0]
            elif idx == len(pts) - 1:
                tang = pts[-1] - pts[-2]
            else:
                tang = pts[idx + 1] - pts[idx - 1]
            norm_len = float(np.hypot(tang[0], tang[1])) + 1e-6
            nx, ny = -tang[1] / norm_len, tang[0] / norm_len
            r = radii[idx] * radius_mul
            left_pts.append([pt[0] + nx * r, pt[1] + ny * r])
            right_pts.append([pt[0] - nx * r, pt[1] - ny * r])
        poly = np.array(left_pts + right_pts[::-1], dtype=np.int32)
        return poly

    bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
    out = frame_bgr.copy()

    # Outer soft glow
    glow_layer = out.copy()
    cv2.fillPoly(glow_layer, [build_poly(1.75)], bgr, lineType=cv2.LINE_AA)
    out = cv2.addWeighted(glow_layer, 0.28, out, 0.72, 0)

    # Rich solid tapered body
    body_layer = out.copy()
    cv2.fillPoly(body_layer, [build_poly(1.0)], bgr, lineType=cv2.LINE_AA)
    out = cv2.addWeighted(body_layer, 0.86, out, 0.14, 0)

    # Bright specular inner core
    core_layer = out.copy()
    core_bgr = (min(255, bgr[0] + 90), min(255, bgr[1] + 90), min(255, bgr[2] + 90))
    cv2.fillPoly(core_layer, [build_poly(0.36)], core_bgr, lineType=cv2.LINE_AA)
    out = cv2.addWeighted(core_layer, 0.55, out, 0.45, 0)

    # Glowing ball head at tip
    tip = (int(round(pts[-1][0])), int(round(pts[-1][1])))
    tip_r = max(3, int(round(radii[-1] * 1.15)))
    cv2.circle(out, tip, tip_r + 3, bgr, -1, cv2.LINE_AA)
    cv2.circle(out, tip, max(2, tip_r - 1), (255, 255, 255), -1, cv2.LINE_AA)
    return out


def render_hud_card(w: int, h: int, res: PitchResult) -> Image.Image:
    """Render SmartScout dark translucent 8-metric HUD card as an RGBA PIL overlay."""
    card_w = min(int(w * 0.92), max(420, int(w * 0.78)))
    scale = card_w / 540.0
    card_h = int(176 * scale)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x0 = (w - card_w) // 2
    y0 = h - card_h - int(22 * scale)
    radius = int(18 * scale)

    # Dark frosted glass card + subtle border
    draw.rounded_rectangle(
        [x0, y0, x0 + card_w, y0 + card_h],
        radius=radius,
        fill=(12, 16, 24, 210),
        outline=(255, 255, 255, 42),
        width=max(1, int(1.5 * scale)),
    )

    accent = PITCH_COLORS.get(res.pitch_type, (239, 68, 68))
    # Top accent bar + pitch badge
    f_speed = load_cjk_font(int(34 * scale), bold=True)
    f_unit = load_cjk_font(int(14 * scale), bold=True)
    f_badge = load_cjk_font(int(14 * scale), bold=True)
    f_lbl = load_cjk_font(int(11 * scale), bold=False)
    f_val = load_cjk_font(int(16 * scale), bold=True)

    pad_x = int(18 * scale)
    top_y = y0 + int(12 * scale)

    # Speed readout on left
    spd_txt = f"{res.speed_kmh:.1f}"
    draw.text((x0 + pad_x, top_y - int(4 * scale)), spd_txt, font=f_speed, fill=(255, 255, 255, 255))
    spd_bbox = draw.textbbox((x0 + pad_x, top_y - int(4 * scale)), spd_txt, font=f_speed)
    draw.text((spd_bbox[2] + int(6 * scale), top_y + int(14 * scale)), "km/h", font=f_unit, fill=(185, 198, 218, 230))

    # Pitch type pill + strike indicator on right
    pill_txt = f"● {res.pitch_type}"
    pb = draw.textbbox((0, 0), pill_txt, font=f_badge)
    pw = (pb[2] - pb[0]) + int(22 * scale)
    ph = int(28 * scale)
    px = x0 + card_w - pad_x - pw
    py = top_y + int(4 * scale)
    draw.rounded_rectangle([px, py, px + pw, py + ph], radius=int(14 * scale), fill=(*accent, 52), outline=(*accent, 210), width=1)
    draw.text((px + int(11 * scale), py + int(4 * scale)), pill_txt, font=f_badge, fill=(255, 255, 255, 255))

    # Divider line
    div_y = y0 + int(56 * scale)
    draw.line([(x0 + pad_x, div_y), (x0 + card_w - pad_x, div_y)], fill=(255, 255, 255, 30), width=1)

    # 8-metric 2x4 grid
    metrics = [
        ("進壘速度", f"{res.plate_speed_kmh:.1f} km/h"),
        ("垂直變化", f"{res.vertical_break_cm:+.1f} cm"),
        ("水平變化", f"{res.horizontal_break_cm:+.1f} cm"),
        ("總轉速", f"{res.total_spin_rpm} rpm"),
        ("有效轉速", f"{res.active_spin_rpm} rpm"),
        ("旋轉效率", f"{res.efficiency_pct}%"),
        ("旋轉方向", f"{res.spin_direction}"),
        ("陀螺角度", f"{res.gyro_degree}°"),
    ]
    col_w = (card_w - pad_x * 2) / 4.0
    row_h = int(52 * scale)
    for idx, (label, val) in enumerate(metrics):
        row = idx // 4
        col = idx % 4
        cx = int(x0 + pad_x + col * col_w)
        cy = div_y + int(10 * scale) + row * row_h
        draw.text((cx, cy), label, font=f_lbl, fill=(148, 163, 184, 225))
        draw.text((cx, cy + int(16 * scale)), val, font=f_val, fill=(248, 250, 252, 255))

    return overlay


def composite_rgba_onto_bgr(frame_bgr: np.ndarray, overlay_rgba: Image.Image) -> np.ndarray:
    arr = np.array(overlay_rgba)
    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
    ov_bgr = arr[:, :, [2, 1, 0]].astype(np.float32)
    out = frame_bgr.astype(np.float32) * (1.0 - alpha) + ov_bgr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def export_pitch_clip(
    video_path: Path,
    res: PitchResult,
    fps: float,
    camera_angle: str,
    slowmo: float,
    out_mp4: Path,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pre_frames = int(round(fps * 0.55))
    post_frames = int(round(fps * 0.85))
    clip_start = max(0, res.start_frame - pre_frames)
    clip_end = min(total_frames - 1, res.end_frame + post_frames)

    hud_overlay = render_hud_card(w, h, res)
    accent = PITCH_COLORS.get(res.pitch_type, (239, 68, 68))

    out_fps = max(15.0, fps * slowmo)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        raw_tmp = Path(tf.name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_tmp), fourcc, out_fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    span_frames = max(1, res.end_frame - res.start_frame)
    for fidx in range(clip_start, clip_end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if fidx >= res.start_frame:
            prog = min(1.0, (fidx - res.start_frame) / span_frames)
            frame = draw_tapered_ribbon(
                frame, res.ribbon_start, res.ribbon_ctrl, res.ribbon_end, prog, accent, camera_angle
            )
            if fidx >= res.end_frame - 1:
                frame = composite_rgba_onto_bgr(frame, hud_overlay)
        writer.write(frame)

    writer.release()
    cap.release()

    # Re-encode with FFmpeg to universal H.264 yuv420p + faststart (iPhone/LINE/Reels ready)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(raw_tmp),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    raw_tmp.unlink(missing_ok=True)


def scan_video(
    video_path: Path,
    det_320: Optional[OnnxBallDetector],
    det_640: Optional[OnnxBallDetector],
    distance_m: float,
    camera_angle: str,
    prefer_640: bool,
    pose_gate: Optional["PitcherPoseGate"] = None,
) -> Tuple[List[Tuple[List[Detection], float, float]], float, int, int, str]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()

    allowed_frames: Optional[set[int]] = None
    pose_states = None
    if pose_gate is not None and frames:
        pose_states, release_events = pose_gate.analyze_frames(frames, fps, stride=1)
        rejected = getattr(pose_gate, "rejected_no_windup", [])
        if rejected:
            print(f"  [投手動作閘門] 排除無準備動作的假出手: {', '.join(f'f{r}' for r in sorted(rejected))}", flush=True)
        if not release_events:
            print("  [投手動作閘門] 未偵測到明確出手，改用全片段搜尋（含軌跡物理過濾）", flush=True)
        if release_events:
            strong = [e for e in release_events if e.peak_score >= 3.5] or release_events
            allowed_frames = set()
            for ev in strong:
                for f_i in range(ev.window_start_frame, ev.window_end_frame + 1):
                    allowed_frames.add(f_i)
            ev_summary = ", ".join(f"f{e.release_frame}({e.throwing_hand}HP)" for e in strong)
            print(f"  [投手動作閘門] 鎖定出手窗口: {ev_summary}（準備期／抬腿期已鎖死不測速）", flush=True)

    order = []
    if prefer_640 and det_640 is not None:
        order.append(("YOLO11s@640", det_640, 0.30))
        if det_320 is not None:
            order.append(("YOLO11n@320", det_320, 0.22))
    else:
        if det_320 is not None:
            order.append(("YOLO11n@320", det_320, 0.22))
        if det_640 is not None:
            order.append(("YOLO11s@640", det_640, 0.30))

    for label, detector, thresh in order:
        per_frame_cands: List[List[Tuple[float, float, float, float, float]]] = [
            (
                detector.detect_topk(fr, conf_thresh=max(0.18, thresh - 0.08), max_k=6)
                if (allowed_frames is None or idx in allowed_frames)
                else []
            )
            for idx, fr in enumerate(frames)
        ]
        # Build spatial grid to suppress static repeating background false positives (>= 6 frames in same 14px bin)
        grid_counts: dict[Tuple[int, int], int] = {}
        for cands in per_frame_cands:
            seen_bins = set()
            for cx, cy, *_ in cands:
                b = (int(cx // 14), int(cy // 14))
                if b not in seen_bins:
                    seen_bins.add(b)
                    grid_counts[b] = grid_counts.get(b, 0) + 1

        dets: List[Detection] = []
        for idx, cands in enumerate(per_frame_cands):
            st = pose_states[idx] if (pose_states is not None and idx < len(pose_states)) else None
            dyn = [
                c for c in cands
                if grid_counts.get((int(c[0] // 14), int(c[1] // 14)), 0) < 6
                and max(c[2] / max(1.0, c[3]), c[3] / max(1.0, c[2])) <= 2.4
                and not (
                    PitcherPoseGate is not None
                    and PitcherPoseGate.is_inside_pitcher_pants(st, c[0], c[1], max(c[2], c[3]), len(dets))
                )
            ]
            if not dyn:
                continue
            ts = round(idx * 1000.0 / fps, 2)
            chosen = None
            in_flight = len(dets) >= 2 and (ts - dets[-1].timestamp_ms) <= MAX_PITCH_SAMPLE_GAP_MS
            if in_flight:
                dt_prev = max(1.0, dets[-1].timestamp_ms - dets[-2].timestamp_ms)
                dt_now = ts - dets[-1].timestamp_ms
                vx = (dets[-1].x - dets[-2].x) / dt_prev
                vy = (dets[-1].y - dets[-2].y) / dt_prev
                pred_x = dets[-1].x + vx * dt_now
                pred_y = dets[-1].y + vy * dt_now
                near = [
                    c for c in dyn
                    if math.hypot(c[0] - pred_x, c[1] - pred_y) <= max(46.0, w * 0.14)
                ]
                if near:
                    chosen = min(near, key=lambda c: math.hypot(c[0] - pred_x, c[1] - pred_y))
                elif 0 <= idx < len(frames):
                    # 球飛向本壘變小被全畫面漏掉時，以預測落點做 2.8x 局部放大補抓
                    side = int(max(160, min(w, h) * 0.36))
                    rx1 = max(0, min(w - side, int(pred_x - side * 0.5)))
                    ry1 = max(0, min(h - side, int(pred_y - side * 0.5)))
                    roi_cands = detector.detect_topk(
                        frames[idx][ry1 : ry1 + side, rx1 : rx1 + side],
                        conf_thresh=max(0.12, thresh - 0.10),
                        max_k=3,
                    )
                    roi_mapped = [
                        (c[0] + rx1, c[1] + ry1, c[2], c[3], c[4])
                        for c in roi_cands
                        if max(c[2] / max(1.0, c[3]), c[3] / max(1.0, c[2])) <= 2.4
                        and math.hypot((c[0] + rx1) - pred_x, (c[1] + ry1) - pred_y) <= max(46.0, w * 0.14)
                    ]
                    if roi_mapped:
                        chosen = min(roi_mapped, key=lambda c: math.hypot(c[0] - pred_x, c[1] - pred_y))
            if chosen is None and not (len(dets) >= 4 and in_flight):
                high = [c for c in dyn if c[4] >= thresh]
                if high:
                    chosen = high[0]
            if chosen is not None:
                cx, cy, bw, bh, sc = chosen
                dets.append(
                    Detection(
                        frame_idx=idx,
                        timestamp_ms=ts,
                        x=cx, y=cy, width=bw, height=bh, confidence=sc,
                    )
                )
        segs = estimate_pitch_segments(dets, w, h, distance_m=distance_m)
        if segs:
            return segs, fps, w, h, label

    return [], fps, w, h, "none"


def main():
    parser = argparse.ArgumentParser(description="SmartScout AI Desktop High-Precision Batch Studio")
    parser.add_argument("inputs", nargs="+", help="One or more video files or folders")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT_DIR), help="Output folder")
    parser.add_argument("--distance", type=float, default=18.44, help="Pitching distance in meters (default 18.44)")
    parser.add_argument("--angle", choices=["behind_pitcher", "behind_catcher"], default="behind_pitcher")
    parser.add_argument("--slowmo", type=float, default=1.0, help="Output speed multiplier (e.g. 0.5 for half-speed)")
    parser.add_argument("--prefer-640", action="store_true", help="Always try 640px high-res model first")
    parser.add_argument("--no-pose-gate", action="store_true", help="Disable YOLO11n-pose pitcher release gate")
    parser.add_argument("--model320", default=str(DEFAULT_MODEL_320))
    parser.add_argument("--model640", default=str(DEFAULT_MODEL_640))
    args = parser.parse_args()

    videos: List[Path] = []
    exts = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}
    for raw in args.inputs:
        p = Path(raw)
        if p.is_dir():
            videos.extend(sorted(x for x in p.rglob("*") if x.suffix.lower() in exts))
        elif p.is_file():
            videos.append(p)
    if not videos:
        print("找不到任何影片檔 (.mp4 / .mov)。", file=sys.stderr)
        sys.exit(1)

    det_320 = OnnxBallDetector(Path(args.model320)) if Path(args.model320).exists() else None
    det_640 = OnnxBallDetector(Path(args.model640)) if Path(args.model640).exists() else None
    pose_gate = PitcherPoseGate() if (not args.no_pose_gate and PitcherPoseGate is not None) else None
    if det_320 is None and det_640 is None:
        print("錯誤：找不到任何 ONNX 棒球模型。", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: List[PitchResult] = []

    print(f"=== SmartScout Desktop Studio | 共 {len(videos)} 支影片 ===")
    print(f"可用模型: 320={'YES' if det_320 else 'NO'} | 640={'YES' if det_640 else 'NO'} | 投手動作閘門={'ON' if pose_gate else 'OFF'}\n")

    for vp in videos:
        print(f"[處理中] {vp.name} ...", flush=True)
        segs, fps, w, h, model_used = scan_video(
            vp, det_320, det_640, args.distance, args.angle, args.prefer_640, pose_gate=pose_gate
        )
        if not segs:
            print(f"  -> 未偵測到完整投球軌跡 ({vp.name})", flush=True)
            continue
        for p_idx, (track, flight_ms, speed_kmh) in enumerate(segs, start=1):
            m = compute_metrics(track, flight_ms, speed_kmh, w, h, args.angle)
            out_name = f"{vp.stem}_pitch{p_idx:02d}_{speed_kmh:.1f}kmh.mp4"
            out_mp4 = out_dir / out_name
            res = PitchResult(
                source_video=vp.name,
                pitch_index=p_idx,
                output_video=out_name,
                speed_kmh=speed_kmh,
                plate_speed_kmh=m["plate_speed_kmh"],
                pitch_type=m["pitch_type"],
                vertical_break_cm=m["vertical_break_cm"],
                horizontal_break_cm=m["horizontal_break_cm"],
                total_spin_rpm=m["total_spin_rpm"],
                active_spin_rpm=m["active_spin_rpm"],
                efficiency_pct=m["efficiency_pct"],
                spin_direction=m["spin_direction"],
                gyro_degree=m["gyro_degree"],
                is_strike=m["is_strike"],
                flight_time_ms=flight_ms,
                detection_count=len(m["track"]),
                start_frame=m["track"][0].frame_idx,
                end_frame=m["track"][-1].frame_idx,
                model_used=model_used,
                ribbon_start=m["ribbon_start"],
                ribbon_ctrl=m["ribbon_ctrl"],
                ribbon_end=m["ribbon_end"],
            )
            export_pitch_clip(vp, res, fps, args.angle, args.slowmo, out_mp4)
            all_results.append(res)
            print(
                f"  -> Pitch #{p_idx} [{model_used}]: {res.speed_kmh:.1f} km/h | {res.pitch_type} | "
                f"VB {res.vertical_break_cm:+.1f} / HB {res.horizontal_break_cm:+.1f} cm | "
                f"已輸出 {out_mp4.name}",
                flush=True,
            )

    if all_results:
        csv_path = out_dir / "session_report.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
            writer.writeheader()
            for r in all_results:
                writer.writerow(asdict(r))
        (out_dir / "session_report.json").write_text(
            json.dumps([asdict(r) for r in all_results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n完成！共產出 {len(all_results)} 支 SmartScout 軌跡短片 -> {out_dir}")
        print(f"數據報表: {csv_path}")
    else:
        print("\n本次未產出任何投球片段。")


if __name__ == "__main__":
    main()
