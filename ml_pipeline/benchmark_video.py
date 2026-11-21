"""
PitchSpeed Web - ONNX 模型與 18.44m 測速引擎自動化驗證腳本
"""
import argparse
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class BallDetection:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    timestamp_ms: float
    frame_idx: int


@dataclass
class WorldSample:
    x: float
    y: float
    timestamp_ms: float
    confidence: float
    frame_idx: int
    pixel_x: float
    pixel_y: float


@dataclass
class PitchBenchmarkResult:
    pitch_index: int
    speed_kmh: float
    speed_mph: float
    flight_time_ms: float
    confidence_score: float
    r_squared_x: float
    detected_frames: int
    expected_frames: int
    flight_recall_pct: float
    start_frame: int
    end_frame: int
    trajectory_world: List[Tuple[float, float]] = field(default_factory=list)
    trajectory_pixel: List[Tuple[float, float]] = field(default_factory=list)


class BallOnnxDetector:
    def __init__(self, model_path: Path, conf_threshold: float = 0.35):
        if not model_path.exists():
            raise FileNotFoundError(f"找不到 ONNX 模型檔：{model_path}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        active_providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(model_path), providers=active_providers)
        self.input_meta = self.session.get_inputs()[0]
        self.input_name = self.input_meta.name
        self.output_name = self.session.get_outputs()[0].name

        shape = self.input_meta.shape
        self.input_size = int(shape[2]) if isinstance(shape[2], int) else 320
        self.conf_threshold = conf_threshold
        self.active_provider = self.session.get_providers()[0]

    def detect(
        self, frame_bgr: np.ndarray, timestamp_ms: float, frame_idx: int
    ) -> Tuple[Optional[BallDetection], float]:
        t0 = time.perf_counter()
        orig_h, orig_w = frame_bgr.shape[:2]

        resized = cv2.resize(
            frame_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        infer_ms = (time.perf_counter() - t0) * 1000.0

        preds = outputs[0][0]
        channels, _ = preds.shape
        num_classes = channels - 4
        target_class_idx = 0 if num_classes == 1 else 32

        if target_class_idx >= num_classes:
            return None, infer_ms

        scores = preds[4 + target_class_idx, :]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < self.conf_threshold:
            return None, infer_ms

        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        det = BallDetection(
            x=float(preds[0, best_idx]) * scale_x,
            y=float(preds[1, best_idx]) * scale_y,
            width=float(preds[2, best_idx]) * scale_x,
            height=float(preds[3, best_idx]) * scale_y,
            confidence=best_score,
            timestamp_ms=timestamp_ms,
            frame_idx=frame_idx,
        )
        return det, infer_ms


class PitchSpeedEvaluator:
    def __init__(
        self,
        image_points: np.ndarray,
        distance_m: float = 18.44,
        ref_height_m: float = 1.5,
        min_confidence: float = 0.35,
        min_samples: int = 4,
        max_gap_ms: float = 180.0,
        max_plausible_kmh: float = 200.0,
        min_plausible_kmh: float = 25.0,
        air_drag_comp: float = 1.0,
    ):
        self.distance_m = distance_m
        self.ref_height_m = ref_height_m
        self.min_confidence = min_confidence
        self.min_samples = min_samples
        self.max_gap_ms = max_gap_ms
        self.max_plausible_kmh = max_plausible_kmh
        self.min_plausible_kmh = min_plausible_kmh
        self.air_drag_comp = air_drag_comp
        self.gravity = 9.81
        self.margin_m = 1.5

        world_pts = np.array(
            [
                [0.0, ref_height_m],
                [distance_m, ref_height_m],
                [distance_m, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.H, _ = cv2.findHomography(image_points.astype(np.float32), world_pts)
        if self.H is None:
            raise ValueError("無法計算單應性矩陣")
        self.H_inv = np.linalg.inv(self.H)

        self.active_track: List[WorldSample] = []
        self.completed_pitches: List[PitchBenchmarkResult] = []
        self.rejected_outliers = 0

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        pt = np.array([[[u, v]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(out[0]), float(out[1])

    def world_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        pt = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H_inv)[0][0]
        return float(out[0]), float(out[1])

    def push_detection(self, det: BallDetection) -> Optional[PitchBenchmarkResult]:
        if det.confidence < self.min_confidence:
            return None

        wx, wy = self.pixel_to_world(det.x, det.y)
        if not (math.isfinite(wx) and math.isfinite(wy)):
            return None

        if (
            wx < -self.margin_m
            or wx > self.distance_m + self.margin_m
            or wy < -0.5
            or wy > 6.0
        ):
            self.rejected_outliers += 1
            return None

        sample = WorldSample(
            x=wx,
            y=wy,
            timestamp_ms=det.timestamp_ms,
            confidence=det.confidence,
            frame_idx=det.frame_idx,
            pixel_x=det.x,
            pixel_y=det.y,
        )

        finalized = None
        if self.active_track:
            prev = self.active_track[-1]
            dt_ms = sample.timestamp_ms - prev.timestamp_ms
            if dt_ms <= 0:
                return None

            if dt_ms > self.max_gap_ms:
                finalized = self._finalize_active_track()
                self.active_track = [sample]
                return finalized

            dist_m = math.hypot(sample.x - prev.x, sample.y - prev.y)
            instant_kmh = (dist_m / (dt_ms / 1000.0)) * 3.6
            if instant_kmh > self.max_plausible_kmh * 1.35:
                self.rejected_outliers += 1
                return None

            if len(self.active_track) >= 2:
                ref_dx = prev.x - self.active_track[0].x
                curr_dx = sample.x - prev.x
                if abs(ref_dx) > 0.25 and math.copysign(1.0, ref_dx) * curr_dx < -0.12:
                    self.rejected_outliers += 1
                    return None

        self.active_track.append(sample)
        return finalized

    def flush(self) -> Optional[PitchBenchmarkResult]:
        res = self._finalize_active_track()
        self.active_track = []
        return res

    def _finalize_active_track(self) -> Optional[PitchBenchmarkResult]:
        n = len(self.active_track)
        if n < self.min_samples:
            return None

        t0 = self.active_track[0].timestamp_ms
        t_last = self.active_track[-1].timestamp_ms
        flight_ms = t_last - t0
        if flight_ms < 20.0:
            return None

        times_sec = np.array([(s.timestamp_ms - t0) / 1000.0 for s in self.active_track])
        xs = np.array([s.x for s in self.active_track])
        ys_comp = np.array(
            [s.y + 0.5 * self.gravity * (t ** 2) for s, t in zip(self.active_track, times_sec)]
        )

        vx, _, r2_x = self._linear_regression(times_sec, xs)
        vy0, _, _ = self._linear_regression(times_sec, ys_comp)

        speed_kmh = math.hypot(vx, vy0) * self.air_drag_comp * 3.6
        if speed_kmh < self.min_plausible_kmh or speed_kmh > self.max_plausible_kmh:
            return None

        avg_det_conf = float(np.mean([s.confidence for s in self.active_track]))
        sample_score = min(1.0, n / 10.0)
        r2_clamped = max(0.0, min(1.0, r2_x))
        conf_score = round(r2_clamped * 0.65 + sample_score * 0.2 + avg_det_conf * 0.15, 3)

        start_f = self.active_track[0].frame_idx
        end_f = self.active_track[-1].frame_idx
        expected_frames = max(1, end_f - start_f + 1)
        recall_pct = min(100.0, (n / expected_frames) * 100.0)

        result = PitchBenchmarkResult(
            pitch_index=len(self.completed_pitches) + 1,
            speed_kmh=round(speed_kmh, 1),
            speed_mph=round(speed_kmh * 0.621371, 1),
            flight_time_ms=round(flight_ms, 1),
            confidence_score=conf_score,
            r_squared_x=round(r2_x, 4),
            detected_frames=n,
            expected_frames=expected_frames,
            flight_recall_pct=round(recall_pct, 1),
            start_frame=start_f,
            end_frame=end_f,
            trajectory_world=[(round(s.x, 3), round(s.y, 3)) for s in self.active_track],
            trajectory_pixel=[(s.pixel_x, s.pixel_y) for s in self.active_track],
        )
        self.completed_pitches.append(result)
        return result

    @staticmethod
    def _linear_regression(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        mean_t = float(np.mean(t))
        mean_y = float(np.mean(y))
        dt = t - mean_t
        dy = y - mean_y
        ss_tt = float(np.sum(dt * dt))
        if ss_tt < 1e-12:
            return 0.0, mean_y, 0.0
        slope = float(np.sum(dt * dy)) / ss_tt
        intercept = mean_y - slope * mean_t
        pred = intercept + slope * t
        ss_res = float(np.sum((y - pred) ** 2))
        ss_yy = float(np.sum(dy * dy))
        r2 = 1.0 if ss_yy < 1e-12 else 1.0 - (ss_res / ss_yy)
        return slope, intercept, r2


def download_video_from_url(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(output_dir / "%(id)s.%(ext)s"),
        "--print",
        "after_move:filepath",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp 下載失敗：\n{proc.stderr}")
    return Path(proc.stdout.strip().splitlines()[-1])


def parse_points_arg(points_str: str) -> np.ndarray:
    pairs = [p.strip() for p in points_str.split(";") if p.strip()]
    if len(pairs) != 4:
        raise ValueError("--points 必須剛好包含 4 組座標")
    return np.array([[float(v) for v in pair.split(",")] for pair in pairs], dtype=np.float32)


def interactive_select_four_points(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_idx = 0
    points: List[Tuple[int, int]] = []
    labels = ["1:Pitcher Top", "2:Home Top", "3:Home Bot", "4:Pitcher Bot"]

    def read_frame(idx: int) -> np.ndarray:
        idx = max(0, min(max(0, total_frames - 1), idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok or img is None:
            raise IOError(f"無法讀取第 {idx} 幀")
        return img

    base_frame = read_frame(current_idx)
    win_name = "Calibration: Click 4 Corners (n/p: Seek, r: Reset, Enter: Confirm)"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        canvas = base_frame.copy()
        for i, (px, py) in enumerate(points):
            cv2.circle(canvas, (px, py), 8, (16, 185, 129), -1)
            cv2.putText(canvas, labels[i], (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if len(points) >= 2:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], len(points) == 4, (52, 211, 153), 2)
        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and len(points) == 4:
            break
        elif key == ord("r"):
            points.clear()
        elif key == ord("n"):
            current_idx = min(max(0, total_frames - 1), current_idx + 15)
            base_frame = read_frame(current_idx)
        elif key == ord("p"):
            current_idx = max(0, current_idx - 15)
            base_frame = read_frame(current_idx)
        elif key in (ord("q"), 27):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("使用者取消校正。")

    cap.release()
    cv2.destroyAllWindows()
    print(f'[校正完成] --points "{";".join(f"{x},{y}" for x, y in points)}"')
    return np.array(points, dtype=np.float32)


def draw_debug_overlay(
    frame: np.ndarray,
    evaluator: PitchSpeedEvaluator,
    det: Optional[BallDetection],
    latest_pitch: Optional[PitchBenchmarkResult],
) -> np.ndarray:
    vis = frame.copy()
    corners = [
        evaluator.world_to_pixel(0, evaluator.ref_height_m),
        evaluator.world_to_pixel(evaluator.distance_m, evaluator.ref_height_m),
        evaluator.world_to_pixel(evaluator.distance_m, 0),
        evaluator.world_to_pixel(0, 0),
    ]
    cv2.polylines(vis, [np.array(corners, dtype=np.int32)], True, (16, 185, 129), 2)
    if len(evaluator.active_track) >= 2:
        pts = np.array([[s.pixel_x, s.pixel_y] for s in evaluator.active_track], dtype=np.int32)
        cv2.polylines(vis, [pts], False, (0, 215, 255), 3)
    if det is not None:
        r = max(6, int(max(det.width, det.height) * 0.6))
        cv2.circle(vis, (int(det.x), int(det.y)), r, (0, 0, 255), 2)
    hud = (
        f"Last Pitch: {latest_pitch.speed_kmh:.1f} km/h ({latest_pitch.speed_mph:.1f} mph)"
        if latest_pitch
        else "Waiting for pitch..."
    )
    cv2.rectangle(vis, (15, 15), (520, 58), (15, 23, 42), -1)
    cv2.putText(vis, hud, (25, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 204, 21), 2)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--url", type=str, default="")
    parser.add_argument("--model", type=str, default="public/models/ball_yolo11n.onnx")
    parser.add_argument("--points", type=str, default="")
    parser.add_argument("--distance", type=float, default=18.44)
    parser.add_argument("--height", type=float, default=1.5)
    parser.add_argument("--slowmo-scale", type=float, default=1.0)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--air-drag", type=float, default=1.0)
    parser.add_argument("--ground-truth-kmh", type=str, default="")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--out-dir", type=str, default="runs/benchmark")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(args.video) if args.video else download_video_from_url(args.url, out_dir / "downloads")
    points = parse_points_arg(args.points) if args.points else interactive_select_four_points(video_path)

    detector = BallOnnxDetector(Path(args.model), conf_threshold=args.conf)
    evaluator = PitchSpeedEvaluator(
        image_points=points,
        distance_m=args.distance,
        ref_height_m=args.height,
        min_confidence=args.conf,
        air_drag_comp=args.air_drag,
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = (
        cv2.VideoWriter(str(out_dir / f"debug_{video_path.stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if args.save_video
        else None
    )

    infer_times: List[float] = []
    frame_idx = 0
    last_ts_ms = -1.0
    latest_pitch: Optional[PitchBenchmarkResult] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if raw_ms <= last_ts_ms or (frame_idx > 0 and raw_ms == 0.0):
            raw_ms = frame_idx * (1000.0 / fps)
        last_ts_ms = raw_ms
        det, infer_ms = detector.detect(frame, raw_ms / args.slowmo_scale, frame_idx)
        infer_times.append(infer_ms)
        if det is not None:
            completed = evaluator.push_detection(det)
            if completed is not None:
                latest_pitch = completed
        if writer is not None:
            writer.write(draw_debug_overlay(frame, evaluator, det, latest_pitch))
        frame_idx += 1

    final_pitch = evaluator.flush()
    if final_pitch is not None:
        latest_pitch = final_pitch

    cap.release()
    if writer is not None:
        writer.release()

    gts = [float(v.strip()) for v in args.ground_truth_kmh.split(",") if v.strip()]
    lines = [
        f"# Benchmark Report: {video_path.name}",
        f"- Avg Infer: `{np.mean(infer_times):.2f} ms` ({detector.active_provider}) | Pitches: `{len(evaluator.completed_pitches)}`",
        "",
        "| # | Est (km/h) | GT (km/h) | Error | Flight Recall | R² | Conf |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for i, r in enumerate(evaluator.completed_pitches):
        gt = gts[i] if i < len(gts) else (gts[-1] if gts else None)
        err = f"{r.speed_kmh - gt:+.1f} ({abs(r.speed_kmh - gt)/gt*100:.2f}%)" if gt else "--"
        lines.append(
            f"| #{r.pitch_index} | **{r.speed_kmh}** | {gt or '--'} | {err} | {r.flight_recall_pct}% ({r.detected_frames}/{r.expected_frames}) | {r.r_squared_x} | {int(r.confidence_score*100)}% |"
        )
    report = "\n".join(lines) + "\n"
    report_path = out_dir / f"report_{video_path.stem}.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"[報告存檔] {report_path}")


if __name__ == "__main__":
    main()
