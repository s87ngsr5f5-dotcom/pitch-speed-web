"""Run private, labeled pitch training and speed evaluation outside the web checkout."""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TOLERANCE_KMH = 1.5
PASS_RATE = 0.99
MIN_TEST_PITCHES = 100
VIEWS = {"behind_pitcher", "behind_catcher", "side_view"}
SPLITS = {"train", "validation", "test"}


def outside_repo(path):
    resolved = Path(path).resolve()
    if resolved == REPO or resolved.is_relative_to(REPO):
        raise ValueError(f"Must be outside the web checkout: {resolved}")
    return resolved


def private_file(root, value):
    path = outside_repo(root / value)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Expected a local file inside the private root: {value}")
    return path


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def positive_number(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def read_manifest(path, root):
    clips = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(clips, list) or not clips:
        raise ValueError("Manifest must be a nonempty JSON array")
    group_splits = {}
    videos = set()
    ids = set()
    physical_pitches = set()
    for clip in clips:
        for field in ("id", "pitch_id", "group", "reference_source"):
            if not isinstance(clip.get(field), str) or not clip[field].strip():
                raise ValueError(f"Each clip needs a nonempty {field}")
        clip_id = clip["id"]
        split = clip.get("split")
        if split not in SPLITS or clip.get("view") not in VIEWS:
            raise ValueError(f"{clip_id}: invalid split or camera view")
        if clip_id in ids:
            raise ValueError(f"Duplicate pitch id: {clip_id}")
        ids.add(clip_id)
        if clip["pitch_id"] in physical_pitches:
            raise ValueError(f"{clip_id}: multiple clips of the same physical pitch")
        physical_pitches.add(clip["pitch_id"])
        if group_splits.setdefault(clip["group"], split) != split:
            raise ValueError(f"{clip_id}: source/session group appears in more than one split")
        clip["video"] = private_file(root, clip["video"])
        digest = file_hash(clip["video"])
        if digest in videos:
            raise ValueError(f"{clip_id}: duplicate video content")
        videos.add(digest)
        positive_number(clip.get("reference_kmh"), f"{clip_id}: reference_kmh")
        positive_number(clip.get("distance_m", 18.44), f"{clip_id}: distance_m")
        positive_number(clip.get("playback_scale", 1), f"{clip_id}: playback_scale")
        if type(clip.get("flight_visible", True)) is not bool:
            raise ValueError(f"{clip_id}: flight_visible must be a boolean")
        if clip.get("extension_m") is not None:
            positive_number(clip["extension_m"], f"{clip_id}: extension_m")
        start_frame = clip.get("start_frame", 0)
        end_frame = clip.get("end_frame")
        if type(start_frame) is not int or start_frame < 0 or (
            end_frame is not None and (type(end_frame) is not int or end_frame < start_frame)
        ):
            raise ValueError(f"{clip_id}: invalid start_frame/end_frame")
        checkpoints = clip.get("checkpoints")
        if checkpoints is not None:
            for name in ("releaseBox", "strikeZoneBox"):
                box = checkpoints[name]
                for field in ("cx", "cy", "width", "height"):
                    value = box[field]
                    if type(value) not in (float, int) or not 0 < value <= 1:
                        raise ValueError(f"{clip_id}: {name}.{field} must be in (0, 1]")
                if any(
                    not span / 2 <= center <= 1 - span / 2
                    for center, span in ((box["cx"], box["width"]), (box["cy"], box["height"]))
                ):
                    raise ValueError(f"{clip_id}: {name} extends outside the video")
    return clips


def open_video(cv2, path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("could not open the video")
    return capture


def detect_clip(cv2, detector, clip):
    capture = open_video(cv2, clip["video"])
    detections = []
    frames = 0
    infer_total_ms = 0.0
    last_pts = -1.0
    start_frame = clip.get("start_frame", 0)
    end_frame = clip.get("end_frame")
    try:
        if start_frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            raise ValueError(f"could not seek to frame {start_frame}")
        while end_frame is None or start_frame + frames <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            pts = capture.get(cv2.CAP_PROP_POS_MSEC)
            if not math.isfinite(pts) or pts <= last_pts:
                raise ValueError("missing or non-monotonic frame timestamps")
            last_pts = pts
            det, infer_ms = detector.detect(frame, pts / clip.get("playback_scale", 1), frames)
            frames += 1
            infer_total_ms += infer_ms
            if det:
                detections.append({
                    "x": det.x, "y": det.y, "width": det.width, "height": det.height,
                    "confidence": det.confidence, "timestampMs": det.timestamp_ms,
                })
    finally:
        capture.release()
    if not frames:
        raise ValueError("no decodable frames")
    if end_frame is not None and start_frame + frames != end_frame + 1:
        raise ValueError(f"video ended before frame {end_frame}")
    arguments = {
        "detections": detections,
        "videoWidth": width,
        "videoHeight": height,
        "cameraAngle": clip["view"],
        "distanceMeters": clip.get("distance_m", 18.44),
    }
    if clip.get("checkpoints") is not None:
        arguments["checkpoints"] = clip["checkpoints"]
    if clip.get("extension_m") is not None:
        arguments["extensionMeters"] = clip["extension_m"]
    return arguments, {
        "frame_count": frames,
        "model_detections": len(detections),
        "mean_inference_ms": round(infer_total_ms / frames, 2),
    }


def estimate(arguments):
    proc = subprocess.run(
        ["node", str(REPO / "ml_pipeline" / "run_live_estimator.mjs")],
        input=json.dumps(arguments, allow_nan=False),
        capture_output=True, text=True, encoding="utf-8", cwd=REPO, check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"Live estimator failed:\n{proc.stderr.strip()}")
    return json.loads(proc.stdout)


def grade(clip, result, details=None, problem=None):
    speed = result["speedKmh"] if result else None
    error = round(speed - clip["reference_kmh"], 9) if speed is not None else None
    passed = error is not None and abs(error) <= TOLERANCE_KMH
    return {
        "id": clip["id"], "pitch_id": clip["pitch_id"], "group": clip["group"], "view": clip["view"],
        "reference_source": clip["reference_source"], "reference_kmh": clip["reference_kmh"],
        "predicted_kmh": speed, "error_kmh": error,
        "status": "within_tolerance" if passed else "outside_tolerance" if result else "missed",
        "problem": problem, "flight_time_ms": result["flightTimeMs"] if result else None,
        "track_detections": result["detectionCount"] if result else None,
        **(details or {}),
    }


def summarize(rows, split):
    errors = [abs(row["error_kmh"]) for row in rows if row["error_kmh"] is not None]
    hits = sum(row["status"] == "within_tolerance" for row in rows)
    misses = len(rows) - len(errors)
    rate = hits / len(rows)
    return {
        "split": split, "count": len(rows), "within_tolerance": hits,
        "missed": misses, "outside_tolerance": len(rows) - hits - misses,
        "within_tolerance_rate": rate, "mae_of_detected_kmh": sum(errors) / len(errors) if errors else None,
        "tolerance_kmh": TOLERANCE_KMH, "required_rate": PASS_RATE,
        "min_test_count": MIN_TEST_PITCHES,
        "test_passed": split == "test" and len(rows) >= MIN_TEST_PITCHES and rate >= PASS_RATE,
    }


def evaluate(args, root, clips):
    import cv2
    from benchmark_video import BallOnnxDetector

    selected = [clip for clip in clips if clip["split"] == args.split]
    if not selected:
        raise ValueError(f"No pitches in split {args.split}")
    model = Path(args.model).resolve()
    detector = BallOnnxDetector(model, conf_threshold=args.confidence, letterbox=None)
    # Fail before processing videos if the website estimator cannot be loaded.
    estimate({
        "detections": [], "videoWidth": 1, "videoHeight": 1,
        "cameraAngle": "behind_pitcher", "distanceMeters": 18.44,
    })
    run = outside_repo(root / "runs" / f"{args.run_id}-{args.split}")
    run.mkdir(parents=True, exist_ok=False)
    rows = []
    with (run / "progress.jsonl").open("x", encoding="utf-8") as progress:
        for index, clip in enumerate(selected, start=1):
            try:
                inputs, details = detect_clip(cv2, detector, clip)
            except (ValueError, cv2.error) as exc:
                row = grade(clip, None, problem=str(exc))
            else:
                if clip["view"] == "side_view":
                    problem = "side view is outside this live estimator"
                elif not clip.get("flight_visible", True):
                    problem = "the video does not show the complete flight"
                else:
                    problem = None
                result = None if problem else estimate(inputs)
                row = grade(clip, result, details, problem or (None if result else "no accepted pitch track"))
            rows.append(row)
            progress.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.flush()
            prediction = row["predicted_kmh"]
            prediction_text = f"{prediction:.1f} km/h ({row['error_kmh']:+.2f})" if prediction is not None else "MISSED"
            print(f"[{index}/{len(selected)}] {row['id']} {row['reference_kmh']:.2f} km/h -> {prediction_text}", flush=True)
    column_order = list(dict.fromkeys(key for row in rows for key in row))
    with (run / "per_pitch.csv").open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=column_order)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, args.split)
    summary.update(
        model_sha256=file_hash(model), inference_provider=detector.active_provider,
        confidence=args.confidence, source_groups=len({row["group"] for row in rows}),
        by_camera={
            view: summarize([row for row in rows if row["view"] == view], args.split)
            for view in sorted({row["view"] for row in rows})
        },
    )
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(run)
    return int(args.split == "test" and not summary["test_passed"])


def read_annotations(path):
    annotations = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            frame = int(row["frame"])
            if frame < 0 or frame in annotations:
                raise ValueError(f"{path.name}: frame indices must be unique and nonnegative")
            coords = [row[field].strip() for field in ("cx", "cy", "width", "height")]
            box = tuple(map(float, coords)) if any(coords) else ()
            if box:
                cx, cy, width, height = box
                if (
                    not all(math.isfinite(value) for value in box)
                    or width <= 0 or height <= 0
                    or not width / 2 <= cx <= 1 - width / 2
                    or not height / 2 <= cy <= 1 - height / 2
                ):
                    raise ValueError(f"{path.name}: invalid normalized box in frame {frame}")
            annotations[frame] = box
    if not annotations:
        raise ValueError(f"{path.name}: no labeled frames")
    return annotations


def export_frames(cv2, clip, annotations, images_dir, labels_dir, index):
    capture = open_video(cv2, clip["video"])
    exported = 0
    try:
        for frame_index in range(max(annotations) + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"{clip['id']}: annotation frame is beyond the end of the video")
            if frame_index not in annotations:
                continue
            stem = f"{index}_{frame_index}"
            if not cv2.imwrite(str(images_dir / f"{stem}.png"), frame):
                raise RuntimeError(f"{clip['id']}: could not export frame {frame_index}")
            box = annotations[frame_index]
            label = "0 " + " ".join(str(value) for value in box) + "\n" if box else ""
            (labels_dir / f"{stem}.txt").write_text(label, encoding="utf-8")
            exported += 1
    finally:
        capture.release()
    return exported


def train(args, root, clips):
    import cv2
    import yaml

    weights = private_file(root, args.weights)
    labeled = []
    positive_splits = set()
    for clip in clips:
        if clip["split"] in {"train", "validation"} and clip.get("annotations"):
            annotations = read_annotations(private_file(root, clip["annotations"]))
            labeled.append((clip, annotations))
            if any(annotations.values()):
                positive_splits.add(clip["split"])
    if positive_splits != {"train", "validation"}:
        raise ValueError("Both train and validation session groups need real baseball box labels")

    run = outside_repo(root / "runs" / f"{args.run_id}-training")
    run.mkdir(parents=True, exist_ok=False)
    dataset = run / "dataset"
    for index, (clip, annotations) in enumerate(labeled):
        images = dataset / "images" / clip["split"]
        labels = dataset / "labels" / clip["split"]
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        count = export_frames(cv2, clip, annotations, images, labels, index)
        print(f"[{index + 1}/{len(labeled)}] {clip['id']}: {count} labeled frames ({clip['split']})", flush=True)
    data_yaml = dataset / "baseball.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": dataset.as_posix(), "train": "images/train",
        "val": "images/validation", "names": {0: "baseball"},
    }), encoding="utf-8")

    os.environ["YOLO_CONFIG_DIR"] = str(run / "ultralytics")
    from ultralytics import YOLO, settings

    offline_keys = ("sync", "clearml", "comet", "dvc", "hub", "mlflow", "neptune", "raytune", "wandb")
    settings.update({key: False for key in offline_keys if key in settings})
    model = YOLO(str(weights))
    model.train(
        data=str(data_yaml), imgsz=args.imgsz, epochs=args.epochs,
        batch=args.batch, device=args.device, seed=args.seed, workers=0,
        project=str(run), name="model", deterministic=True,
    )
    best = run / "model" / "weights" / "best.pt"
    output = YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, opset=12, simplify=False, dynamic=False)
    print(f"Candidate ONNX: {output}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--manifest", default="manifest.json", help="Path inside private root")
    parser.add_argument("--run-id", required=True, help="Unique report name in private root/runs")
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("evaluate")
    benchmark.add_argument("--split", choices=("validation", "test"), required=True)
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--confidence", type=float, default=0.22)
    training = commands.add_parser("train")
    training.add_argument("--weights", required=True, help="Local YOLO11 checkpoint inside private root")
    training.add_argument("--imgsz", type=int, default=320)
    training.add_argument("--epochs", type=int, default=60)
    training.add_argument("--batch", type=int, default=16)
    training.add_argument("--device", default="cpu")
    training.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.run_id or not all(char.isascii() and (char.isalnum() or char in "_-") for char in args.run_id):
        parser.error("--run-id may use only ASCII letters, digits, hyphens and underscores")
    if args.command == "evaluate" and not 0 < args.confidence <= 1:
        parser.error("--confidence must be in (0, 1]")
    try:
        root = outside_repo(args.private_root)
        manifest = private_file(root, args.manifest)
        clips = read_manifest(manifest, root)
        handler = evaluate if args.command == "evaluate" else train
        return handler(args, root, clips)
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
