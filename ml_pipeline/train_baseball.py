"""
PitchSpeed Web - YOLO11n 棒球高速偵測微調腳本
1. 支援從 Roboflow 下載或讀取本地 YOLO 資料集
2. 自動清洗並統一為單類別 (0: baseball)
3. 離線合成方向性動態模糊 (Motion Blur) 並自適應拉長標註框
4. 訓練 YOLO11n 並自動匯出至 public/models/ball_yolo11n.onnx
"""
import argparse
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def download_roboflow_dataset(
    api_key: str, workspace: str, project: str, version: int, dest_dir: Path
) -> Path:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8", location=str(dest_dir))
    return Path(dataset.location)


def resolve_split_dir(yaml_path: Path, split_value: str) -> Optional[Path]:
    base = yaml_path.parent
    candidates = [
        (base / split_value).resolve(),
        (base / split_value.lstrip("../")).resolve(),
        (base / Path(split_value).name).resolve(),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def filter_dataset_to_single_ball_class(
    dataset_dir: Path, target_keywords: Tuple[str, ...] = ("ball", "baseball")
) -> Tuple[Path, Path]:
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到 data.yaml：{yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_names = cfg.get("names", {})
    if isinstance(raw_names, list):
        id_to_name = {i: str(n).lower() for i, n in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        id_to_name = {int(k): str(v).lower() for k, v in raw_names.items()}
    else:
        raise ValueError("無法解析 data.yaml 中的 names 欄位")

    keep_ids = {
        cid
        for cid, name in id_to_name.items()
        if any(kw in name for kw in target_keywords)
    }
    if not keep_ids:
        if len(id_to_name) == 1:
            keep_ids = set(id_to_name.keys())
        else:
            raise ValueError(
                f"在類別 {id_to_name} 中找不到包含 {target_keywords} 的類別，請用 --ball-keywords 指定。"
            )

    print(f"[資料清洗] 保留類別 ID: {keep_ids} -> 重映射為 0: baseball")

    train_images_dir: Optional[Path] = None
    updated_splits = {}

    for split in ("train", "val", "test"):
        if split not in cfg or not cfg[split]:
            continue
        img_dir = resolve_split_dir(yaml_path, str(cfg[split]))
        if not img_dir:
            continue

        updated_splits[split] = str(img_dir)
        if split == "train":
            train_images_dir = img_dir

        label_dir = img_dir.parent / "labels"
        if not label_dir.exists():
            continue

        kept_boxes = 0
        for txt_file in label_dir.glob("*.txt"):
            new_lines: List[str] = []
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    if cls_id in keep_ids:
                        new_lines.append("0 " + " ".join(parts[1:5]))
                        kept_boxes += 1
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
        print(f"  - {split}: 保留 {kept_boxes} 個棒球標註框")

    if train_images_dir is None:
        raise FileNotFoundError("無法定位 train/images 資料夾")

    clean_yaml = dataset_dir / "data_single_ball.yaml"
    clean_cfg = {
        "path": str(dataset_dir.resolve()),
        "train": updated_splits.get("train"),
        "val": updated_splits.get("val", updated_splits.get("train")),
        "nc": 1,
        "names": {0: "baseball"},
    }
    if "test" in updated_splits:
        clean_cfg["test"] = updated_splits["test"]

    with open(clean_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean_cfg, f, sort_keys=False, allow_unicode=True)

    return clean_yaml, train_images_dir


def build_motion_blur_kernel(kernel_size: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    rad = math.radians(angle_deg)
    dx = math.cos(rad)
    dy = math.sin(rad)

    x1 = int(round(center - dx * center))
    y1 = int(round(center - dy * center))
    x2 = int(round(center + dx * center))
    y2 = int(round(center + dy * center))

    cv2.line(kernel, (x1, y1), (x2, y2), 1.0, thickness=1)
    total = float(kernel.sum())
    if total > 0:
        kernel /= total
    return kernel


def apply_ball_motion_blur(
    img: np.ndarray,
    boxes: List[Tuple[float, float, float, float]],
    kernel_size: int,
    angle_deg: float,
    local_only: bool = True,
) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
    h, w = img.shape[:2]
    kernel = build_motion_blur_kernel(kernel_size, angle_deg)
    rad = math.radians(angle_deg)
    stretch_x = abs(math.cos(rad)) * kernel_size * 0.65
    stretch_y = abs(math.sin(rad)) * kernel_size * 0.65

    out_img = img.copy()
    new_boxes: List[Tuple[float, float, float, float]] = []

    if not local_only:
        out_img = cv2.filter2D(out_img, -1, kernel)

    for cx, cy, bw, bh in boxes:
        if local_only:
            pad_x = int(max(bw * w * 1.2, kernel_size * 1.5))
            pad_y = int(max(bh * h * 1.2, kernel_size * 1.5))
            x1 = max(0, int(cx * w - pad_x))
            y1 = max(0, int(cy * h - pad_y))
            x2 = min(w, int(cx * w + pad_x))
            y2 = min(h, int(cy * h + pad_y))

            if x2 - x1 > kernel_size and y2 - y1 > kernel_size:
                roi = out_img[y1:y2, x1:x2]
                blurred_roi = cv2.filter2D(roi, -1, kernel)
                out_img[y1:y2, x1:x2] = cv2.addWeighted(blurred_roi, 0.9, roi, 0.1, 0)

        new_bw = min(1.0, bw + stretch_x / w)
        new_bh = min(1.0, bh + stretch_y / h)
        new_cx = max(new_bw / 2, min(1.0 - new_bw / 2, cx))
        new_cy = max(new_bh / 2, min(1.0 - new_bh / 2, cy))
        new_boxes.append((new_cx, new_cy, new_bw, new_bh))

    return out_img, new_boxes


def augment_train_set_with_motion_blur(
    train_images_dir: Path, blur_ratio: float = 0.5, seed: int = 42
) -> int:
    rng = random.Random(seed)
    label_dir = train_images_dir.parent / "labels"
    exts = {".jpg", ".jpeg", ".png", ".webp"}

    orig_images = [
        p
        for p in train_images_dir.iterdir()
        if p.suffix.lower() in exts and not p.stem.startswith("mblur_")
    ]

    sample_count = int(len(orig_images) * blur_ratio)
    selected = rng.sample(orig_images, min(sample_count, len(orig_images)))
    generated = 0

    for img_path in selected:
        lbl_path = label_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        boxes: List[Tuple[float, float, float, float]] = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append(
                        (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                    )
        if not boxes:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        angle = rng.uniform(-25.0, 25.0) if rng.random() < 0.85 else rng.uniform(-60.0, 60.0)
        w = img.shape[1]
        ksize = int(rng.uniform(0.015, 0.045) * w)
        ksize = max(7, min(35, ksize | 1))

        local_only = rng.random() < 0.8
        aug_img, aug_boxes = apply_ball_motion_blur(img, boxes, ksize, angle, local_only)

        alpha = rng.uniform(0.85, 1.15)
        beta = rng.uniform(-12, 12)
        aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

        out_img_path = train_images_dir / f"mblur_{img_path.name}"
        out_lbl_path = label_dir / f"mblur_{img_path.stem}.txt"

        cv2.imwrite(str(out_img_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(out_lbl_path, "w", encoding="utf-8") as f:
            for cx, cy, bw, bh in aug_boxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        generated += 1

    print(f"[動態模糊增強] 完成，共新增 {generated} 張動態模糊訓練圖。")
    return generated


def train_and_export(
    data_yaml: Path,
    weights: str,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str,
    onnx_out: Path,
) -> None:
    model = YOLO(weights)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/pitchspeed",
        name="ball_yolo11n",
        exist_ok=True,
        mosaic=1.0,
        close_mosaic=12,
        scale=0.6,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        erasing=0.1,
        patience=20,
        cos_lr=True,
        workers=4,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best_pt))
    metrics = best_model.val(data=str(data_yaml), imgsz=imgsz)
    print(f"Precision={metrics.box.mp:.4f} | Recall={metrics.box.mr:.4f} | mAP50={metrics.box.map50:.4f}")

    exported = best_model.export(
        format="onnx",
        imgsz=imgsz,
        opset=12,
        simplify=True,
        dynamic=False,
    )

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, onnx_out)
    size_mb = onnx_out.stat().st_size / (1024 * 1024)
    print(f"[匯出成功] {onnx_out} ({size_mb:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--rf-workspace", type=str, default="")
    parser.add_argument("--rf-project", type=str, default="")
    parser.add_argument("--rf-version", type=int, default=1)
    parser.add_argument("--rf-api-key", type=str, default=os.getenv("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--ball-keywords", type=str, default="ball,baseball")
    parser.add_argument("--blur-ratio", type=float, default=0.5)
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--out", type=str, default="public/models/ball_yolo11n.onnx")
    args = parser.parse_args()

    if args.data_dir:
        dataset_dir = Path(args.data_dir)
    elif args.rf_workspace and args.rf_project:
        if not args.rf_api_key:
            raise ValueError("請設定環境變數 ROBOFLOW_API_KEY 或傳入 --rf-api-key")
        dataset_dir = download_roboflow_dataset(
            args.rf_api_key,
            args.rf_workspace,
            args.rf_project,
            args.rf_version,
            Path("datasets/roboflow_baseball"),
        )
    else:
        raise ValueError("請指定 --data-dir 或 --rf-workspace 與 --rf-project")

    keywords = tuple(k.strip().lower() for k in args.ball_keywords.split(",") if k.strip())
    clean_yaml, train_images_dir = filter_dataset_to_single_ball_class(dataset_dir, keywords)

    if args.blur_ratio > 0:
        augment_train_set_with_motion_blur(train_images_dir, blur_ratio=args.blur_ratio)

    train_and_export(
        data_yaml=clean_yaml,
        weights=args.weights,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        onnx_out=Path(args.out),
    )


if __name__ == "__main__":
    main()
