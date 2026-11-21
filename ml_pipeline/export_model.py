"""
PitchSpeed Web - YOLO11n 瀏覽器端 ONNX 匯出腳本
用法：
  python ml_pipeline/export_model.py --weights yolo11n.pt --imgsz 320
"""
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--out", type=str, default="public/models/ball_yolo11n.onnx")
    args = parser.parse_args()

    model = YOLO(args.weights)
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=12,
        simplify=True,
        dynamic=False,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_path, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Exported ONNX to {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
