from pathlib import Path
import cv2
import numpy as np


def render_icon(size: int) -> np.ndarray:
    img = np.full((size, size, 3), (23, 6, 2), dtype=np.uint8)  # #020617 (BGR)
    c = size // 2
    cv2.circle(img, (c, c), int(size * 0.39), (129, 185, 16), max(3, size // 28), cv2.LINE_AA)
    cv2.circle(img, (c, c), int(size * 0.27), (90, 120, 40), max(2, size // 64), cv2.LINE_AA)
    cv2.circle(img, (c, c), int(size * 0.14), (245, 245, 245), -1, cv2.LINE_AA)
    r_seam = int(size * 0.13)
    t_seam = max(2, size // 45)
    cv2.ellipse(img, (c - int(size * 0.11), c), (r_seam, r_seam), 0, -45, 45, (68, 68, 239), t_seam, cv2.LINE_AA)
    cv2.ellipse(img, (c + int(size * 0.11), c), (r_seam, r_seam), 0, 135, 225, (68, 68, 239), t_seam, cv2.LINE_AA)
    return img


def main() -> None:
    out_dir = Path("public/icons")
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "apple-touch-icon.png": 180,
        "icon-192.png": 192,
        "icon-512.png": 512,
    }
    for name, sz in targets.items():
        cv2.imwrite(str(out_dir / name), render_icon(sz))
        print(f"Generated {out_dir / name} ({sz}x{sz})")


if __name__ == "__main__":
    main()
