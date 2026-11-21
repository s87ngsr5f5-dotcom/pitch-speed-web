# PitchSpeed Web (18.44m)

支援 **iOS Safari 後置鏡頭**、**iPhone 120/240fps 慢動作影片** 與 **離線 PWA 橫向全螢幕 HUD** 的純前端棒球測速系統，透過「**垂直飛行切面 Homography 矩陣 + 重力補償拋物線最小平方法**」消除斜角拍攝與可變球路高度誤差，可直接部署至 GitHub Pages（HTTPS）。

## 內建核心特色
- **18.44m 垂直切面空間校正**：零 `opencv.js` 依賴，以純 TypeScript 實作 Hartley 正規化 DLT，支援任意高度球路與斜角拍攝。
- **PWA 離線球場測速**：內建 `public/manifest.json`、PNG 圖示與 `public/sw.js` 雙層快取（App Shell 與 ONNX 模型分開管理），儀表板支援檢視離線狀態與一鍵強制更新模型。
- **iOS 橫向全螢幕 HUD + 螢幕恆亮**：`fullHeight` 模式嚴格以 `min(100dvw, calc(100dvh * aspectRatio))` 鎖定比例，四點校正零偏移，並整合 Screen Wake Lock 防止腳架鎖屏。
- **localStorage 跨解析度自動記憶**：以 0~1 正規化比例記住四點校正與距離／高度設定，重開 PWA 免重拉。
- **自動化上線前檢查（`npm run preflight`）**：零依賴解析 ONNX Protobuf 真實輸入維度與類別數，自動比對 `src/App.tsx` 的 `inputSize`、跑完 `tsc`、`Vitest` 與 `dist/wasm/` 打包驗證。

## 快速啟動（3 步驟）

```bash
# 1. 安裝前端相依套件（同時產生 package-lock.json）
npm install

# 2. 匯出第一版可直接跑通的輕量 ONNX 模型至 public/models/ball_yolo11n.onnx
pip install -r ml_pipeline/requirements.txt
python ml_pipeline/export_model.py --weights yolo11n.pt --imgsz 320

# 3. 跑完完整 6 大關上線前檢查（含 ONNX 維度比對 + tsc + Vitest + dist/wasm）
npm run preflight
```

本機預覽：
```bash
npm run dev       # 開發模式（不啟用 Service Worker）
npm run preview   # 正式預覽（啟用離線 PWA 與模型快取檢查）
```

## 部署到 GitHub Pages（取得 iOS 必需的 HTTPS）
1. 確認 `npm run preflight` 顯示 `PREFLIGHT PASS`。
2. 將專案（包含 `package-lock.json`、`public/icons/*.png` 與 `public/models/ball_yolo11n.onnx`）推上 GitHub Repo 的 `main` 分支。
3. 前往 GitHub Repo **Settings → Pages → Build and deployment → Source**，選擇 **GitHub Actions**。
4. 等待 Workflow 綠燈後，使用 **iPhone 原生 Safari** 開啟 HTTPS 網址，點底部「分享 → 加入主畫面」即可像原生測速 App 使用。

## 進階模型微調與實拍影片跑分
```bash
# 使用 Roboflow 棒球資料集 + 方向性動態模糊增強微調 YOLO11n
export ROBOFLOW_API_KEY="YOUR_KEY"
python ml_pipeline/train_baseball.py \
  --rf-workspace "WORKSPACE" --rf-project "PROJECT" --rf-version 1 \
  --blur-ratio 0.6 --imgsz 416 --epochs 60

# 若改用 --imgsz 416，請同步把 src/App.tsx 的 inputSize 改為 416，再跑 npm run preflight:fast 驗證！

# 實拍或網路影片誤差跑分（輸出除錯軌跡影片與 Markdown 報告）
python ml_pipeline/benchmark_video.py \
  --video ./pitch.mp4 --ground-truth-kmh 142.0 --save-video
```
