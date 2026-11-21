# PitchSpeed Web × SmartScout Baseball AI

- **線上公開網址 (GitHub Pages)**：https://s87ngsr5f5-dotcom.github.io/pitch-speed-web/
- **GitHub 程式碼倉庫**：https://github.com/s87ngsr5f5-dotcom/pitch-speed-web

結合 **SmartScout Baseball 投手正後方 / 捕手正前方縱深追蹤介面** 與 **18.44m 物理測速引擎** 的全功能棒球投打 AI 分析 PWA：
- **SmartScout 經典 3D 紅色漸變彗星彩帶軌跡 (`投球 #1`)**：支援從投手正後方或正前方拍攝，自動繪製立體紅色彩帶軌跡、4 顆右側懸浮控制鈕（紅圈軌跡開關、1x/0.5x/0.25x 慢動作錶盤、下載含數據圖卡、數據卡開關）、底部深色半透明 8 大數據卡（`初速 km/h`、`球種下拉選單`、`垂直變化量 VB [cm]`、`水平變化量 HB [cm]`、`轉速 S [rpm]`、`有效轉速 AS [rpm]`、`效率 E [%]`、`旋轉方向 SD`、`陀螺角度 GD [°]`）與白色逐幀時間軸滑桿。
- **完整 5 大底部導覽分頁 (`追蹤` | `場次` | `影片` | `球員` | `設定`)**：
  1. **`追蹤`**：`您要錄製什麼？`（`⚾ 投球` / `🏏 打擊`、投手正後方或正前方視角、`使用上次設定`、`🔊 語音朗讀結果` 自動語音播報、上傳影片或開啟即時鏡頭）。
  2. **`場次`**：月份分組場次總覽、搜尋與篩選；點入單一場次具備 **`列表`**（2×4 八格數據卡 + `◇ 比較結果` 並排比較）、**`圖表`**（`-60~+60 cm` 投球變化量圖 & 九宮格好球帶進壘分布圖，點按圓點查看詳情、再次點按前往影片）、以及 **100% 免費無上限解鎖的 `AI 教練`**（`球種特徵` + `優勢 3 / 重點改善 3` 生物力學建議）。
  3. **`影片`**：全畫面 3D 紅色彩帶軌跡播放器，支援上/下一球快速切換與圖卡匯出。
  4. **`球員`**：可自訂球員檔案（投打習慣、年齡、守備位置、身高體重）、年度訓練活躍熱力日曆與 `1個月 / 3個月 / 12個月` 數據成長統計。
  5. **`設定`**：即時鏡頭與 100 部投球模型練習判斷實驗室（`MAE 0.40 km/h`）、ONNX WebGPU/WASM 引擎與離線 PWA 快取管理。

## 100 部投球模型跑分驗證結果
透過 `ml_pipeline/batch_benchmark_100.py` 對 100 部涵蓋 5 大球種（四縫線速球、伸卡球、滑球、變速球、曲球）、3 種幀率（60/120/240fps）與 3 種斜角視角之投球模型實測（完整報告見 `ml_pipeline/BENCHMARK_100_REPORT.md`）：
- **平均絕對誤差 (MAE)**：**`0.40 km/h`**
- **均方根誤差 (RMSE)**：**`0.55 km/h`**
- **誤差 ≤ ±1.5 km/h 比例**：**`99.0%`**（240fps 慢動作平均誤差僅 **`0.23 km/h`**）

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
