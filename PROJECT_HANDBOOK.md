# SmartScout Baseball AI — 專案總覽、禁動清單與後續優化手冊

> 最後更新：2026-11-24｜線上版：https://s87ngsr5f5-dotcom.github.io/pitch-speed-web/｜Repo：https://github.com/s87ngsr5f5-dotcom/pitch-speed-web

---

## 一、三端完整架構與目前狀態

| 平台 | 定位與優勢 | 開啟 / 安裝 / 更新方式 | 內建 AI 模型 | 實測驗證狀態 |
|---|---|---|---|---|
| **1. 網頁 PWA 版** | 免安裝、手機 Safari / 電腦即開即用；支援連續測速、自動存片至 IndexedDB、上傳影片 0.25x 流暢畫軌跡 | 網址即開；iPhone Safari「分享 → 加入主畫面」；更新只要重新整理 | `YOLO11n @ 320×320 FP32`（`9.96 MB`，2.38 秒載入，~20 FPS）＋ **`YOLO11n-pose INT8`（`3.07 MB`，背景懶載入出手閘門）** | 已知球場 10/10 命中，雷達對照 **MAE = 4.79 km/h**；已整合三端統一出手閘門防準備期誤測 |
| **2. Windows 電腦工作站** | **準度與畫質天花板**；批次吃單支或整個資料夾，自動輸出 SmartScout 3D 漸細彩帶＋繁中 8 項數據卡 `.mp4`（iPhone/LINE/Reels 相容）＋ `session_report.csv` | 雙擊桌面 **`SmartScout AI 測速工作站`** 捷徑；日後點右上角 **「🔄 檢查並更新」** 自動同步 GitHub 最新版 | 雙球體模型（`YOLO11n@320` ＋ `YOLO11s@640`）＋ **`YOLO11n-pose` 17 點骨架出手閘門**（含靜態背景抑制＋速度外推追蹤） | 已實測通過 `act_01`、`act_03`、`act_04` 王建凱（`125.3 km/h`）、`act_05`、`act_06` 並自動攔截無準備動作假出手 |
| **3. iPhone 原生 60fps App** | **跳脫網頁算力上限**；Apple Neural Engine (CoreML) 真正跑滿 **60fps**、低發熱、螢幕恆亮、中文語音報速、每球無縫直存 iPhone「照片」App | GitHub Actions 雲端 macOS 自動編譯 `.ipa`（免 Mac）→ Windows 用 **Sideloadly** 裝進 iPhone（詳見 `ios-app/README.md`） | 雙 CoreML 模型（FP16 ANE：`640 高精度` ＋ `320 極速`）＋ **Apple Vision `VNDetectHumanBodyPoseRequest` 17 點骨架出手閘門** | 320 首版 `.ipa` 已 100% 綠燈；雙模型 `640+320` ＋ Vision 出手閘門整合完成 |

---

## 一之二、🎯 投手動作出手閘門（Pitch Release Gate）

為解決「球還在投手手上／準備期就提早開測」的問題，三端採用統一的 17 點人體骨架四階段狀態機：

- **四階段定義**：`IDLE`（站姿／看暗號）→ `WINDUP`（抬腿／拉臂蓄力）→ `RELEASE`（投球手腕高位甩動峰值）→ `FOLLOW_THROUGH`（隨球減速）。
- **核心判斷與門檻**：
  1. **自動判別左右投**：依雙手腕高位甩動能量自動鎖定投球手（`RHP` / `LHP`）。
  2. **出手分數（Release Score）**：投球手腕速度（以軀幹高度／秒正規化，3 幀平滑）× 手腕高於肩膀加權，取局部峰值（桌面版 ≥ `1.6`、強訊號 ≥ `3.5`；Web / iOS 即時串流 ≥ `2.0`）。
  3. **準備動作防誤判守門（Windup Guard）**：出手前 `1.2s` 內必須觀察到抬腿（`knee_lift > 0.35`）或明顯拉臂移動（`smooth_spd > 0.8`）達至少 3 幀／樣本，且不可落在影片最初 `0.2s`，徹底杜絕第一幀假出手。
  4. **出手測速窗口**：僅接受 `[release - 70ms, release + 1150ms]` 內的球體偵測點；兩次出手至少間隔 `1.5s`。
  5. **遠景自動放行**：若連續 `2.0s` 畫面中看不到清楚的投手骨架（例如捕手後方遠景），閘門自動放行並交由物理軌跡引擎把關，不會擋住測速。
- **三端整合位置**：
  - **桌面工作站**：`desktop-studio/pitcher_pose_gate.py` → `desktop-studio/smartscout_studio.py`（可用 `--no-pose-gate` 關閉）。
  - **網頁 PWA**：`public/models/pitcher_pose_int8.onnx`（3.07 MB，背景懶載入）→ `src/workers/inference.worker.ts` → `src/core/pitcherPoseGate.ts` → `src/App.tsx` / `src/components/SmartScoutLiveCamera.tsx`。
  - **iPhone 原生 App**：`ios-app/SmartScoutPro/Engine/PitcherPoseGate.swift`（Apple Vision ~33Hz 節流）→ `ios-app/SmartScoutPro/Camera/CameraManager.swift`（即時測速與匯入影片皆啟用）。

---

## 二、🔒 確認「絕對不要動」的核心地雷清單

以下皆為本專案實測踩坑後驗證過的硬規則，**後續任何改版、重構或訓練都不可違反**：

### 1. 網頁版模型與推論引擎（`src/workers/inference.worker.ts`、`public/models/`）
- **不要對棒球偵測 ONNX 做 `INT8 quantize_dynamic`**：動態量化會直接把 YOLO11 小球輸出信心壓成 `0.000`（完全偵測不到球）。`public/models/ball_yolo11n.onnx` 必須永遠保持**無損 FP32（~9.96 MB）**。
- **不要把網頁版換成 `YOLO11s @ 640×640` 或 `onnxruntime-web/webgpu`**：
  - `onnxruntime-web/webgpu`（28.7 MB Asyncify WASM）在 Intel 內顯 Chrome 會卡死在 WGSL 編譯，並拉長載入至 115 秒。
  - `640×640` 在瀏覽器單執行緒 WASM 每幀要 ~2 秒（0.5 FPS），完全不能用。
  - **必守設定**：維持 `onnxruntime-web/wasm`、`ort.env.wasm.numThreads = 1`（相容 GitHub Pages 無 COOP/COEP）、`YOLO11n @ 320×320 FP32`（2.38 秒載入、~20 FPS）。
- **不要讓動作模型（`pitch_action_cls.onnx`）阻擋啟動**：必須維持在 `READY` 之後才背景懶載入（lazy-load）。
- **不要移除 `bitmap.close()` 與 `<video playsInline muted>`**：移除會導致 iOS Safari 記憶體閃退（OOM）或鏡頭黑畫面。
- **不要忘記更新快取版本號**：每次替換 `public/models/ball_yolo11n.onnx` 時，必同步遞增 `public/sw.js` 的快取名稱與 `src/App.tsx` 的模型 query string（例如 `?v=v3fp32`），否則手機會一直讀舊快取。

### 2. 物理測速與變化球引擎（`src/core/smartScoutEngine.ts`、`src/data/smartScoutDefaultSessions.ts`）
- **不要讓 `estimateAutomaticPitchFlight` 在 `isFinalized: false`（球還在飛）時提早結算**：
  - 遠端單幀微小位移曾導致 `e < segEnd` 提早成立，截斷飛行時間算出 `156.9–167 km/h` 假高速。
  - 必守規則：`const hasEndStop = isFinalized && (e < segEnd || sizeRatio >= 1.45);`（等進手套消失 200ms 才結算，MAE 才能壓在 `4.79 km/h`）。
- **不要反轉垂直變化量（IVB）的重力拋物線正負號**：
  - 必守公式：`gravitySagCm = 0.125 * 981 * flightSec^2`，`observedArcCm = (lineMidY - mid.y) * pxToCm * 1.6`，`rawVbCm = (gravitySagCm - observedArcCm) * 1.35 + 18`。平直速球弧高小 → IVB 為正；大曲球／指叉球彩虹弧高大 → IVB 為負。
- **不要移除 `cleanAndSmoothTrajectory` 跳躍點濾除**：用來消掉單幀誤判的飛點，維持 3D 紅軌跡貼合真實球路。

### 3. 相機連續測速與自動錄影（`src/components/SmartScoutLiveCamera.tsx`）
- **不要把即時 `MediaRecorder` 改回 `.shift()` 丟掉第 0 個 chunk**：chunk 0 是 MP4/WebM 容器檔頭，丟掉會讓 4.5 秒以後錄的片全部壞檔無法播放。必守目前的**閒置每 6 秒無縫換段＋進手套 `stop()` 封存完整 `moov` 檔頭存入 IndexedDB 並馬上續錄**。
- **不要在 `actionPhase === 'pitching_start'` 時切換 `uploadedVideoUrl` 中斷 `getUserMedia`**：那會導致每投一球相機就重開閃黑；僅清掉上一球軌跡疊層，保持像測速槍一樣不中斷串流。
- **不要把使用者手動上傳的影片改回逐幀 `seekTo()` 頓挫模式**：維持 `0.25x` 流暢播放即時畫出軌跡＋馬上出速。

### 4. 模型訓練與版控紀律（`pitch-private/`、`.gitignore`、`npm run preflight`）
- **不要把未通過「全集實測把關」的模型直接覆蓋上線**：
  - 新模型必須同時滿足：① 已知雷達片（`rebas-r1..r5` + `academy-000003`）命中率不退化且 MAE < 5 km/h；② 陌生球場命中率有實質提升且無嚴重背景誤報（前車之鑑：`cv320_v2` 雖抓到 10 支新片但 `rebas` 退化成 0/8，已攔截未部署）。
- **不要在 320px 小球微調時開 `degrees > 0`、`mosaic = 0.8` 或 `erasing = 0.4`**：會把 3–6px 的棒球直接縮沒或擦掉。
- **不要把私人影片、訓練集、`pitch-private/` 或 `desktop-studio/output/` 提交進 GitHub**：嚴格維持 `.gitignore`，每次推前端程式碼前必跑 `npm run preflight`（0 錯誤才准 `git push`）。
- **不要在 `desktop-studio/smartscout_studio.py` 拿掉「靜態背景像素格抑制」與「飛行中防遠距跳點保護」**：這兩道防線是電腦版與 iPhone 640px 版能在陌生球場抓出王建凱 `act_04` 的關鍵。

---

## 三、🚀 後續優化清單（依實戰效益排序）

### P0｜最高投資報酬率（立即可做）
1. **專屬常去球場 2～3 球實拍微調**
   - 用 iPhone 原生 60fps 在自己平常練球的牛棚錄 2～3 球丟進來。
   - 採用「低學習率 (`lr0=0.0002`) ＋ 凍結骨幹前 6 層 (`freeze=6`) ＋ 混入原本 `dataset_v3_9videos` 防遺忘」精調 5～8 epochs，讓該球場實戰命中率衝上 95%+ 同時守住雷達準度。
2. **iPhone 原生 640px 版實機測試與校調**
   - 裝上 `SmartScoutPro-unsigned.ipa`（640+320 雙模型版）實測 iPhone 實機 FPS、溫度與相簿自動存片延遲，微調 640px 信心門檻與 `flush` 延遲（目前 220ms）。

### P1｜核心體驗升級
3. **iPhone 原生版自動燒錄「帶 3D 紅軌跡＋數據卡」影片存相簿**
   - 目前 iPhone 自動存進相簿的是乾淨原始 60fps 影片；下一步在 `CameraManager.swift` 加入背景 `AVAssetExportSession` + `AVVideoCompositionCoreAnimationTool`，自動把 3D 漸細彩帶與 8 項數據卡一起燒進 `.mp4` 再存進相簿（完全等同 SmartScout 付費版成品！）。
4. **iPhone 原生版加入 120fps / 240fps 選項與距離快速選單**
   - 在頂部列加入 `18.44m（成棒/青棒）/ 14.0m（少棒）/ 自訂` 快速切換，以及支援 iPhone 120fps 高速攝影模式（時間解析度從 `±16.7ms` 進一步壓縮到 `±8.3ms`，球速誤差可望壓進 `±2 km/h` 內）。
5. **變化球 7 球種實拍門檻校準**
   - 收集 3～5 顆已知真實球種（直球、滑球、曲球、指叉球、變速球）的 60fps 影片，微調 `classifyPitchTypeFromTrajectory` 的垂直／水平變化 cm 門檻，讓球種辨識從「參考估算」升級為「實戰準確」。

### P2｜進階功能與長期演進
6. **跨球場 320px 網頁模型 v3（防遺忘配方）**
   - 修正 `cv320_v2` 災難性遺忘原因：維持原框比例不硬放大、凍結骨幹前 6 層 (`freeze=6`)、以 `dataset_v3_9videos` 為主軸搭配 25% 困難負樣本與精選 640 蒸餾高信心樣本，讓網頁版 320px 也能同時通吃已知與陌生球場。
7. **電腦工作站進階功能**
   - 支援長牛棚影片自動切分後一鍵合併輸出「當日投球精華 Reel（含球速排行榜封面）」。
   - 加入簡易「單幀點選補標註」工具，遇到電腦版漏抓的球點幾下就能直接加入微調資料集。
   - 支援把 `session_report.csv` 一鍵繪製成球速趨勢圖與九宮格進壘分布 PNG 圖卡。

---

## 四、專案重要檔案地圖

```text
pitch-speed-web/
├── PROJECT_HANDBOOK.md              # 本文件（架構、禁動清單、優化路線圖）
├── README.md                        # 專案入口簡介
├── public/
│   ├── models/ball_yolo11n.onnx      # 線上網頁版 320px 無損 FP32 棒球模型 (9.96 MB)
│   ├── models/pitch_action_cls.onnx  # 動作偵測模型 (1.57 MB，懶載入)
│   ├── models/pitcher_pose_int8.onnx # 投手 17 點骨架出手閘門模型 (3.07 MB，懶載入)
│   └── sw.js                         # PWA 雙層快取（換模型必升版號）
├── src/
│   ├── core/smartScoutEngine.ts      # 物理測速核心（isFinalized 結算保護）
│   ├── core/pitcherPoseGate.ts       # Web 端 4 階段投手出手閘門狀態機
│   ├── data/smartScoutDefaultSessions.ts # 變化球重力拋物線分類與軌跡平滑
│   ├── components/SmartScoutLiveCamera.tsx # 連續相機、6秒無縫換段錄影、0.25x 流暢播放
│   └── workers/inference.worker.ts   # 純 WASM 單執行緒推論 Worker（球體＋動作＋骨架）
├── desktop-studio/                   # Windows 電腦高精度工作站（桌面捷徑指向此處）
│   ├── studio_gui.py                 # 暗黑介面 GUI＋一鍵「檢查並更新」
│   ├── pitcher_pose_gate.py          # 桌面端 YOLO11n-pose 17 點骨架出手閘門
│   ├── smartscout_studio.py          # 320+640 雙模型＋出手閘門＋FFmpeg 3D 軌跡渲染引擎
│   ├── 拖曳影片批次產片.bat           # 拖曳影片/資料夾快速啟動器
│   └── app_icon.ico                  # 桌面捷徑圖示
├── ios-app/                          # iPhone 原生 SwiftUI + CoreML 60fps App
│   ├── project.yml                   # XcodeGen 專案設定檔
│   ├── README.md                     # Windows 免 Mac 用 Sideloadly 安裝教學
│   └── SmartScoutPro/
│       ├── Camera/CameraManager.swift       # 60fps 相機＋靜態抗雜訊追蹤＋直存相簿
│       ├── Engine/CoreMLBallDetector.swift  # Apple Neural Engine 640px/320px 雙 CoreML
│       ├── Engine/PitcherPoseGate.swift     # Apple Vision 17 點骨架出手閘門
│       ├── Engine/PitchPhysicsEngine.swift  # 測速、7球種變化量＋中文語音報速
│       ├── Views/ContentView.swift          # 全螢幕 3D 彩帶 HUD＋640/320 切換鈕
│       └── Models/                          # ball_yolo11s_640.pt + ball_yolo11n.pt
└── .github/workflows/
    ├── deploy.yml                   # 自動部署網頁版至 GitHub Pages
    └── build-ios-ipa.yml            # 雲端 macOS 自動轉 CoreML 並打包 .ipa
```
