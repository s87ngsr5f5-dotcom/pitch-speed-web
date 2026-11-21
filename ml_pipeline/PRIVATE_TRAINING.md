# 私下訓練與實拍驗證

在後端工作機或本機執行 `private_pitch_pipeline.py`。在網站 checkout 外建立私有目錄，將影片、標註、權重、執行紀錄和報告放在該目錄。這個流程沒有 GitHub Actions 發布步驟。網站維持使用瀏覽器內推論，實拍評估使用 Python ONNX 偵測與網站共用的球速估算函式。瀏覽器的影像縮放、硬體推論與即時丟幀仍須另外驗證。

1. 收集有合法訓練用途的影片。每段保留一球完整球路及出手和進壘的畫面。使用同一球的測速槍或 Statcast 出手球速作為參考。轉播影片須確認畫面是原速，保留原始幀率與原始時間戳。
2. 建立私有 `manifest.json`，內容是一個陣列。每個項目都要有 `id`、`pitch_id`、`group`、`split`、`view`、`video`、`reference_kmh` 和 `reference_source`。`id` 識別影片，`pitch_id` 識別實際投出的每一球。同一球即使有多角度影片，也只能在同一份名單出現一次。在 `reference_source` 記錄逐球可追溯的資料來源，例如比賽與投球 ID，或訓練場次和雷達紀錄。數值以 km/h 記錄，mph 可乘以 1.609344。
3. 將同一錄影場次、比賽、攝影機位置或其剪輯及重新編碼副本放入同一 `group`。每組只能出現在 `train`、`validation`、`test` 其中一個分區。用 validation 修正參數，凍結候選模型後再跑 test。

| 欄位 | 格式 |
| --- | --- |
| `view` | `behind_pitcher` 或 `behind_catcher`。若為 `side_view`，仍會執行球偵測，但這套即時估算器會回報無測速結果。 |
| `video` | 私有目錄下的影片相對路徑 |
| `distance_m` | 選填，投手板到本壘板距離，預設 18.44 |
| `extension_m` | 選填，投手向前跨步後出手點超出投手板的距離，須另外量測，預設 1.75 |
| `playback_scale` | 選填，原速為 1。影片時間軸被拉長 4 倍時填 4。 |
| `start_frame`、`end_frame` | 選填，影片中要評估的單球範圍，第一幀為 0，含末幀。 |
| `flight_visible` | 選填。沒有拍到出手至本壘的完整路徑時填 `false`，程式仍會執行球偵測並計為漏測。 |
| `checkpoints` | 選填，依兩個實際位置校正網站使用的 `releaseBox`、`strikeZoneBox`。每個框含相對於影片尺寸的 `cx`、`cy`、`width`、`height`（0 到 1）。 |
| `annotations` | 訓練或驗證用的棒球框 CSV，填私有目錄下的相對路徑。 |

標註 CSV 表頭為 `frame,cx,cy,width,height`。幀從 0 起算，座標除以影片寬高。只有明確確認畫面沒有可見棒球時，才將該幀的四個座標留白。其餘未標幀不會進入訓練。標註棒球的位置供偵測訓練，參考球速用於檢驗完整測速流程。

安裝 Python 相依套件後，從 checkout 執行 `npm ci` 以載入共用估算器。先用已部署的 ONNX 基準模型評估 validation，再以實拍標註微調。替換以下權重及目錄路徑。

```powershell
python -m pip install -r ml_pipeline/requirements.txt
npm ci
python ml_pipeline/private_pitch_pipeline.py --private-root ../pitch-private --run-id baseline evaluate --split validation --model public/models/ball_yolo11n.onnx
python ml_pipeline/private_pitch_pipeline.py --private-root ../pitch-private --run-id candidate train --weights yolo11n.pt --device cpu
python ml_pipeline/private_pitch_pipeline.py --private-root ../pitch-private --run-id candidate evaluate --split validation --model ../pitch-private/runs/candidate-training/model/weights/best.onnx
python ml_pipeline/private_pitch_pipeline.py --private-root ../pitch-private --run-id frozen evaluate --split test --model ../pitch-private/runs/candidate-training/model/weights/best.onnx
```

每測完一球即寫入 `progress.jsonl` 並輸出參考值、估測值與差值。完成後用 `per_pitch.csv` 查看逐球結果、推論延遲和漏測，用 `summary.json` 查看候選模型的 SHA-256、推論後端與整體統計。只有至少 100 球的獨立實拍測試，且至少 99% 的球在 ±1.5 km/h 內，才通過腳本門檻。每個漏測都計入分母。這個門檻是實拍樣本的通過比例，如需推廣到不同手機、場地和拍攝角度，還應擴充獨立測試場次並逐類評估。
