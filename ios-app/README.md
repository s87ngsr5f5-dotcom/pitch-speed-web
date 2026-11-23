# SmartScout AI — iPhone 原生 60fps CoreML 測速 App

原生 SwiftUI + AVFoundation + Apple Neural Engine (CoreML) 版本，跳脫瀏覽器 WASM 的 20fps、發熱與無法直存相簿限制：

- **真正 60fps 硬體級追蹤**：後置鏡頭鎖定 `60fps`，CoreML FP16 模型在 iPhone Neural Engine 上單幀推論僅 **~2–4ms**。
- **像測速槍一樣連續不斷測**：自動開啟防休眠，球進手套自動出速、畫 3D 漸細紅軌跡、顯示 8 項數據卡片＋中文語音即時報速。
- **每球自動存進 iPhone 相簿**：背景使用硬體 H.264 `AVAssetWriter` 每 6 秒無縫換段，每球進手套直接寫入 iPhone「照片」App，完全不中斷相機。
- **支援從相簿匯入 60/120/240fps 影片**：用 Apple Neural Engine 高速零掉幀掃描。

## 免 Mac！從 Windows 電腦安裝到 iPhone（3 步驟）

1. 打開 GitHub Repo → **Actions** 分頁 → 點選左側 **`Build iPhone Native App (.ipa + CoreML)`** → 點最新一次成功建置（或按 `Run workflow`）→ 下載底部 Artifacts 的 **`SmartScoutPro-iPhone-IPA`**，解壓取得 `SmartScoutPro-unsigned.ipa`。
2. 在 Windows 安裝免費工具 **[Sideloadly](https://sideloadly.io/)**（需安裝 iTunes / Apple Devices 驅動）。
3. iPhone 接上電腦 USB → 開啟 Sideloadly → 把 `SmartScoutPro-unsigned.ipa` 拖進去 → 輸入自己的 Apple ID → 按 **Start**。安裝完成後到 iPhone「設定 → 一般 → VPN 與裝置管理」信任開發者憑證即可打開使用！

> 註：免費 Apple ID 側載憑證有效 7 天，到期重新插線按一次 Start（或改用 AltStore 自動 Wi-Fi 續簽）；若有 Apple Developer 帳號則有效 1 年。
