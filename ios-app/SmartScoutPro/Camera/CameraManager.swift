import AVFoundation
import Foundation
import Photos
import SwiftUI
import Vision

/// Continuous 60fps rear camera + CoreML Apple Neural Engine 640px/320px ball tracker +
/// static background noise suppression + seamless hardware H.264 auto-recording into iPhone Photos.
final class CameraManager: NSObject, ObservableObject {
    let session = AVCaptureSession()

    @Published var isRunning = false
    @Published var activeFps: Int = 60
    @Published var isModelReady = false
    @Published var statusText = "準備中..."
    @Published var liveDots: [CGPoint] = []
    @Published var currentPitch: MeasuredPitch?
    @Published var pitchHistory: [MeasuredPitch] = []
    @Published var videoSize: CGSize = CGSize(width: 720, height: 1280)

    // User settings
    @Published var modelMode: DetectorModelMode = .highRes640
    @Published var cameraAngle: CameraAngleMode = .behindPitcher
    @Published var distanceMeters: Double = 18.44
    @Published var preferredFps: Int = 60
    @Published var autoSaveToPhotos: Bool = true
    @Published var voiceAnnounce: Bool = true

    private let videoQueue = DispatchQueue(label: "com.smartscout.camera.video", qos: .userInteractive)
    private let detector = CoreMLBallDetector()
    private let physics = PitchPhysicsEngine()
    /// 投手骨架出手閘門：球還在手上（準備期 / 抬腿期）一律不測速
    private let poseGate = PitcherPoseGate()

    private var activeTrack: [BallDetectionSample] = []
    private var flushWorkItem: DispatchWorkItem?
    private var nextPitchNumber: Int = 1

    // Rolling 90-frame static background pixel-bin suppressor (prevents lights/logos stealing the ball)
    private var recentBinWindow: [Set<Int>] = []
    private var binCounts: [Int: Int] = [:]

    // Rolling hardware H.264 AVAssetWriter segment recorder (never interrupts camera!)
    private var assetWriter: AVAssetWriter?
    private var writerInput: AVAssetWriterInput?
    private var currentSegmentUrl: URL?
    private var segmentStartedAt: Date = Date()
    private var isRotatingWriter = false

    override init() {
        super.init()
        self.isModelReady = detector.isReady
        if !detector.hasModel(.highRes640) && detector.hasModel(.fast320) {
            self.modelMode = .fast320
        }
        self.statusText = detector.isReady ? "60fps \(modelMode.label) 待命中" : "尚未載入 CoreML 模型"
    }

    func toggleModelMode() {
        let target: DetectorModelMode = (modelMode == .highRes640) ? .fast320 : .highRes640
        guard detector.hasModel(target) else { return }
        modelMode = target
        videoQueue.async { [weak self] in
            self?.recentBinWindow.removeAll()
            self?.binCounts.removeAll()
            self?.activeTrack.removeAll()
        }
        statusText = "\(activeFps)fps \(target.label) 連續測速中"
    }

    /// 切換投球距離：18.44m（成棒/青棒）→ 16.0m（青少棒）→ 14.0m（少棒）
    func cycleDistance() {
        if abs(distanceMeters - 18.44) < 0.1 {
            distanceMeters = 16.0
        } else if abs(distanceMeters - 16.0) < 0.1 {
            distanceMeters = 14.0
        } else {
            distanceMeters = 18.44
        }
    }

    /// 切換 60fps / 120fps 高速錄影（若裝置支援 120fps，時間誤差可減半）
    func toggleHighSpeedFps() {
        preferredFps = (preferredFps == 60) ? 120 : 60
        videoQueue.async { [weak self] in
            guard let self else { return }
            self.finishCurrentSegment(saveToPhotos: false)
            if self.session.isRunning { self.session.stopRunning() }
            for input in self.session.inputs { self.session.removeInput(input) }
            for output in self.session.outputs { self.session.removeOutput(output) }
            self.configureSession()
        }
    }

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self else { return }
            guard granted else {
                DispatchQueue.main.async { self.statusText = "請在設定中開啟相機權限" }
                return
            }
            self.videoQueue.async {
                self.configureSession()
            }
        }
    }

    func stop() {
        videoQueue.async { [weak self] in
            guard let self else { return }
            self.finishCurrentSegment(saveToPhotos: false)
            if self.session.isRunning {
                self.session.stopRunning()
            }
            DispatchQueue.main.async { self.isRunning = false }
        }
    }

    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .inputPriority

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            DispatchQueue.main.async { self.statusText = "無法啟動後置鏡頭" }
            return
        }
        session.addInput(input)

        let targetFps = Double(preferredFps)
        var chosenFormat: AVCaptureDevice.Format?
        var chosenFps: Double = 60.0
        for format in device.formats {
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            guard dims.width >= 1280, dims.width <= 1920 else { continue }
            for range in format.videoSupportedFrameRateRanges {
                if range.maxFrameRate >= targetFps {
                    if chosenFormat == nil || dims.width == 1280 {
                        chosenFormat = format
                        chosenFps = min(targetFps, range.maxFrameRate)
                    }
                } else if chosenFormat == nil && range.maxFrameRate >= 60.0 {
                    chosenFormat = format
                    chosenFps = min(60.0, range.maxFrameRate)
                }
            }
        }

        if let fmt = chosenFormat {
            do {
                try device.lockForConfiguration()
                device.activeFormat = fmt
                let dur = CMTime(value: 1, timescale: CMTimeScale(chosenFps))
                device.activeVideoMinFrameDuration = dur
                device.activeVideoMaxFrameDuration = dur
                if device.isFocusModeSupported(.continuousAutoFocus) {
                    device.focusMode = .continuousAutoFocus
                }
                device.unlockForConfiguration()
            } catch {
                // fallback
            }
        }

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)

        if session.canAddOutput(output) {
            session.addOutput(output)
            if let conn = output.connection(with: .video), conn.isVideoOrientationSupported {
                conn.videoOrientation = .portrait
            }
        }

        session.commitConfiguration()
        session.startRunning()

        DispatchQueue.main.async {
            self.activeFps = Int(chosenFps.rounded())
            self.isRunning = true
            self.statusText = self.detector.isReady ? "\(self.activeFps)fps \(self.modelMode.label) 連續測速待命中" : "相機已啟動（等待模型）"
        }
    }

    // MARK: - Rolling AVAssetWriter (Zero Camera Interruption)

    private func ensureWriterReady(for sampleBuffer: CMSampleBuffer, width: Int, height: Int) {
        if assetWriter != nil || isRotatingWriter { return }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pitch_\(UUID().uuidString).mp4")
        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return }

        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 6_000_000,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            ],
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        guard writer.canAdd(input) else { return }
        writer.add(input)

        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        writer.startWriting()
        writer.startSession(atSourceTime: pts)

        self.assetWriter = writer
        self.writerInput = input
        self.currentSegmentUrl = url
        self.segmentStartedAt = Date()
    }

    private func appendToWriter(_ sampleBuffer: CMSampleBuffer, width: Int, height: Int) {
        guard autoSaveToPhotos else { return }
        ensureWriterReady(for: sampleBuffer, width: width, height: height)
        if let input = writerInput, input.isReadyForMoreMediaData, assetWriter?.status == .writing {
            input.append(sampleBuffer)
        }
        if activeTrack.isEmpty && Date().timeIntervalSince(segmentStartedAt) >= 6.0 {
            finishCurrentSegment(saveToPhotos: false)
        }
    }

    private func finishCurrentSegment(saveToPhotos: Bool, completion: ((Bool) -> Void)? = nil) {
        guard let writer = assetWriter, let url = currentSegmentUrl else {
            completion?(false)
            return
        }
        isRotatingWriter = true
        assetWriter = nil
        writerInput = nil
        currentSegmentUrl = nil

        if writer.status == .writing {
            writer.finishWriting { [weak self] in
                self?.videoQueue.async {
                    self?.isRotatingWriter = false
                }
                if saveToPhotos {
                    self?.saveVideoToPhotosLibrary(url: url, completion: completion)
                } else {
                    try? FileManager.default.removeItem(at: url)
                    completion?(false)
                }
            }
        } else {
            isRotatingWriter = false
            try? FileManager.default.removeItem(at: url)
            completion?(false)
        }
    }

    private func saveVideoToPhotosLibrary(url: URL, completion: ((Bool) -> Void)?) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                try? FileManager.default.removeItem(at: url)
                completion?(false)
                return
            }
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
            }) { success, _ in
                try? FileManager.default.removeItem(at: url)
                completion?(success)
            }
        }
    }

    // MARK: - Static-Background Bin Suppressor & Trajectory Selector

    /// 依影片 preferredTransform 推算 Vision 需要的方向
    static func orientation(for t: CGAffineTransform) -> CGImagePropertyOrientation {
        if t.a == 0 && t.b == 1 && t.c == -1 && t.d == 0 { return .right }
        if t.a == 0 && t.b == -1 && t.c == 1 && t.d == 0 { return .left }
        if t.a == -1 && t.b == 0 && t.c == 0 && t.d == -1 { return .down }
        return .up
    }

    private static func binKey(x: CGFloat, y: CGFloat) -> Int {
        let bx = Int(x / 14.0)
        let by = Int(y / 14.0)
        return (bx & 0xFFFF) << 16 | (by & 0xFFFF)
    }

    private func updateLiveStaticBins(candidates: [BallDetectionSample]) {
        var frameBins = Set<Int>()
        for c in candidates {
            frameBins.insert(Self.binKey(x: c.x, y: c.y))
        }
        recentBinWindow.append(frameBins)
        for b in frameBins {
            binCounts[b, default: 0] += 1
        }
        if recentBinWindow.count > 90 {
            let old = recentBinWindow.removeFirst()
            for b in old {
                if let cnt = binCounts[b], cnt > 1 {
                    binCounts[b] = cnt - 1
                } else {
                    binCounts.removeValue(forKey: b)
                }
            }
        }
    }

    private func selectBestDynamicCandidate(
        candidates: [BallDetectionSample],
        staticCounts: [Int: Int],
        track: [BallDetectionSample],
        frameWidth: CGFloat,
        primaryConf: Float
    ) -> BallDetectionSample? {
        let dynamic = candidates.filter { (staticCounts[Self.binKey(x: $0.x, y: $0.y)] ?? 0) < 6 }
        guard !dynamic.isEmpty else { return nil }
        let ts = dynamic[0].timestampMs
        let inFlight = track.count >= 2 && (ts - track.last!.timestampMs) <= PitchPhysicsEngine.maxSampleGapMs
        if inFlight {
            let p1 = track[track.count - 2]
            let p2 = track[track.count - 1]
            let dtPrev = max(1.0, p2.timestampMs - p1.timestampMs)
            let dtNow = ts - p2.timestampMs
            let vx = (p2.x - p1.x) / CGFloat(dtPrev)
            let vy = (p2.y - p1.y) / CGFloat(dtPrev)
            let predX = p2.x + vx * CGFloat(dtNow)
            let predY = p2.y + vy * CGFloat(dtNow)
            let radius = max(46.0, frameWidth * 0.14)
            let near = dynamic.filter { hypot($0.x - predX, $0.y - predY) <= radius }
            if let closest = near.min(by: { hypot($0.x - predX, $0.y - predY) < hypot($1.x - predX, $1.y - predY) }) {
                return closest
            }
        }
        if !(track.count >= 4 && inFlight) {
            return dynamic.first(where: { $0.confidence >= primaryConf })
        }
        return nil
    }

    // MARK: - Offline Video Import Analysis (Auto 640px -> 320px Fallback on Neural Engine)

    func analyzeImportedVideo(url: URL) {
        let preferred = self.modelMode
        DispatchQueue.main.async {
            self.statusText = "正在用 Apple Neural Engine (\(preferred.label)) 高速掃描影片..."
            self.liveDots = []
        }
        videoQueue.async { [weak self] in
            guard let self else { return }
            let modesToTry: [DetectorModelMode] = preferred == .highRes640 ? [.highRes640, .fast320] : [.fast320, .highRes640]
            let importGate = PitcherPoseGate()
            var poseScanned = false

            for mode in modesToTry where self.detector.hasModel(mode) {
                let asset = AVURLAsset(url: url)
                guard let track = asset.tracks(withMediaType: .video).first,
                      let reader = try? AVAssetReader(asset: asset) else {
                    continue
                }
                let orientation = Self.orientation(for: track.preferredTransform)
                let output = AVAssetReaderTrackOutput(
                    track: track,
                    outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
                )
                reader.add(output)
                reader.startReading()

                var perFrame: [[BallDetectionSample]] = []
                var frameTs: [Double] = []
                var counts: [Int: Int] = [:]
                var w: CGFloat = 720
                var h: CGFloat = 1280

                while let sb = output.copyNextSampleBuffer(),
                      let pb = CMSampleBufferGetImageBuffer(sb) {
                    w = CGFloat(CVPixelBufferGetWidth(pb))
                    h = CGFloat(CVPixelBufferGetHeight(pb))
                    let tsMs = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sb)) * 1000.0
                    if !poseScanned {
                        importGate.process(pixelBuffer: pb, timestampMs: tsMs, orientation: orientation)
                    }
                    let cands = self.detector.detectTopK(pixelBuffer: pb, timestampMs: tsMs, mode: mode)
                    perFrame.append(cands)
                    frameTs.append(tsMs)
                    var seen = Set<Int>()
                    for c in cands {
                        let k = Self.binKey(x: c.x, y: c.y)
                        if seen.insert(k).inserted {
                            counts[k, default: 0] += 1
                        }
                    }
                }
                poseScanned = true

                // 出手閘門：只在投手真正出手後的窗口內接受球點
                let poseSeen = frameTs.contains { importGate.isPoseAvailable(nearMs: $0) }
                if poseSeen && !importGate.releases.isEmpty {
                    for i in perFrame.indices where !importGate.isMeasurementAllowed(atMs: frameTs[i]) {
                        perFrame[i] = []
                    }
                }

                var selected: [BallDetectionSample] = []
                for cands in perFrame {
                    if let hit = self.selectBestDynamicCandidate(
                        candidates: cands,
                        staticCounts: counts,
                        track: selected,
                        frameWidth: w,
                        primaryConf: mode.defaultConfidence
                    ) {
                        selected.append(hit)
                    }
                }

                if let pitch = self.physics.finalizePitch(
                    detections: selected,
                    videoWidth: w,
                    videoHeight: h,
                    cameraAngle: self.cameraAngle,
                    distanceMeters: self.distanceMeters,
                    pitchNumber: self.nextPitchNumber
                ) {
                    self.nextPitchNumber += 1
                    DispatchQueue.main.async {
                        self.videoSize = CGSize(width: w, height: h)
                        self.currentPitch = pitch
                        self.pitchHistory.insert(pitch, at: 0)
                        self.statusText = "完成 (\(mode.label))：\(String(format: "%.1f", pitch.speedKmh)) km/h · \(pitch.pitchType)"
                        if self.voiceAnnounce {
                            self.physics.speakPitch(pitch)
                        }
                    }
                    return
                }
            }

            DispatchQueue.main.async {
                self.statusText = "影片掃描完成，未偵測到完整出手至接球軌跡"
            }
        }
    }

    // MARK: - Live Pitch Finalization

    private func scheduleFlush(videoWidth: CGFloat, videoHeight: CGFloat) {
        flushWorkItem?.cancel()
        let item = DispatchWorkItem { [weak self] in
            guard let self else { return }
            let buffered = self.poseGate.filter(self.activeTrack)
            self.activeTrack.removeAll()
            DispatchQueue.main.async { self.liveDots = [] }

            guard buffered.count >= 4 else { return }
            guard let measured = self.physics.finalizePitch(
                detections: buffered,
                videoWidth: videoWidth,
                videoHeight: videoHeight,
                cameraAngle: self.cameraAngle,
                distanceMeters: self.distanceMeters,
                pitchNumber: self.nextPitchNumber
            ) else { return }

            self.nextPitchNumber += 1
            let shouldSave = self.autoSaveToPhotos
            self.finishCurrentSegment(saveToPhotos: shouldSave) { saved in
                var finalPitch = measured
                finalPitch.savedToPhotos = saved
                DispatchQueue.main.async {
                    self.currentPitch = finalPitch
                    self.pitchHistory.insert(finalPitch, at: 0)
                    self.statusText = saved ? "已自動存入 iPhone 相簿" : "\(self.activeFps)fps \(self.modelMode.label) 連續測速中"
                    if self.voiceAnnounce {
                        self.physics.speakPitch(finalPitch)
                    }
                }
            }
        }
        flushWorkItem = item
        videoQueue.asyncAfter(deadline: .now() + 0.22, execute: item)
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let tsMs = CMTimeGetSeconds(pts) * 1000.0

        appendToWriter(sampleBuffer, width: w, height: h)
        poseGate.process(pixelBuffer: pb, timestampMs: tsMs)

        let mode = self.modelMode
        let cands = detector.detectTopK(pixelBuffer: pb, timestampMs: tsMs, mode: mode)
        updateLiveStaticBins(candidates: cands)

        var det = selectBestDynamicCandidate(
            candidates: cands,
            staticCounts: binCounts,
            track: activeTrack,
            frameWidth: CGFloat(w),
            primaryConf: mode.defaultConfidence
        )
        // 球飛向本壘縮小被全畫面漏抓時，自動以預測落點做 2.8x 局部放大追焦
        if det == nil, activeTrack.count >= 2 {
            let p1 = activeTrack[activeTrack.count - 2]
            let p2 = activeTrack[activeTrack.count - 1]
            let dtNow = tsMs - p2.timestampMs
            let dtPrev = max(1.0, p2.timestampMs - p1.timestampMs)
            if dtNow > 0 && dtNow <= PitchPhysicsEngine.maxSampleGapMs {
                let predX = p2.x + ((p2.x - p1.x) / CGFloat(dtPrev)) * CGFloat(dtNow)
                let predY = p2.y + ((p2.y - p1.y) / CGFloat(dtPrev)) * CGFloat(dtNow)
                det = detector.detectInROI(
                    pixelBuffer: pb,
                    timestampMs: tsMs,
                    mode: mode,
                    predX: predX,
                    predY: predY,
                    radius: max(50.0, CGFloat(w) * 0.14)
                )
            }
        }
        guard let det else { return }
        // 屏蔽前景投手白球褲反光誤判
        if poseGate.isInsidePitcherPants(
            x: det.x,
            y: det.y,
            atMs: tsMs,
            boxSize: max(det.width, det.height),
            trackCount: activeTrack.count
        ) { return }

        if let prev = activeTrack.last {
            let gap = det.timestampMs - prev.timestampMs
            if gap <= 0 || gap > PitchPhysicsEngine.maxSampleGapMs || det.timestampMs - activeTrack[0].timestampMs > 1600 {
                activeTrack.removeAll()
            }
        }
        activeTrack.append(det)

        let normDots = activeTrack.map { CGPoint(x: $0.x / CGFloat(w), y: $0.y / CGFloat(h)) }
        DispatchQueue.main.async {
            self.videoSize = CGSize(width: w, height: h)
            self.liveDots = normDots
        }
        scheduleFlush(videoWidth: CGFloat(w), videoHeight: CGFloat(h))
    }
}
