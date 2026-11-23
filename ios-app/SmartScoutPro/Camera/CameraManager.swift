import AVFoundation
import Foundation
import Photos
import SwiftUI

/// Continuous 60fps/120fps rear camera + CoreML Apple Neural Engine ball tracker +
/// seamless hardware H.264 auto-recording directly into iPhone Photos Library.
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
    @Published var cameraAngle: CameraAngleMode = .behindPitcher
    @Published var distanceMeters: Double = 18.44
    @Published var autoSaveToPhotos: Bool = true
    @Published var voiceAnnounce: Bool = true

    private let videoQueue = DispatchQueue(label: "com.smartscout.camera.video", qos: .userInteractive)
    private let detector = CoreMLBallDetector()
    private let physics = PitchPhysicsEngine()

    private var activeTrack: [BallDetectionSample] = []
    private var lastDetectionWallMs: Double = 0
    private var flushWorkItem: DispatchWorkItem?
    private var nextPitchNumber: Int = 1

    // Rolling hardware H.264 AVAssetWriter segment recorder (never interrupts camera!)
    private var assetWriter: AVAssetWriter?
    private var writerInput: AVAssetWriterInput?
    private var currentSegmentUrl: URL?
    private var segmentStartPts: CMTime?
    private var segmentStartedAt: Date = Date()
    private var isRotatingWriter = false

    override init() {
        super.init()
        self.isModelReady = detector.isReady
        self.statusText = detector.isReady ? "60fps AI 待命中" : "尚未載入 CoreML 模型"
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

        // Lock highest clean 60fps (or 120fps) 720p/1080p format
        var chosenFormat: AVCaptureDevice.Format?
        var chosenFps: Double = 60.0
        for format in device.formats {
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            guard dims.width >= 1280, dims.width <= 1920 else { continue }
            for range in format.videoSupportedFrameRateRanges {
                if range.maxFrameRate >= 60.0 {
                    if chosenFormat == nil || dims.width == 1280 {
                        chosenFormat = format
                        chosenFps = min(60.0, range.maxFrameRate)
                    }
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
                // fallback to default format
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
            self.statusText = self.detector.isReady ? "\(self.activeFps)fps 連續測速待命中" : "相機已啟動（等待模型）"
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
        self.segmentStartPts = pts
        self.segmentStartedAt = Date()
    }

    private func appendToWriter(_ sampleBuffer: CMSampleBuffer, width: Int, height: Int) {
        guard autoSaveToPhotos else { return }
        ensureWriterReady(for: sampleBuffer, width: width, height: height)
        if let input = writerInput, input.isReadyForMoreMediaData, assetWriter?.status == .writing {
            input.append(sampleBuffer)
        }
        // Rotate every 6 seconds while idle so clips stay compact (~3-6s) and valid from byte 0
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
        segmentStartPts = nil

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

    // MARK: - Offline Video Import Analysis (60/120/240fps from Photos)

    func analyzeImportedVideo(url: URL) {
        DispatchQueue.main.async {
            self.statusText = "正在用 Apple Neural Engine 高速掃描影片..."
            self.liveDots = []
        }
        videoQueue.async { [weak self] in
            guard let self else { return }
            let asset = AVURLAsset(url: url)
            guard let track = asset.tracks(withMediaType: .video).first,
                  let reader = try? AVAssetReader(asset: asset) else {
                DispatchQueue.main.async { self.statusText = "無法讀取該影片檔案" }
                return
            }
            let output = AVAssetReaderTrackOutput(
                track: track,
                outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            )
            reader.add(output)
            reader.startReading()

            var detections: [BallDetectionSample] = []
            var w: CGFloat = 720
            var h: CGFloat = 1280
            while let sb = output.copyNextSampleBuffer(),
                  let pb = CMSampleBufferGetImageBuffer(sb) {
                w = CGFloat(CVPixelBufferGetWidth(pb))
                h = CGFloat(CVPixelBufferGetHeight(pb))
                let tsMs = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sb)) * 1000.0
                if let det = self.detector.detect(pixelBuffer: pb, timestampMs: tsMs) {
                    detections.append(det)
                }
            }
            if let pitch = self.physics.finalizePitch(
                detections: detections,
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
                    self.statusText = "影片測速完成：\(String(format: "%.1f", pitch.speedKmh)) km/h (\(pitch.pitchType))"
                    if self.voiceAnnounce {
                        self.physics.speakPitch(pitch)
                    }
                }
            } else {
                DispatchQueue.main.async {
                    self.statusText = "影片掃描完成（偵測 \(detections.count) 幀），未構成完整投球軌跡"
                }
            }
        }
    }

    // MARK: - Live Pitch Finalization

    private func scheduleFlush(videoWidth: CGFloat, videoHeight: CGFloat) {
        flushWorkItem?.cancel()
        let item = DispatchWorkItem { [weak self] in
            guard let self else { return }
            let buffered = self.activeTrack
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
                    self.statusText = saved ? "已自動存入 iPhone 相簿" : "\(self.activeFps)fps 連續測速中"
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

        guard let det = detector.detect(pixelBuffer: pb, timestampMs: tsMs) else { return }

        if let prev = activeTrack.last {
            let gap = det.timestampMs - prev.timestampMs
            let jump = hypot(det.x - prev.x, det.y - prev.y)
            if gap <= 0 || gap > PitchPhysicsEngine.maxSampleGapMs || det.timestampMs - activeTrack[0].timestampMs > 1600 {
                activeTrack.removeAll()
            } else if jump > CGFloat(w) * 0.22 {
                return
            }
        }
        activeTrack.append(det)
        lastDetectionWallMs = tsMs

        let normDots = activeTrack.map { CGPoint(x: $0.x / CGFloat(w), y: $0.y / CGFloat(h)) }
        DispatchQueue.main.async {
            self.videoSize = CGSize(width: w, height: h)
            self.liveDots = normDots
        }
        scheduleFlush(videoWidth: CGFloat(w), videoHeight: CGFloat(h))
    }
}
