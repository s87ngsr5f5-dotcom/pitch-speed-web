import CoreGraphics
import CoreVideo
import Foundation
import Vision

/// 投手動作出手閘門（Apple Vision 17 點人體骨架，與桌面版 pitcher_pose_gate.py 同邏輯）
///
/// 四階段：IDLE → WINDUP（抬腿 / 拉臂）→ RELEASE（投球手腕甩動峰值）→ FOLLOW_THROUGH
/// 規則：
///  1. 出手 = 投球手腕速度（以軀幹高度 / 秒正規化）× 手腕高度加權 的局部峰值，且 ≥ 門檻。
///  2. 出手前 1.2 秒內必須看到準備動作（抬腿或手臂明顯移動）至少 3 個樣本，否則視為假出手。
///  3. 兩次出手至少間隔 1.5 秒。
///  4. 測速只接受 [出手 − 70ms, 出手 + 1150ms] 內的球點。
///  5. 若最近 2 秒完全看不到投手（例如捕手後方遠距離），閘門自動放行，不擋測速。
struct PoseReleaseEvent {
    let releaseMs: Double
    let throwingHand: String  // "R" / "L"
    let score: Double
}

final class PitcherPoseGate {
    static let windowBeforeMs: Double = 70
    static let windowAfterMs: Double = 1150
    static let releaseScoreThreshold: Double = 2.0
    static let windupLookbackMs: Double = 1200
    static let minWindupSamples = 3
    static let minReleaseGapMs: Double = 1500
    static let poseIntervalMs: Double = 30  // ~33Hz 骨架推論，60fps 相機下每 2 幀做一次

    private struct Sample {
        let ts: Double
        let personSeen: Bool
        let rSpeed: Double
        let lSpeed: Double
        let rAbove: Double
        let lAbove: Double
        let kneeLift: Double
    }

    private struct Joints {
        var rWrist: CGPoint?
        var lWrist: CGPoint?
        var rShoulder: CGPoint?
        var lShoulder: CGPoint?
        var rHip: CGPoint?
        var lHip: CGPoint?
        var rKnee: CGPoint?
        var lKnee: CGPoint?
        var rAnkle: CGPoint?
        var lAnkle: CGPoint?
    }

    private struct PantsBox {
        let x1: CGFloat
        let y1: CGFloat
        let x2: CGFloat
        let y2: CGFloat
        let torso: CGFloat
        let ts: Double
    }

    private struct HandState {
        let wx: CGFloat
        let wy: CGFloat
        let torso: CGFloat
        let speed: Double
        let ts: Double
    }

    private var samples: [Sample] = []
    private var lastJoints: Joints?
    private var lastJointsTs: Double = -1
    private var lastPoseTs: Double = -1_000_000
    private var lastPersonSeenTs: Double = -1_000_000
    private var lastPantsBox: PantsBox?
    private var lastHand: HandState?
    private(set) var releases: [PoseReleaseEvent] = []

    func isBallStillInHand(x: CGFloat, y: CGFloat, atMs ts: Double) -> Bool {
        guard let h = lastHand, abs(ts - h.ts) <= 280 else { return false }
        guard hypot(x - h.wx, y - h.wy) <= h.torso * 0.24 else { return false }
        let recentRelease = releases.contains { ts >= $0.releaseMs - 80 && ts <= $0.releaseMs + 450 }
        if recentRelease || h.speed >= 3.2 { return false }
        return true
    }

    func isInsidePitcherPants(x: CGFloat, y: CGFloat, atMs ts: Double, boxSize: CGFloat, trackCount: Int) -> Bool {
        guard let b = lastPantsBox, abs(ts - b.ts) <= 400 else { return false }
        guard x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2 else { return false }
        return trackCount < 3 || boxSize >= b.torso * 0.14
    }
    private let request = VNDetectHumanBodyPoseRequest()

    func reset() {
        samples.removeAll()
        releases.removeAll()
        lastJoints = nil
        lastJointsTs = -1
        lastPoseTs = -1_000_000
        lastPersonSeenTs = -1_000_000
        lastPantsBox = nil
        lastHand = nil
    }

    /// 每一幀呼叫；內部自動節流到 ~33Hz。
    func process(pixelBuffer: CVPixelBuffer, timestampMs: Double, orientation: CGImagePropertyOrientation = .up) {
        if timestampMs - lastPoseTs < Self.poseIntervalMs { return }
        lastPoseTs = timestampMs

        let bufW = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let bufH = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let rotatedSet: [CGImagePropertyOrientation] = [.left, .right, .leftMirrored, .rightMirrored]
        let rotated = rotatedSet.contains(orientation)
        let imgW = rotated ? bufH : bufW
        let imgH = rotated ? bufW : bufH
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
        var joints: Joints?
        do {
            try handler.perform([request])
            if let obs = pickPitcher(request.results ?? []) {
                joints = extractJoints(obs, imgW: imgW, imgH: imgH)
            }
        } catch {
            joints = nil
        }
        ingest(joints: joints, timestampMs: timestampMs)
    }

    /// 這個時間點是否允許開始 / 繼續測速
    func isMeasurementAllowed(atMs ts: Double) -> Bool {
        if !isPoseAvailable(nearMs: ts) || releases.isEmpty { return true }
        return releases.contains { ts >= $0.releaseMs - Self.windowBeforeMs && ts <= $0.releaseMs + Self.windowAfterMs }
    }

    /// 投手是否在畫面中（最近 2 秒內有骨架）；看不到投手時閘門放行
    func isPoseAvailable(nearMs ts: Double) -> Bool {
        return abs(ts - lastPersonSeenTs) <= 2000
    }

    /// 依出手事件裁切「出手前」準備期雜點，完整保留出手後進壘尾段；未捕捉到對應出手峰值時交由物理引擎把關
    func filter(_ detections: [BallDetectionSample]) -> [BallDetectionSample] {
        guard let first = detections.first, let last = detections.last, !releases.isEmpty else { return detections }
        if !isPoseAvailable(nearMs: first.timestampMs) && !isPoseAvailable(nearMs: last.timestampMs) {
            return detections
        }
        let matching = releases.filter {
            $0.releaseMs >= first.timestampMs - 150 && $0.releaseMs <= last.timestampMs - 180
        }
        guard let rel = matching.last else { return detections }
        let kept = detections.filter { $0.timestampMs >= rel.releaseMs - Self.windowBeforeMs }
        return kept.count >= 4 ? kept : detections
    }

    // MARK: - Internals

    private func pickPitcher(_ observations: [VNHumanBodyPoseObservation]) -> VNHumanBodyPoseObservation? {
        var best: VNHumanBodyPoseObservation?
        var bestScore: CGFloat = -1
        for obs in observations {
            guard let pts = try? obs.recognizedPoints(.all) else { continue }
            let valid = pts.values.filter { $0.confidence > 0.25 }
            guard valid.count >= 6 else { continue }
            let xs = valid.map { $0.location.x }
            let ys = valid.map { $0.location.y }
            let bw = (xs.max() ?? 0) - (xs.min() ?? 0)
            let bh = (ys.max() ?? 0) - (ys.min() ?? 0)
            guard bh >= 0.16 else { continue }
            let cx = ((xs.max() ?? 0) + (xs.min() ?? 0)) * 0.5
            let centerBias = max(0.2, 1.0 - abs(cx - 0.48) * 1.4)
            let score = bw * bh * centerBias
            if score > bestScore {
                bestScore = score
                best = obs
            }
        }
        return best
    }

    private func extractJoints(_ obs: VNHumanBodyPoseObservation, imgW: CGFloat, imgH: CGFloat) -> Joints {
        // Vision 座標：正規化、左下為原點；轉成像素且 y 向下（與桌面版一致）
        func p(_ name: VNHumanBodyPoseObservation.JointName) -> CGPoint? {
            guard let pt = try? obs.recognizedPoint(name), pt.confidence > 0.25 else { return nil }
            return CGPoint(x: pt.location.x * imgW, y: (1 - pt.location.y) * imgH)
        }
        return Joints(
            rWrist: p(.rightWrist), lWrist: p(.leftWrist),
            rShoulder: p(.rightShoulder), lShoulder: p(.leftShoulder),
            rHip: p(.rightHip), lHip: p(.leftHip),
            rKnee: p(.rightKnee), lKnee: p(.leftKnee),
            rAnkle: p(.rightAnkle), lAnkle: p(.leftAnkle)
        )
    }

    private static func midY(_ a: CGPoint?, _ b: CGPoint?) -> CGFloat? {
        let vals = [a, b].compactMap { $0?.y }
        guard !vals.isEmpty else { return nil }
        return vals.reduce(0, +) / CGFloat(vals.count)
    }

    private func ingest(joints: Joints?, timestampMs ts: Double) {
        guard let j = joints else {
            samples.append(Sample(ts: ts, personSeen: false, rSpeed: 0, lSpeed: 0, rAbove: -1, lAbove: -1, kneeLift: 0))
            lastJoints = nil
            trimAndDetect()
            return
        }
        lastPersonSeenTs = ts
        let shY = Self.midY(j.lShoulder, j.rShoulder)
        let hpY = Self.midY(j.lHip, j.rHip)
        let torso: CGFloat = {
            if let s = shY, let h = hpY { return max(24, abs(h - s)) }
            return 120
        }()

        let rAbove: Double = {
            guard let w = j.rWrist, let s = j.rShoulder else { return -1 }
            return Double((s.y - w.y) / torso)
        }()
        let lAbove: Double = {
            guard let w = j.lWrist, let s = j.lShoulder else { return -1 }
            return Double((s.y - w.y) / torso)
        }()

        var knee: Double = 0
        if let h = hpY {
            let knees = [j.lKnee, j.rKnee].compactMap { $0?.y }
            if let highest = knees.min() {
                knee = max(0, 1.0 - Double((highest - h) / max(24, torso * 1.1)))
            }
        }

        var rs = 0.0, ls = 0.0
        if let prev = lastJoints, lastJointsTs > 0 {
            let dt = max(0.008, (ts - lastJointsTs) / 1000.0)
            if let a = j.rWrist, let b = prev.rWrist {
                rs = min(18, Double(hypot(a.x - b.x, a.y - b.y) / torso) / dt)
            }
            if let a = j.lWrist, let b = prev.lWrist {
                ls = min(18, Double(hypot(a.x - b.x, a.y - b.y) / torso) / dt)
            }
        }
        lastJoints = j
        lastJointsTs = ts
        if torso >= 48 {
            if rs >= ls, let rw = j.rWrist {
                lastHand = HandState(wx: rw.x, wy: rw.y, torso: torso, speed: rs, ts: ts)
            } else if let lw = j.lWrist {
                lastHand = HandState(wx: lw.x, wy: lw.y, torso: torso, speed: ls, ts: ts)
            } else if let rw = j.rWrist {
                lastHand = HandState(wx: rw.x, wy: rw.y, torso: torso, speed: rs, ts: ts)
            }
        }
        let lowerPts = [j.lHip, j.rHip, j.lKnee, j.rKnee, j.lAnkle, j.rAnkle].compactMap { $0 }
        if lowerPts.count >= 2, let h = hpY, torso >= 48 {
            let xs = lowerPts.map { $0.x }
            let ys = lowerPts.map { $0.y }
            let padX = torso * 0.22
            lastPantsBox = PantsBox(
                x1: (xs.min() ?? 0) - padX,
                y1: h - torso * 0.05,
                x2: (xs.max() ?? 0) + padX,
                y2: (ys.max() ?? h) + torso * 0.12,
                torso: torso,
                ts: ts
            )
        }
        samples.append(Sample(ts: ts, personSeen: true, rSpeed: rs, lSpeed: ls, rAbove: rAbove, lAbove: lAbove, kneeLift: knee))
        trimAndDetect()
    }

    private func trimAndDetect() {
        if let last = samples.last {
            samples.removeAll { last.ts - $0.ts > 4000 }
            releases.removeAll { last.ts - $0.releaseMs > 8000 }
        }
        guard samples.count >= 5 else { return }

        // 投球手 = 高位甩臂能量較大者
        var rE = 0.0, lE = 0.0
        for s in samples {
            rE += s.rSpeed * (s.rAbove > 0.05 ? 1.2 : 0.3)
            lE += s.lSpeed * (s.lAbove > 0.05 ? 1.2 : 0.3)
        }
        let useR = rE >= lE
        let n = samples.count
        func speed(_ i: Int) -> Double { useR ? samples[i].rSpeed : samples[i].lSpeed }
        func above(_ i: Int) -> Double { useR ? samples[i].rAbove : samples[i].lAbove }
        func smooth(_ i: Int) -> Double {
            let lo = max(0, i - 1), hi = min(n - 1, i + 1)
            var sum = 0.0
            for k in lo...hi { sum += speed(k) }
            return sum / Double(hi - lo + 1)
        }
        func score(_ i: Int) -> Double {
            let a = above(i)
            let s = smooth(i)
            if a <= -0.25 { return s * 0.2 }
            return s * (1.0 + max(-0.4, min(1.0, a + 0.15)))
        }

        // 檢查倒數第 2 個樣本是否為局部峰值（需要下一個樣本確認）
        let c = n - 2
        let sc = score(c)
        guard sc >= Self.releaseScoreThreshold, sc >= score(c - 1), sc > score(c + 1) else { return }
        let peakTs = samples[c].ts
        // 準備動作守門：出手前必須看到抬腿或手臂移動
        var windup = 0
        for i in 0..<max(0, c - 1) {
            let s = samples[i]
            guard s.personSeen, peakTs - s.ts <= Self.windupLookbackMs else { continue }
            if s.kneeLift > 0.35 || smooth(i) > 0.8 { windup += 1 }
        }
        guard windup >= Self.minWindupSamples else { return }

        let ev = PoseReleaseEvent(releaseMs: peakTs, throwingHand: useR ? "R" : "L", score: sc)
        if let lastRel = releases.last, peakTs - lastRel.releaseMs < Self.minReleaseGapMs {
            // 若 1.5 秒內出現明顯更強的真正甩臂出手峰值，立即取代先前的拉臂小峰
            if sc > lastRel.score * 1.25 {
                releases[releases.count - 1] = ev
            }
            return
        }
        releases.append(ev)
    }
}
