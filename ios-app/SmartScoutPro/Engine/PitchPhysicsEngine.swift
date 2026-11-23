import Foundation
import AVFoundation
import SwiftUI

enum CameraAngleMode: String, CaseIterable, Identifiable {
    case behindPitcher = "behind_pitcher"
    case behindCatcher = "behind_catcher"
    var id: String { rawValue }
    var label: String {
        switch self {
        case .behindPitcher: return "投手正後方"
        case .behindCatcher: return "捕手正後方"
        }
    }
}

struct BallDetectionSample: Identifiable {
    let id = UUID()
    let timestampMs: Double
    let x: CGFloat
    let y: CGFloat
    let width: CGFloat
    let height: CGFloat
    let confidence: Float
}

struct RibbonPathData {
    let startX: CGFloat
    let startY: CGFloat
    let ctrlX: CGFloat
    let ctrlY: CGFloat
    let endX: CGFloat
    let endY: CGFloat
}

struct MeasuredPitch: Identifiable {
    let id = UUID()
    let pitchNumber: Int
    let timestamp: Date
    let speedKmh: Double
    let plateSpeedKmh: Double
    let pitchType: String
    let verticalBreakCm: Double
    let horizontalBreakCm: Double
    let totalSpinRpm: Int
    let activeSpinRpm: Int
    let efficiencyPct: Int
    let spinDirection: String
    let gyroDegree: Int
    let isStrike: Bool
    let flightTimeMs: Double
    let detectionCount: Int
    let cameraAngle: CameraAngleMode
    let ribbon: RibbonPathData
    var savedToPhotos: Bool = false

    var accentColor: Color {
        switch pitchType {
        case "四縫線快速球": return Color(red: 0.94, green: 0.27, blue: 0.27)
        case "伸卡球": return Color(red: 0.98, green: 0.45, blue: 0.09)
        case "卡特球": return Color(red: 0.66, green: 0.33, blue: 0.97)
        case "滑球": return Color(red: 0.92, green: 0.70, blue: 0.03)
        case "曲球": return Color(red: 0.22, green: 0.74, blue: 0.97)
        case "指叉球": return Color(red: 0.93, green: 0.28, blue: 0.60)
        case "變速球": return Color(red: 0.06, green: 0.73, blue: 0.51)
        default: return Color(red: 0.94, green: 0.27, blue: 0.27)
        }
    }
}

final class PitchPhysicsEngine {
    static let maxSampleGapMs: Double = 145.0
    private let synthesizer = AVSpeechSynthesizer()

    func speakPitch(_ pitch: MeasuredPitch) {
        let speedRounded = Int(pitch.speedKmh.rounded())
        let utterance = AVSpeechUtterance(string: "\(speedRounded) 公里，\(pitch.pitchType)")
        utterance.voice = AVSpeechSynthesisVoice(language: "zh-TW")
        utterance.rate = 0.54
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        synthesizer.speak(utterance)
    }

    func cleanAndSmooth(_ track: [BallDetectionSample], videoWidth: CGFloat) -> [BallDetectionSample] {
        guard track.count >= 4 else { return track }
        let maxJump = max(24.0, videoWidth * 0.16)
        var cleaned: [BallDetectionSample] = [track[0]]
        for i in 1..<(track.count - 1) {
            let prev = cleaned.last!
            let curr = track[i]
            let next = track[i + 1]
            let dtTotal = max(1.0, next.timestampMs - prev.timestampMs)
            let alpha = CGFloat((curr.timestampMs - prev.timestampMs) / dtTotal)
            let ix = prev.x + (next.x - prev.x) * alpha
            let iy = prev.y + (next.y - prev.y) * alpha
            let dev = hypot(curr.x - ix, curr.y - iy)
            let step = hypot(curr.x - prev.x, curr.y - prev.y)
            if dev <= maxJump * 0.65 && step <= maxJump {
                cleaned.append(curr)
            }
        }
        let last = track.last!
        if hypot(last.x - cleaned.last!.x, last.y - cleaned.last!.y) <= maxJump {
            cleaned.append(last)
        }
        return cleaned.count >= 4 ? cleaned : track
    }

    func classifyPitchType(speedKmh: Double, hbCm: Double, vbCm: Double) -> String {
        let absHb = abs(hbCm)
        if speedKmh < 112 && vbCm <= -12 { return "曲球" }
        if absHb >= 16 && vbCm <= 14 && speedKmh < 132 { return "滑球" }
        if vbCm <= -10 && absHb < 12 && speedKmh >= 112 { return "指叉球" }
        if speedKmh < 118 && hbCm >= 8 && vbCm < 16 { return "變速球" }
        if speedKmh >= 118 && hbCm >= 14 && vbCm < 24 { return "伸卡球" }
        if speedKmh >= 118 && hbCm <= -10 && vbCm >= 8 { return "卡特球" }
        return "四縫線快速球"
    }

    private func spinAxisToClock(_ deg: Double) -> String {
        let norm = deg.truncatingRemainder(dividingBy: 360.0)
        let pos = norm < 0 ? norm + 360.0 : norm
        let totalMins = Int(((pos / 360.0) * 720.0).rounded()) % 720
        var hh = totalMins / 60
        let mm = ((totalMins % 60) / 15) * 15
        if hh == 0 { hh = 12 }
        return String(format: "%d:%02d", hh, mm)
    }

    func finalizePitch(
        detections: [BallDetectionSample],
        videoWidth: CGFloat,
        videoHeight: CGFloat,
        cameraAngle: CameraAngleMode,
        distanceMeters: Double,
        pitchNumber: Int
    ) -> MeasuredPitch? {
        guard detections.count >= 5, videoWidth > 0, videoHeight > 0 else { return nil }
        let flightDistance = max(5.0, distanceMeters - 1.75)

        let points = detections.map { CGPoint(x: $0.x / videoWidth, y: $0.y / videoHeight) }
        struct StepInfo {
            let dt: Double
            let valid: Bool
            let rate: Double
        }
        var steps: [StepInfo] = []
        for i in 0..<(detections.count - 1) {
            let dt = detections[i + 1].timestampMs - detections[i].timestampMs
            let dx = Double(points[i + 1].x - points[i].x)
            let dy = Double(points[i + 1].y - points[i].y)
            let valid = dt > 0 && dt <= Self.maxSampleGapMs &&
                detections[i].confidence >= 0.18 && detections[i + 1].confidence >= 0.18
            let rate = dt > 0 ? hypot(dx, dy) / dt : 0.0
            steps.append(StepInfo(dt: dt, valid: valid, rate: rate))
        }
        func depthStationary(_ idx: Int) -> Bool {
            idx >= 0 && idx < steps.count && steps[idx].valid && steps[idx].rate < 0.000038
        }

        // Find longest finalized valid flight segment
        var bestTrack: [BallDetectionSample]?
        var bestFlightMs: Double = 0
        var bestSpeed: Double = 0

        var segStart = 0
        for i in 0...steps.count {
            let isBreak = (i == steps.count) || !steps[i].valid
            if !isBreak { continue }
            let segEnd = i
            if segEnd - segStart >= 4 {
                var s = segStart
                var e = segEnd
                while s < e && depthStationary(s) { s += 1 }
                while e > s && depthStationary(e - 1) { e -= 1 }
                if e - s >= 4 {
                    let firstDet = detections[s]
                    let lastDet = detections[e]
                    let sSize = max(firstDet.width, firstDet.height)
                    let eSize = max(lastDet.width, lastDet.height)
                    let sizeRatio = max(sSize / max(1, eSize), eSize / max(1, sSize))
                    let span = hypot(Double(points[e].x - points[s].x), Double(points[e].y - points[s].y))
                    let isDepth = sizeRatio >= 1.32 && span >= 0.014
                    let isOblique = sizeRatio >= 1.12 && span >= 0.12
                    let relPad = (s == segStart && steps[s].dt > 45) ? 33.37 : 0.0
                    let catPad = (e == segEnd && steps[e - 1].dt > 45 && sizeRatio >= 1.50) ? 33.37 : 0.0
                    let flightMs = lastDet.timestampMs - firstDet.timestampMs - relPad - catPad
                    if flightMs >= 320 && flightMs <= 920 {
                        let spd = (flightDistance * 3600.0) / flightMs / 0.955
                        if (isDepth || isOblique) && spd >= 45 && spd <= 158 {
                            bestTrack = Array(detections[s...e])
                            bestFlightMs = (flightMs * 10).rounded() / 10
                            bestSpeed = (spd * 10).rounded() / 10
                            break
                        }
                    }
                }
            }
            segStart = i + 1
        }

        // Clean fallback when track itself is a single short valid flight
        if bestTrack == nil {
            let cleaned = cleanAndSmooth(detections, videoWidth: videoWidth)
            if cleaned.count >= 4 {
                let flightMs = cleaned.last!.timestampMs - cleaned.first!.timestampMs
                if flightMs >= 360 && flightMs <= 880 {
                    let spd = (flightDistance * 3600.0) / flightMs / 0.955
                    if spd >= 45 && spd <= 158 {
                        bestTrack = cleaned
                        bestFlightMs = (flightMs * 10).rounded() / 10
                        bestSpeed = (spd * 10).rounded() / 10
                    }
                }
            }
        }

        guard let rawTrack = bestTrack else { return nil }
        let track = cleanAndSmooth(rawTrack, videoWidth: videoWidth)
        let first = track.first!
        let last = track.last!
        let midIdx = track.count / 2
        let midLo = max(0, midIdx - 1)
        let midHi = min(track.count - 1, midIdx + 1)
        let midSlice = track[midLo...midHi]
        let midX = midSlice.reduce(CGFloat(0)) { $0 + $1.x } / CGFloat(midSlice.count)
        let midY = midSlice.reduce(CGFloat(0)) { $0 + $1.y } / CGFloat(midSlice.count)

        let sx = first.x / videoWidth
        let sy = first.y / videoHeight
        let ex = last.x / videoWidth
        let ey = last.y / videoHeight
        let cx = max(0, min(1, 2 * midX / videoWidth - (sx + ex) / 2))
        let cy = max(0, min(1, 2 * midY / videoHeight - (sy + ey) / 2))

        let behindPitcher = cameraAngle == .behindPitcher
        let lineMidX = (first.x + last.x) * 0.5
        let lineMidY = (first.y + last.y) * 0.5
        let pxToCm = 86.0 / max(110.0, Double(videoWidth) * 0.2)
        let signX: Double = behindPitcher ? -1.0 : 1.0
        let rawHb = Double(midX - lineMidX) * pxToCm * signX * 1.8
        let flightSec = bestFlightMs / 1000.0
        let gravitySagCm = 0.125 * 981.0 * flightSec * flightSec
        let observedArcCm = Double(lineMidY - midY) * pxToCm * 1.6
        let rawVb = (gravitySagCm - observedArcCm) * 1.35 + 18.0

        let hbCm = (max(-58.0, min(58.0, rawHb)) * 10).rounded() / 10
        let vbCm = (max(-58.0, min(68.0, rawVb)) * 10).rounded() / 10
        let plateX = max(-42.0, min(42.0, (Double(ex) - 0.48) * 140.0))
        let plateY = max(25.0, min(130.0, 76.0 - (Double(ey) - 0.50) * 120.0))
        let isStrike = abs(plateX) <= 21.6 && abs(plateY - 76.0) <= 30.0

        let totalBreak = hypot(hbCm, vbCm)
        let activeSpin = max(900, min(3100, Int((620 + totalBreak * 32.0 + bestSpeed * 4.2).rounded())))
        let effPct = max(32, min(98, Int((54 + min(42, totalBreak * 0.85)).rounded())))
        let totalSpin = Int((Double(activeSpin) / (Double(effPct) / 100.0)).rounded())
        let gyroDeg = Int((acos(max(-1.0, min(1.0, Double(effPct) / 100.0))) * 180.0 / .pi).rounded())
        let axisDeg = (atan2(hbCm, vbCm) * 180.0 / .pi + 360.0).truncatingRemainder(dividingBy: 360.0)
        let spinClock = spinAxisToClock(axisDeg)
        let plateSpeed = (bestSpeed * 0.916 * 10).rounded() / 10
        let pitchType = classifyPitchType(speedKmh: bestSpeed, hbCm: hbCm, vbCm: vbCm)

        return MeasuredPitch(
            pitchNumber: pitchNumber,
            timestamp: Date(),
            speedKmh: bestSpeed,
            plateSpeedKmh: plateSpeed,
            pitchType: pitchType,
            verticalBreakCm: vbCm,
            horizontalBreakCm: hbCm,
            totalSpinRpm: totalSpin,
            activeSpinRpm: activeSpin,
            efficiencyPct: effPct,
            spinDirection: spinClock,
            gyroDegree: gyroDeg,
            isStrike: isStrike,
            flightTimeMs: bestFlightMs,
            detectionCount: track.count,
            cameraAngle: cameraAngle,
            ribbon: RibbonPathData(startX: sx, startY: sy, ctrlX: cx, ctrlY: cy, endX: ex, endY: ey)
        )
    }
}
