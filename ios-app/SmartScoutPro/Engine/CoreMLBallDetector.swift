import CoreImage
import CoreML
import CoreVideo
import Foundation

/// Runs YOLO11n (320x320) baseball detection on Apple Neural Engine (ANE) at ~2-4ms/frame.
final class CoreMLBallDetector {
    private let model: MLModel?
    private let inputSize: Int = 320
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let inputFeatureName: String
    private let outputFeatureName: String

    var isReady: Bool { model != nil }

    init() {
        let config = MLModelConfiguration()
        config.computeUnits = .all // Apple Neural Engine + GPU + CPU

        var loaded: MLModel?
        if let compiledUrl = Bundle.main.url(forResource: "BaseballDetector", withExtension: "mlmodelc") {
            loaded = try? MLModel(contentsOf: compiledUrl, configuration: config)
        } else if let pkgUrl = Bundle.main.url(forResource: "BaseballDetector", withExtension: "mlpackage"),
                  let compiledUrl = try? MLModel.compileModel(at: pkgUrl) {
            loaded = try? MLModel(contentsOf: compiledUrl, configuration: config)
        }
        self.model = loaded
        self.inputFeatureName = loaded?.modelDescription.inputDescriptionsByName.keys.first ?? "image"
        self.outputFeatureName = loaded?.modelDescription.outputDescriptionsByName.keys.first ?? "var_916"
    }

    func detect(
        pixelBuffer: CVPixelBuffer,
        timestampMs: Double,
        confThreshold: Float = 0.22
    ) -> BallDetectionSample? {
        guard let model else { return nil }
        let srcW = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let srcH = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        guard srcW > 0, srcH > 0 else { return nil }

        let size = CGFloat(inputSize)
        let scale = min(size / srcW, size / srcH)
        let newW = (srcW * scale).rounded()
        let newH = (srcH * scale).rounded()
        let padX = floor((size - newW) / 2.0)
        let padY = floor((size - newH) / 2.0)

        guard let letterboxed = makeLetterboxedPixelBuffer(
            from: pixelBuffer,
            targetSize: inputSize,
            scale: scale,
            padX: padX,
            padY: padY
        ) else { return nil }

        guard let provider = try? MLDictionaryFeatureProvider(
            dictionary: [inputFeatureName: MLFeatureValue(pixelBuffer: letterboxed)]
        ),
        let output = try? model.prediction(from: provider),
        let multiArray = output.featureValue(for: outputFeatureName)?.multiArrayValue else {
            return nil
        }

        // Expect YOLO11 single-class shape [1, 5, N] (e.g. N=2100 for 320x320)
        let shape = multiArray.shape.map { $0.intValue }
        guard shape.count == 3, shape[1] >= 5 else { return nil }
        let numPreds = shape[2]
        let strideC = multiArray.strides[1].intValue
        let strideN = multiArray.strides[2].intValue

        var bestScore: Float = confThreshold
        var bestIdx: Int = -1

        if multiArray.dataType == .float32 {
            let ptr = multiArray.dataPointer.bindMemory(to: Float32.self, capacity: multiArray.count)
            let scoreBase = 4 * strideC
            for i in 0..<numPreds {
                let sc = ptr[scoreBase + i * strideN]
                if sc > bestScore {
                    bestScore = sc
                    bestIdx = i
                }
            }
            guard bestIdx >= 0 else { return nil }
            let cx320 = CGFloat(ptr[0 * strideC + bestIdx * strideN])
            let cy320 = CGFloat(ptr[1 * strideC + bestIdx * strideN])
            let bw320 = CGFloat(ptr[2 * strideC + bestIdx * strideN])
            let bh320 = CGFloat(ptr[3 * strideC + bestIdx * strideN])
            return mapBack(cx320, cy320, bw320, bh320, bestScore, srcW, srcH, newW, newH, padX, padY, timestampMs)
        } else if multiArray.dataType == .float16 {
            #if arch(arm64)
            let ptr = multiArray.dataPointer.bindMemory(to: Float16.self, capacity: multiArray.count)
            let scoreBase = 4 * strideC
            for i in 0..<numPreds {
                let sc = Float(ptr[scoreBase + i * strideN])
                if sc > bestScore {
                    bestScore = sc
                    bestIdx = i
                }
            }
            guard bestIdx >= 0 else { return nil }
            let cx320 = CGFloat(ptr[0 * strideC + bestIdx * strideN])
            let cy320 = CGFloat(ptr[1 * strideC + bestIdx * strideN])
            let bw320 = CGFloat(ptr[2 * strideC + bestIdx * strideN])
            let bh320 = CGFloat(ptr[3 * strideC + bestIdx * strideN])
            return mapBack(cx320, cy320, bw320, bh320, bestScore, srcW, srcH, newW, newH, padX, padY, timestampMs)
            #else
            return nil
            #endif
        } else {
            // Float16 / Double safe fallback
            for i in 0..<numPreds {
                let sc = multiArray[[0, 4, i] as [NSNumber]].floatValue
                if sc > bestScore {
                    bestScore = sc
                    bestIdx = i
                }
            }
            guard bestIdx >= 0 else { return nil }
            let cx320 = CGFloat(multiArray[[0, 0, bestIdx] as [NSNumber]].floatValue)
            let cy320 = CGFloat(multiArray[[0, 1, bestIdx] as [NSNumber]].floatValue)
            let bw320 = CGFloat(multiArray[[0, 2, bestIdx] as [NSNumber]].floatValue)
            let bh320 = CGFloat(multiArray[[0, 3, bestIdx] as [NSNumber]].floatValue)
            return mapBack(cx320, cy320, bw320, bh320, bestScore, srcW, srcH, newW, newH, padX, padY, timestampMs)
        }
    }

    private func mapBack(
        _ cx320: CGFloat, _ cy320: CGFloat, _ bw320: CGFloat, _ bh320: CGFloat,
        _ score: Float,
        _ srcW: CGFloat, _ srcH: CGFloat,
        _ newW: CGFloat, _ newH: CGFloat,
        _ padX: CGFloat, _ padY: CGFloat,
        _ timestampMs: Double
    ) -> BallDetectionSample? {
        let sx = srcW / max(1, newW)
        let sy = srcH / max(1, newH)
        let cx = (cx320 - padX) * sx
        let cy = (cy320 - padY) * sy
        let bw = bw320 * sx
        let bh = bh320 * sy
        let ny = cy / srcH
        guard ny >= 0.08, ny <= 0.78, bw <= srcW * 0.14, bh <= srcH * 0.10 else {
            return nil
        }
        return BallDetectionSample(
            timestampMs: timestampMs,
            x: cx, y: cy, width: bw, height: bh, confidence: score
        )
    }

    private func makeLetterboxedPixelBuffer(
        from src: CVPixelBuffer,
        targetSize: Int,
        scale: CGFloat,
        padX: CGFloat,
        padY: CGFloat
    ) -> CVPixelBuffer? {
        var dst: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            targetSize,
            targetSize,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &dst
        )
        guard let outputBuffer = dst else { return nil }

        let gray = CIImage(color: CIColor(red: 114.0/255.0, green: 114.0/255.0, blue: 114.0/255.0))
            .cropped(to: CGRect(x: 0, y: 0, width: targetSize, height: targetSize))
        let scaled = CIImage(cvPixelBuffer: src)
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(translationX: padX, y: padY))
        let composed = scaled.composited(over: gray)
        ciContext.render(composed, to: outputBuffer)
        return outputBuffer
    }
}
