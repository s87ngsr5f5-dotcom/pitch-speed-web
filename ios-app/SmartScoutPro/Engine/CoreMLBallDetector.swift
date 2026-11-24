import CoreImage
import CoreML
import CoreVideo
import Foundation

enum DetectorModelMode: String, CaseIterable, Identifiable {
    case highRes640 = "640"
    case fast320 = "320"
    var id: String { rawValue }
    var label: String {
        switch self {
        case .highRes640: return "640 高精度"
        case .fast320: return "320 極速"
        }
    }
    var resourceName: String {
        switch self {
        case .highRes640: return "BaseballDetector640"
        case .fast320: return "BaseballDetector"
        }
    }
    var defaultSize: Int {
        switch self {
        case .highRes640: return 640
        case .fast320: return 320
        }
    }
    var defaultConfidence: Float {
        switch self {
        case .highRes640: return 0.28
        case .fast320: return 0.22
        }
    }
}

private struct LoadedCoreMLModel {
    let model: MLModel
    let inputSize: Int
    let inputFeatureName: String
    let outputFeatureName: String
}

/// Runs YOLO11s (640x640) or YOLO11n (320x320) baseball detection on Apple Neural Engine (ANE).
final class CoreMLBallDetector {
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private var loadedModels: [DetectorModelMode: LoadedCoreMLModel] = [:]

    var isReady: Bool { !loadedModels.isEmpty }

    func hasModel(_ mode: DetectorModelMode) -> Bool {
        loadedModels[mode] != nil
    }

    init() {
        for mode in DetectorModelMode.allCases {
            if let loaded = Self.loadModel(mode: mode) {
                loadedModels[mode] = loaded
            }
        }
    }

    private static func loadModel(mode: DetectorModelMode) -> LoadedCoreMLModel? {
        let config = MLModelConfiguration()
        config.computeUnits = .all // Apple Neural Engine + GPU + CPU

        var mlModel: MLModel?
        if let compiledUrl = Bundle.main.url(forResource: mode.resourceName, withExtension: "mlmodelc") {
            mlModel = try? MLModel(contentsOf: compiledUrl, configuration: config)
        } else if let pkgUrl = Bundle.main.url(forResource: mode.resourceName, withExtension: "mlpackage"),
                  let compiledUrl = try? MLModel.compileModel(at: pkgUrl) {
            mlModel = try? MLModel(contentsOf: compiledUrl, configuration: config)
        }
        guard let model = mlModel else { return nil }
        let inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"
        let outputName = model.modelDescription.outputDescriptionsByName.keys.first ?? "var_916"
        let detectedSize = model.modelDescription.inputDescriptionsByName[inputName]?.imageConstraint?.pixelsWide ?? mode.defaultSize
        return LoadedCoreMLModel(
            model: model,
            inputSize: detectedSize > 0 ? detectedSize : mode.defaultSize,
            inputFeatureName: inputName,
            outputFeatureName: outputName
        )
    }

    /// Returns up to `maxK` non-overlapping ball candidates sorted by confidence descending.
    func detectTopK(
        pixelBuffer: CVPixelBuffer,
        timestampMs: Double,
        mode: DetectorModelMode,
        confThreshold: Float? = nil,
        maxK: Int = 6
    ) -> [BallDetectionSample] {
        guard let entry = loadedModels[mode] ?? loadedModels[.highRes640] ?? loadedModels[.fast320] else {
            return []
        }
        let srcW = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let srcH = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        guard srcW > 0, srcH > 0 else { return [] }

        let inputSize = entry.inputSize
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
        ) else { return [] }

        guard let provider = try? MLDictionaryFeatureProvider(
            dictionary: [entry.inputFeatureName: MLFeatureValue(pixelBuffer: letterboxed)]
        ),
        let output = try? entry.model.prediction(from: provider),
        let multiArray = output.featureValue(for: entry.outputFeatureName)?.multiArrayValue else {
            return []
        }

        let shape = multiArray.shape.map { $0.intValue }
        guard shape.count == 3, shape[1] >= 5 else { return [] }
        let numPreds = shape[2]
        let strideC = multiArray.strides[1].intValue
        let strideN = multiArray.strides[2].intValue
        let minConf = confThreshold ?? max(0.18, mode.defaultConfidence - 0.08)

        var hits: [(Int, Float)] = []
        hits.reserveCapacity(16)

        if multiArray.dataType == .float32 {
            let ptr = multiArray.dataPointer.bindMemory(to: Float32.self, capacity: multiArray.count)
            let scoreBase = 4 * strideC
            for i in 0..<numPreds {
                let sc = ptr[scoreBase + i * strideN]
                if sc >= minConf {
                    hits.append((i, sc))
                }
            }
            guard !hits.isEmpty else { return [] }
            hits.sort { $0.1 > $1.1 }
            return filterNMS(hits: hits, maxK: maxK, srcW: srcW, srcH: srcH, newW: newW, newH: newH, padX: padX, padY: padY, timestampMs: timestampMs) { idx in
                (
                    CGFloat(ptr[0 * strideC + idx * strideN]),
                    CGFloat(ptr[1 * strideC + idx * strideN]),
                    CGFloat(ptr[2 * strideC + idx * strideN]),
                    CGFloat(ptr[3 * strideC + idx * strideN])
                )
            }
        } else if multiArray.dataType == .float16 {
            #if arch(arm64)
            let ptr = multiArray.dataPointer.bindMemory(to: Float16.self, capacity: multiArray.count)
            let scoreBase = 4 * strideC
            for i in 0..<numPreds {
                let sc = Float(ptr[scoreBase + i * strideN])
                if sc >= minConf {
                    hits.append((i, sc))
                }
            }
            guard !hits.isEmpty else { return [] }
            hits.sort { $0.1 > $1.1 }
            return filterNMS(hits: hits, maxK: maxK, srcW: srcW, srcH: srcH, newW: newW, newH: newH, padX: padX, padY: padY, timestampMs: timestampMs) { idx in
                (
                    CGFloat(ptr[0 * strideC + idx * strideN]),
                    CGFloat(ptr[1 * strideC + idx * strideN]),
                    CGFloat(ptr[2 * strideC + idx * strideN]),
                    CGFloat(ptr[3 * strideC + idx * strideN])
                )
            }
            #else
            return []
            #endif
        } else {
            for i in 0..<numPreds {
                let sc = multiArray[[0, 4, i] as [NSNumber]].floatValue
                if sc >= minConf {
                    hits.append((i, sc))
                }
            }
            guard !hits.isEmpty else { return [] }
            hits.sort { $0.1 > $1.1 }
            return filterNMS(hits: hits, maxK: maxK, srcW: srcW, srcH: srcH, newW: newW, newH: newH, padX: padX, padY: padY, timestampMs: timestampMs) { idx in
                (
                    CGFloat(multiArray[[0, 0, idx] as [NSNumber]].floatValue),
                    CGFloat(multiArray[[0, 1, idx] as [NSNumber]].floatValue),
                    CGFloat(multiArray[[0, 2, idx] as [NSNumber]].floatValue),
                    CGFloat(multiArray[[0, 3, idx] as [NSNumber]].floatValue)
                )
            }
        }
    }

    private func filterNMS(
        hits: [(Int, Float)],
        maxK: Int,
        srcW: CGFloat, srcH: CGFloat,
        newW: CGFloat, newH: CGFloat,
        padX: CGFloat, padY: CGFloat,
        timestampMs: Double,
        readBox: (Int) -> (CGFloat, CGFloat, CGFloat, CGFloat)
    ) -> [BallDetectionSample] {
        let sx = srcW / max(1, newW)
        let sy = srcH / max(1, newH)
        let minDist = max(18.0, srcW * 0.03)
        var picked: [BallDetectionSample] = []
        for (idx, sc) in hits {
            let (cxIn, cyIn, bwIn, bhIn) = readBox(idx)
            let cx = (cxIn - padX) * sx
            let cy = (cyIn - padY) * sy
            let bw = bwIn * sx
            let bh = bhIn * sy
            let ny = cy / srcH
            let aspect = max(bw / max(1, bh), bh / max(1, bw))
            guard ny >= 0.08, ny <= 0.78, aspect <= 2.4, bw <= srcW * 0.14, bh <= srcH * 0.10 else { continue }
            if picked.contains(where: { hypot(cx - $0.x, cy - $0.y) < minDist }) { continue }
            picked.append(BallDetectionSample(timestampMs: timestampMs, x: cx, y: cy, width: bw, height: bh, confidence: sc))
            if picked.count >= maxK { break }
        }
        return picked
    }

    /// 遠端小球 2.8 倍局部放大追焦（以預測落點為中心裁切 36% 方塊，再用 ANE 補抓縮小至 4–8px 的遠端球）
    func detectInROI(
        pixelBuffer: CVPixelBuffer,
        timestampMs: Double,
        mode: DetectorModelMode,
        predX: CGFloat,
        predY: CGFloat,
        radius: CGFloat
    ) -> BallDetectionSample? {
        guard let entry = loadedModels[mode] ?? loadedModels[.highRes640] ?? loadedModels[.fast320] else {
            return nil
        }
        let srcW = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let srcH = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        guard srcW > 0, srcH > 0 else { return nil }

        let side = max(160.0, min(srcW, srcH) * 0.36)
        let rx1 = max(0, min(srcW - side, predX - side * 0.5))
        let ry1 = max(0, min(srcH - side, predY - side * 0.5))
        let cropW = min(side, srcW - rx1)
        let cropH = min(side, srcH - ry1)

        // CIImage 原點在左下角，須把 top-down 的 ry1 轉成 bottom-up
        let ciY = max(0, srcH - (ry1 + cropH))
        let cropped = CIImage(cvPixelBuffer: pixelBuffer)
            .cropped(to: CGRect(x: rx1, y: ciY, width: cropW, height: cropH))
            .transformed(by: CGAffineTransform(translationX: -rx1, y: -ciY))

        let inputSize = entry.inputSize
        let size = CGFloat(inputSize)
        let scale = min(size / cropW, size / cropH)
        let newW = (cropW * scale).rounded()
        let newH = (cropH * scale).rounded()
        let padX = floor((size - newW) / 2.0)
        let padY = floor((size - newH) / 2.0)

        var dst: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        CVPixelBufferCreate(kCFAllocatorDefault, inputSize, inputSize, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &dst)
        guard let outputBuffer = dst else { return nil }

        let gray = CIImage(color: CIColor(red: 114.0/255.0, green: 114.0/255.0, blue: 114.0/255.0))
            .cropped(to: CGRect(x: 0, y: 0, width: inputSize, height: inputSize))
        let composed = cropped
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(translationX: padX, y: padY))
            .composited(over: gray)
        ciContext.render(composed, to: outputBuffer)

        guard let provider = try? MLDictionaryFeatureProvider(
            dictionary: [entry.inputFeatureName: MLFeatureValue(pixelBuffer: outputBuffer)]
        ),
        let output = try? entry.model.prediction(from: provider),
        let multiArray = output.featureValue(for: entry.outputFeatureName)?.multiArrayValue else {
            return nil
        }

        let thresh: Float = max(0.12, mode.defaultConfidence - 0.10)
        let shape = multiArray.shape.map { $0.intValue }
        guard shape.count == 3 else { return nil }
        let isChannelsLast = shape[2] == 5 || shape[2] == 84
        let anchors = isChannelsLast ? shape[1] : shape[2]
        let channels = isChannelsLast ? shape[2] : shape[1]
        let targetClassIdx = (channels - 4) == 1 ? 0 : 32
        guard targetClassIdx < (channels - 4) else { return nil }

        let sx = cropW / max(1, newW)
        let sy = cropH / max(1, newH)
        var best: BallDetectionSample?
        var bestDist = radius

        for i in 0..<anchors {
            let sc: Float = isChannelsLast
                ? multiArray[[0, i, 4 + targetClassIdx] as [NSNumber]].floatValue
                : multiArray[[0, 4 + targetClassIdx, i] as [NSNumber]].floatValue
            guard sc >= thresh else { continue }
            let cxIn = CGFloat(isChannelsLast ? multiArray[[0, i, 0] as [NSNumber]].floatValue : multiArray[[0, 0, i] as [NSNumber]].floatValue)
            let cyIn = CGFloat(isChannelsLast ? multiArray[[0, i, 1] as [NSNumber]].floatValue : multiArray[[0, 1, i] as [NSNumber]].floatValue)
            let bwIn = CGFloat(isChannelsLast ? multiArray[[0, i, 2] as [NSNumber]].floatValue : multiArray[[0, 2, i] as [NSNumber]].floatValue)
            let bhIn = CGFloat(isChannelsLast ? multiArray[[0, i, 3] as [NSNumber]].floatValue : multiArray[[0, 3, i] as [NSNumber]].floatValue)
            let cx = rx1 + (cxIn - padX) * sx
            let cy = ry1 + (cyIn - padY) * sy
            let bw = bwIn * sx
            let bh = bhIn * sy
            let aspect = max(bw / max(1, bh), bh / max(1, bw))
            guard aspect <= 2.4 else { continue }
            let d = hypot(cx - predX, cy - predY)
            if d <= bestDist {
                bestDist = d
                best = BallDetectionSample(timestampMs: timestampMs, x: cx, y: cy, width: bw, height: bh, confidence: sc)
            }
        }
        return best
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
