import AVFoundation
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}
}

struct TaperedRibbonOverlay: View {
    let pitch: MeasuredPitch

    var body: some View {
        Canvas { context, size in
            let p0 = CGPoint(x: pitch.ribbon.startX * size.width, y: pitch.ribbon.startY * size.height)
            let p1 = CGPoint(x: pitch.ribbon.ctrlX * size.width, y: pitch.ribbon.ctrlY * size.height)
            let p2 = CGPoint(x: pitch.ribbon.endX * size.width, y: pitch.ribbon.endY * size.height)
            let behindPitcher = pitch.cameraAngle == .behindPitcher
            let startR: CGFloat = behindPitcher ? 18 : 4
            let endR: CGFloat = behindPitcher ? 4 : 18

            let steps = 32
            var pts: [CGPoint] = []
            var radii: [CGFloat] = []
            for i in 0...steps {
                let t = CGFloat(i) / CGFloat(steps)
                let inv = 1 - t
                let x = inv * inv * p0.x + 2 * inv * t * p1.x + t * t * p2.x
                let y = inv * inv * p0.y + 2 * inv * t * p1.y + t * t * p2.y
                pts.append(CGPoint(x: x, y: y))
                radii.append(startR + (endR - startR) * t)
            }

            func ribbonPath(mul: CGFloat) -> Path {
                var left: [CGPoint] = []
                var right: [CGPoint] = []
                for idx in 0..<pts.count {
                    let prev = pts[max(0, idx - 1)]
                    let next = pts[min(pts.count - 1, idx + 1)]
                    let tx = next.x - prev.x
                    let ty = next.y - prev.y
                    let len = max(0.001, hypot(tx, ty))
                    let nx = -ty / len
                    let ny = tx / len
                    let r = radii[idx] * mul
                    left.append(CGPoint(x: pts[idx].x + nx * r, y: pts[idx].y + ny * r))
                    right.append(CGPoint(x: pts[idx].x - nx * r, y: pts[idx].y - ny * r))
                }
                var path = Path()
                guard let first = left.first else { return path }
                path.move(to: first)
                for p in left.dropFirst() { path.addLine(to: p) }
                for p in right.reversed() { path.addLine(to: p) }
                path.closeSubpath()
                return path
            }

            context.fill(ribbonPath(mul: 1.7), with: .color(pitch.accentColor.opacity(0.28)))
            context.fill(ribbonPath(mul: 1.0), with: .color(pitch.accentColor.opacity(0.88)))
            context.fill(ribbonPath(mul: 0.35), with: .color(Color.white.opacity(0.55)))

            if let tip = pts.last {
                let r = (radii.last ?? 5) * 1.15
                let rect = CGRect(x: tip.x - r, y: tip.y - r, width: r * 2, height: r * 2)
                context.fill(Path(ellipseIn: rect), with: .color(.white))
            }
        }
        .allowsHitTesting(false)
    }
}

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @State private var showHistory = false
    @State private var selectedVideoItem: PhotosPickerItem?

    var body: some View {
        ZStack {
            CameraPreviewView(session: camera.session)
                .ignoresSafeArea()

            // Live ball trail dots during flight
            GeometryReader { geo in
                ForEach(Array(camera.liveDots.enumerated()), id: \.offset) { _, pt in
                    Circle()
                        .fill(Color.red)
                        .frame(width: 10, height: 10)
                        .shadow(color: .red, radius: 4)
                        .position(x: pt.x * geo.size.width, y: pt.y * geo.size.height)
                }
            }
            .ignoresSafeArea()
            .allowsHitTesting(false)

            // Finalized 3D tapered trajectory ribbon
            if let pitch = camera.currentPitch {
                TaperedRibbonOverlay(pitch: pitch)
                    .ignoresSafeArea()
            }

            // HUD Overlay
            VStack(spacing: 10) {
                topBar
                Spacer()
                if let pitch = camera.currentPitch {
                    statsCard(pitch)
                } else {
                    Text(camera.statusText)
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(.ultraThinMaterial, in: Capsule())
                        .padding(.bottom, 24)
                }
            }
            .padding(.horizontal, 14)
        }
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
        .sheet(isPresented: $showHistory) {
            historySheet
        }
        .onChange(of: selectedVideoItem) { newItem in
            guard let newItem else { return }
            Task {
                if let movie = try? await newItem.loadTransferable(type: VideoTransferable.self) {
                    camera.analyzeImportedVideo(url: movie.url)
                }
            }
        }
    }

    private var topBar: some View {
        HStack(spacing: 8) {
            Label("\(camera.activeFps) FPS · ANE", systemImage: "bolt.fill")
                .font(.caption.weight(.bold))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.black.opacity(0.62), in: Capsule())
                .foregroundColor(camera.isModelReady ? .green : .orange)

            Button {
                camera.cameraAngle = camera.cameraAngle == .behindPitcher ? .behindCatcher : .behindPitcher
            } label: {
                Text(camera.cameraAngle.label)
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.black.opacity(0.62), in: Capsule())
            }

            Spacer()

            Button {
                camera.autoSaveToPhotos.toggle()
            } label: {
                Image(systemName: camera.autoSaveToPhotos ? "record.circle.fill" : "record.circle")
                    .foregroundColor(camera.autoSaveToPhotos ? .red : .gray)
                    .padding(8)
                    .background(Color.black.opacity(0.62), in: Circle())
            }

            Button {
                camera.voiceAnnounce.toggle()
            } label: {
                Image(systemName: camera.voiceAnnounce ? "speaker.wave.2.fill" : "speaker.slash.fill")
                    .foregroundColor(.white)
                    .padding(8)
                    .background(Color.black.opacity(0.62), in: Circle())
            }

            PhotosPicker(selection: $selectedVideoItem, matching: .videos) {
                Image(systemName: "photo.on.rectangle")
                    .foregroundColor(.white)
                    .padding(8)
                    .background(Color.black.opacity(0.62), in: Circle())
            }

            Button {
                showHistory = true
            } label: {
                Text("\(camera.pitchHistory.count) 球")
                    .font(.caption.weight(.bold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.black.opacity(0.62), in: Capsule())
            }
        }
        .foregroundColor(.white)
        .padding(.top, 8)
    }

    private func statsCard(_ p: MeasuredPitch) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(String(format: "%.1f", p.speedKmh))
                    .font(.system(size: 38, weight: .heavy, design: .rounded))
                    .foregroundColor(.white)
                Text("km/h")
                    .font(.subheadline.weight(.bold))
                    .foregroundColor(.gray)
                Spacer()
                HStack(spacing: 6) {
                    Circle().fill(p.accentColor).frame(width: 8, height: 8)
                    Text(p.pitchType)
                        .font(.subheadline.weight(.bold))
                        .foregroundColor(.white)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(p.accentColor.opacity(0.22), in: Capsule())
                .overlay(Capsule().stroke(p.accentColor, lineWidth: 1))
            }

            Divider().background(Color.white.opacity(0.2))

            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 4), spacing: 10) {
                metricCell("進壘速度", String(format: "%.1f", p.plateSpeedKmh))
                metricCell("垂直變化", String(format: "%+.1f cm", p.verticalBreakCm))
                metricCell("水平變化", String(format: "%+.1f cm", p.horizontalBreakCm))
                metricCell("總轉速", "\(p.totalSpinRpm)")
                metricCell("有效轉速", "\(p.activeSpinRpm)")
                metricCell("旋轉效率", "\(p.efficiencyPct)%")
                metricCell("旋轉方向", p.spinDirection)
                metricCell("陀螺角度", "\(p.gyroDegree)°")
            }

            HStack {
                Text("飛行 \(Int(p.flightTimeMs))ms · \(p.detectionCount) 幀")
                    .font(.caption2)
                    .foregroundColor(.gray)
                Spacer()
                if p.savedToPhotos {
                    Label("已自動存進相簿", systemImage: "checkmark.circle.fill")
                        .font(.caption2.weight(.semibold))
                        .foregroundColor(.green)
                }
            }
        }
        .padding(14)
        .background(Color(red: 0.05, green: 0.07, blue: 0.11).opacity(0.84), in: RoundedRectangle(cornerRadius: 18))
        .overlay(
            RoundedRectangle(cornerRadius: 18)
                .stroke(Color.white.opacity(0.18), lineWidth: 1)
        )
        .padding(.bottom, 12)
    }

    private func metricCell(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption2).foregroundColor(.gray)
            Text(value).font(.subheadline.weight(.bold)).foregroundColor(.white)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var historySheet: some View {
        NavigationStack {
            List(camera.pitchHistory) { pitch in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("#\(pitch.pitchNumber) · \(pitch.pitchType)")
                            .font(.headline)
                        Text(String(format: "VB %+.1f / HB %+.1f cm · %d rpm", pitch.verticalBreakCm, pitch.horizontalBreakCm, pitch.totalSpinRpm))
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    Text(String(format: "%.1f km/h", pitch.speedKmh))
                        .font(.title3.weight(.heavy))
                        .foregroundColor(pitch.accentColor)
                }
            }
            .navigationTitle("本次投球紀錄")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { showHistory = false }
                }
            }
        }
    }
}

struct VideoTransferable: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let dst = FileManager.default.temporaryDirectory.appendingPathComponent("import_\(UUID().uuidString).mov")
            try FileManager.default.copyItem(at: received.file, to: dst)
            return Self(url: dst)
        }
    }
}
