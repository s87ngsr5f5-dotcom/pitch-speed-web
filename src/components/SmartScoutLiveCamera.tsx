import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  SMARTSCOUT_PITCH_TYPES,
  SmartScoutPitchRecord,
  buildTaperedRibbonPolygon,
  computeRealPitchFromDetections,
  getPitchTypeColor,
  sampleQuadraticRibbon,
} from '../data/smartScoutDefaultSessions';
import { DEFAULT_TUNNEL_CONFIGS, MAX_PITCH_SAMPLE_GAP_MS } from '../core/smartScoutEngine';
import { BallDetection, Point2D, TunnelBoxConfig } from '../types';
import { VideoFlightReview } from './VideoFlightReview';

interface VideoFrameMetadata {
  presentationTime: number;
  expectedDisplayTime: number;
  width: number;
  height: number;
  mediaTime: number;
  presentedFrames: number;
}

type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback?: (
    callback: (now: DOMHighResTimeStamp, metadata: VideoFrameMetadata) => void
  ) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

let lastSentTimestampMs = 0;
const CHECKPOINTS_KEY = 'pitch-speed-checkpoints';
const MIN_CHECKPOINT_DISTANCE = 0.08;

function readCheckpoints(cameraAngle: 'behind_pitcher' | 'behind_catcher'): TunnelBoxConfig {
  const defaults = DEFAULT_TUNNEL_CONFIGS[cameraAngle];
  try {
    const saved = JSON.parse(localStorage.getItem(`${CHECKPOINTS_KEY}:${cameraAngle}`) ?? 'null');
    if (Array.isArray(saved) && saved.length === 4 && saved.every(
      (value) => typeof value === 'number' && value >= 0 && value <= 1,
    ) && Math.hypot(saved[0] - saved[2], saved[1] - saved[3]) >= MIN_CHECKPOINT_DISTANCE) {
      return {
        releaseBox: { ...defaults.releaseBox, cx: saved[0], cy: saved[1] },
        strikeZoneBox: { ...defaults.strikeZoneBox, cx: saved[2], cy: saved[3] },
      };
    }
  } catch {
    // Fall back to the camera guides.
  }
  return defaults;
}

interface SmartScoutLiveCameraProps {
  currentPitch: SmartScoutPitchRecord | null;
  cameraAngle: 'behind_pitcher' | 'behind_catcher';
  discipline: 'pitching' | 'hitting';
  distanceMeters: number;
  isWorkerReady: boolean;
  workerError?: string;
  isBusyRef: React.MutableRefObject<boolean>;
  latestBall: BallDetection | null;
  onBackToSession: () => void;
  onDeletePitch: (pitchId: string) => void;
  onFrame: (
    bitmap: ImageBitmap,
    timestampMs: number,
    width: number,
    height: number
  ) => void;
  onNewMeasuredPitch: (
    pitch: Omit<SmartScoutPitchRecord, 'id' | 'pitchNumber' | 'timestamp'>
  ) => void;
  onChangePitchType: (pitchId: string, newType: string) => void;
}

export const SmartScoutLiveCamera: React.FC<SmartScoutLiveCameraProps> = ({
  currentPitch,
  cameraAngle,
  discipline,
  distanceMeters,
  isWorkerReady,
  workerError,
  isBusyRef,
  latestBall,
  onBackToSession,
  onDeletePitch,
  onFrame,
  onNewMeasuredPitch,
  onChangePitchType,
}) => {
  const videoRef = useRef<VideoWithRVFC | null>(null);
  const [hasLiveCamera, setHasLiveCamera] = useState<boolean>(false);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string | null>(null);
  const [selectedPitchType, setSelectedPitchType] = useState('未標記');
  const [captureMessage, setCaptureMessage] = useState('');
  const [fileEnded, setFileEnded] = useState(false);
  const [filePlaying, setFilePlaying] = useState(false);
  const [reviewingFlight, setReviewingFlight] = useState(false);
  const [region, setRegion] = useState<{ x: number; y: number; width: number; height: number } | null>(null);
  const [choosingRegion, setChoosingRegion] = useState(false);
  const [corner, setCorner] = useState<Point2D | null>(null);
  const [frameSize, setFrameSize] = useState({ width: 1280, height: 720 });
  const [showRibbon, setShowRibbon] = useState<boolean>(true);
  const [showStatsCard, setShowStatsCard] = useState<boolean>(true);
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 0.5 | 0.25>(1);
  const [progress, setProgress] = useState<number>(1);
  const [isAnimatingRibbon, setIsAnimatingRibbon] = useState<boolean>(false);
  const [liveTrackDots, setLiveTrackDots] = useState<BallDetection[]>([]);

  const rafRef = useRef<number | null>(null);
  const detectedInFileRef = useRef(0);
  const measuredInFileRef = useRef(0);
  const activeTrackRef = useRef<BallDetection[]>([]);
  const flushTimerRef = useRef<number | null>(null);
  const sentTimestampsRef = useRef(new Set<number>());
  const [cameraCheckpoints, setCameraCheckpoints] = useState(() => ({
    behind_pitcher: readCheckpoints('behind_pitcher'),
    behind_catcher: readCheckpoints('behind_catcher'),
  }));
  const [fileCheckpoints, setFileCheckpoints] = useState<TunnelBoxConfig | null>(null);
  const [calibration, setCalibration] = useState<{ release?: Point2D } | null>(null);
  const calibrating = calibration !== null;
  const checkpoints = uploadedVideoUrl
    ? fileCheckpoints ?? DEFAULT_TUNNEL_CONFIGS[cameraAngle]
    : cameraCheckpoints[cameraAngle];
  const manualCheckpoints = checkpoints === DEFAULT_TUNNEL_CONFIGS[cameraAngle] ? undefined : checkpoints;

  const resetTrack = useCallback(() => {
    if (flushTimerRef.current !== null) window.clearTimeout(flushTimerRef.current);
    flushTimerRef.current = null;
    activeTrackRef.current = [];
    setLiveTrackDots([]);
  }, []);

  // 1. 一進入即啟動相機 (getUserMedia 優先 60fps 1280x720)
  useEffect(() => {
    const video = videoRef.current;
    if (!video || uploadedVideoUrl) return;

    let activeStream: MediaStream | null = null;
    let cancelled = false;

    const stopStream = () => {
      if (activeStream) {
        activeStream.getTracks().forEach((t) => t.stop());
        if (video.srcObject === activeStream) video.srcObject = null;
        activeStream = null;
      }

    };

    const startCamera = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setHasLiveCamera(false);
        setCaptureMessage('無法使用鏡頭。請確認相機權限，或選擇影片。');
        return;
      }

      const tryConstraints: MediaStreamConstraints[] = [
        {
          video: {
            facingMode: { exact: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 60 },
          },
          audio: false,
        },
        {
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 60 },
          },
          audio: false,
        },
        { video: true, audio: false },
      ];

      for (const c of tryConstraints) {
        try {
          activeStream = await navigator.mediaDevices.getUserMedia(c);
          break;
        } catch {
          // fallback
        }
      }

      if (cancelled) {
        stopStream();
        return;
      }

      if (!activeStream) {
        setHasLiveCamera(false);
        setCaptureMessage('無法使用鏡頭。請確認相機權限，或選擇影片。');
        return;
      }

      video.srcObject = activeStream;
      try {
        await video.play();
        if (!cancelled) {
          setHasLiveCamera(true);
          setCaptureMessage('');
        }
      } catch {
        if (!cancelled) {
          stopStream();
          setHasLiveCamera(false);
          setCaptureMessage('無法播放鏡頭畫面。');
        }
      }
    };

    startCamera();

    return () => {
      cancelled = true;
      stopStream();
    };
  }, [uploadedVideoUrl]);

  // 2. 以 requestVideoFrameCallback 擷取 ImageBitmap 送入 Web Worker 即時偵測
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let stopped = false;
    let rvfcId: number | null = null;
    let rafId: number | null = null;
    let lastMediaTime = -1;
    let timestampOffset: number | null = null;
    const sentTimestamps = sentTimestampsRef.current;
    const fallbackCanvas = document.createElement('canvas');
    const resetSource = () => {
      timestampOffset = null;
      lastMediaTime = -1;
      sentTimestamps.clear();
      resetTrack();
    };
    resetSource();
    video.addEventListener('seeking', resetSource);

    const captureBitmap = async () => {
      if (region) {
        const cropWidth = Math.max(1, Math.round(region.width * video.videoWidth));
        const cropHeight = Math.max(1, Math.round(region.height * video.videoHeight));
        if (fallbackCanvas.width !== cropWidth) fallbackCanvas.width = cropWidth;
        if (fallbackCanvas.height !== cropHeight) fallbackCanvas.height = cropHeight;
        const ctx = fallbackCanvas.getContext('2d');
        if (!ctx) throw new Error('Cannot crop video frame');
        ctx.drawImage(video, region.x * video.videoWidth, region.y * video.videoHeight,
          region.width * video.videoWidth, region.height * video.videoHeight,
          0, 0, fallbackCanvas.width, fallbackCanvas.height);
        return createImageBitmap(fallbackCanvas);
      }
      try {
        return await createImageBitmap(video);
      } catch {
        fallbackCanvas.width = video.videoWidth;
        fallbackCanvas.height = video.videoHeight;
        const context = fallbackCanvas.getContext('2d');
        if (!context) throw new Error('Cannot capture video frame');
        context.drawImage(video, 0, 0);
        return createImageBitmap(fallbackCanvas);
      }
    };

    const sendFrame = (mediaTime: number) => {
      if (
        !isWorkerReady ||
        choosingRegion || reviewingFlight ||
        calibrating ||
        discipline !== 'pitching' ||
        isBusyRef.current ||
        video.readyState < 2 ||
        video.paused ||
        video.ended ||
        video.seeking ||
        video.videoWidth === 0
      ) {
        return;
      }
      isBusyRef.current = true;
      const w = video.videoWidth;
      const h = video.videoHeight;
      const mediaTimeMs = mediaTime * 1000;
      timestampOffset ??= lastSentTimestampMs + 1 - mediaTimeMs;
      const offset = timestampOffset;
      const timestampMs = mediaTimeMs + offset;
      lastSentTimestampMs = timestampMs;
      captureBitmap().then((bitmap) => {
        if (stopped || offset !== timestampOffset) {
          bitmap.close();
          isBusyRef.current = false;
          return;
        }
        // Only accept worker responses from this capture source.
        if (sentTimestamps.size >= 32) {
          sentTimestamps.delete(sentTimestamps.values().next().value!);
        }
        sentTimestamps.add(timestampMs);
        try {
          onFrame(bitmap, timestampMs, region ? bitmap.width : w, region ? bitmap.height : h);
        } catch (error) {
          bitmap.close();
          throw error;
        }
      }).catch(() => {
        isBusyRef.current = false;
        if (!stopped) setCaptureMessage('無法擷取影片畫面。');
      });
    };

    if (typeof video.requestVideoFrameCallback === 'function') {
      const onVideoFrame = (_now: DOMHighResTimeStamp, meta: VideoFrameMetadata) => {
        if (stopped) return;
        if (meta.mediaTime !== lastMediaTime) {
          lastMediaTime = meta.mediaTime;
          sendFrame(meta.mediaTime);
        }
        rvfcId = video.requestVideoFrameCallback!(onVideoFrame);
      };
      rvfcId = video.requestVideoFrameCallback(onVideoFrame);
    } else {
      const onRaf = () => {
        if (stopped) return;
        if (video.currentTime !== lastMediaTime) {
          lastMediaTime = video.currentTime;
          sendFrame(lastMediaTime);
        }
        rafId = requestAnimationFrame(onRaf);
      };
      rafId = requestAnimationFrame(onRaf);
    }

    return () => {
      stopped = true;
      sentTimestamps.clear();
      video.removeEventListener('seeking', resetSource);
      if (rvfcId !== null && video.cancelVideoFrameCallback) {
        video.cancelVideoFrameCallback(rvfcId);
      }
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
      }
    };
  }, [region, choosingRegion, calibrating, cameraAngle, discipline, isWorkerReady, isBusyRef, onFrame, resetTrack, reviewingFlight, uploadedVideoUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !uploadedVideoUrl || !isWorkerReady || discipline !== 'pitching') return;
    void video.play().catch((error: DOMException) => {
      if (error.name !== 'AbortError') {
        setCaptureMessage('無法自動播放。請按「播放並分析」。');
      }
    });
  }, [discipline, isWorkerReady, uploadedVideoUrl]);

  const captureVideoSnapshot = useCallback((): string | undefined => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return undefined;
    try {
      const c = document.createElement('canvas');
      c.width = 590;
      c.height = 1050;
      const ctx = c.getContext('2d');
      if (!ctx) return undefined;
      ctx.drawImage(video, 0, 0, c.width, c.height);
      return c.toDataURL('image/jpeg', 0.82);
    } catch {
      return undefined;
    }
  }, []);

  useEffect(() => {
    if (!latestBall || !sentTimestampsRef.current.delete(latestBall.timestampMs)) return;
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;

    const track = activeTrackRef.current;
    const betweenSamples = track.length
      ? latestBall.timestampMs - track[track.length - 1].timestampMs
      : 0;
    if (track.length && (betweenSamples <= 0 || betweenSamples > MAX_PITCH_SAMPLE_GAP_MS ||
      latestBall.timestampMs - track[0].timestampMs > 1600)) {
      resetTrack();
    }

    const activeTrack = activeTrackRef.current;
    if (region) {
      const width = region.width * video.videoWidth;
      const height = region.height * video.videoHeight;
      const sx = width / Math.max(1, Math.round(width));
      const sy = height / Math.max(1, Math.round(height));
      if (latestBall.x < 0 || latestBall.x * sx > width || latestBall.y < 0 || latestBall.y * sy > height) return;
      activeTrack.push({ ...latestBall, x: region.x * video.videoWidth + latestBall.x * sx,
        y: region.y * video.videoHeight + latestBall.y * sy, width: latestBall.width * sx, height: latestBall.height * sy });
    } else {
      activeTrack.push(latestBall);
    }
    if (uploadedVideoUrl) detectedInFileRef.current += 1;
    setLiveTrackDots([...activeTrack]);
    const record = computeRealPitchFromDetections({
      detections: activeTrack,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      cameraAngle,
      discipline,
      distanceMeters,
      checkpoints: manualCheckpoints,
      preferredPitchType: selectedPitchType,
      videoUrl: uploadedVideoUrl ?? undefined,
    });

    if (record) {
      if (uploadedVideoUrl) measuredInFileRef.current += 1;
      onNewMeasuredPitch({ ...record, snapshotUrl: captureVideoSnapshot() });
      setCaptureMessage('');
      resetTrack();
      setProgress(0.06);
      setIsAnimatingRibbon(true);
      return;
    }

    if (flushTimerRef.current !== null) window.clearTimeout(flushTimerRef.current);
    flushTimerRef.current = window.setTimeout(() => {
      if (activeTrackRef.current.length >= 3) {
        setCaptureMessage('有偵測到球，但還無法確認出手與接球。可調整測速框後再試。');
      }
      resetTrack();
    }, uploadedVideoUrl ? 500 : 180);
  }, [
    cameraAngle,
    captureVideoSnapshot,
    manualCheckpoints,
    discipline,
    distanceMeters,
    latestBall, region,
    onNewMeasuredPitch,
    resetTrack,
    selectedPitchType,
    uploadedVideoUrl,
  ]);

  useEffect(() => () => {
    if (flushTimerRef.current !== null) window.clearTimeout(flushTimerRef.current);
  }, []);

  useEffect(() => {
    if (!isAnimatingRibbon) {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      return;
    }

    let lastTime = performance.now();
    const durationMs = 1000 / playbackSpeed;

    const tick = (now: number) => {
      const dt = now - lastTime;
      lastTime = now;
      setProgress((prev) => {
        const next = prev + dt / durationMs;
        if (next >= 1) {
          setIsAnimatingRibbon(false);
          return 1;
        }
        return next;
      });
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [isAnimatingRibbon, playbackSpeed]);

  const stageW = 590;
  const stageH = 1050;

  const ribbonSamples = useMemo(
    () =>
      currentPitch
        ? sampleQuadraticRibbon(currentPitch.ribbonPath, stageW, stageH, progress)
        : [],
    [currentPitch, progress]
  );

  const polygonPoints = useMemo(
    () => buildTaperedRibbonPolygon(ribbonSamples),
    [ribbonSamples]
  );

  const firstSample = ribbonSamples[0];
  const lastSample = ribbonSamples[ribbonSamples.length - 1];

  const handleVideoFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.currentTarget.value = '';
    const video = videoRef.current;
    if (video?.srcObject instanceof MediaStream) {
      video.srcObject.getTracks().forEach((track) => track.stop());
      video.srcObject = null;
    }
    setCaptureMessage('');
    setHasLiveCamera(false);
    setFileEnded(false);
    setFilePlaying(false);
    measuredInFileRef.current = 0;
    detectedInFileRef.current = 0;
    setCalibration(null);
    setFileCheckpoints(null);
    resetTrack();
    setRegion(null); setChoosingRegion(false); setCorner(null);
    setUploadedVideoUrl(URL.createObjectURL(file));
  };

  const restartAnalysis = () => {
    const video = videoRef.current;
    if (!video || !isWorkerReady) return;
    resetTrack();
    setFileEnded(false);
    measuredInFileRef.current = 0;
    detectedInFileRef.current = 0;
    setCaptureMessage('');
    if (video.readyState >= 1) video.currentTime = 0;
    void video.play().catch(() => setCaptureMessage(
      '無法播放這部影片。請試用 H.264 編碼的 MP4。'
    ));
  };

  const applyCalibration = (event: React.MouseEvent<SVGRectElement>) => {
    if (!calibration) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const point = {
      x: (event.clientX - bounds.left) / bounds.width,
      y: (event.clientY - bounds.top) / bounds.height,
    };
    const { release } = calibration;
    if (!release) {
      setCalibration({ release: point });
      return;
    }
    if (Math.hypot(release.x - point.x, release.y - point.y) < MIN_CHECKPOINT_DISTANCE) {
      setCaptureMessage('兩個測速點太近，請重新點選本壘位置。');
      return;
    }

    const corrected = {
      releaseBox: { ...checkpoints.releaseBox, cx: release.x, cy: release.y },
      strikeZoneBox: { ...checkpoints.strikeZoneBox, cx: point.x, cy: point.y },
    };
    if (uploadedVideoUrl) {
      setFileCheckpoints(corrected);
      restartAnalysis();
    } else {
      setCameraCheckpoints((previous) => ({ ...previous, [cameraAngle]: corrected }));
      try {
        localStorage.setItem(
          `${CHECKPOINTS_KEY}:${cameraAngle}`,
          JSON.stringify([release.x, release.y, point.x, point.y]),
        );
      } catch {
        // Keep the new checkpoints for this session.
      }
    }
    resetTrack();
    setCaptureMessage('');
    setCalibration(null);
  };

  const handleDownloadSnapshot = () => {
    if (!currentPitch) return;
    const canvas = document.createElement('canvas');
    canvas.width = stageW;
    canvas.height = stageH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (videoRef.current && videoRef.current.videoWidth > 0) {
      ctx.drawImage(videoRef.current, 0, 0, stageW, stageH);
    } else {
      const grad = ctx.createLinearGradient(0, 0, 0, stageH);
      grad.addColorStop(0, '#0f172a');
      grad.addColorStop(1, '#1e293b');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, stageW, stageH);
    }

    if (showRibbon && ribbonSamples.length >= 2) {
      const rGrad = ctx.createLinearGradient(
        firstSample.x,
        firstSample.y,
        lastSample.x,
        lastSample.y
      );
      rGrad.addColorStop(0, 'rgba(127, 16, 34, 0.92)');
      rGrad.addColorStop(1, 'rgba(239, 35, 60, 0.92)');
      ctx.fillStyle = rGrad;
      for (const s of ribbonSamples) {
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (showStatsCard) {
      ctx.fillStyle = 'rgba(24, 16, 14, 0.88)';
      ctx.beginPath();
      ctx.roundRect(22, stageH - 320, stageW - 44, 245, 22);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.font = '900 52px sans-serif';
      ctx.fillText(`${currentPitch.speedKmh.toFixed(1)} km/h`, 46, stageH - 230);
      ctx.font = 'bold 20px sans-serif';
      ctx.fillText(
        currentPitch.measurement
          ? `${Math.round(currentPitch.measurement.flightTimeMs)} ms | ${currentPitch.measurement.detectionCount} 幀`
          : currentPitch.pitchType,
        46,
        stageH - 150
      );
    }

    const link = document.createElement('a');
    link.download = `smartscout-pitch-${currentPitch.pitchNumber}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  const { width: vw, height: vh } = frameSize;

  return (
    <div className="relative mx-auto flex h-[calc(100dvh-64px)] max-h-[920px] w-full max-w-[460px] flex-col overflow-hidden rounded-none sm:rounded-3xl bg-slate-950 shadow-xl select-none">
      {/* 真實相機串流或載入影片 */}
      <video
        ref={videoRef}
        src={uploadedVideoUrl ?? undefined}
        playsInline
        muted
        preload="metadata"
        autoPlay={!uploadedVideoUrl}
        onPlaying={() => setFilePlaying(true)}
        onPause={() => setFilePlaying(false)}
        onEnded={() => {
          setFilePlaying(false);
          setFileEnded(true);
          if (measuredInFileRef.current === 0) {
            const area = region ? '所選範圍內' : '全畫面';
            const detected = detectedInFileRef.current;
            setCaptureMessage(detected
              ? `${area}在 ${detected} 幀找到疑似球的位置，但無法確認完整球路。可重選範圍或手動核對。`
              : `${area}沒有偵測到球。請確認球路在畫面內，或手動核對影片時間。`);
          }
        }}
        onLoadedMetadata={(event) => {
          const video = event.currentTarget;
          setFrameSize({ width: video.videoWidth, height: video.videoHeight });
          video.playbackRate = uploadedVideoUrl ? 0.5 : 1;
        }}
        onResize={(event) => {
          const video = event.currentTarget;
          if (video.videoWidth && video.videoHeight) {
            setFrameSize({ width: video.videoWidth, height: video.videoHeight });
          }
        }}
        onError={() => {
          setFilePlaying(false);
          setCaptureMessage('無法讀取影片。若檔案在雲端，請先下載到手機；也可試用 H.264 編碼的 MP4。');
        }}
        className="absolute inset-0 h-full w-full object-contain bg-slate-950"
      />

      {(region || choosingRegion) && <svg viewBox={`0 0 ${vw} ${vh}`}
        className={`absolute inset-0 h-full w-full ${choosingRegion ? 'z-30' : 'pointer-events-none z-10'}`}>
        {region && <rect x={region.x * vw} y={region.y * vh} width={region.width * vw} height={region.height * vh}
          fill="none" stroke="#34d399" strokeWidth={2} vectorEffect="non-scaling-stroke" />}
        {corner && <circle cx={corner.x * vw} cy={corner.y * vh} r={vw / 100} fill="#34d399" />}
        {choosingRegion && <rect width={vw} height={vh} fill="transparent" onClick={event => {
          const bounds = event.currentTarget.getBoundingClientRect();
          const x = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width));
          const y = Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height));
          if (!corner) { setCorner({ x, y }); return; }
          const width = Math.abs(x - corner.x), height = Math.abs(y - corner.y);
          if (width * vw < 4 || height * vh < 4) { setCaptureMessage('範圍太小，請重新點選對角。'); return; }
          setRegion({ x: Math.min(x, corner.x), y: Math.min(y, corner.y), width, height });
          setCorner(null); setChoosingRegion(false); setCaptureMessage('已套用球路範圍。');
        }} />}
      </svg>}
      {/* 無實體相機時顯示清爽的縱深投球通道對準格線 */}
      {!hasLiveCamera && !uploadedVideoUrl && (
        <div
          className="pointer-events-none absolute inset-0 opacity-45"
          style={{
            backgroundImage:
              'radial-gradient(circle at 50% 45%, rgba(11, 114, 185, 0.28), transparent 70%), linear-gradient(to right, rgba(148, 163, 184, 0.12) 1px, transparent 1px), linear-gradient(to bottom, rgba(148, 163, 184, 0.12) 1px, transparent 1px)',
            backgroundSize: '100% 100%, 44px 44px, 44px 44px',
          }}
        />
      )}

      {/* 頂部導覽與快捷測試/上傳按鈕 */}
      <div className="relative z-20 flex items-center justify-between px-3.5 pt-3.5 pb-2 text-white bg-gradient-to-b from-black/65 to-transparent">
        <button
          type="button"
          onClick={onBackToSession}
          className="flex h-9 w-9 items-center justify-center rounded-full bg-black/30 hover:bg-black/50 transition"
          aria-label="返回場次列表"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>

        <div className="flex items-center gap-2">
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              isWorkerReady ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
            }`}
          />
          <h2 className="text-base font-bold tracking-wide drop-shadow">
            {workerError ? '模型無法啟動' : isWorkerReady ? '投球追蹤' : '正在載入模型'}
          </h2>
        </div>

        <div className="flex items-center gap-1.5">
          {uploadedVideoUrl && (
            <button
              type="button"
              onClick={() => {
                resetTrack();
                setCalibration(null);
                setCaptureMessage('');
                setRegion(null); setChoosingRegion(false); setCorner(null);
                setUploadedVideoUrl(null);
              }}
              className="rounded-xl bg-white/15 px-2.5 py-1.5 text-xs font-bold text-white"
            >
              開啟鏡頭
            </button>
          )}

          <label
            title="上傳影片"
            className="cursor-pointer rounded-xl bg-white/15 hover:bg-white/25 px-2.5 py-1.5 text-xs font-bold text-white backdrop-blur-sm transition"
          >
            上傳影片
            <input
              type="file"
              accept="video/*"
              onChange={handleVideoFileUpload}
              className="hidden"
            />
          </label>

          {currentPitch && (
            <button
              type="button"
              onClick={() => onDeletePitch(currentPitch.id)}
              className="flex h-8 w-8 items-center justify-center rounded-full bg-black/30 hover:bg-rose-600/80 transition"
              aria-label="刪除"
            >
              🗑
            </button>
          )}
        </div>
      </div>

      <div className="relative z-40 mx-auto max-w-[92%] rounded-lg bg-black/65 px-3 py-1.5 text-center text-xs text-white">
        <p role="status">
          {workerError || captureMessage || (discipline === 'hitting'
            ? '目前支援投球測速。請在設定中切換為投球。'
            : calibration
              ? calibration.release ? '點選球通過本壘的位置' : '點選投手放球的位置'
              : manualCheckpoints
                ? '使用你設定的出手與本壘位置。'
                : '自動辨識出手和接球。請固定鏡頭，讓投手、球路與捕手都入鏡。')}
        </p>
        {(uploadedVideoUrl || hasLiveCamera) && !calibrating && discipline === 'pitching' && <>
          <button type="button" className="mt-1 rounded px-2 py-1 text-sky-200 underline"
            onClick={() => {
              if (uploadedVideoUrl) videoRef.current?.pause();
              resetTrack(); setCorner(null); setChoosingRegion(!choosingRegion); setCaptureMessage('');
            }}>{choosingRegion ? '取消框選' : region ? '重選球路範圍' : '選取球路範圍'}</button>
          {region && !choosingRegion && <button type="button" className="px-2 text-sky-200 underline"
            onClick={() => { setRegion(null); resetTrack(); }}>使用全畫面</button>}
          {choosingRegion && <p>{corner ? '再點對角，完成選取' : '點選球路範圍的一角，讓出手到接球都在框內'}</p>}
        </>}
        {!choosingRegion && uploadedVideoUrl && !filePlaying && !calibrating && !workerError && (
          <button
            type="button"
            onClick={restartAnalysis}
            disabled={!isWorkerReady}
            className="mt-1 rounded px-2 py-1 font-semibold text-sky-200 underline disabled:opacity-60"
          >
            {!isWorkerReady ? '等待模型載入' : fileEnded ? '重新分析' : '播放並分析'}
          </button>
        )}
        {!choosingRegion && !workerError && discipline === 'pitching' && (hasLiveCamera || uploadedVideoUrl) && (
          <button
            type="button"
            onClick={() => {
              resetTrack();
              setCaptureMessage('');
              setCalibration(calibrating ? null : {});
            }}
            className="mt-1 rounded px-2 py-1 font-semibold text-sky-200 underline"
          >
            {calibrating ? '取消調整' : '調整測速框'}
          </button>
        )}
        {manualCheckpoints && !calibrating && (
          <button
            type="button"
            className="mt-1 rounded px-2 py-1 font-semibold text-sky-200 underline"
            onClick={() => {
              if (uploadedVideoUrl) {
                setFileCheckpoints(null);
              } else {
                setCameraCheckpoints((saved) => ({
                  ...saved, [cameraAngle]: DEFAULT_TUNNEL_CONFIGS[cameraAngle],
                }));
                try {
                  localStorage.removeItem(`${CHECKPOINTS_KEY}:${cameraAngle}`);
                } catch {
                  // Still switch back to automatic mode for the current session.
                }
              }
              setCaptureMessage('');
              resetTrack();
            }}
          >
            改回自動辨識
          </button>
        )}
        {uploadedVideoUrl && !calibrating && !choosingRegion && (
          <button type="button" className="mt-1 rounded px-2 py-1 font-semibold text-sky-200 underline"
            onClick={() => { videoRef.current?.pause(); resetTrack(); setReviewingFlight(true); }}>
            手動核對影片時間
          </button>
        )}
      </div>
      {reviewingFlight && uploadedVideoUrl && (
        <VideoFlightReview url={uploadedVideoUrl} onClose={() => setReviewingFlight(false)} />
      )}

      {/* 3D 漸變紅色彗星彩帶軌跡 (#7f1022 -> #ef233c) + 末端立體小白球(含縫線與陰影) */}
      <svg
        viewBox={`0 0 ${vw} ${vh}`}
        className={`absolute inset-0 h-full w-full ${calibrating ? 'z-30' : 'z-10 pointer-events-none'}`}
      >
        <defs>
          <linearGradient id="liveRibbonGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#7f1022" stopOpacity="0.92" />
            <stop offset="55%" stopColor="#d9182e" stopOpacity="0.90" />
            <stop offset="100%" stopColor="#ef233c" stopOpacity="0.96" />
          </linearGradient>
          <filter id="liveRibbonShadow" x="-15%" y="-15%" width="130%" height="130%">
            <feDropShadow dx="0" dy="4" stdDeviation="5" floodColor="#4c0519" floodOpacity="0.55" />
          </filter>
        </defs>

        <g transform={`scale(${vw / stageW} ${vh / stageH})`}>
        {calibrating && (
          <rect width={stageW} height={stageH} fill="transparent" onClick={applyCalibration} />
        )}
        {discipline === 'pitching' && (manualCheckpoints || calibrating) && [checkpoints.releaseBox, checkpoints.strikeZoneBox].map((box, i) => (
          <rect
            key={i}
            x={(box.cx - box.width / 2) * stageW}
            y={(box.cy - box.height / 2) * stageH}
            width={box.width * stageW}
            height={box.height * stageH}
            fill="none"
            stroke="white"
            strokeDasharray="6 6"
            strokeWidth="2"
            opacity="0.75"
            pointerEvents="none"
          />
        ))}
        {calibration?.release && (
          <circle
            cx={calibration.release.x * stageW}
            cy={calibration.release.y * stageH}
            r={8}
            fill="#38bdf8"
            pointerEvents="none"
          />
        )}
        {!calibrating && currentPitch?.measurement?.automaticEndpoints && [
          { x: currentPitch.ribbonPath.startX, y: currentPitch.ribbonPath.startY, label: '推估出手' },
          { x: currentPitch.ribbonPath.endX, y: currentPitch.ribbonPath.endY, label: '推估接球' },
        ].map(({ x, y, label }) => (
          <g key={label} transform={`translate(${x * stageW}, ${y * stageH})`}>
            <circle r={6} fill="#38bdf8" stroke="white" strokeWidth={2} />
            <text x={10} y={-5} fontSize={14} fill="white" stroke="#0f172a" strokeWidth={3}
              paintOrder="stroke">{label}</text>
          </g>
        ))}
        {!calibrating && showRibbon && currentPitch && ribbonSamples.length >= 2 && (
          <g filter="url(#liveRibbonShadow)">
            <polygon points={polygonPoints} fill="url(#liveRibbonGrad)" />
            {ribbonSamples.map((s, idx) => (
              <circle
                key={idx}
                cx={s.x}
                cy={s.y}
                r={s.r}
                fill={idx < ribbonSamples.length * 0.35 ? '#9f1229' : '#ef233c'}
                fillOpacity={idx === 0 ? 0.65 : 0.85}
              />
            ))}
            {firstSample && lastSample && (
              <circle
                cx={firstSample.r >= lastSample.r ? firstSample.x : lastSample.x}
                cy={firstSample.r >= lastSample.r ? firstSample.y : lastSample.y}
                r={Math.max(firstSample.r, lastSample.r)}
                fill="#be123c"
                fillOpacity="0.62"
                stroke="rgba(255, 160, 170, 0.5)"
                strokeWidth="1.5"
              />
            )}
            {/* 末端立體小白球 (含棒球紅色縫線與陰影) */}
            {lastSample && (
              <g>
                <circle
                  cx={lastSample.x}
                  cy={lastSample.y}
                  r={Math.max(7, lastSample.r * 0.95)}
                  fill="#ffffff"
                  stroke="#e2e8f0"
                  strokeWidth="1.5"
                />
                <path
                  d={`M ${lastSample.x - lastSample.r * 0.45} ${lastSample.y - lastSample.r * 0.55} Q ${lastSample.x} ${lastSample.y} ${lastSample.x - lastSample.r * 0.45} ${lastSample.y + lastSample.r * 0.55}`}
                  fill="none"
                  stroke="#dc2626"
                  strokeWidth="1.6"
                  strokeDasharray="2.5 2"
                />
                <path
                  d={`M ${lastSample.x + lastSample.r * 0.45} ${lastSample.y - lastSample.r * 0.55} Q ${lastSample.x} ${lastSample.y} ${lastSample.x + lastSample.r * 0.45} ${lastSample.y + lastSample.r * 0.55}`}
                  fill="none"
                  stroke="#dc2626"
                  strokeWidth="1.6"
                  strokeDasharray="2.5 2"
                />
              </g>
            )}
          </g>
        )}

        {liveTrackDots.map((d, i) => (
          <circle
            key={i}
            cx={(d.x / vw) * stageW}
            cy={(d.y / vh) * stageH}
            r={8}
            fill="#ef233c"
            stroke="#ffffff"
            strokeWidth="2"
          />
        ))}
        </g>
      </svg>

      {/* 右側 4 顆懸浮圓形控制鈕 */}
      <div className="absolute right-4 top-[36%] z-20 flex flex-col items-center gap-5">
        <button
          type="button"
          onClick={() => setShowRibbon((v) => !v)}
          title="紅圈軌跡顯示開關"
          className="flex h-11 w-11 items-center justify-center rounded-full transition active:scale-90"
        >
          <span
            className={`flex h-9 w-9 items-center justify-center rounded-full border-[2.5px] border-white shadow-lg ${
              showRibbon ? 'bg-[#ef233c]' : 'bg-black/40'
            }`}
          />
        </button>

        <button
          type="button"
          onClick={() => {
            const nextSpeed: 1 | 0.5 | 0.25 =
              playbackSpeed === 1 ? 0.5 : playbackSpeed === 0.5 ? 0.25 : 1;
            setPlaybackSpeed(nextSpeed);
            if (currentPitch) {
              setProgress(0.06);
              setIsAnimatingRibbon(true);
            }
          }}
          title="慢動作播放倍率切換"
          className="relative flex h-11 w-11 items-center justify-center rounded-full text-white drop-shadow-[0_2px_6px_rgba(0,0,0,0.8)] hover:scale-105 active:scale-95 transition"
        >
          <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 15l3.5-3.5" />
            <path d="M20.3 18A9 9 0 1 0 3.7 18Z" />
          </svg>
          {playbackSpeed !== 1 && (
            <span className="absolute -bottom-2 rounded bg-black/80 px-1 text-[10px] font-bold text-amber-300">
              {playbackSpeed}x
            </span>
          )}
        </button>

        <button
          type="button"
          onClick={handleDownloadSnapshot}
          title="下載 PNG 分享圖卡"
          className="flex h-11 w-11 items-center justify-center rounded-full text-white drop-shadow-[0_2px_6px_rgba(0,0,0,0.8)] hover:scale-105 active:scale-95 transition"
        >
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
        </button>

        <button
          type="button"
          onClick={() => setShowStatsCard((v) => !v)}
          title="底部數據卡展開/收合"
          className={`flex h-11 w-11 items-center justify-center rounded-full drop-shadow-[0_2px_6px_rgba(0,0,0,0.8)] hover:scale-105 active:scale-95 transition ${
            showStatsCard ? 'text-white' : 'text-white/45'
          }`}
        >
          <svg width="28" height="28" viewBox="0 0 24 24" fill="currentColor">
            <rect x="4" y="10" width="4" height="10" rx="1" />
            <rect x="10" y="5" width="4" height="15" rx="1" />
            <rect x="16" y="13" width="4" height="7" rx="1" />
          </svg>
        </button>
      </div>

      {/* 底部深色半透明 2x4 八大數據面板 + 時間軸滑桿 */}
      <div className="relative z-20 mt-auto px-3.5 pb-3.5 space-y-2.5">
        {showStatsCard && (
          <div className="rounded-2xl bg-[#18110e]/88 backdrop-blur-md px-4 py-3 text-white shadow-2xl border border-white/10">
            {/* 頂部：初速 km/h + 球種下拉選單 */}
            <div className="flex items-center justify-between gap-2">
              <div>
                <div className="text-xs font-medium text-neutral-300">預估初速</div>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-[42px] font-bold leading-tight tracking-tight tabular-nums">
                    {currentPitch ? currentPitch.speedKmh.toFixed(1) : '--.-'}
                  </span>
                  <span className="text-base font-medium text-neutral-300">km/h</span>
                </div>
              </div>

              <div>
                <select
                  value={currentPitch ? currentPitch.pitchType : selectedPitchType}
                  onChange={(e) => {
                    setSelectedPitchType(e.target.value);
                    if (currentPitch) {
                      onChangePitchType(currentPitch.id, e.target.value);
                    }
                  }}
                  style={{
                    backgroundColor: getPitchTypeColor(
                      currentPitch ? currentPitch.pitchType : selectedPitchType
                    ),
                  }}
                  className="rounded-lg px-3.5 py-1.5 text-xs font-bold text-white shadow-md focus:outline-none cursor-pointer"
                >
                  {SMARTSCOUT_PITCH_TYPES.map((pt) => (
                    <option key={pt.name} value={pt.name} className="bg-neutral-900 text-white">
                      {pt.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {currentPitch?.measurement && (
              <p className="mt-2 text-xs text-neutral-300">
                飛行 {Math.round(currentPitch.measurement.flightTimeMs)} ms · 偵測 {currentPitch.measurement.detectionCount} 幀
              </p>
            )}
          </div>
        )}

        {/* 最下方：播放/暫停鈕與白色逐幀時間軸滑桿 */}
        <div className="flex items-center gap-2.5 px-1">
          <button
            type="button"
            disabled={!currentPitch}
            onClick={() => {
              if (progress >= 0.99) setProgress(0.05);
              setIsAnimatingRibbon((v) => !v);
            }}
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white text-neutral-950 font-bold text-xs shadow disabled:opacity-40"
          >
            {isAnimatingRibbon ? '❚❚' : '▶'}
          </button>
          <input
            type="range"
            min={0.05}
            max={1}
            step={0.01}
            value={progress}
            disabled={!currentPitch}
            onChange={(e) => {
              setIsAnimatingRibbon(false);
              setProgress(Number(e.target.value));
            }}
            className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/40 accent-white disabled:opacity-40"
          />
        </div>
      </div>
    </div>
  );
};
