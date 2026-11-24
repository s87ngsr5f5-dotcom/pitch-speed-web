import { measureCheckpointFlight } from '../core/checkpointFlight';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  SMARTSCOUT_PITCH_TYPES,
  SmartScoutPitchRecord,
  buildTaperedRibbonPolygon,
  computeRealPitchFromDetections,
  getPitchTypeColor,
  sampleQuadraticRibbon,
} from '../data/smartScoutDefaultSessions';
import {
  DEFAULT_TUNNEL_CONFIGS,
  MAX_PITCH_SAMPLE_GAP_MS,
} from '../core/smartScoutEngine';
import { BallDetection, PitchActionPhase, Point2D, TunnelBoxConfig } from '../types';
import { PitcherPoseGate } from '../core/pitcherPoseGate';
import { savePitchVideoClip } from '../utils/pitchVideoStore';
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

function isFlightCorridorBall(
  det: BallDetection,
  videoWidth: number,
  videoHeight: number,
  hasRegion: boolean
): boolean {
  if (videoWidth <= 0 || videoHeight <= 0) return false;
  // 濾除長條狀燈管、白線與過大球褲反光
  const bw = Math.max(1, det.width);
  const bh = Math.max(1, det.height);
  if (Math.max(bw / bh, bh / bw) > 2.4) return false;
  if (det.width > videoWidth * 0.14 || det.height > videoHeight * 0.10) return false;
  if (hasRegion) return true;
  const ny = det.y / videoHeight;
  return ny >= 0.08 && ny <= 0.78;
}

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
  /** 投手骨架出手閘門：只有投手真正出手後的球點才會被拿去測速 */
  poseGate?: PitcherPoseGate;
  actionPhase?: PitchActionPhase;
  actionConfidence?: number;
  onBackToSession: () => void;
  onDeletePitch: (pitchId: string) => void;
  onFrame: (
    bitmap: ImageBitmap,
    timestampMs: number,
    width: number,
    height: number,
    predictedRoi?: { x: number; y: number; radius: number }
  ) => void;
  onPoseFrame?: (
    bitmap: ImageBitmap,
    timestampMs: number,
    width: number,
    height: number
  ) => void;
  onScanFrame: (
    bitmap: ImageBitmap,
    timestampMs: number,
    width: number,
    height: number
  ) => Promise<BallDetection | null>;
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
  poseGate,
  actionPhase = 'idle',
  actionConfidence = 0,
  onBackToSession,
  onDeletePitch,
  onFrame,
  onPoseFrame,
  onScanFrame,
  onNewMeasuredPitch,
  onChangePitchType,
}) => {
  const videoRef = useRef<VideoWithRVFC | null>(null);
  const liveStreamRef = useRef<MediaStream | null>(null);
  const liveRecorderRef = useRef<MediaRecorder | null>(null);
  const clipStartTsRef = useRef<number>(0);
  const autoReturnToCameraRef = useRef<boolean>(false);
  const isAutoRecordingRef = useRef<boolean>(false);
  const clipChunksRef = useRef<Blob[]>([]);
  const clipMimeRef = useRef<string>('video/webm');
  const actionStartStreakRef = useRef<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [cameraRequested, setCameraRequested] = useState(false);
  const [hasLiveCamera, setHasLiveCamera] = useState<boolean>(false);
  const [cameraFps, setCameraFps] = useState<number>(60);
  const [autoRecordLeftSec, setAutoRecordLeftSec] = useState<number | null>(null);
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
  // 一打開相機預設不顯示上次留存的舊球紅點／軌跡，只顯示當次新測到的球
  const [dismissedPitchId, setDismissedPitchId] = useState<string | null>(
    () => currentPitch?.id ?? null
  );
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 0.5 | 0.25>(1);
  const [progress, setProgress] = useState<number>(1);
  const [isAnimatingRibbon, setIsAnimatingRibbon] = useState<boolean>(false);
  const [liveTrackDots, setLiveTrackDots] = useState<BallDetection[]>([]);
  const displayPitch =
    currentPitch && currentPitch.id !== dismissedPitchId ? currentPitch : null;

  const rafRef = useRef<number | null>(null);
  const detectedInFileRef = useRef(0);
  const measuredInFileRef = useRef(0);
  const endpointSamplesRef = useRef<BallDetection[]>([]);
  const activeTrackRef = useRef<BallDetection[]>([]);
  const flushTimerRef = useRef<number | null>(null);
  const sentTimestampsRef = useRef(new Set<number>());
  // 滾動 90 幀靜態背景抑制器（14px 格點）：同一位置反覆亮 >= 5 次一律判為球場燈光／招牌，絕不當成棒球
  const staticBinWindowRef = useRef<number[]>([]);
  const staticBinCountsRef = useRef<Map<number, number>>(new Map());
  const [cameraCheckpoints, setCameraCheckpoints] = useState(() => ({
    behind_pitcher: readCheckpoints('behind_pitcher'),
    behind_catcher: readCheckpoints('behind_catcher'),
  }));
  const [fileCheckpoints, setFileCheckpoints] = useState<TunnelBoxConfig | null>(null);
  const [calibrationTime, setCalibrationTime] = useState(0);
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
    endpointSamplesRef.current = [];
    setLiveTrackDots([]);
  }, []);

  const startFreshLiveRecorder = useCallback((stream: MediaStream) => {
    if (typeof MediaRecorder === 'undefined') return;
    try {
      const preferredTypes = [
        'video/mp4',
        'video/webm;codecs=vp8',
        'video/webm',
      ];
      const mimeType =
        preferredTypes.find((t) => MediaRecorder.isTypeSupported?.(t)) ?? '';
      clipMimeRef.current = mimeType || 'video/webm';
      clipChunksRef.current = [];
      clipStartTsRef.current = Date.now();
      const rec = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      rec.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) {
          clipChunksRef.current.push(ev.data);
        }
      };
      rec.start(500);
      liveRecorderRef.current = rec;
    } catch {
      liveRecorderRef.current = null;
    }
  }, []);

  const finalizeAndSaveLiveClip = useCallback((): Promise<{
    videoUrl?: string;
    videoClipId?: string;
  }> => {
    if (uploadedVideoUrl) return Promise.resolve({});
    const stream = liveStreamRef.current;
    const rec = liveRecorderRef.current;
    const mime = clipMimeRef.current || 'video/webm';

    const persistBlob = (chunks: Blob[]) => {
      if (chunks.length === 0) return {};
      try {
        const clipBlob = new Blob(chunks, { type: mime });
        if (clipBlob.size <= 0) return {};
        const clipId = `clip-${Date.now()}`;
        const url = URL.createObjectURL(clipBlob);
        void savePitchVideoClip(clipId, clipBlob);
        return { videoUrl: url, videoClipId: clipId };
      } catch {
        return {};
      }
    };

    if (!rec || rec.state === 'inactive') {
      const saved = persistBlob([...clipChunksRef.current]);
      clipChunksRef.current = [];
      if (stream) startFreshLiveRecorder(stream);
      return Promise.resolve(saved);
    }

    return new Promise((resolve) => {
      let settled = false;
      const done = () => {
        if (settled) return;
        settled = true;
        const saved = persistBlob([...clipChunksRef.current]);
        clipChunksRef.current = [];
        if (liveStreamRef.current) {
          startFreshLiveRecorder(liveStreamRef.current);
        }
        resolve(saved);
      };
      const timer = window.setTimeout(done, 350);
      rec.onstop = () => {
        window.clearTimeout(timer);
        done();
      };
      try {
        liveRecorderRef.current = null;
        rec.stop();
      } catch {
        window.clearTimeout(timer);
        done();
      }
    });
  }, [cameraRequested, startFreshLiveRecorder, uploadedVideoUrl]);

  // 1. 一進入即啟動相機 (getUserMedia 優先 60fps 1280x720)
  useEffect(() => {
    const video = videoRef.current;
    if (!video || uploadedVideoUrl || !cameraRequested) return;

    let activeStream: MediaStream | null = null;
    let rotateTimer: number | null = null;
    let cancelled = false;

    const stopStream = () => {
      if (rotateTimer !== null) {
        window.clearInterval(rotateTimer);
        rotateTimer = null;
      }
      const rec = liveRecorderRef.current;
      liveRecorderRef.current = null;
      if (rec && rec.state !== 'inactive') {
        try {
          rec.onstop = null;
          rec.stop();
        } catch {
          // ignore
        }
      }
      clipChunksRef.current = [];
      liveStreamRef.current = null;
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

      // 強制優先鎖定 60fps 高幀率（支援 iOS Safari / Android Chrome / 筆電高幀率鏡頭）
      const tryConstraints: MediaStreamConstraints[] = [
        {
          video: {
            facingMode: { exact: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { exact: 60 },
          },
          audio: false,
        },
        {
          video: {
            facingMode: { exact: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { min: 50, ideal: 60 },
          },
          audio: false,
        },
        {
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { exact: 60 },
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

      liveStreamRef.current = activeStream;
      const vTrack = activeStream.getVideoTracks()[0];
      if (vTrack) {
        try {
          await vTrack.applyConstraints({ frameRate: { ideal: 60 } });
        } catch {
          // ignore if hardware locks fps
        }
        const actualFps = Math.round(vTrack.getSettings()?.frameRate ?? 60);
        if (actualFps > 0) setCameraFps(actualFps);
      }

      video.srcObject = activeStream;
      try {
        await video.play();
        if (!cancelled) {
          // 啟動 6 秒完整片段自動錄影（閒置時自動無縫換段，永不裁切檔頭，每球進手套自動存進 IndexedDB）
          startFreshLiveRecorder(activeStream);
          rotateTimer = window.setInterval(() => {
            if (cancelled || !liveStreamRef.current) return;
            if (activeTrackRef.current.length > 0 || isAutoRecordingRef.current) return;
            if (Date.now() - clipStartTsRef.current < 6000) return;
            const oldRec = liveRecorderRef.current;
            if (!oldRec || oldRec.state === 'inactive') {
              startFreshLiveRecorder(liveStreamRef.current);
              return;
            }
            liveRecorderRef.current = null;
            oldRec.onstop = () => {
              if (!cancelled && liveStreamRef.current) {
                startFreshLiveRecorder(liveStreamRef.current);
              }
            };
            try {
              oldRec.stop();
            } catch {
              startFreshLiveRecorder(liveStreamRef.current);
            }
          }, 2000);
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
  }, [cameraRequested, startFreshLiveRecorder, uploadedVideoUrl]);

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
      staticBinWindowRef.current = [];
      staticBinCountsRef.current.clear();
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
        (uploadedVideoUrl && autoReturnToCameraRef.current) ||
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
          // 球體飛行中（已有 >= 2 幀）推算下一幀預測位置，供 Worker 在遠端小球縮小漏抓時自動 2.8x 局部放大補抓
          let predictedRoi: { x: number; y: number; radius: number } | undefined;
          const trk = activeTrackRef.current;
          if (!region && trk.length >= 2) {
            const p1 = trk[trk.length - 2];
            const p2 = trk[trk.length - 1];
            const dtNow = timestampMs - p2.timestampMs;
            const dtPrev = Math.max(1, p2.timestampMs - p1.timestampMs);
            if (dtNow > 0 && dtNow <= 220) {
              predictedRoi = {
                x: p2.x + ((p2.x - p1.x) / dtPrev) * dtNow,
                y: p2.y + ((p2.y - p1.y) / dtPrev) * dtNow,
                radius: Math.max(54, w * 0.14),
              };
            }
          }
          onFrame(bitmap, timestampMs, region ? bitmap.width : w, region ? bitmap.height : h, predictedRoi);
        } catch (error) {
          bitmap.close();
          throw error;
        }
        // 非阻塞送入獨立骨架 Worker（僅全畫面模式，完全不影響球體偵測 FPS）
        if (onPoseFrame && !region) {
          void createImageBitmap(video)
            .then((poseBmp) => {
              if (stopped || offset !== timestampOffset) {
                poseBmp.close();
                return;
              }
              onPoseFrame(poseBmp, timestampMs, w, h);
            })
            .catch(() => {});
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

  // 上傳影片預設流暢播放即時畫出軌跡＋馬上出速；僅背景 5 秒自動錄影才走離線逐幀 seek
  const [scanRunId, setScanRunId] = useState(0);
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !uploadedVideoUrl || !isWorkerReady || discipline !== 'pitching') return;

    if (!autoReturnToCameraRef.current) {
      // 以 0.25x 平滑慢動作播放（每幀間隔 133ms > WASM 50ms），確保全幀不掉幀且即時出軌跡與精準球速
      video.playbackRate = 0.25;
      setCaptureMessage('');
      void video.play().catch((error: DOMException) => {
        if (error.name !== 'AbortError') {
          setCaptureMessage('無法自動播放。請按「播放並分析」。');
        }
      });
      return;
    }

    let cancelled = false;
    const seekTo = (timeSec: number) =>
      new Promise<void>((resolve) => {
        if (Math.abs(video.currentTime - timeSec) < 0.002) {
          resolve();
          return;
        }
        const onSeeked = () => {
          video.removeEventListener('seeked', onSeeked);
          resolve();
        };
        video.addEventListener('seeked', onSeeked, { once: true });
        video.currentTime = timeSec;
      });

    const waitLoaded = () =>
      new Promise<void>((resolve) => {
        if (video.readyState >= 2 && video.duration > 0) {
          resolve();
          return;
        }
        const onReady = () => {
          video.removeEventListener('loadeddata', onReady);
          resolve();
        };
        video.addEventListener('loadeddata', onReady, { once: true });
      });

    (async () => {
      video.pause();
      await waitLoaded();
      if (cancelled || !video.videoWidth || !video.duration || !Number.isFinite(video.duration)) return;

      const duration = Math.min(video.duration, 15);
      const scanFps = autoReturnToCameraRef.current && cameraFps >= 50 ? 60 : 30;
      const stepSec = 1 / scanFps;
      const totalSteps = Math.max(1, Math.floor(duration / stepSec));
      const allDetections: BallDetection[] = [];

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx2d = canvas.getContext('2d');

      setCaptureMessage('AI 逐幀掃描中 0%（零掉幀模式）...');
      setLiveTrackDots([]);

      for (let i = 0; i <= totalSteps; i++) {
        if (cancelled) return;
        const tSec = Math.min(duration, i * stepSec);
        await seekTo(tSec);
        if (cancelled) return;

        const tsMs = Math.round(tSec * 1000);
        let bitmap: ImageBitmap;
        try {
          bitmap = await createImageBitmap(video);
        } catch {
          if (!ctx2d) continue;
          ctx2d.drawImage(video, 0, 0, canvas.width, canvas.height);
          bitmap = await createImageBitmap(canvas);
        }

        const rawDet = await onScanFrame(bitmap, tsMs, video.videoWidth, video.videoHeight);
        if (cancelled) return;
        const det =
          rawDet && isFlightCorridorBall(rawDet, video.videoWidth, video.videoHeight, Boolean(region))
            ? rawDet
            : null;
        if (det) {
          // 若與上一幀間隔過長，視為新一球出手，清掉前面準備期零星雜點
          if (
            allDetections.length > 0 &&
            tsMs - allDetections[allDetections.length - 1].timestampMs > MAX_PITCH_SAMPLE_GAP_MS
          ) {
            allDetections.length = 0;
          }
          // 單幀空間跳躍過大（> 22% 畫面寬度）視為背景雜訊，不納入軌跡
          const prevDet = allDetections[allDetections.length - 1];
          if (prevDet && Math.hypot(det.x - prevDet.x, det.y - prevDet.y) > video.videoWidth * 0.22) {
            continue;
          }
          allDetections.push(det);
          detectedInFileRef.current += 1;
          setLiveTrackDots([...allDetections]);

        } else if (allDetections.length >= 5) {
          // 球進捕手手套後連續消失 >= 3 幀 (~100ms)，立即結算出手到進手套飛行速度
          const silenceMs = tsMs - allDetections[allDetections.length - 1].timestampMs;
          if (silenceMs >= 100) {
            const catchRecord =
              computeRealPitchFromDetections({
                detections: poseGate ? poseGate.filter(allDetections) : allDetections,
                videoWidth: video.videoWidth,
                videoHeight: video.videoHeight,
                cameraAngle,
                discipline,
                distanceMeters,
                checkpoints: manualCheckpoints,
                preferredPitchType: selectedPitchType,
                videoUrl: uploadedVideoUrl,
              });
            if (catchRecord) {
              measuredInFileRef.current += 1;
              setDismissedPitchId(null);
              onNewMeasuredPitch({ ...catchRecord, snapshotUrl: captureVideoSnapshot() });
              setCaptureMessage('');
              setProgress(0.06);
              setIsAnimatingRibbon(true);
              setFileEnded(true);
              if (autoReturnToCameraRef.current) {
                autoReturnToCameraRef.current = false;
                setUploadedVideoUrl(null);
              }
              return;
            }
            allDetections.length = 0;
          }
        }
        const pct = Math.min(99, Math.round((i / totalSteps) * 100));
        setCaptureMessage(`AI 逐幀掃描中 ${pct}%（已偵測到 ${allDetections.length} 幀棒球）`);
      }

      if (!cancelled) {
        setFileEnded(true);
        setCaptureMessage(
          detectedInFileRef.current === 0
            ? '掃描完成：此影片未偵測到飛行中的棒球，請確認光線與鏡頭固定。'
            : `掃描完成：偵測到 ${detectedInFileRef.current} 幀候選點，但未構成完整出手到進壘飛行軌跡。`
        );
        if (autoReturnToCameraRef.current) {
          autoReturnToCameraRef.current = false;
          setUploadedVideoUrl(null);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [discipline, isWorkerReady, uploadedVideoUrl, scanRunId]);

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

  const triggerAutoRecord5s = useCallback(() => {
    const stream = liveStreamRef.current;
    if (!stream || isAutoRecordingRef.current || typeof MediaRecorder === 'undefined') return;
    isAutoRecordingRef.current = true;
    if (currentPitch) setDismissedPitchId(currentPitch.id);
    resetTrack();

    const preferredTypes = ['video/mp4', 'video/webm;codecs=vp8', 'video/webm'];
    const mimeType = preferredTypes.find((t) => MediaRecorder.isTypeSupported?.(t)) ?? '';
    const chunks: Blob[] = [];
    let rec: MediaRecorder;
    try {
      rec = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    } catch {
      isAutoRecordingRef.current = false;
      return;
    }
    rec.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) chunks.push(ev.data);
    };
    const startedAt = performance.now();
    setAutoRecordLeftSec(5.0);
    setCaptureMessage(`${cameraFps}fps 自動錄影 5 秒中...`);
    const countdownTimer = window.setInterval(() => {
      const left = Math.max(0, 5.0 - (performance.now() - startedAt) / 1000);
      setAutoRecordLeftSec(Number(left.toFixed(1)));
      setCaptureMessage(`${cameraFps}fps 自動錄影中 ${left.toFixed(1)}s（結束後自動零掉幀測速）`);
    }, 200);

    rec.onstop = () => {
      window.clearInterval(countdownTimer);
      setAutoRecordLeftSec(null);
      isAutoRecordingRef.current = false;
      const blob = new Blob(chunks, { type: mimeType || 'video/webm' });
      if (blob.size > 0) {
        autoReturnToCameraRef.current = true;
        setUploadedVideoUrl(URL.createObjectURL(blob));
      }
    };
    rec.start(250);
    window.setTimeout(() => {
      if (rec.state !== 'inactive') {
        try {
          rec.stop();
        } catch {
          // ignore
        }
      }
    }, 5000);
  }, [cameraFps, currentPitch, resetTrack]);

  // 隨時感知投球動作：一開始投球立刻清除上一球軌跡，但「絕不中斷相機串流」，保持像測速槍一樣連續不斷測速＋背景自動存片
  useEffect(() => {
    if (!hasLiveCamera || uploadedVideoUrl || !isWorkerReady) return;
    if (actionPhase === 'pitching_start' && actionConfidence >= 0.55) {
      actionStartStreakRef.current += 1;
      if (actionStartStreakRef.current === 2 && currentPitch) {
        setDismissedPitchId(currentPitch.id);
      }
    } else {
      actionStartStreakRef.current = 0;
    }
  }, [actionConfidence, actionPhase, currentPitch, hasLiveCamera, isWorkerReady, uploadedVideoUrl]);

  useEffect(() => {
    if (!latestBall || !sentTimestampsRef.current.delete(latestBall.timestampMs)) return;
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;
    if (!isFlightCorridorBall(latestBall, video.videoWidth, video.videoHeight, Boolean(region))) {
      return;
    }

    // 1. 靜態燈光／看板抑制（14px 格點，90 幀滑動視窗內重複 >= 5 次直接封殺）
    const binKey = (Math.floor(latestBall.x / 14) << 16) | (Math.floor(latestBall.y / 14) & 0xffff);
    const binWin = staticBinWindowRef.current;
    const binMap = staticBinCountsRef.current;
    const prevCount = binMap.get(binKey) ?? 0;
    binMap.set(binKey, prevCount + 1);
    binWin.push(binKey);
    if (binWin.length > 90) {
      const oldest = binWin.shift()!;
      const c = (binMap.get(oldest) ?? 1) - 1;
      if (c <= 0) binMap.delete(oldest);
      else binMap.set(oldest, c);
    }
    if (prevCount >= 5) {
      return;
    }

    // 2. 前景投手球褲骨架屏蔽（球褲／膝蓋／白鞋反光絕不啟動假軌跡）
    const detSize = Math.max(latestBall.width, latestBall.height);
    if (
      !region &&
      poseGate?.isInsidePitcherPants(
        latestBall.x,
        latestBall.y,
        latestBall.timestampMs,
        detSize,
        activeTrackRef.current.length,
      )
    ) {
      return;
    }

    const track = activeTrackRef.current;
    const betweenSamples = track.length
      ? latestBall.timestampMs - track[track.length - 1].timestampMs
      : 0;
    // 允許 WASM 偶發掉 1–2 幀（最高 260ms 間隔不斷軌），避免 5–9 幀真實飛行球被中途清空
    const trackGapLimitMs = Math.max(MAX_PITCH_SAMPLE_GAP_MS, 260);
    if (track.length && (betweenSamples <= 0 || (!manualCheckpoints && betweenSamples > trackGapLimitMs) ||
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
    if (manualCheckpoints) {
      const sample = activeTrack[activeTrack.length - 1];
      const samples = endpointSamplesRef.current.filter(
        previous => sample.timestampMs - previous.timestampMs <= 1600,
      );
      samples.push(sample);
      endpointSamplesRef.current = samples;
      const result = measureCheckpointFlight(
        samples, manualCheckpoints, video.videoWidth, video.videoHeight, distanceMeters,
      );
      if (result) {
        if (uploadedVideoUrl) measuredInFileRef.current += 1;
        setCaptureMessage(`平均球速約 ${result.averageKmh.toFixed(1)} km/h（依影格估計 ${result.minKmh.toFixed(1)}–${result.maxKmh.toFixed(1)} km/h，距離 ${distanceMeters} 公尺）`);
        resetTrack();
      }
      return;
    }
    const record = computeRealPitchFromDetections({
      detections: poseGate ? poseGate.filter(activeTrack) : activeTrack,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      cameraAngle,
      discipline,
      distanceMeters,
      checkpoints: manualCheckpoints,
      preferredPitchType: selectedPitchType,
      videoUrl: uploadedVideoUrl ?? undefined,
      isFinalized: false,
    });

    if (record) {
      if (uploadedVideoUrl) measuredInFileRef.current += 1;
      const snap = captureVideoSnapshot();
      setDismissedPitchId(null);
      setCaptureMessage('');
      resetTrack();
      setProgress(0.06);
      setIsAnimatingRibbon(true);
      void finalizeAndSaveLiveClip().then((clip) => {
        onNewMeasuredPitch({
          ...record,
          snapshotUrl: snap,
          videoUrl: clip.videoUrl ?? record.videoUrl,
          videoClipId: clip.videoClipId ?? record.videoClipId,
        });
      });
      return;
    }

    if (flushTimerRef.current !== null) window.clearTimeout(flushTimerRef.current);
    flushTimerRef.current = window.setTimeout(() => {
      const rawBuffered = [...activeTrackRef.current];
      const buffered = poseGate ? poseGate.filter(rawBuffered) : rawBuffered;
      if (buffered.length >= 4 && video.videoWidth > 0 && video.videoHeight > 0) {
        const autoRec =
          computeRealPitchFromDetections({
            detections: buffered,
            videoWidth: video.videoWidth,
            videoHeight: video.videoHeight,
            cameraAngle,
            discipline,
            distanceMeters,
            checkpoints: manualCheckpoints,
            preferredPitchType: selectedPitchType,
            videoUrl: uploadedVideoUrl ?? undefined,
            isFinalized: true,
          });
        if (autoRec) {
          if (uploadedVideoUrl) measuredInFileRef.current += 1;
          const snap = captureVideoSnapshot();
          setDismissedPitchId(null);
          setCaptureMessage('');
          resetTrack();
          setProgress(0.06);
          setIsAnimatingRibbon(true);
          void finalizeAndSaveLiveClip().then((clip) => {
            onNewMeasuredPitch({
              ...autoRec,
              snapshotUrl: snap,
              videoUrl: clip.videoUrl ?? autoRec.videoUrl,
              videoClipId: clip.videoClipId ?? autoRec.videoClipId,
            });
          });
          return;
        }
      }
      if (buffered.length >= 3) {
        setCaptureMessage('有偵測到球，但還無法確認出手與接球。可調整測速框後再試。');
      }
      resetTrack();
    }, uploadedVideoUrl ? 750 : 200);
  }, [
    cameraAngle,
    captureVideoSnapshot,
    finalizeAndSaveLiveClip,
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
      displayPitch
        ? sampleQuadraticRibbon(displayPitch.ribbonPath, stageW, stageH, progress)
        : [],
    [displayPitch, progress]
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
    setDismissedPitchId(currentPitch?.id ?? null);
    resetTrack();
    setRegion(null); setChoosingRegion(false); setCorner(null);
    setUploadedVideoUrl(URL.createObjectURL(file));
  };

  const restartAnalysis = () => {
    const video = videoRef.current;
    if (!video || !isWorkerReady) return;
    setDismissedPitchId(currentPitch?.id ?? null);
    resetTrack();
    setFileEnded(false);
    measuredInFileRef.current = 0;
    detectedInFileRef.current = 0;
    setCaptureMessage('');
    if (uploadedVideoUrl && autoReturnToCameraRef.current) {
      setScanRunId((n) => n + 1);
      return;
    }
    if (video.readyState >= 1) video.currentTime = 0;
    video.playbackRate = uploadedVideoUrl ? 0.25 : 1;
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
  const choosingSource = !cameraRequested && !uploadedVideoUrl;

  return (
    <div className="relative mx-auto flex h-[calc(100dvh-64px)] max-h-[920px] w-full max-w-[460px] flex-col overflow-hidden rounded-none sm:rounded-3xl bg-slate-950 shadow-xl select-none">
      <input ref={fileInputRef} type="file" accept="video/*" onChange={handleVideoFileUpload} hidden />
      {choosingSource && (
        <section className="absolute inset-0 z-35 flex flex-col overflow-y-auto bg-[#101c18] text-[#f3f7f1]">
          <header className="flex items-center justify-between border-b border-white/10 px-6 py-4">
            <span className="text-lg font-extrabold tracking-tight">SmartScout</span>
            <span role="status" className="flex items-center gap-2 text-xs text-[#c5d2c8]">
              <span className={`h-2 w-2 rounded-full ${isWorkerReady ? 'bg-[#c6fc78]' : 'bg-amber-300'}`} />
              {workerError ? '模型載入失敗' : isWorkerReady ? '模型就緒' : '模型載入中'}
            </span>
          </header>
          <div className="relative flex flex-1 flex-col justify-center overflow-hidden px-6 py-6 sm:px-9">
            <div aria-hidden="true" className="pointer-events-none absolute inset-x-0 top-0 h-64 bg-[radial-gradient(ellipse_at_top_right,rgba(128,190,88,.22),transparent_70%)]" />
            <div aria-hidden="true" className="pointer-events-none absolute -right-16 -top-8 h-60 w-60 rotate-45 rounded-[28px] border border-[#c6fc78]/25">
              <div className="absolute inset-8 rounded-[18px] border border-[#c6fc78]/25" />
              <div className="absolute inset-[4.5rem] rounded-xl border border-[#c6fc78]/25" />
            </div>
            <div className="relative">
              <h1 className="text-5xl font-extrabold tracking-tight sm:text-6xl">投球測速</h1>
              <p className="mt-3 max-w-[19rem] text-sm leading-6 text-[#c5d2c8]">使用相機即時追蹤，或匯入投球影片分析球路。</p>
              <dl className="my-6 grid grid-cols-2 divide-x divide-white/10 rounded-2xl border border-white/10 bg-[#182821]">
                <div className="px-4 py-3">
                  <dt className="text-xs text-[#b0c4b5]">拍攝視角</dt>
                  <dd className="mt-1 text-sm font-bold">{cameraAngle === 'behind_pitcher' ? '投手後方' : '捕手後方'}</dd>
                </div>
                <div className="px-4 py-3">
                  <dt className="text-xs text-[#b0c4b5]">設定距離</dt>
                  <dd className="mt-1 text-sm font-bold tabular-nums">{distanceMeters} 公尺</dd>
                </div>
              </dl>
              <div className="grid gap-3">
                <button type="button" onClick={() => { setCameraRequested(true); setCaptureMessage('正在開啟相機…'); }}
                  className="min-h-12 rounded-xl bg-[#c6fc78] px-5 py-3 text-left font-bold text-[#132013] transition hover:bg-[#dbffa4] focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#c6fc78]">
                  開啟相機
                </button>
                <button type="button" onClick={() => fileInputRef.current?.click()}
                  className="min-h-12 rounded-xl border border-white/20 bg-white/5 px-5 py-3 text-left font-bold transition hover:bg-white/10 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#c6fc78]">
                  匯入影片
                </button>
              </div>
            </div>
          </div>
          <footer className="border-t border-white/10 px-6 py-4 text-xs leading-5 text-[#b0c4b5]">
            請固定手機，讓投手、球路和捕手都入鏡。
          </footer>
        </section>
      )}
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
              ? `${area}在 ${detected} 幀找到疑似球的位置，但無法確認完整球路。可按「手動標記出手與接球時間」計算。`
              : `${area}沒有偵測到球。可按「手動標記出手與接球時間」計算。`);
          }
        }}
        onLoadedMetadata={(event) => {
          const video = event.currentTarget;
          setFrameSize({ width: video.videoWidth, height: video.videoHeight });
          video.playbackRate = uploadedVideoUrl ? 0.25 : 1;
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
      <div hidden={choosingSource} className="relative z-20 flex items-center justify-between px-3.5 pt-3.5 pb-2 text-white bg-gradient-to-b from-black/65 to-transparent">
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
          {hasLiveCamera && !uploadedVideoUrl && (
            <button
              type="button"
              onClick={triggerAutoRecord5s}
              disabled={!isWorkerReady || autoRecordLeftSec !== null}
              className={`rounded-full px-2.5 py-1 text-[11px] font-bold transition ${
                autoRecordLeftSec !== null
                  ? 'bg-red-600 text-white animate-pulse'
                  : 'bg-emerald-600/85 hover:bg-emerald-500 text-white'
              }`}
            >
              {autoRecordLeftSec !== null ? `錄影 ${autoRecordLeftSec.toFixed(1)}s` : `${cameraFps}fps 錄5秒`}
            </button>
          )}
          {displayPitch && (
            <button
              type="button"
              onClick={() => setDismissedPitchId(displayPitch.id)}
              className="rounded-full bg-black/45 border border-white/25 px-2.5 py-1 text-[11px] font-bold text-white hover:bg-black/65 transition"
            >
              ✕ 清除軌跡
            </button>
          )}
          {uploadedVideoUrl && (
            <button
              type="button"
              onClick={() => {
                setDismissedPitchId(currentPitch?.id ?? null);
                resetTrack();
                setCalibration(null);
                setCaptureMessage('');
                setRegion(null); setChoosingRegion(false); setCorner(null);
                setCameraRequested(true);
                setUploadedVideoUrl(null);
              }}
              className="rounded-xl bg-white/15 px-2.5 py-1.5 text-xs font-bold text-white"
            >
              開啟鏡頭
            </button>
          )}

          <button type="button" onClick={() => fileInputRef.current?.click()}
            className="min-h-9 rounded-lg border border-white/20 bg-black/45 px-3 text-xs font-bold text-white transition hover:bg-black/65">
            匯入影片
          </button>

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

      <div hidden={choosingSource} className="relative z-40 mx-auto max-w-[92%] rounded-xl border border-white/15 bg-[#101c18]/90 px-3 py-2 text-center text-xs leading-5 text-white">
        <p role="status">
          {workerError || captureMessage || (discipline === 'hitting'
            ? '目前支援投球測速。請在設定中切換為投球。'
            : calibration
              ? calibration.release ? '點選球到達捕手手套的位置' : '點選投手放球的位置'
              : manualCheckpoints
                ? '使用你設定的出手及接球位置，依設定距離估算平均球速。'
                : '自動辨識出手和接球。請固定鏡頭，讓投手、球路與捕手都入鏡。')}
        </p>
        {(uploadedVideoUrl || hasLiveCamera) && !calibrating && discipline === 'pitching' && <>
          <button type="button" className="mt-2 rounded-lg border border-white/20 px-3 py-2 font-semibold text-white hover:bg-white/10"
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
            className="mt-2 rounded-lg bg-[#c6fc78] px-3 py-2 font-bold text-[#132013] disabled:opacity-60"
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
              if (!calibrating && uploadedVideoUrl && videoRef.current) {
                videoRef.current.pause();
                setCalibrationTime(videoRef.current.currentTime);
              }
              setCalibration(calibrating ? null : {});
            }}
            className="mt-1 rounded px-2 py-1 font-semibold text-sky-200 underline"
          >
            {calibrating ? '取消調整' : '調整測速框'}
          </button>
        )}
        {calibrating && uploadedVideoUrl && Number.isFinite(videoRef.current?.duration) && (
          <label className="block">
            拖曳到出手或接球畫面（{calibrationTime.toFixed(2)} 秒）
            <input className="block w-full" type="range" min={0} max={videoRef.current?.duration}
              step={0.001} value={calibrationTime} onChange={event => {
                const time = Number(event.currentTarget.value);
                if (videoRef.current) videoRef.current.currentTime = time;
                setCalibrationTime(time);
              }} />
          </label>
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
            手動標記出手與接球時間
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
        {!calibrating && displayPitch?.measurement?.automaticEndpoints && [
          { x: displayPitch.ribbonPath.startX, y: displayPitch.ribbonPath.startY, label: '推估出手' },
          { x: displayPitch.ribbonPath.endX, y: displayPitch.ribbonPath.endY, label: '推估接球' },
        ].map(({ x, y, label }) => (
          <g key={label} transform={`translate(${x * stageW}, ${y * stageH})`}>
            <circle r={6} fill="#38bdf8" stroke="white" strokeWidth={2} />
            <text x={10} y={-5} fontSize={14} fill="white" stroke="#0f172a" strokeWidth={3}
              paintOrder="stroke">{label}</text>
          </g>
        ))}
        {!calibrating && showRibbon && displayPitch && ribbonSamples.length >= 2 && (
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

        {!displayPitch && showRibbon && liveTrackDots.length >= 2 && (
          <g filter="url(#liveRibbonShadow)">
            <polyline
              fill="none"
              stroke="url(#liveRibbonGrad)"
              strokeWidth="12"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={liveTrackDots
                .map((d) => `${((d.x / vw) * stageW).toFixed(1)},${((d.y / vh) * stageH).toFixed(1)}`)
                .join(' ')}
            />
            <polyline
              fill="none"
              stroke="#ffffff"
              strokeOpacity="0.65"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={liveTrackDots
                .map((d) => `${((d.x / vw) * stageW).toFixed(1)},${((d.y / vh) * stageH).toFixed(1)}`)
                .join(' ')}
            />
          </g>
        )}
        {!displayPitch && showRibbon && liveTrackDots.map((d, i) => (
          <circle
            key={i}
            cx={(d.x / vw) * stageW}
            cy={(d.y / vh) * stageH}
            r={i === liveTrackDots.length - 1 ? 10 : 6}
            fill={i === liveTrackDots.length - 1 ? '#ffffff' : '#ef233c'}
            stroke={i === liveTrackDots.length - 1 ? '#ef233c' : '#ffffff'}
            strokeWidth={i === liveTrackDots.length - 1 ? 3 : 2}
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
                <div className="text-xs font-medium text-neutral-300">
                  預估初速 · 距離 {distanceMeters}m
                </div>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-[42px] font-bold leading-tight tracking-tight tabular-nums">
                    {displayPitch ? displayPitch.speedKmh.toFixed(1) : '--.-'}
                  </span>
                  <span className="text-base font-medium text-neutral-300">km/h</span>
                </div>
              </div>

              <div>
                <select
                  value={displayPitch ? displayPitch.pitchType : selectedPitchType}
                  onChange={(e) => {
                    setSelectedPitchType(e.target.value);
                    if (displayPitch) {
                      onChangePitchType(displayPitch.id, e.target.value);
                    }
                  }}
                  style={{
                    backgroundColor: getPitchTypeColor(
                      displayPitch ? displayPitch.pitchType : selectedPitchType
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

            {displayPitch && (
              <div className="mt-2.5 grid grid-cols-4 gap-x-2 gap-y-2 border-t border-white/10 pt-2.5 text-left">
                <div>
                  <div className="text-[10px] text-neutral-400">垂直變化量</div>
                  <div className="text-sm font-bold tabular-nums">
                    {displayPitch.verticalBreakCm !== undefined ? `${displayPitch.verticalBreakCm.toFixed(1)}` : '--'}
                    <span className="ml-0.5 text-[10px] font-normal text-neutral-300">cm</span>
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-neutral-400">水平變化量</div>
                  <div className="text-sm font-bold tabular-nums">
                    {displayPitch.horizontalBreakCm !== undefined ? `${displayPitch.horizontalBreakCm.toFixed(1)}` : '--'}
                    <span className="ml-0.5 text-[10px] font-normal text-neutral-300">cm</span>
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-neutral-400">轉速</div>
                  <div className="text-sm font-bold tabular-nums">
                    {displayPitch.totalSpinRpm ?? '--'}
                    <span className="ml-0.5 text-[10px] font-normal text-neutral-300">rpm</span>
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-neutral-400">有效轉速</div>
                  <div className="text-sm font-bold tabular-nums">
                    {displayPitch.activeSpinRpm ?? '--'}
                    <span className="ml-0.5 text-[10px] font-normal text-neutral-300">rpm</span>
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-neutral-400">效率</div>
                  <div className="text-sm font-bold tabular-nums">
                    {displayPitch.efficiencyPct ?? '--'}
                    <span className="ml-0.5 text-[10px] font-normal text-neutral-300">%</span>
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-neutral-400">旋轉方向</div>
                  <div className="text-sm font-bold tabular-nums">
                    {displayPitch.spinDirection ?? '--:--'}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-neutral-400">陀螺角度</div>
                  <div className="text-sm font-bold tabular-nums">
                    {displayPitch.gyroDegree !== undefined ? `${displayPitch.gyroDegree}°` : '--'}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-neutral-400">飛行 / 幀數</div>
                  <div className="text-xs font-semibold text-neutral-200 tabular-nums">
                    {displayPitch.measurement ? `${Math.round(displayPitch.measurement.flightTimeMs)}ms · ${displayPitch.measurement.detectionCount}幀` : '--'}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* 最下方：播放/暫停鈕與白色逐幀時間軸滑桿 */}
        <div className="flex items-center gap-2.5 px-1">
          <button
            type="button"
            disabled={!displayPitch}
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
            disabled={!displayPitch}
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
