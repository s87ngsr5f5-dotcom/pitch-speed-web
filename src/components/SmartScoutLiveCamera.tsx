import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  SMARTSCOUT_PITCH_TYPES,
  SmartScoutPitchRecord,
  computeRealPitchFromDetections,
  getPitchTypeColor,
} from '../data/smartScoutDefaultSessions';
import { BallDetection } from '../types';

interface RibbonSample {
  x: number;
  y: number;
  r: number;
}

function sampleQuadraticRibbon(
  path: SmartScoutPitchRecord['ribbonPath'],
  width: number,
  height: number,
  maxT: number,
  steps = 44
): RibbonSample[] {
  const sx = path.startX * width;
  const sy = path.startY * height;
  const cx = path.ctrlX * width;
  const cy = path.ctrlY * height;
  const ex = path.endX * width;
  const ey = path.endY * height;

  const scale = Math.min(width, height) / 420;
  const sr = path.startR * scale;
  const er = path.endR * scale;

  const out: RibbonSample[] = [];
  const count = Math.max(2, Math.round(steps * Math.max(0.05, maxT)));
  for (let i = 0; i <= count; i++) {
    const t = (i / count) * maxT;
    const inv = 1 - t;
    const x = inv * inv * sx + 2 * inv * t * cx + t * t * ex;
    const y = inv * inv * sy + 2 * inv * t * cy + t * t * ey;
    const r = sr + (er - sr) * Math.pow(t, 0.85);
    out.push({ x, y, r });
  }
  return out;
}

function buildTaperedRibbonPolygon(samples: RibbonSample[]): string {
  if (samples.length < 2) return '';
  const leftPts: Array<{ x: number; y: number }> = [];
  const rightPts: Array<{ x: number; y: number }> = [];

  for (let i = 0; i < samples.length; i++) {
    const curr = samples[i];
    const prev = samples[Math.max(0, i - 1)];
    const next = samples[Math.min(samples.length - 1, i + 1)];
    const dx = next.x - prev.x;
    const dy = next.y - prev.y;
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len;
    const ny = dx / len;

    leftPts.push({ x: curr.x + nx * curr.r, y: curr.y + ny * curr.r });
    rightPts.push({ x: curr.x - nx * curr.r, y: curr.y - ny * curr.r });
  }

  const all = [...leftPts, ...rightPts.reverse()];
  return all.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
}

interface SmartScoutLiveCameraProps {
  currentPitch: SmartScoutPitchRecord | null;
  cameraAngle: 'behind_pitcher' | 'behind_catcher';
  discipline: 'pitching' | 'hitting';
  distanceMeters: number;
  isWorkerReady: boolean;
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
  isBusyRef,
  latestBall,
  onBackToSession,
  onDeletePitch,
  onFrame,
  onNewMeasuredPitch,
  onChangePitchType,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [cameraState, setCameraState] = useState<'starting' | 'active' | 'no_camera'>('starting');
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string | null>(null);
  const [selectedPitchType, setSelectedPitchType] = useState<string>('四縫線快速球');
  const [showRibbon, setShowRibbon] = useState<boolean>(true);
  const [showStatsCard, setShowStatsCard] = useState<boolean>(true);
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 0.5 | 0.25>(1);
  const [progress, setProgress] = useState<number>(1);
  const [isAnimatingRibbon, setIsAnimatingRibbon] = useState<boolean>(false);
  const [liveTrackDots, setLiveTrackDots] = useState<BallDetection[]>([]);

  const rafRef = useRef<number | null>(null);
  const activeTrackRef = useRef<BallDetection[]>([]);
  const flushTimerRef = useRef<number | null>(null);

  // 1. 自動開啟真實相機
  useEffect(() => {
    const video = videoRef.current;
    if (!video || uploadedVideoUrl) return;

    let activeStream: MediaStream | null = null;
    let cancelled = false;

    const stopStream = () => {
      if (activeStream) {
        activeStream.getTracks().forEach((t) => t.stop());
        activeStream = null;
      }
      if (video.srcObject) {
        video.srcObject = null;
      }
    };

    const startCamera = async () => {
      setCameraState('starting');
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraState('no_camera');
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
        setCameraState('no_camera');
        return;
      }

      video.srcObject = activeStream;
      try {
        await video.play();
        setCameraState('active');
      } catch {
        setCameraState('no_camera');
      }
    };

    startCamera();

    return () => {
      cancelled = true;
      stopStream();
    };
  }, [uploadedVideoUrl]);

  // 2. 將真實相機或載入的影片畫面持續送入 ONNX Worker 進行球體偵測
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let stopped = false;
    let rafId: number | null = null;

    const loop = (now: DOMHighResTimeStamp) => {
      if (stopped) return;
      if (
        isWorkerReady &&
        !isBusyRef.current &&
        video.readyState >= 2 &&
        !video.paused &&
        !video.ended &&
        video.videoWidth > 0
      ) {
        isBusyRef.current = true;
        const w = video.videoWidth;
        const h = video.videoHeight;
        const ts = uploadedVideoUrl ? video.currentTime * 1000 : now;
        createImageBitmap(video)
          .then((bmp) => onFrame(bmp, ts, w, h))
          .catch(() => {
            isBusyRef.current = false;
          });
      }
      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => {
      stopped = true;
      if (rafId !== null) cancelAnimationFrame(rafId);
    };
  }, [isWorkerReady, isBusyRef, onFrame, uploadedVideoUrl]);

  // 擷取當前相機畫面作為該球紀錄快照
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

  // 結算一段真實偵測到的球體軌跡
  const finalizeActiveTrack = useCallback(() => {
    const track = activeTrackRef.current;
    activeTrackRef.current = [];
    setLiveTrackDots([]);

    if (track.length < 4) return;
    const video = videoRef.current;
    const vw = video?.videoWidth || 1280;
    const vh = video?.videoHeight || 720;

    // 確認軌跡具有足夠位移（過濾靜止背景誤檢）
    const totalDisp = Math.hypot(
      track[track.length - 1].x - track[0].x,
      track[track.length - 1].y - track[0].y
    );
    if (totalDisp < Math.min(vw, vh) * 0.08) return;

    const snapshotUrl = captureVideoSnapshot();
    const record = computeRealPitchFromDetections({
      detections: track,
      videoWidth: vw,
      videoHeight: vh,
      cameraAngle,
      discipline,
      distanceMeters,
      preferredPitchType: selectedPitchType,
      snapshotUrl,
      videoUrl: uploadedVideoUrl ?? undefined,
    });

    if (record) {
      onNewMeasuredPitch(record);
      setProgress(0.06);
      setIsAnimatingRibbon(true);
    }
  }, [
    cameraAngle,
    captureVideoSnapshot,
    discipline,
    distanceMeters,
    onNewMeasuredPitch,
    selectedPitchType,
    uploadedVideoUrl,
  ]);

  // 3. 接收 ONNX Worker 真實偵測到的每一幀球體座標
  useEffect(() => {
    if (!latestBall) return;

    const track = activeTrackRef.current;
    if (track.length > 0) {
      const prev = track[track.length - 1];
      const dt = latestBall.timestampMs - prev.timestampMs;
      if (dt <= 0) return;

      // 若間隔超過 260ms，先結算上一球
      if (dt > 260) {
        finalizeActiveTrack();
      } else {
        // 檢查移動連續性，排除畫面遠處突發雜訊
        const video = videoRef.current;
        const maxJump = Math.max(video?.videoWidth || 1280, video?.videoHeight || 720) * 0.38;
        const jump = Math.hypot(latestBall.x - prev.x, latestBall.y - prev.y);
        if (jump > maxJump) return;
      }
    }

    activeTrackRef.current.push(latestBall);
    setLiveTrackDots([...activeTrackRef.current]);

    if (flushTimerRef.current !== null) {
      window.clearTimeout(flushTimerRef.current);
    }
    flushTimerRef.current = window.setTimeout(() => {
      finalizeActiveTrack();
    }, 280);
  }, [latestBall, finalizeActiveTrack]);

  useEffect(() => {
    return () => {
      if (flushTimerRef.current !== null) {
        window.clearTimeout(flushTimerRef.current);
      }
    };
  }, []);

  // 紅色彩帶延伸動畫
  useEffect(() => {
    if (!isAnimatingRibbon) {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      return;
    }

    let lastTime = performance.now();
    const durationMs = 950 / playbackSpeed;

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
    if (uploadedVideoUrl) URL.revokeObjectURL(uploadedVideoUrl);
    activeTrackRef.current = [];
    setLiveTrackDots([]);
    setUploadedVideoUrl(URL.createObjectURL(file));
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
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, stageW, stageH);
    }

    if (showRibbon && ribbonSamples.length >= 2) {
      ctx.fillStyle = 'rgba(255, 38, 28, 0.88)';
      for (const s of ribbonSamples) {
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (showStatsCard) {
      ctx.fillStyle = 'rgba(24, 16, 14, 0.86)';
      ctx.beginPath();
      ctx.roundRect(22, stageH - 310, stageW - 44, 235, 22);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.font = '900 54px sans-serif';
      ctx.fillText(`${currentPitch.speedKmh.toFixed(1)} km/h`, 46, stageH - 220);
    }
    const link = document.createElement('a');
    link.download = `smartscout-pitch-${currentPitch.pitchNumber}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  const videoEl = videoRef.current;
  const vw = videoEl?.videoWidth || 1280;
  const vh = videoEl?.videoHeight || 720;

  return (
    <div className="relative mx-auto flex h-[calc(100dvh-64px)] max-h-[920px] w-full max-w-[460px] flex-col overflow-hidden rounded-none sm:rounded-3xl bg-slate-900 shadow-xl select-none">
      {/* 真實相機串流或使用者載入的真實投球影片 */}
      <video
        ref={videoRef}
        src={uploadedVideoUrl ?? undefined}
        playsInline
        muted
        autoPlay
        onEnded={() => finalizeActiveTrack()}
        className="absolute inset-0 h-full w-full object-cover bg-slate-950"
      />

      {/* 若裝置無相機且尚未載入影片，顯示乾淨簡潔的開啟相機/載入影片按鈕 */}
      {cameraState === 'no_camera' && !uploadedVideoUrl && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center p-6 text-center bg-slate-900/90">
          <div className="rounded-3xl bg-white p-6 text-slate-900 shadow-2xl max-w-xs w-full space-y-4">
            <div className="text-base font-extrabold text-slate-900">
              未偵測到相機權限
            </div>
            <label className="cursor-pointer block w-full rounded-2xl bg-[#0b72b9] py-3 text-sm font-extrabold text-white shadow hover:bg-[#0962a0] transition">
              載入投球影片測速
              <input
                type="file"
                accept="video/*"
                onChange={handleVideoFileUpload}
                className="hidden"
              />
            </label>
          </div>
        </div>
      )}

      {/* 1:1 SmartScout 頂部列：<   投球 #N (或 自動測速中)   📂 / 🗑 */}
      <div className="relative z-20 flex items-center justify-between px-4 pt-4 pb-2 text-white bg-gradient-to-b from-black/55 to-transparent">
        <button
          type="button"
          onClick={onBackToSession}
          className="flex h-10 w-10 items-center justify-center rounded-full hover:bg-black/25 transition"
          aria-label="返回列表"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>

        <div className="flex items-center gap-2">
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              isWorkerReady ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
            }`}
          />
          <h2 className="text-lg font-bold tracking-wide drop-shadow">
            {currentPitch ? `投球 #${currentPitch.pitchNumber}` : '自動測速中'}
          </h2>
        </div>

        <div className="flex items-center gap-1">
          {uploadedVideoUrl && (
            <button
              type="button"
              onClick={() => {
                URL.revokeObjectURL(uploadedVideoUrl);
                setUploadedVideoUrl(null);
              }}
              className="rounded-lg bg-white/20 px-2 py-1 text-[11px] font-bold text-white backdrop-blur-sm"
            >
              切回相機
            </button>
          )}
          <label
            title="載入影片測速"
            className="cursor-pointer flex h-9 w-9 items-center justify-center rounded-full hover:bg-black/25 transition"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
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
              className="flex h-9 w-9 items-center justify-center rounded-full hover:bg-black/25 transition"
              aria-label="刪除"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                <path d="M10 11v6M14 11v6" />
                <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* 實測 3D 紅色漸變彗星彩帶軌跡 + 即時偵測點 */}
      <svg
        viewBox={`0 0 ${stageW} ${stageH}`}
        className="pointer-events-none absolute inset-0 z-10 h-full w-full"
      >
        <defs>
          <linearGradient id="liveRibbonGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#ff2e24" stopOpacity="0.95" />
            <stop offset="70%" stopColor="#ff2419" stopOpacity="0.88" />
            <stop offset="100%" stopColor="#d9231b" stopOpacity="0.76" />
          </linearGradient>
          <filter id="liveRibbonShadow" x="-15%" y="-15%" width="130%" height="130%">
            <feDropShadow dx="0" dy="3" stdDeviation="4" floodColor="#7f1d1d" floodOpacity="0.45" />
          </filter>
        </defs>

        {/* 已完成測速之 3D 紅色彩帶軌跡 */}
        {showRibbon && currentPitch && ribbonSamples.length >= 2 && (
          <g filter="url(#liveRibbonShadow)">
            <polygon points={polygonPoints} fill="url(#liveRibbonGrad)" />
            {ribbonSamples.map((s, idx) => (
              <circle
                key={idx}
                cx={s.x}
                cy={s.y}
                r={s.r}
                fill="#ff281e"
                fillOpacity={idx === 0 ? 0.65 : 0.86}
              />
            ))}
            {firstSample && lastSample && (
              <circle
                cx={firstSample.r >= lastSample.r ? firstSample.x : lastSample.x}
                cy={firstSample.r >= lastSample.r ? firstSample.y : lastSample.y}
                r={Math.max(firstSample.r, lastSample.r)}
                fill="#c82219"
                fillOpacity="0.62"
                stroke="rgba(255, 120, 110, 0.45)"
                strokeWidth="1.5"
              />
            )}
          </g>
        )}

        {/* 球飛行過程中的即時追蹤軌跡點 */}
        {liveTrackDots.map((d, i) => (
          <circle
            key={i}
            cx={(d.x / vw) * stageW}
            cy={(d.y / vh) * stageH}
            r={8}
            fill="#ff281e"
            stroke="#ffffff"
            strokeWidth="2"
          />
        ))}
      </svg>

      {/* 右側 4 顆 SmartScout 懸浮操作按鈕 */}
      <div className="absolute right-4 top-[38%] z-20 flex flex-col items-center gap-5">
        <button
          type="button"
          onClick={() => setShowRibbon((v) => !v)}
          className="flex h-11 w-11 items-center justify-center rounded-full transition active:scale-90"
        >
          <span
            className={`flex h-9 w-9 items-center justify-center rounded-full border-[2.5px] border-white shadow-lg ${
              showRibbon ? 'bg-[#d9261c]' : 'bg-black/40'
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

      {/* 1:1 SmartScout 底部深色半透明數據卡 (未測得球前顯示 --.- 待命，實測後顯示真實數據) */}
      <div className="relative z-20 mt-auto px-3.5 pb-3.5 space-y-3">
        {showStatsCard && (
          <div className="rounded-2xl bg-[#18110e]/85 backdrop-blur-md px-4 py-3.5 text-white shadow-2xl border border-white/10">
            <div className="flex items-center justify-between gap-2">
              <div>
                <div className="text-xs font-medium text-neutral-300">初速</div>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-[44px] font-bold leading-tight tracking-tight tabular-nums">
                    {currentPitch ? currentPitch.speedKmh.toFixed(1) : '--.-'}
                  </span>
                  <span className="text-lg font-medium text-neutral-300">km/h</span>
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

            <div className="mt-2 grid grid-cols-4 gap-1 text-center">
              <div>
                <div className="text-[11px] text-neutral-400">垂直變化量</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {currentPitch ? currentPitch.verticalBreakCm.toFixed(1) : '--'}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">cm</span>
                </div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">水平變化量</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {currentPitch ? currentPitch.horizontalBreakCm.toFixed(1) : '--'}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">cm</span>
                </div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">轉速</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {currentPitch ? currentPitch.totalSpinRpm : '--'}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">rpm</span>
                </div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">有效轉速</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {currentPitch ? currentPitch.activeSpinRpm : '--'}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">rpm</span>
                </div>
              </div>
            </div>

            <div className="mt-3 grid grid-cols-4 gap-1 text-center">
              <div>
                <div className="text-[11px] text-neutral-400">效率</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {currentPitch ? currentPitch.efficiencyPct : '--'}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">%</span>
                </div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">旋轉方向</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {currentPitch ? currentPitch.spinDirection : '--:--'}
                </div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">陀螺角度</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {currentPitch ? currentPitch.gyroDegree : '--'}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">°</span>
                </div>
              </div>
              <div />
            </div>
          </div>
        )}

        {/* 底部白色滑桿 */}
        <div className="px-1">
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
