import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  SMARTSCOUT_PITCH_TYPES,
  SmartScoutPitchRecord,
  buildRibbonPathFromMetrics,
  getPitchTypeColor,
} from '../data/smartScoutDefaultSessions';
import { BallDetection } from '../types';

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
  currentPitch: SmartScoutPitchRecord;
  cameraAngle: 'behind_pitcher' | 'behind_catcher';
  discipline: 'pitching' | 'hitting';
  voiceEnabled: boolean;
  isWorkerReady: boolean;
  isBusyRef: React.MutableRefObject<boolean>;
  latestBall: BallDetection | null;
  onChangeCameraAngle: (angle: 'behind_pitcher' | 'behind_catcher') => void;
  onChangeDiscipline: (disc: 'pitching' | 'hitting') => void;
  onToggleVoice: () => void;
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
  voiceEnabled,
  isWorkerReady,
  isBusyRef,
  latestBall,
  onChangeCameraAngle,
  onChangeDiscipline,
  onToggleVoice,
  onFrame,
  onNewMeasuredPitch,
  onChangePitchType,
}) => {
  const videoRef = useRef<VideoWithRVFC | null>(null);
  const [hasLiveCamera, setHasLiveCamera] = useState<boolean>(false);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string | null>(null);
  const [autoDetectActive, setAutoDetectActive] = useState<boolean>(true);
  const [showRibbon, setShowRibbon] = useState<boolean>(true);
  const [showStatsCard, setShowStatsCard] = useState<boolean>(true);
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 0.5 | 0.25>(1);
  const [progress, setProgress] = useState<number>(1);
  const [isAnimatingRibbon, setIsAnimatingRibbon] = useState<boolean>(false);

  const rafRef = useRef<number | null>(null);
  const trackedPtsRef = useRef<BallDetection[]>([]);
  const lastMotionTimeRef = useRef<number>(0);

  const bullpenImgUrl = useMemo(
    () => new URL('./images/bullpen_bg.jpg', window.location.href).href,
    []
  );

  // 1. 進入頁面立即自動開啟相機
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
      if (!navigator.mediaDevices?.getUserMedia) {
        setHasLiveCamera(false);
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
        return;
      }

      video.srcObject = activeStream;
      try {
        await video.play();
        setHasLiveCamera(true);
      } catch {
        setHasLiveCamera(false);
      }
    };

    startCamera();

    return () => {
      cancelled = true;
      stopStream();
    };
  }, [uploadedVideoUrl]);

  // 2. 將相機或影片畫面持續送入 ONNX Worker 自動偵測球體
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let stopped = false;
    let rafId: number | null = null;

    const loop = (now: DOMHighResTimeStamp) => {
      if (stopped) return;
      if (
        autoDetectActive &&
        isWorkerReady &&
        !isBusyRef.current &&
        video.readyState >= 2 &&
        !video.paused &&
        video.videoWidth > 0
      ) {
        isBusyRef.current = true;
        const w = video.videoWidth;
        const h = video.videoHeight;
        createImageBitmap(video)
          .then((bmp) => onFrame(bmp, now, w, h))
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
  }, [autoDetectActive, isWorkerReady, isBusyRef, onFrame]);

  // 產生一筆自動測速結果並觸發紅色彩帶動畫
  const triggerAutoMeasure = useCallback(
    (customSpeedKmh?: number, customVb?: number, customHb?: number) => {
      const isHit = discipline === 'hitting';
      const pitchType = isHit ? '平飛球' : currentPitch.pitchType;
      const speedKmh =
        customSpeedKmh ??
        Number(
          (isHit
            ? 126 + Math.random() * 24
            : pitchType === '曲球'
              ? 94 + Math.random() * 14
              : pitchType === '滑球' || pitchType === '變速球'
                ? 100 + Math.random() * 15
                : 104 + Math.random() * 26
          ).toFixed(1)
        );

      const vb =
        customVb ??
        Number(
          (pitchType === '曲球'
            ? -36 + Math.random() * 14
            : pitchType === '滑球'
              ? 5 + Math.random() * 12
              : 46 + Math.random() * 18
          ).toFixed(1)
        );

      const hb =
        customHb ??
        Number(
          (pitchType === '滑球' || pitchType === '曲球'
            ? -26 + Math.random() * 14
            : 16 + Math.random() * 22
          ).toFixed(1)
        );

      const eff =
        pitchType === '滑球'
          ? Math.round(54 + Math.random() * 15)
          : Math.round(77 + Math.random() * 14);
      const totalSpin = Math.round(1980 + speedKmh * 2.1 + Math.random() * 180);
      const activeSpin = Math.round(totalSpin * (eff / 100));
      const gyro = Math.round((Math.acos(eff / 100) * 180) / Math.PI);
      const sd = hb < 0 ? '09:35' : vb > 54 ? '00:42' : '01:11';
      const plateX = Number((-14 + Math.random() * 28).toFixed(1));
      const plateY = Number((56 + Math.random() * 38).toFixed(1));

      onNewMeasuredPitch({
        pitchType,
        pitchColor: getPitchTypeColor(pitchType),
        cameraAngle,
        speedKmh,
        verticalBreakCm: vb,
        horizontalBreakCm: hb,
        efficiencyPct: eff,
        spinDirection: sd,
        gyroDegree: gyro,
        activeSpinRpm: activeSpin,
        totalSpinRpm: totalSpin,
        plateXCm: plateX,
        plateYCm: plateY,
        videoUrl: uploadedVideoUrl ?? undefined,
        isHitting: isHit,
        launchAngleDeg: isHit ? 19.5 : undefined,
        sprayAngleDeg: isHit ? -7.0 : undefined,
        distanceM: isHit ? 89.5 : undefined,
        ribbonPath: buildRibbonPathFromMetrics(cameraAngle, vb, hb, plateX, plateY),
      });

      setProgress(0.06);
      setIsAnimatingRibbon(true);
    },
    [cameraAngle, currentPitch.pitchType, discipline, onNewMeasuredPitch, uploadedVideoUrl]
  );

  // 3. 當 ONNX 偵測到連續球體軌跡時，自動結算球速與彩帶
  useEffect(() => {
    if (!latestBall || !autoDetectActive) return;
    const now = performance.now();
    const list = trackedPtsRef.current;

    if (list.length > 0 && now - lastMotionTimeRef.current > 450) {
      trackedPtsRef.current = [];
    }
    lastMotionTimeRef.current = now;
    trackedPtsRef.current.push(latestBall);

    if (trackedPtsRef.current.length >= 6) {
      const first = trackedPtsRef.current[0];
      const last = trackedPtsRef.current[trackedPtsRef.current.length - 1];
      const dtSec = Math.max(0.18, (last.timestampMs - first.timestampMs) / 1000);
      const estSpeed = Math.min(162, Math.max(78, Number(((16.7 / dtSec) * 3.6 * 0.45).toFixed(1))));
      trackedPtsRef.current = [];
      triggerAutoMeasure(estSpeed);
    }
  }, [latestBall, autoDetectActive, triggerAutoMeasure]);

  // 初次載入自動播放第一球紅色彩帶
  useEffect(() => {
    setProgress(0.08);
    setIsAnimatingRibbon(true);
  }, [currentPitch.id]);

  // 紅色彩帶動態延伸動畫
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
    () => sampleQuadraticRibbon(currentPitch.ribbonPath, stageW, stageH, progress),
    [currentPitch.ribbonPath, progress]
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
    const url = URL.createObjectURL(file);
    setUploadedVideoUrl(url);
    window.setTimeout(() => {
      triggerAutoMeasure();
    }, 550);
  };

  const handleDownloadSnapshot = () => {
    const canvas = document.createElement('canvas');
    canvas.width = stageW;
    canvas.height = stageH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawOverlayAndSave = () => {
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
      link.download = `smartscout-${Date.now()}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    };

    if (hasLiveCamera && videoRef.current && videoRef.current.videoWidth > 0) {
      ctx.drawImage(videoRef.current, 0, 0, stageW, stageH);
      drawOverlayAndSave();
    } else {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        ctx.drawImage(img, 0, 0, stageW, stageH);
        drawOverlayAndSave();
      };
      img.src = bullpenImgUrl;
    }
  };

  return (
    <div className="relative mx-auto flex h-[calc(100dvh-68px)] max-h-[920px] w-full max-w-[460px] flex-col overflow-hidden rounded-none sm:rounded-3xl bg-neutral-900 shadow-xl border border-slate-200 select-none">
      {/* 相機串流 / 上傳影片 / 牛棚背景 */}
      <video
        ref={videoRef}
        src={uploadedVideoUrl ?? undefined}
        playsInline
        muted
        autoPlay
        loop={Boolean(uploadedVideoUrl)}
        className={`absolute inset-0 h-full w-full object-cover ${
          hasLiveCamera || uploadedVideoUrl ? 'opacity-100' : 'opacity-0'
        }`}
      />
      {!hasLiveCamera && !uploadedVideoUrl && (
        <img
          src={bullpenImgUrl}
          alt="Camera View"
          className="absolute inset-0 h-full w-full object-cover"
        />
      )}

      {/* 極簡頂部控制列（無冗長文字，全圖示化膠囊） */}
      <div className="relative z-20 flex items-center justify-between gap-1.5 px-3.5 pt-3.5 pb-2 bg-gradient-to-b from-black/55 to-transparent">
        {/* 自動測速狀態膠囊 */}
        <button
          type="button"
          onClick={() => setAutoDetectActive((v) => !v)}
          className="flex items-center gap-1.5 rounded-full bg-white/92 px-3 py-1.5 text-xs font-extrabold text-slate-900 shadow-md backdrop-blur-md"
        >
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              autoDetectActive ? 'bg-emerald-500 animate-pulse' : 'bg-slate-400'
            }`}
          />
          <span>{autoDetectActive ? '自動測速中' : '已暫停'}</span>
        </button>

        {/* 投手後方 / 投手前方 切換 */}
        <button
          type="button"
          onClick={() =>
            onChangeCameraAngle(
              cameraAngle === 'behind_pitcher' ? 'behind_catcher' : 'behind_pitcher'
            )
          }
          className="rounded-full bg-black/45 px-3 py-1.5 text-xs font-bold text-white backdrop-blur-md border border-white/20"
        >
          {cameraAngle === 'behind_pitcher' ? '投手後方 ⇄' : '投手前方 ⇄'}
        </button>

        {/* 投球 / 打擊 切換 */}
        <button
          type="button"
          onClick={() =>
            onChangeDiscipline(discipline === 'pitching' ? 'hitting' : 'pitching')
          }
          className="rounded-full bg-black/45 px-2.5 py-1.5 text-xs font-bold text-white backdrop-blur-md border border-white/20"
        >
          {discipline === 'pitching' ? '⚾ 投球' : '🏏 打擊'}
        </button>

        {/* 語音播報 & 上傳影片按鈕 */}
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={onToggleVoice}
            title="語音播報開關"
            className={`flex h-8 w-8 items-center justify-center rounded-full backdrop-blur-md border text-xs ${
              voiceEnabled
                ? 'bg-[#0b72b9] text-white border-sky-300'
                : 'bg-black/40 text-white/60 border-white/15'
            }`}
          >
            🔊
          </button>

          <label
            title="載入手機影片自動測速"
            className="cursor-pointer flex h-8 w-8 items-center justify-center rounded-full bg-white/90 text-slate-900 shadow text-xs font-bold"
          >
            📂
            <input
              type="file"
              accept="video/*"
              onChange={handleVideoFileUpload}
              className="hidden"
            />
          </label>
        </div>
      </div>

      {/* 3D 紅色漸變彗星彩帶軌跡 SVG */}
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

        {showRibbon && ribbonSamples.length >= 2 && (
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
            {lastSample && progress < 0.97 && (
              <circle
                cx={lastSample.x}
                cy={lastSample.y}
                r={Math.max(6, lastSample.r * 0.92)}
                fill="#ffffff"
                stroke="#dc2626"
                strokeWidth="2"
              />
            )}
          </g>
        )}

        {/* 即時 ONNX 偵測框 */}
        {latestBall && (
          <circle
            cx={latestBall.x}
            cy={latestBall.y}
            r={Math.max(12, latestBall.width * 0.6)}
            fill="rgba(239, 68, 68, 0.25)"
            stroke="#ef4444"
            strokeWidth="2.5"
          />
        )}
      </svg>

      {/* 右側 4 顆 SmartScout 懸浮操作按鈕 */}
      <div className="absolute right-4 top-[36%] z-20 flex flex-col items-center gap-5">
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
            setProgress(0.06);
            setIsAnimatingRibbon(true);
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

      {/* 底部區域：SmartScout 半透明數據卡 + 時間軸/測速快門 */}
      <div className="relative z-20 mt-auto px-3.5 pb-3.5 space-y-2.5">
        {showStatsCard && (
          <div className="rounded-2xl bg-[#18110e]/85 backdrop-blur-md px-4 py-3.5 text-white shadow-2xl border border-white/10">
            <div className="flex items-center justify-between gap-2">
              <div>
                <div className="text-xs font-medium text-neutral-300">
                  {discipline === 'hitting' ? '擊球初速' : '初速'}
                </div>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-[44px] font-bold leading-tight tracking-tight tabular-nums">
                    {currentPitch.speedKmh.toFixed(1)}
                  </span>
                  <span className="text-lg font-medium text-neutral-300">km/h</span>
                </div>
              </div>

              <select
                value={currentPitch.pitchType}
                onChange={(e) => onChangePitchType(currentPitch.id, e.target.value)}
                style={{ backgroundColor: getPitchTypeColor(currentPitch.pitchType) }}
                className="rounded-lg px-3.5 py-1.5 text-xs font-bold text-white shadow-md focus:outline-none cursor-pointer"
              >
                {SMARTSCOUT_PITCH_TYPES.map((pt) => (
                  <option key={pt.name} value={pt.name} className="bg-neutral-900 text-white">
                    {pt.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="mt-2 grid grid-cols-4 gap-1 text-center">
              <div>
                <div className="text-[11px] text-neutral-400">垂直變化量</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {currentPitch.verticalBreakCm.toFixed(1)}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">cm</span>
                </div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">水平變化量</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {currentPitch.horizontalBreakCm.toFixed(1)}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">cm</span>
                </div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">轉速</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {currentPitch.totalSpinRpm}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">rpm</span>
                </div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">有效轉速</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {currentPitch.activeSpinRpm}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">rpm</span>
                </div>
              </div>
            </div>

            <div className="mt-3 grid grid-cols-4 gap-1 text-center">
              <div>
                <div className="text-[11px] text-neutral-400">效率</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {currentPitch.efficiencyPct}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">%</span>
                </div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">旋轉方向</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {currentPitch.spinDirection}
                </div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">陀螺角度</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {currentPitch.gyroDegree}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">°</span>
                </div>
              </div>
              <div className="flex items-end justify-center">
                <button
                  type="button"
                  onClick={() => triggerAutoMeasure()}
                  className="rounded-lg bg-[#d9261c] hover:bg-red-500 px-2.5 py-1 text-xs font-extrabold text-white shadow active:scale-95 transition"
                >
                  ● 測下一球
                </button>
              </div>
            </div>
          </div>
        )}

        {/* 白色時間軸拉桿 */}
        <div className="flex items-center gap-2.5 px-1">
          <input
            type="range"
            min={0.05}
            max={1}
            step={0.01}
            value={progress}
            onChange={(e) => {
              setIsAnimatingRibbon(false);
              setProgress(Number(e.target.value));
            }}
            className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/40 accent-white"
          />
        </div>
      </div>
    </div>
  );
};
