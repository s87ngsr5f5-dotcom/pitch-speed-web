import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  SMARTSCOUT_PITCH_TYPES,
  SmartScoutPitchRecord,
  buildRibbonPathFromMetrics,
  buildTaperedRibbonPolygon,
  computeRealPitchFromDetections,
  getPitchTypeColor,
  sampleQuadraticRibbon,
} from '../data/smartScoutDefaultSessions';
import { computeStrikeZoneGrid, degToClockString } from '../core/smartScoutEngine';
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
  const videoRef = useRef<VideoWithRVFC | null>(null);
  const [hasLiveCamera, setHasLiveCamera] = useState<boolean>(false);
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

  // 1. 一進入即啟動相機 (getUserMedia 優先 60fps 1280x720)
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

  // 2. 以 requestVideoFrameCallback 擷取 ImageBitmap 送入 Web Worker 即時偵測
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let stopped = false;
    let rvfcId: number | null = null;
    let rafId: number | null = null;
    let lastMediaTime = -1;

    const sendFrame = (timestampMs: number) => {
      if (
        !isWorkerReady ||
        isBusyRef.current ||
        video.readyState < 2 ||
        video.paused ||
        video.ended ||
        video.videoWidth === 0
      ) {
        return;
      }
      isBusyRef.current = true;
      const w = video.videoWidth;
      const h = video.videoHeight;
      createImageBitmap(video)
        .then((bmp) => onFrame(bmp, timestampMs, w, h))
        .catch(() => {
          isBusyRef.current = false;
        });
    };

    if (typeof video.requestVideoFrameCallback === 'function') {
      const onVideoFrame = (_now: DOMHighResTimeStamp, meta: VideoFrameMetadata) => {
        if (stopped) return;
        if (meta.mediaTime !== lastMediaTime) {
          lastMediaTime = meta.mediaTime;
          sendFrame(meta.mediaTime * 1000);
        }
        rvfcId = video.requestVideoFrameCallback!(onVideoFrame);
      };
      rvfcId = video.requestVideoFrameCallback(onVideoFrame);
    } else {
      const onRaf = (now: DOMHighResTimeStamp) => {
        if (stopped) return;
        const ts = uploadedVideoUrl ? video.currentTime * 1000 : now;
        sendFrame(ts);
        rafId = requestAnimationFrame(onRaf);
      };
      rafId = requestAnimationFrame(onRaf);
    }

    return () => {
      stopped = true;
      if (rvfcId !== null && video.cancelVideoFrameCallback) {
        video.cancelVideoFrameCallback(rvfcId);
      }
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
      }
    };
  }, [isWorkerReady, isBusyRef, onFrame, uploadedVideoUrl]);

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

  const finalizeActiveTrack = useCallback(() => {
    const track = activeTrackRef.current;
    activeTrackRef.current = [];
    setLiveTrackDots([]);

    if (track.length < 4) return;
    const video = videoRef.current;
    const vw = video?.videoWidth || 1280;
    const vh = video?.videoHeight || 720;

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

  useEffect(() => {
    if (!latestBall) return;

    const track = activeTrackRef.current;
    if (track.length > 0) {
      const prev = track[track.length - 1];
      const dt = latestBall.timestampMs - prev.timestampMs;
      if (dt <= 0) return;

      if (dt > 260) {
        finalizeActiveTrack();
      } else {
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

  // 快速測試下一球按鈕（無實體相機或快速驗證物理引擎時使用）
  const handleQuickTestNextPitch = useCallback(() => {
    const pitchType = currentPitch ? currentPitch.pitchType : selectedPitchType;
    const dEff = Math.max(8, distanceMeters - 1.75);
    const flightSec =
      pitchType === '曲球'
        ? 0.48 + Math.random() * 0.05
        : pitchType === '滑球' || pitchType === '變速球'
          ? 0.43 + Math.random() * 0.04
          : 0.37 + Math.random() * 0.05;

    const speedKmh = Number((((dEff / flightSec) * 3.6) / 0.955).toFixed(1));
    const plateSpeedKmh = Number((speedKmh * 0.916).toFixed(1));

    const vb = Number(
      (pitchType === '曲球'
        ? -35 + Math.random() * 14
        : pitchType === '滑球'
          ? 4 + Math.random() * 12
          : 44 + Math.random() * 18
      ).toFixed(1)
    );
    const hb = Number(
      (pitchType === '滑球' || pitchType === '曲球'
        ? -26 + Math.random() * 14
        : 16 + Math.random() * 20
      ).toFixed(1)
    );

    const eff =
      pitchType === '滑球'
        ? Math.round(52 + Math.random() * 15)
        : Math.round(78 + Math.random() * 15);
    const spinAxisDeg = Math.round((Math.atan2(vb, hb) * 180) / Math.PI);
    const sd = degToClockString(spinAxisDeg);
    const activeSpin = Math.round(
      Math.max(950, Math.min(2650, Math.hypot(vb, hb) * 20 + speedKmh * 7.2))
    );
    const totalSpin = Math.round(activeSpin / (eff / 100));
    const gyro = Math.round((Math.acos(eff / 100) * 180) / Math.PI);
    const plateX = Number((-16 + Math.random() * 32).toFixed(1));
    const plateY = Number((52 + Math.random() * 44).toFixed(1));
    const { zoneId, isStrike } = computeStrikeZoneGrid(plateX, plateY);

    onNewMeasuredPitch({
      pitchType,
      pitchColor: getPitchTypeColor(pitchType),
      cameraAngle,
      speedKmh,
      plateSpeedKmh,
      verticalBreakCm: vb,
      horizontalBreakCm: hb,
      efficiencyPct: eff,
      spinDirection: sd,
      spinAxisDeg,
      gyroDegree: gyro,
      activeSpinRpm: activeSpin,
      totalSpinRpm: totalSpin,
      plateXCm: plateX,
      plateYCm: plateY,
      zoneId,
      isStrike,
      snapshotUrl: captureVideoSnapshot(),
      videoUrl: uploadedVideoUrl ?? undefined,
      isHitting: discipline === 'hitting',
      ribbonPath: buildRibbonPathFromMetrics(cameraAngle, vb, hb, plateX, plateY),
    });

    setProgress(0.06);
    setIsAnimatingRibbon(true);
  }, [
    cameraAngle,
    captureVideoSnapshot,
    currentPitch,
    discipline,
    distanceMeters,
    onNewMeasuredPitch,
    selectedPitchType,
    uploadedVideoUrl,
  ]);

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
        `VB ${currentPitch.verticalBreakCm}cm | HB ${currentPitch.horizontalBreakCm}cm | ${currentPitch.totalSpinRpm} rpm | SD ${currentPitch.spinDirection}`,
        46,
        stageH - 150
      );
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
    <div className="relative mx-auto flex h-[calc(100dvh-64px)] max-h-[920px] w-full max-w-[460px] flex-col overflow-hidden rounded-none sm:rounded-3xl bg-slate-950 shadow-xl select-none">
      {/* 真實相機串流或載入影片 */}
      <video
        ref={videoRef}
        src={uploadedVideoUrl ?? undefined}
        playsInline
        muted
        autoPlay
        onEnded={() => finalizeActiveTrack()}
        className="absolute inset-0 h-full w-full object-cover bg-slate-950"
      />

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
            {currentPitch ? `投球 #${currentPitch.pitchNumber}` : '自動測速中'}
          </h2>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={handleQuickTestNextPitch}
            className="rounded-xl bg-[#0b72b9] hover:bg-[#0962a0] px-2.5 py-1.5 text-xs font-extrabold text-white shadow transition active:scale-95"
          >
            ⚡ 測試下一球
          </button>

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

      {/* 3D 漸變紅色彗星彩帶軌跡 (#7f1022 -> #ef233c) + 末端立體小白球(含縫線與陰影) */}
      <svg
        viewBox={`0 0 ${stageW} ${stageH}`}
        className="pointer-events-none absolute inset-0 z-10 h-full w-full"
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

        {showRibbon && currentPitch && ribbonSamples.length >= 2 && (
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
                <div className="text-xs font-medium text-neutral-300">初速</div>
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

            {/* 2x4 八格網格：
                Row 1: VB [cm] 垂直變化量 | HB [cm] 水平變化量 | E [%] 效率 | SD 旋轉方向(附箭頭方向圓圈)
                Row 2: GD [°] 陀螺角度 | AS [rpm] 有效轉速 | S [rpm] 轉速 | 九宮格進壘小圖示(標示紅點落點)
            */}
            <div className="mt-2 grid grid-cols-4 gap-1.5 text-center">
              <div className="rounded-xl bg-white/5 py-1.5 px-1">
                <div className="text-[10px] text-neutral-400">VB [cm] 垂直</div>
                <div className="mt-0.5 text-lg font-bold tabular-nums">
                  {currentPitch ? currentPitch.verticalBreakCm.toFixed(1) : '--'}
                </div>
              </div>

              <div className="rounded-xl bg-white/5 py-1.5 px-1">
                <div className="text-[10px] text-neutral-400">HB [cm] 水平</div>
                <div className="mt-0.5 text-lg font-bold tabular-nums">
                  {currentPitch ? currentPitch.horizontalBreakCm.toFixed(1) : '--'}
                </div>
              </div>

              <div className="rounded-xl bg-white/5 py-1.5 px-1">
                <div className="text-[10px] text-neutral-400">E [%] 效率</div>
                <div className="mt-0.5 text-lg font-bold tabular-nums">
                  {currentPitch ? `${currentPitch.efficiencyPct}%` : '--'}
                </div>
              </div>

              <div className="rounded-xl bg-white/5 py-1.5 px-1 flex flex-col items-center justify-center">
                <div className="text-[10px] text-neutral-400">SD 旋轉方向</div>
                <div className="mt-0.5 flex items-center gap-1">
                  <span className="text-base font-bold tabular-nums">
                    {currentPitch ? currentPitch.spinDirection : '--:--'}
                  </span>
                  {currentPitch && (
                    <span
                      className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-white/50 text-[9px]"
                      style={{
                        transform: `rotate(${90 - (currentPitch.spinAxisDeg ?? 45)}deg)`,
                      }}
                    >
                      ↑
                    </span>
                  )}
                </div>
              </div>

              <div className="rounded-xl bg-white/5 py-1.5 px-1">
                <div className="text-[10px] text-neutral-400">GD [°] 陀螺角</div>
                <div className="mt-0.5 text-lg font-bold tabular-nums">
                  {currentPitch ? `${currentPitch.gyroDegree}°` : '--'}
                </div>
              </div>

              <div className="rounded-xl bg-white/5 py-1.5 px-1">
                <div className="text-[10px] text-neutral-400">AS [rpm] 有效</div>
                <div className="mt-0.5 text-lg font-bold tabular-nums">
                  {currentPitch ? currentPitch.activeSpinRpm : '--'}
                </div>
              </div>

              <div className="rounded-xl bg-white/5 py-1.5 px-1">
                <div className="text-[10px] text-neutral-400">S [rpm] 轉速</div>
                <div className="mt-0.5 text-lg font-bold tabular-nums">
                  {currentPitch ? currentPitch.totalSpinRpm : '--'}
                </div>
              </div>

              {/* 九宮格進壘小圖示（標示紅點落點） */}
              <div className="rounded-xl bg-white/5 py-1 px-1 flex flex-col items-center justify-center">
                <div className="text-[10px] text-neutral-400">
                  進壘 #{currentPitch ? currentPitch.zoneId : '-'}
                </div>
                <svg width="26" height="28" viewBox="0 0 30 32" className="mt-0.5">
                  <rect
                    x="4"
                    y="3"
                    width="22"
                    height="25"
                    fill="none"
                    stroke="rgba(255,255,255,0.65)"
                    strokeWidth="1.4"
                  />
                  <line x1="11.3" y1="3" x2="11.3" y2="28" stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
                  <line x1="18.6" y1="3" x2="18.6" y2="28" stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
                  <line x1="4" y1="11.3" x2="26" y2="11.3" stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
                  <line x1="4" y1="19.6" x2="26" y2="19.6" stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
                  {currentPitch && (
                    <circle
                      cx={Math.max(2, Math.min(28, 15 + (currentPitch.plateXCm / 21.6) * 11))}
                      cy={Math.max(2, Math.min(30, 15.5 - ((currentPitch.plateYCm - 76) / 30) * 12))}
                      r="3.2"
                      fill="#ef233c"
                      stroke="#ffffff"
                      strokeWidth="1"
                    />
                  )}
                </svg>
              </div>
            </div>
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
