import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  SMARTSCOUT_PITCH_TYPES,
  SmartScoutPitchRecord,
  getPitchTypeColor,
} from '../data/smartScoutDefaultSessions';

interface SmartScoutPitchPlayerProps {
  pitch: SmartScoutPitchRecord | null;
  totalInSession?: number;
  onBack: () => void;
  onDelete?: (pitchId: string) => void;
  onChangePitchType?: (pitchId: string, newType: string) => void;
  onPrevPitch?: () => void;
  onNextPitch?: () => void;
}

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
  steps = 48
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

export const SmartScoutPitchPlayer: React.FC<SmartScoutPitchPlayerProps> = ({
  pitch,
  onBack,
  onDelete,
  onChangePitchType,
  onPrevPitch,
  onNextPitch,
}) => {
  const [showRibbon, setShowRibbon] = useState<boolean>(true);
  const [showStatsCard, setShowStatsCard] = useState<boolean>(true);
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 0.5 | 0.25>(1);
  const [progress, setProgress] = useState<number>(1);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (!pitch) return;
    setProgress(0.08);
    setIsPlaying(true);
  }, [pitch?.id]);

  useEffect(() => {
    if (!isPlaying) {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      return;
    }

    let lastTime = performance.now();
    const durationMs = 1150 / playbackSpeed;

    const tick = (now: number) => {
      const dt = now - lastTime;
      lastTime = now;
      setProgress((prev) => {
        const next = prev + dt / durationMs;
        if (next >= 1) {
          setIsPlaying(false);
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
  }, [isPlaying, playbackSpeed]);

  const stageW = 590;
  const stageH = 1050;

  const ribbonSamples = useMemo(
    () =>
      pitch
        ? sampleQuadraticRibbon(pitch.ribbonPath, stageW, stageH, progress)
        : [],
    [pitch, progress]
  );

  const polygonPoints = useMemo(
    () => buildTaperedRibbonPolygon(ribbonSamples),
    [ribbonSamples]
  );

  if (!pitch) {
    return (
      <div className="mx-auto max-w-md px-4 pt-12 text-center">
        <div className="rounded-3xl bg-white p-8 text-slate-900 shadow-sm border border-slate-200 space-y-4">
          <div className="text-lg font-extrabold text-slate-900">尚無投球影片紀錄</div>
          <button
            type="button"
            onClick={onBack}
            className="rounded-2xl bg-[#0b72b9] px-6 py-3 text-sm font-extrabold text-white shadow hover:bg-[#0962a0] transition"
          >
            前往相機開始測速
          </button>
        </div>
      </div>
    );
  }

  const firstSample = ribbonSamples[0];
  const lastSample = ribbonSamples[ribbonSamples.length - 1];

  const handleCycleSpeed = () => {
    const nextSpeed: 1 | 0.5 | 0.25 =
      playbackSpeed === 1 ? 0.5 : playbackSpeed === 0.5 ? 0.25 : 1;
    setPlaybackSpeed(nextSpeed);
    setProgress(0.06);
    setIsPlaying(true);
  };

  return (
    <div className="relative mx-auto flex h-[calc(100dvh-64px)] max-h-[920px] w-full max-w-[460px] flex-col overflow-hidden rounded-none sm:rounded-3xl bg-slate-900 shadow-2xl select-none">
      {/* 實測擷取之相機畫面或上傳之真實影片 */}
      {pitch.videoUrl ? (
        <video
          src={pitch.videoUrl}
          playsInline
          muted
          autoPlay
          loop
          className="absolute inset-0 h-full w-full object-cover"
        />
      ) : pitch.snapshotUrl ? (
        <img
          src={pitch.snapshotUrl}
          alt="實測畫面"
          className="absolute inset-0 h-full w-full object-cover"
        />
      ) : (
        <div className="absolute inset-0 bg-slate-900" />
      )}

      {/* 頂部導覽列 (< 投球 #1 🗑) */}
      <div className="relative z-20 flex items-center justify-between px-4 pt-4 pb-3 bg-gradient-to-b from-black/60 via-black/25 to-transparent text-white">
        <button
          type="button"
          onClick={onBack}
          className="flex h-10 w-10 items-center justify-center rounded-full bg-black/25 backdrop-blur-sm hover:bg-black/45 transition"
          aria-label="返回"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>

        <div className="flex items-center gap-2">
          {onPrevPitch && (
            <button
              type="button"
              onClick={onPrevPitch}
              className="rounded-full bg-black/25 px-2 py-0.5 text-xs font-bold text-white/80 hover:bg-black/45"
            >
              ◀
            </button>
          )}
          <h2 className="text-lg font-bold tracking-wide drop-shadow">
            投球 #{pitch.pitchNumber}
          </h2>
          {onNextPitch && (
            <button
              type="button"
              onClick={onNextPitch}
              className="rounded-full bg-black/25 px-2 py-0.5 text-xs font-bold text-white/80 hover:bg-black/45"
            >
              ▶
            </button>
          )}
        </div>

        <button
          type="button"
          onClick={() => {
            onDelete?.(pitch.id);
            onBack();
          }}
          className="flex h-10 w-10 items-center justify-center rounded-full bg-black/25 backdrop-blur-sm hover:bg-rose-600/70 transition"
          aria-label="刪除此球"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="3 6 5 6 21 6" />
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
            <path d="M10 11v6M14 11v6" />
            <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
          </svg>
        </button>
      </div>

      {/* 3D 紅色漸變彗星彩帶軌跡 */}
      <svg
        viewBox={`0 0 ${stageW} ${stageH}`}
        className="pointer-events-none absolute inset-0 z-10 h-full w-full"
      >
        <defs>
          <linearGradient id="smartScoutRibbonGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#ff2e24" stopOpacity="0.95" />
            <stop offset="70%" stopColor="#ff2419" stopOpacity="0.88" />
            <stop offset="100%" stopColor="#d9231b" stopOpacity="0.76" />
          </linearGradient>
          <filter id="ribbonShadow" x="-15%" y="-15%" width="130%" height="130%">
            <feDropShadow dx="0" dy="3" stdDeviation="4" floodColor="#7f1d1d" floodOpacity="0.45" />
          </filter>
        </defs>

        {showRibbon && ribbonSamples.length >= 2 && (
          <g filter="url(#ribbonShadow)">
            <polygon points={polygonPoints} fill="url(#smartScoutRibbonGrad)" />
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
          onClick={handleCycleSpeed}
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

      {/* 底部半透明深色數據卡 + 白色時間軸拉桿 */}
      <div className="relative z-20 mt-auto px-3.5 pb-4 space-y-3">
        {showStatsCard && (
          <div className="rounded-2xl bg-[#18110e]/85 backdrop-blur-md px-4 py-3.5 text-white shadow-2xl border border-white/10">
            <div className="flex items-center justify-between gap-2">
              <div>
                <div className="text-xs font-medium text-neutral-300">初速</div>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-[44px] font-bold leading-tight tracking-tight tabular-nums">
                    {pitch.speedKmh.toFixed(1)}
                  </span>
                  <span className="text-lg font-medium text-neutral-300">km/h</span>
                </div>
              </div>

              <div>
                <select
                  value={pitch.pitchType}
                  onChange={(e) => onChangePitchType?.(pitch.id, e.target.value)}
                  style={{ backgroundColor: getPitchTypeColor(pitch.pitchType) }}
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
                  {pitch.verticalBreakCm.toFixed(1)}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">cm</span>
                </div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">水平變化量</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {pitch.horizontalBreakCm.toFixed(1)}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">cm</span>
                </div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">轉速</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {pitch.totalSpinRpm}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">rpm</span>
                </div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">有效轉速</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {pitch.activeSpinRpm}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">rpm</span>
                </div>
              </div>
            </div>

            <div className="mt-3 grid grid-cols-4 gap-1 text-center">
              <div>
                <div className="text-[11px] text-neutral-400">效率</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {pitch.efficiencyPct}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">%</span>
                </div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">旋轉方向</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">{pitch.spinDirection}</div>
              </div>
              <div>
                <div className="text-[11px] text-neutral-400">陀螺角度</div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {pitch.gyroDegree}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">°</span>
                </div>
              </div>
              <div />
            </div>
          </div>
        )}

        <div className="px-1">
          <input
            type="range"
            min={0.05}
            max={1}
            step={0.01}
            value={progress}
            onChange={(e) => {
              setIsPlaying(false);
              setProgress(Number(e.target.value));
            }}
            className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/40 accent-white"
          />
        </div>
      </div>
    </div>
  );
};
