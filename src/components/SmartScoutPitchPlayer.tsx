import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  SMARTSCOUT_PITCH_TYPES,
  SmartScoutPitchRecord,
  buildTaperedRibbonPolygon,
  getPitchTypeColor,
  sampleQuadraticRibbon,
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

  const handleDownloadSnapshot = () => {
    const canvas = document.createElement('canvas');
    canvas.width = stageW;
    canvas.height = stageH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const grad = ctx.createLinearGradient(0, 0, 0, stageH);
    grad.addColorStop(0, '#0f172a');
    grad.addColorStop(1, '#1e293b');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, stageW, stageH);

    if (showRibbon && ribbonSamples.length >= 2 && firstSample && lastSample) {
      const rGrad = ctx.createLinearGradient(
        firstSample.x,
        firstSample.y,
        lastSample.x,
        lastSample.y
      );
      rGrad.addColorStop(0, 'rgba(127, 16, 34, 0.92)');
      rGrad.addColorStop(1, 'rgba(239, 35, 60, 0.95)');
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
      ctx.fillText(`${pitch.speedKmh.toFixed(1)} km/h`, 46, stageH - 230);
      ctx.font = 'bold 20px sans-serif';
      ctx.fillText(
        `${pitch.pitchType} | VB ${pitch.verticalBreakCm}cm | HB ${pitch.horizontalBreakCm}cm | ${pitch.totalSpinRpm} rpm`,
        46,
        stageH - 150
      );
    }

    const link = document.createElement('a');
    link.download = `smartscout-pitch-${pitch.pitchNumber}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  return (
    <div className="relative mx-auto flex h-[calc(100dvh-64px)] max-h-[920px] w-full max-w-[460px] flex-col overflow-hidden rounded-none sm:rounded-3xl bg-slate-950 shadow-2xl select-none">
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
        <div
          className="absolute inset-0 opacity-45"
          style={{
            backgroundImage:
              'radial-gradient(circle at 50% 45%, rgba(11, 114, 185, 0.28), transparent 70%), linear-gradient(to right, rgba(148, 163, 184, 0.12) 1px, transparent 1px), linear-gradient(to bottom, rgba(148, 163, 184, 0.12) 1px, transparent 1px)',
            backgroundSize: '100% 100%, 44px 44px, 44px 44px',
          }}
        />
      )}

      {/* 頂部導覽列 (< 投球 #N 🗑) */}
      <div className="relative z-20 flex items-center justify-between px-4 pt-4 pb-3 bg-gradient-to-b from-black/65 via-black/25 to-transparent text-white">
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
              className="rounded-full bg-black/25 px-2.5 py-0.5 text-xs font-bold text-white/90 hover:bg-black/45"
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
              className="rounded-full bg-black/25 px-2.5 py-0.5 text-xs font-bold text-white/90 hover:bg-black/45"
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
          🗑
        </button>
      </div>

      {/* 3D 漸變紅色彗星彩帶軌跡 (#7f1022 -> #ef233c) + 末端立體小白球 */}
      <svg
        viewBox={`0 0 ${stageW} ${stageH}`}
        className="pointer-events-none absolute inset-0 z-10 h-full w-full"
      >
        <defs>
          <linearGradient id="smartScoutRibbonGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#7f1022" stopOpacity="0.92" />
            <stop offset="55%" stopColor="#d9182e" stopOpacity="0.90" />
            <stop offset="100%" stopColor="#ef233c" stopOpacity="0.96" />
          </linearGradient>
          <filter id="ribbonShadow" x="-15%" y="-15%" width="130%" height="130%">
            <feDropShadow dx="0" dy="4" stdDeviation="5" floodColor="#4c0519" floodOpacity="0.55" />
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
      </svg>

      {/* 右側 4 顆懸浮圓形控制鈕 */}
      <div className="absolute right-4 top-[36%] z-20 flex flex-col items-center gap-5">
        <button
          type="button"
          onClick={() => setShowRibbon((v) => !v)}
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

      {/* 底部深色半透明 2x4 八大數據面板 + 時間軸滑桿 */}
      <div className="relative z-20 mt-auto px-3.5 pb-3.5 space-y-2.5">
        {showStatsCard && (
          <div className="rounded-2xl bg-[#18110e]/88 backdrop-blur-md px-4 py-3 text-white shadow-2xl border border-white/10">
            <div className="flex items-center justify-between gap-2">
              <div>
                <div className="text-xs font-medium text-neutral-300">初速</div>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-[42px] font-bold leading-tight tracking-tight tabular-nums">
                    {pitch.speedKmh.toFixed(1)}
                  </span>
                  <span className="text-base font-medium text-neutral-300">km/h</span>
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

            <div className="mt-2 grid grid-cols-4 gap-1.5 text-center">
              <div className="rounded-xl bg-white/5 py-1.5 px-1">
                <div className="text-[10px] text-neutral-400">VB [cm] 垂直</div>
                <div className="mt-0.5 text-lg font-bold tabular-nums">
                  {pitch.verticalBreakCm.toFixed(1)}
                </div>
              </div>

              <div className="rounded-xl bg-white/5 py-1.5 px-1">
                <div className="text-[10px] text-neutral-400">HB [cm] 水平</div>
                <div className="mt-0.5 text-lg font-bold tabular-nums">
                  {pitch.horizontalBreakCm.toFixed(1)}
                </div>
              </div>

              <div className="rounded-xl bg-white/5 py-1.5 px-1">
                <div className="text-[10px] text-neutral-400">E [%] 效率</div>
                <div className="mt-0.5 text-lg font-bold tabular-nums">
                  {pitch.efficiencyPct}%
                </div>
              </div>

              <div className="rounded-xl bg-white/5 py-1.5 px-1 flex flex-col items-center justify-center">
                <div className="text-[10px] text-neutral-400">SD 旋轉方向</div>
                <div className="mt-0.5 flex items-center gap-1">
                  <span className="text-base font-bold tabular-nums">{pitch.spinDirection}</span>
                  <span
                    className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-white/50 text-[9px]"
                    style={{
                      transform: `rotate(${90 - (pitch.spinAxisDeg ?? 45)}deg)`,
                    }}
                  >
                    ↑
                  </span>
                </div>
              </div>

              <div className="rounded-xl bg-white/5 py-1.5 px-1">
                <div className="text-[10px] text-neutral-400">GD [°] 陀螺角</div>
                <div className="mt-0.5 text-lg font-bold tabular-nums">{pitch.gyroDegree}°</div>
              </div>

              <div className="rounded-xl bg-white/5 py-1.5 px-1">
                <div className="text-[10px] text-neutral-400">AS [rpm] 有效</div>
                <div className="mt-0.5 text-lg font-bold tabular-nums">{pitch.activeSpinRpm}</div>
              </div>

              <div className="rounded-xl bg-white/5 py-1.5 px-1">
                <div className="text-[10px] text-neutral-400">S [rpm] 轉速</div>
                <div className="mt-0.5 text-lg font-bold tabular-nums">{pitch.totalSpinRpm}</div>
              </div>

              <div className="rounded-xl bg-white/5 py-1 px-1 flex flex-col items-center justify-center">
                <div className="text-[10px] text-neutral-400">進壘 #{pitch.zoneId ?? 5}</div>
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
                  <circle
                    cx={Math.max(2, Math.min(28, 15 + (pitch.plateXCm / 21.6) * 11))}
                    cy={Math.max(2, Math.min(30, 15.5 - ((pitch.plateYCm - 76) / 30) * 12))}
                    r="3.2"
                    fill="#ef233c"
                    stroke="#ffffff"
                    strokeWidth="1"
                  />
                </svg>
              </div>
            </div>
          </div>
        )}

        <div className="flex items-center gap-2.5 px-1">
          <button
            type="button"
            onClick={() => {
              if (progress >= 0.99) setProgress(0.05);
              setIsPlaying((v) => !v);
            }}
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white text-neutral-950 font-bold text-xs shadow"
          >
            {isPlaying ? '❚❚' : '▶'}
          </button>
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
