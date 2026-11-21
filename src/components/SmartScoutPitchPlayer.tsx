import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  SMARTSCOUT_PITCH_TYPES,
  SmartScoutPitchRecord,
  getPitchTypeColor,
} from '../data/smartScoutDefaultSessions';

interface SmartScoutPitchPlayerProps {
  pitch: SmartScoutPitchRecord;
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
    // 半徑隨透視深度平滑過渡
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
  const bullpenImgUrl = useMemo(
    () => new URL('./images/bullpen_bg.jpg', window.location.href).href,
    []
  );

  // 切換球時自動播放一次紅色彩帶生長動畫
  useEffect(() => {
    setProgress(0.08);
    setIsPlaying(true);
  }, [pitch.id]);

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
    () => sampleQuadraticRibbon(pitch.ribbonPath, stageW, stageH, progress),
    [pitch.ribbonPath, progress]
  );

  const polygonPoints = useMemo(
    () => buildTaperedRibbonPolygon(ribbonSamples),
    [ribbonSamples]
  );

  const firstSample = ribbonSamples[0];
  const lastSample = ribbonSamples[ribbonSamples.length - 1];

  const handleCycleSpeed = () => {
    const nextSpeed: 1 | 0.5 | 0.25 =
      playbackSpeed === 1 ? 0.5 : playbackSpeed === 0.5 ? 0.25 : 1;
    setPlaybackSpeed(nextSpeed);
    setProgress(0.06);
    setIsPlaying(true);
  };

  const handleReplay = () => {
    setProgress(0.06);
    setIsPlaying(true);
  };

  const handleDownloadSnapshot = () => {
    const canvas = document.createElement('canvas');
    canvas.width = stageW;
    canvas.height = stageH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      ctx.drawImage(img, 0, 0, stageW, stageH);

      // 繪製 3D 紅色彩帶
      if (showRibbon && ribbonSamples.length >= 2) {
        ctx.save();
        ctx.fillStyle = 'rgba(255, 38, 28, 0.88)';
        for (const s of ribbonSamples) {
          ctx.beginPath();
          ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
          ctx.fill();
        }
        if (firstSample) {
          ctx.fillStyle = 'rgba(210, 35, 28, 0.78)';
          ctx.beginPath();
          ctx.arc(firstSample.x, firstSample.y, firstSample.r, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }

      // 繪製下方數據黑卡
      if (showStatsCard) {
        ctx.fillStyle = 'rgba(24, 16, 14, 0.86)';
        ctx.beginPath();
        ctx.roundRect(22, stageH - 310, stageW - 44, 240, 22);
        ctx.fill();

        ctx.fillStyle = '#d6d3d1';
        ctx.font = 'bold 16px sans-serif';
        ctx.fillText('初速', 48, stageH - 274);

        ctx.fillStyle = '#ffffff';
        ctx.font = '900 54px sans-serif';
        ctx.fillText(`${pitch.speedKmh.toFixed(1)}`, 46, stageH - 218);
        ctx.font = 'bold 22px sans-serif';
        ctx.fillText('km/h', 202, stageH - 218);

        ctx.fillStyle = getPitchTypeColor(pitch.pitchType);
        ctx.beginPath();
        ctx.roundRect(stageW - 235, stageH - 258, 190, 38, 10);
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 17px sans-serif';
        ctx.fillText(pitch.pitchType, stageW - 205, stageH - 233);

        const row1 = [
          { label: '垂直變化量', val: `${pitch.verticalBreakCm.toFixed(1)} cm` },
          { label: '水平變化量', val: `${pitch.horizontalBreakCm.toFixed(1)} cm` },
          { label: '轉速', val: `${pitch.totalSpinRpm} rpm` },
          { label: '有效轉速', val: `${pitch.activeSpinRpm} rpm` },
        ];
        row1.forEach((item, idx) => {
          const x = 48 + idx * 132;
          ctx.fillStyle = '#a8a29e';
          ctx.font = '14px sans-serif';
          ctx.fillText(item.label, x, stageH - 168);
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 24px sans-serif';
          ctx.fillText(item.val, x, stageH - 138);
        });

        const row2 = [
          { label: '效率', val: `${pitch.efficiencyPct} %` },
          { label: '旋轉方向', val: pitch.spinDirection },
          { label: '陀螺角度', val: `${pitch.gyroDegree}°` },
        ];
        row2.forEach((item, idx) => {
          const x = 64 + idx * 145;
          ctx.fillStyle = '#a8a29e';
          ctx.font = '14px sans-serif';
          ctx.fillText(item.label, x, stageH - 105);
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 24px sans-serif';
          ctx.fillText(item.val, x, stageH - 76);
        });
      }

      const link = document.createElement('a');
      link.download = `smartscout-pitch-${pitch.pitchNumber}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    };
    img.src = bullpenImgUrl;
  };

  return (
    <div className="relative mx-auto flex h-[calc(100dvh-64px)] max-h-[920px] w-full max-w-[460px] flex-col overflow-hidden rounded-none sm:rounded-3xl bg-neutral-950 shadow-2xl border border-neutral-800 select-none">
      {/* 背景牛棚畫面或使用者上傳影片 */}
      {pitch.videoUrl ? (
        <video
          src={pitch.videoUrl}
          playsInline
          muted
          autoPlay
          loop
          className="absolute inset-0 h-full w-full object-cover"
        />
      ) : (
        <img
          src={bullpenImgUrl}
          alt="投球視角畫面"
          className="absolute inset-0 h-full w-full object-cover"
        />
      )}

      {/* 頂部漸層與導覽列 (< 投球 #1 🗑) */}
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
              title="上一球"
            >
              ◀
            </button>
          )}
          <h2 className="text-lg font-bold tracking-wide drop-shadow">
            {pitch.isHitting ? `打擊 #${pitch.pitchNumber}` : `投球 #${pitch.pitchNumber}`}
          </h2>
          {onNextPitch && (
            <button
              type="button"
              onClick={onNextPitch}
              className="rounded-full bg-black/25 px-2 py-0.5 text-xs font-bold text-white/80 hover:bg-black/45"
              title="下一球"
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

      {/* 視角標籤膠囊 */}
      <div className="relative z-20 flex justify-center">
        <span className="rounded-full bg-black/45 px-3 py-0.5 text-[11px] font-semibold text-white/90 backdrop-blur-sm border border-white/15">
          {pitch.cameraAngle === 'behind_pitcher'
            ? '📹 投手正後方視角 (Behind Pitcher)'
            : '📹 投手正前方 / 捕手視角 (Behind Catcher)'}
        </span>
      </div>

      {/* 3D 紅色漸變彗星彩帶軌跡 SVG */}
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
            {/* 連續圓柱體包絡面 */}
            <polygon points={polygonPoints} fill="url(#smartScoutRibbonGrad)" />
            {/* 逐點圓球疊加使轉折處完全圓滑立體 */}
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
            {/* 近端球體立體半透明圓形端蓋 (完全重現 SmartScout 紅色彩帶頭端立體感) */}
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
            {/* 動態播放時的飛行棒球球體 */}
            {lastSample && progress < 0.97 && (
              <g>
                <circle
                  cx={lastSample.x}
                  cy={lastSample.y}
                  r={Math.max(6, lastSample.r * 0.92)}
                  fill="#f8fafc"
                  stroke="#dc2626"
                  strokeWidth="2"
                />
              </g>
            )}
          </g>
        )}
      </svg>

      {/* 右側 4 顆 SmartScout 懸浮操作按鈕 (紅圈軌跡開關 / 慢動作錶盤 / 下載 / 數據圖表開關) */}
      <div className="absolute right-4 top-[38%] z-20 flex flex-col items-center gap-5">
        {/* 1. 紅色彩帶軌跡開關 */}
        <button
          type="button"
          onClick={() => setShowRibbon((v) => !v)}
          title={showRibbon ? '隱藏紅色彩帶軌跡' : '顯示紅色彩帶軌跡'}
          className="flex h-11 w-11 items-center justify-center rounded-full transition active:scale-90"
        >
          <span
            className={`flex h-9 w-9 items-center justify-center rounded-full border-[2.5px] border-white shadow-lg ${
              showRibbon ? 'bg-[#d9261c]' : 'bg-black/40'
            }`}
          />
        </button>

        {/* 2. 慢動作速度切換與重播 */}
        <button
          type="button"
          onClick={handleCycleSpeed}
          title="切換回放速度 (1x / 0.5x / 0.25x)"
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

        {/* 3. 下載含軌跡與數據卡的圖片 */}
        <button
          type="button"
          onClick={handleDownloadSnapshot}
          title="下載含軌跡與數據之圖卡"
          className="flex h-11 w-11 items-center justify-center rounded-full text-white drop-shadow-[0_2px_6px_rgba(0,0,0,0.8)] hover:scale-105 active:scale-95 transition"
        >
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
        </button>

        {/* 4. 顯示/隱藏下方數據黑卡 */}
        <button
          type="button"
          onClick={() => setShowStatsCard((v) => !v)}
          title={showStatsCard ? '隱藏下方數據卡' : '顯示下方數據卡'}
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

      {/* 底部區域：SmartScout 半透明深色數據卡 + 白色時間軸拉桿 */}
      <div className="relative z-20 mt-auto px-3.5 pb-4 space-y-3">
        {showStatsCard && (
          <div
            onClick={handleReplay}
            className="rounded-2xl bg-[#18110e]/85 backdrop-blur-md px-4 py-3.5 text-white shadow-2xl border border-white/10 cursor-pointer"
          >
            {/* 第一列：初速 107.0 km/h + 球種紅色下拉選單 */}
            <div className="flex items-center justify-between gap-2">
              <div>
                <div className="text-xs font-medium text-neutral-300">
                  {pitch.isHitting ? '擊球初速' : '初速'}
                </div>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-[44px] font-bold leading-tight tracking-tight tabular-nums">
                    {pitch.speedKmh.toFixed(1)}
                  </span>
                  <span className="text-lg font-medium text-neutral-300">km/h</span>
                </div>
              </div>

              <div onClick={(e) => e.stopPropagation()}>
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

            {/* 第二列：垂直變化量 / 水平變化量 / 轉速 / 有效轉速 */}
            <div className="mt-2 grid grid-cols-4 gap-1 text-center">
              <div>
                <div className="text-[11px] text-neutral-400">
                  {pitch.isHitting ? '擊球仰角' : '垂直變化量'}
                </div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {pitch.isHitting
                    ? `${(pitch.launchAngleDeg ?? pitch.verticalBreakCm).toFixed(1)}`
                    : `${pitch.verticalBreakCm.toFixed(1)}`}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">
                    {pitch.isHitting ? '°' : 'cm'}
                  </span>
                </div>
              </div>

              <div>
                <div className="text-[11px] text-neutral-400">
                  {pitch.isHitting ? '水平噴射角' : '水平變化量'}
                </div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {pitch.isHitting
                    ? `${(pitch.sprayAngleDeg ?? pitch.horizontalBreakCm).toFixed(1)}`
                    : `${pitch.horizontalBreakCm.toFixed(1)}`}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">
                    {pitch.isHitting ? '°' : 'cm'}
                  </span>
                </div>
              </div>

              <div>
                <div className="text-[11px] text-neutral-400">
                  {pitch.isHitting ? '飛行距離' : '轉速'}
                </div>
                <div className="mt-0.5 text-xl font-bold tabular-nums">
                  {pitch.isHitting ? `${pitch.distanceM ?? 88}` : pitch.totalSpinRpm}
                  <span className="ml-0.5 text-xs font-normal text-neutral-300">
                    {pitch.isHitting ? 'm' : 'rpm'}
                  </span>
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

            {/* 第三列：效率 / 旋轉方向 / 陀螺角度 */}
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

              <div className="flex items-end justify-center pb-1">
                <span className="rounded-md bg-white/10 px-2 py-0.5 text-[10px] text-neutral-300">
                  點按重播軌跡
                </span>
              </div>
            </div>
          </div>
        )}

        {/* 底部白色時間軸進度拉桿 (可拖曳逐幀檢視紅色彩帶生長過程) */}
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
