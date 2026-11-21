import React from 'react';
import { SpeedResult } from '../types';

interface LandscapeHudBarProps {
  latestResult: SpeedResult | null;
  inferTimeMs: number;
  locked: boolean;
  wakeLockActive: boolean;
  onToggleLock: () => void;
  onExitHud: () => void;
}

export const LandscapeHudBar: React.FC<LandscapeHudBarProps> = ({
  latestResult,
  inferTimeMs,
  locked,
  wakeLockActive,
  onToggleLock,
  onExitHud,
}) => {
  return (
    <div className="pointer-events-none fixed inset-0 z-40 flex flex-col justify-between safe-p">
      <div className="flex items-start justify-between gap-2">
        <div className="pointer-events-auto flex items-center gap-2 rounded-xl bg-slate-950/70 px-3 py-1.5 backdrop-blur-md border border-slate-700/70 text-xs">
          <span className="font-bold text-emerald-400">18.44m HUD</span>
          <span className="font-mono text-slate-300">
            {inferTimeMs > 0 ? `${inferTimeMs}ms` : '--'}
          </span>
          {wakeLockActive && (
            <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] text-emerald-300">
              恆亮中
            </span>
          )}
        </div>

        <div className="pointer-events-auto flex items-center gap-2">
          <button
            type="button"
            onClick={onToggleLock}
            className={`rounded-xl px-3 py-1.5 text-xs font-bold shadow-lg backdrop-blur-md ${
              locked
                ? 'bg-amber-500/90 text-slate-950'
                : 'bg-emerald-600/90 text-white'
            }`}
          >
            {locked ? '解鎖四點' : '鎖定測速'}
          </button>
          <button
            type="button"
            onClick={onExitHud}
            className="rounded-xl bg-slate-900/80 px-3 py-1.5 text-xs font-semibold text-slate-200 backdrop-blur-md border border-slate-700"
          >
            退出全螢幕
          </button>
        </div>
      </div>

      <div className="flex justify-end">
        <div className="rounded-2xl bg-slate-950/75 px-4 py-2 backdrop-blur-md border border-emerald-500/40 shadow-2xl">
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-black tabular-nums text-white leading-none">
              {latestResult ? latestResult.speedKmh.toFixed(1) : '--.-'}
            </span>
            <span className="text-sm font-bold text-emerald-400">km/h</span>
          </div>
          <div className="mt-0.5 flex items-center justify-between gap-3 text-[11px] font-mono text-slate-300">
            <span>{latestResult ? `${latestResult.speedMph.toFixed(1)} mph` : '--.- mph'}</span>
            <span>
              {latestResult ? `${Math.round(latestResult.confidenceScore * 100)}%` : '--%'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
