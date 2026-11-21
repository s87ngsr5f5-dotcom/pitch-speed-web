import React from 'react';
import { SpeedResult } from '../types';

interface SpeedDashboardProps {
  latestResult: SpeedResult | null;
  history: SpeedResult[];
  inferTimeMs: number;
  workerStatus: 'loading' | 'ready' | 'error';
  backend: string;
  workerError: string | null;
  isModelCached: boolean | null;
  cachedSizeMb: number | null;
  isOnline: boolean;
  isUpdatingModel: boolean;
  onForceUpdateModel: () => void;
  onClearHistory: () => void;
  onSelectHistoryItem?: (item: SpeedResult) => void;
}

function getSpeedTier(speedKmh: number | null): { label: string; badgeClass: string } {
  if (!speedKmh) {
    return {
      label: '待命偵測中',
      badgeClass: 'bg-slate-800 text-slate-400 border-slate-700',
    };
  }
  if (speedKmh >= 150) {
    return {
      label: '🔥 職業級火球 (150+ km/h)',
      badgeClass: 'bg-rose-950/80 text-rose-300 border-rose-700/70',
    };
  }
  if (speedKmh >= 138) {
    return {
      label: '⚡ 高速直球 (138+ km/h)',
      badgeClass: 'bg-amber-950/80 text-amber-300 border-amber-700/70',
    };
  }
  if (speedKmh >= 125) {
    return {
      label: '⚾ 實戰速球 / 高速變化球',
      badgeClass: 'bg-emerald-950/80 text-emerald-300 border-emerald-700/70',
    };
  }
  return {
    label: '🎯 曲球 / 變速球路',
    badgeClass: 'bg-sky-950/80 text-sky-300 border-sky-700/70',
  };
}

export const SpeedDashboard: React.FC<SpeedDashboardProps> = ({
  latestResult,
  history,
  inferTimeMs,
  workerStatus,
  backend,
  workerError,
  isModelCached,
  cachedSizeMb,
  isOnline,
  isUpdatingModel,
  onForceUpdateModel,
  onClearHistory,
  onSelectHistoryItem,
}) => {
  const confPct = latestResult ? Math.round(latestResult.confidenceScore * 100) : 0;
  const confColor =
    confPct >= 80 ? 'text-emerald-400' : confPct >= 60 ? 'text-amber-400' : 'text-rose-400';

  const maxKmh =
    history.length > 0
      ? Math.max(...history.map((h) => h.speedKmh)).toFixed(1)
      : latestResult
        ? latestResult.speedKmh.toFixed(1)
        : '--.-';

  const avgKmh =
    history.length > 0
      ? (history.reduce((acc, h) => acc + h.speedKmh, 0) / history.length).toFixed(1)
      : latestResult
        ? latestResult.speedKmh.toFixed(1)
        : '--.-';

  const tier = getSpeedTier(latestResult?.speedKmh ?? null);

  const exportCsv = () => {
    if (history.length === 0) return;
    const headers = ['編號,球種/標籤,預測球速(km/h),預測球速(mph),真實球速(km/h),誤差(km/h),飛行時間(ms),追蹤幀數,可靠度(%)'];
    const rows = history.map((item, idx) => {
      const no = history.length - idx;
      const gt = item.groundTruthKmh ?? '';
      const err = item.groundTruthKmh ? (item.speedKmh - item.groundTruthKmh).toFixed(2) : '';
      return `${no},"${item.label ?? '實測投球'}",${item.speedKmh.toFixed(1)},${item.speedMph.toFixed(1)},${gt},${err},${item.flightTimeMs},${item.trajectoryWorld.length},${Math.round(item.confidenceScore * 100)}`;
    });
    const blob = new Blob(['\uFEFF' + headers.concat(rows).join('\n')], {
      type: 'text/csv;charset=utf-8;',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pitchspeed-records-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // 建立球速趨勢折線圖座標 (取最近 12 球由舊到新)
  const chartItems = [...history].slice(0, 12).reverse();
  let sparklinePts = '';
  if (chartItems.length >= 2) {
    const speeds = chartItems.map((c) => c.speedKmh);
    const minS = Math.min(...speeds) - 3;
    const maxS = Math.max(...speeds) + 3;
    const span = Math.max(6, maxS - minS);
    sparklinePts = chartItems
      .map((c, i) => {
        const x = 10 + (i / (chartItems.length - 1)) * 220;
        const y = 38 - ((c.speedKmh - minS) / span) * 28;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }

  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
      <div className="md:col-span-2 flex flex-col justify-between rounded-2xl bg-slate-900/95 p-4 border border-slate-800 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                workerStatus === 'ready'
                  ? 'bg-emerald-400 animate-pulse'
                  : workerStatus === 'loading'
                    ? 'bg-amber-400 animate-bounce'
                    : 'bg-rose-500'
              }`}
            />
            <span className="font-semibold text-slate-200">
              {workerStatus === 'ready'
                ? `引擎就緒 (${backend})`
                : workerStatus === 'loading'
                  ? '載入 ONNX 模型中...'
                  : '模型載入失敗'}
            </span>

            {isModelCached !== null && (
              <span
                className={`rounded-full px-2 py-0.5 text-[11px] font-semibold border ${
                  isModelCached
                    ? 'bg-emerald-950/70 text-emerald-300 border-emerald-700/70'
                    : 'bg-slate-800 text-slate-400 border-slate-700'
                }`}
              >
                {isModelCached
                  ? `離線可用${cachedSizeMb ? ` (${cachedSizeMb}MB)` : ''}`
                  : '尚未快取'}
              </span>
            )}

            <span className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold border ${tier.badgeClass}`}>
              {tier.label}
            </span>

            {!isOnline && (
              <span className="rounded-full bg-amber-950/80 px-2 py-0.5 text-[11px] font-semibold text-amber-300 border border-amber-700/70">
                離線模式
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className="font-mono text-slate-300">
              延遲: {inferTimeMs > 0 ? `${inferTimeMs} ms` : '--'}
            </span>
            <button
              type="button"
              onClick={onForceUpdateModel}
              disabled={!isOnline || isUpdatingModel}
              title={!isOnline ? '需連線網路才能更新模型' : '清除舊模型快取並重新下載最新 ONNX'}
              className="rounded-lg bg-slate-800 px-2.5 py-1 text-[11px] font-semibold text-slate-200 border border-slate-700 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
            >
              {isUpdatingModel ? '更新中...' : '強制更新模型'}
            </button>
          </div>
        </div>

        {workerError && (
          <div className="mt-2 rounded-lg bg-rose-950/70 p-2 text-xs text-rose-300 border border-rose-800">
            {workerError}（請確認 public/models/ball_yolo11n.onnx 已存在）
          </div>
        )}

        <div className="my-3 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-baseline gap-4">
            <div className="text-6xl font-black tracking-tight tabular-nums text-white drop-shadow">
              {latestResult ? latestResult.speedKmh.toFixed(1) : '--.-'}
            </div>
            <div className="flex flex-col">
              <span className="text-lg font-extrabold text-emerald-400">km/h</span>
              <span className="text-sm font-mono text-slate-400">
                {latestResult ? `${latestResult.speedMph.toFixed(1)} mph` : '--.- mph'}
              </span>
            </div>
          </div>

          {/* MAX / AVG / COUNT 統計卡 */}
          <div className="grid grid-cols-3 gap-2 min-w-[230px]">
            <div className="rounded-xl bg-slate-950/70 px-3 py-2 border border-slate-800">
              <div className="text-[10px] font-semibold text-slate-400">最高 MAX</div>
              <div className="text-base font-black font-mono text-emerald-400">{maxKmh}</div>
            </div>
            <div className="rounded-xl bg-slate-950/70 px-3 py-2 border border-slate-800">
              <div className="text-[10px] font-semibold text-slate-400">平均 AVG</div>
              <div className="text-base font-black font-mono text-sky-400">{avgKmh}</div>
            </div>
            <div className="rounded-xl bg-slate-950/70 px-3 py-2 border border-slate-800">
              <div className="text-[10px] font-semibold text-slate-400">累計球數</div>
              <div className="text-base font-black font-mono text-white">{history.length}</div>
            </div>
          </div>
        </div>

        {latestResult?.groundTruthKmh && (
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2 rounded-xl bg-emerald-950/40 px-3 py-2 border border-emerald-700/50 text-xs">
            <span className="font-bold text-emerald-300">
              ⚾ {latestResult.label ?? '模型驗證球路'}
            </span>
            <div className="flex items-center gap-3 font-mono">
              <span className="text-slate-300">
                真實初速: <strong className="text-white">{latestResult.groundTruthKmh.toFixed(1)} km/h</strong>
              </span>
              <span
                className={`rounded-md px-2 py-0.5 font-bold ${
                  Math.abs(latestResult.speedKmh - latestResult.groundTruthKmh) <= 1.5
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'bg-amber-500/20 text-amber-300'
                }`}
              >
                誤差 {latestResult.speedKmh - latestResult.groundTruthKmh >= 0 ? '+' : ''}
                {(latestResult.speedKmh - latestResult.groundTruthKmh).toFixed(2)} km/h
              </span>
            </div>
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-4 text-xs text-slate-300 border-t border-slate-800 pt-2.5">
          <div className="flex flex-wrap items-center gap-4">
            <span>
              可靠度：<strong className={confColor}>{latestResult ? `${confPct}%` : '--'}</strong>
            </span>
            <span>
              追蹤點數：<strong>{latestResult ? latestResult.trajectoryWorld.length : 0} 幀</strong>
            </span>
            <span>
              區間飛行：<strong>{latestResult ? `${latestResult.flightTimeMs} ms` : '--'}</strong>
            </span>
          </div>

          {chartItems.length >= 2 && (
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-slate-400">球速趨勢</span>
              <svg width="120" height="24" viewBox="0 0 240 44" className="overflow-visible">
                <polyline
                  fill="none"
                  stroke="#34d399"
                  strokeWidth="3.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={sparklinePts}
                />
              </svg>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-col rounded-2xl bg-slate-900/95 p-4 border border-slate-800 shadow-xl">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-bold text-slate-200">歷史球速紀錄</h2>
          {history.length > 0 && (
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={exportCsv}
                className="rounded-md bg-slate-800 px-2 py-0.5 text-[11px] font-semibold text-emerald-300 border border-slate-700 hover:bg-slate-700 transition"
              >
                匯出 CSV
              </button>
              <button
                type="button"
                onClick={onClearHistory}
                className="text-xs text-slate-400 hover:text-white"
              >
                清除
              </button>
            </div>
          )}
        </div>
        <div className="flex-1 space-y-1.5 max-h-44 overflow-y-auto pr-1">
          {history.length === 0 ? (
            <p className="text-xs text-slate-500 py-6 text-center">
              尚無測速紀錄（可點上方「100 球模型練習庫」立即實測）
            </p>
          ) : (
            history.map((item, idx) => (
              <div
                key={idx}
                onClick={() => onSelectHistoryItem?.(item)}
                className="flex items-center justify-between rounded-xl bg-slate-800/75 hover:bg-slate-800 px-3 py-1.5 text-xs cursor-pointer border border-slate-700/50 transition"
              >
                <div className="flex items-center gap-2 truncate">
                  <span className="text-slate-400 font-mono">#{history.length - idx}</span>
                  {item.label && (
                    <span className="truncate text-[11px] font-semibold text-slate-300 max-w-[95px]">
                      {item.label}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <span className="font-mono font-bold text-white text-sm">
                    {item.speedKmh.toFixed(1)} <span className="text-[10px] text-emerald-400">km/h</span>
                  </span>
                  {item.groundTruthKmh ? (
                    <span className="rounded bg-emerald-950/90 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300 border border-emerald-800/70">
                      Δ{(item.speedKmh - item.groundTruthKmh >= 0 ? '+' : '') +
                        (item.speedKmh - item.groundTruthKmh).toFixed(1)}
                    </span>
                  ) : (
                    <span className="text-slate-400 font-mono">
                      {Math.round(item.confidenceScore * 100)}%
                    </span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
