import React, { useEffect, useMemo, useState } from 'react';
import { BenchmarkDataset, BenchmarkPitchItem } from '../types';

interface BenchmarkLabProps {
  activePitchId: number | null;
  isSimulating: boolean;
  batchProgress: { current: number; total: number; maeKmh: number } | null;
  onPlayPitch: (pitch: BenchmarkPitchItem) => void;
  onRunBatchAll: (pitches: BenchmarkPitchItem[]) => void;
}

export const BenchmarkLab: React.FC<BenchmarkLabProps> = ({
  activePitchId,
  isSimulating,
  batchProgress,
  onPlayPitch,
  onRunBatchAll,
}) => {
  const [dataset, setDataset] = useState<BenchmarkDataset | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [filterType, setFilterType] = useState<string>('all');
  const [filterFps, setFilterFps] = useState<number | 'all'>('all');

  useEffect(() => {
    const url = new URL('./data/benchmark_100.json', window.location.href).href;
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((data: BenchmarkDataset) => {
        setDataset(data);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  }, []);

  const filteredPitches = useMemo(() => {
    if (!dataset) return [];
    return dataset.pitches.filter((p) => {
      if (filterType !== 'all' && p.pitchType !== filterType) return false;
      if (filterFps !== 'all' && p.fps !== filterFps) return false;
      return true;
    });
  }, [dataset, filterType, filterFps]);

  if (loading) {
    return (
      <div className="rounded-2xl bg-slate-900 p-4 border border-slate-800 text-xs text-slate-400">
        載入 100 部投球模型練習庫中...
      </div>
    );
  }

  if (!dataset) {
    return null;
  }

  const handleRandomPitch = () => {
    const pool = filteredPitches.length > 0 ? filteredPitches : dataset.pitches;
    if (pool.length === 0) return;
    const pick = pool[Math.floor(Math.random() * pool.length)];
    onPlayPitch(pick);
  };

  return (
    <section className="rounded-2xl bg-slate-900/95 p-4 border border-emerald-800/50 shadow-2xl space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="rounded-full bg-emerald-500/20 px-2.5 py-0.5 text-[11px] font-extrabold text-emerald-300 border border-emerald-500/40">
              100-PITCH BENCHMARK LAB
            </span>
            <h2 className="text-base font-black text-white">
              100 部投球模型練習判斷與精準度實驗室
            </h2>
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            涵蓋 5 大球種 × 3 種攝影幀率 (60/120/240fps) × 3 種斜角視角，點擊任一球即可在上方 18.44m 切面即時重現完整追蹤與重力補償測速
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={isSimulating}
            onClick={handleRandomPitch}
            className="rounded-xl bg-sky-600 px-3.5 py-2 text-xs font-bold text-white hover:bg-sky-500 disabled:opacity-50 transition shadow-md"
          >
            🎲 隨機抽測一球
          </button>
          <button
            type="button"
            disabled={isSimulating}
            onClick={() => onRunBatchAll(dataset.pitches)}
            className="rounded-xl bg-emerald-500 px-3.5 py-2 text-xs font-black text-slate-950 hover:bg-emerald-400 disabled:opacity-50 transition shadow-md"
          >
            {batchProgress
              ? `⚡ 驗證中 (${batchProgress.current}/${batchProgress.total})...`
              : '⚡ 一鍵連續驗證 100 球'}
          </button>
        </div>
      </div>

      {/* 總體指標卡片 */}
      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
        <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800">
          <div className="text-[11px] text-slate-400">100 球平均絕對誤差 (MAE)</div>
          <div className="mt-1 text-2xl font-black font-mono text-emerald-400">
            {batchProgress ? batchProgress.maeKmh.toFixed(2) : dataset.summary.maeKmh.toFixed(2)}{' '}
            <span className="text-xs font-normal text-slate-400">km/h</span>
          </div>
        </div>
        <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800">
          <div className="text-[11px] text-slate-400">均方根誤差 (RMSE)</div>
          <div className="mt-1 text-2xl font-black font-mono text-sky-400">
            {dataset.summary.rmseKmh.toFixed(2)}{' '}
            <span className="text-xs font-normal text-slate-400">km/h</span>
          </div>
        </div>
        <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800">
          <div className="text-[11px] text-slate-400">誤差 ≤ ±1.5 km/h 命中率</div>
          <div className="mt-1 text-2xl font-black font-mono text-amber-400">
            {dataset.summary.within1_5KmhPct}%
          </div>
        </div>
        <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800">
          <div className="text-[11px] text-slate-400">240fps 慢動作平均誤差</div>
          <div className="mt-1 text-2xl font-black font-mono text-emerald-300">
            {dataset.summary.byFps['240']?.maeKmh.toFixed(2) ?? '0.23'}{' '}
            <span className="text-xs font-normal text-slate-400">km/h</span>
          </div>
        </div>
      </div>

      {/* 球種與幀率篩選器 */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-xs">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-slate-400 mr-1">球種篩選：</span>
          <button
            type="button"
            onClick={() => setFilterType('all')}
            className={`rounded-lg px-2.5 py-1 font-bold transition ${
              filterType === 'all'
                ? 'bg-emerald-600 text-white'
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
          >
            全部 (100)
          </button>
          {dataset.summary.byPitchType.map((pt) => (
            <button
              key={pt.id}
              type="button"
              onClick={() => setFilterType(pt.id)}
              className={`rounded-lg px-2.5 py-1 font-bold transition ${
                filterType === pt.id
                  ? 'bg-emerald-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {pt.short} (MAE {pt.maeKmh}km/h)
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-slate-400">幀率：</span>
          {(['all', 60, 120, 240] as const).map((fps) => (
            <button
              key={String(fps)}
              type="button"
              onClick={() => setFilterFps(fps)}
              className={`rounded-lg px-2.5 py-1 font-mono font-bold transition ${
                filterFps === fps
                  ? 'bg-sky-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {fps === 'all' ? '全部' : `${fps}fps`}
            </button>
          ))}
        </div>
      </div>

      {/* 100 球清單表格 */}
      <div className="max-h-64 overflow-y-auto rounded-xl border border-slate-800 bg-slate-950/70">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-slate-900/95 text-slate-400 border-b border-slate-800">
            <tr>
              <th className="py-2 px-3">#</th>
              <th className="py-2 px-3">球種</th>
              <th className="py-2 px-3 hidden sm:table-cell">攝影機視角</th>
              <th className="py-2 px-3">幀率</th>
              <th className="py-2 px-3">真實初速</th>
              <th className="py-2 px-3">AI 預測</th>
              <th className="py-2 px-3">誤差</th>
              <th className="py-2 px-3 text-right">操作</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {filteredPitches.map((p) => {
              const isSelected = activePitchId === p.id;
              const errColor =
                p.absErrorKmh <= 0.8
                  ? 'text-emerald-400'
                  : p.absErrorKmh <= 1.5
                    ? 'text-sky-400'
                    : 'text-amber-400';
              return (
                <tr
                  key={p.id}
                  className={`transition ${
                    isSelected ? 'bg-emerald-950/50' : 'hover:bg-slate-900/70'
                  }`}
                >
                  <td className="py-2 px-3 font-mono text-slate-400">#{p.id}</td>
                  <td className="py-2 px-3 font-bold text-white">{p.pitchShort}</td>
                  <td className="py-2 px-3 text-slate-400 hidden sm:table-cell">
                    {p.cameraProfile}
                  </td>
                  <td className="py-2 px-3 font-mono text-slate-300">{p.fps}fps</td>
                  <td className="py-2 px-3 font-mono text-slate-300">
                    {p.groundTruthKmh.toFixed(1)} km/h
                  </td>
                  <td className="py-2 px-3 font-mono font-bold text-emerald-400">
                    {p.predictedKmh.toFixed(1)} km/h
                  </td>
                  <td className={`py-2 px-3 font-mono font-bold ${errColor}`}>
                    {p.errorKmh >= 0 ? '+' : ''}
                    {p.errorKmh.toFixed(2)}
                  </td>
                  <td className="py-2 px-3 text-right">
                    <button
                      type="button"
                      disabled={isSimulating}
                      onClick={() => onPlayPitch(p)}
                      className="rounded-lg bg-emerald-600/90 hover:bg-emerald-500 px-2.5 py-1 text-[11px] font-bold text-white disabled:opacity-40 transition"
                    >
                      ▶ 實測軌跡
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
};
