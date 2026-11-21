import React, { useState } from 'react';
import { SmartScoutSession } from '../data/smartScoutDefaultSessions';

interface SmartScoutPlayerProfileProps {
  sessions: SmartScoutSession[];
}

export const SmartScoutPlayerProfile: React.FC<SmartScoutPlayerProfileProps> = ({
  sessions,
}) => {
  const [name, setName] = useState<string>('你');
  const [throwsHand, setThrowsHand] = useState<string>('投右');
  const [batsHand, setBatsHand] = useState<string>('打右');
  const [age, setAge] = useState<number>(17);
  const [position, setPosition] = useState<string>('投手 / 捕手');
  const [heightCm, setHeightCm] = useState<number>(170);
  const [weightKg, setWeightKg] = useState<number>(69);
  const [period, setPeriod] = useState<'1m' | '3m' | '12m'>('3m');
  const [editing, setEditing] = useState<boolean>(false);

  const allPitches = sessions.flatMap((s) => s.pitches.filter((p) => !p.isHitting));
  const totalCount = allPitches.length;
  const maxSpeed =
    totalCount > 0
      ? Math.max(...allPitches.map((p) => p.speedKmh)).toFixed(1)
      : '--.-';
  const avgSpeed =
    totalCount > 0
      ? (allPitches.reduce((a, p) => a + p.speedKmh, 0) / totalCount).toFixed(1)
      : '--.-';
  const spinRates = allPitches.flatMap((p) => p.totalSpinRpm ?? []);
  const verticalBreaks = allPitches.flatMap((p) => p.verticalBreakCm ?? []);
  const maxSpin = spinRates.length ? Math.max(...spinRates) : '—';
  const avgVB = verticalBreaks.length
    ? (verticalBreaks.reduce((sum, value) => sum + value, 0) / verticalBreaks.length).toFixed(1)
    : '—';

  // 活躍熱力方格 (4 列 x 11 欄)
  const months = ['2月', '5月', '8月', '11月'];

  return (
    <div className="mx-auto max-w-xl pb-24 space-y-4">
      <div className="rounded-3xl bg-white p-5 text-slate-900 shadow-md border border-slate-200 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-extrabold text-[#0b72b9]">{name}</h2>
          <button
            type="button"
            onClick={() => setEditing((v) => !v)}
            className="rounded-xl bg-slate-100 px-3 py-1 text-xs font-bold text-[#0b72b9] hover:bg-slate-200"
          >
            {editing ? '完成儲存' : '✎ 編輯球員資料'}
          </button>
        </div>

        {/* 頭像 + 訓練活躍熱力日曆 */}
        <div className="flex items-center gap-4">
          <div className="flex h-20 w-20 shrink-0 items-center justify-center rounded-2xl bg-slate-300 text-2xl font-extrabold text-[#0b72b9]">
            {name.slice(0, 2)}
          </div>

          <div className="flex-1">
            <div className="flex justify-between text-[11px] font-semibold text-slate-500 mb-1 px-1">
              {months.map((m) => (
                <span key={m}>{m}</span>
              ))}
            </div>
            <div className="grid grid-cols-11 gap-1.5">
              {Array.from({ length: 44 }).map((_, idx) => {
                const isActive = idx === 29 || idx === 41 || idx === 43;
                return (
                  <div
                    key={idx}
                    className={`h-3.5 rounded-sm ${
                      isActive ? 'bg-[#0b72b9]' : 'bg-slate-200'
                    }`}
                  />
                );
              })}
            </div>
          </div>
        </div>

        <div className="text-xs font-semibold text-slate-600">
          {throwsHand} / {batsHand} | {age}歲 | {position} | {heightCm} cm | {weightKg} kg
        </div>

        {editing && (
          <div className="grid grid-cols-2 gap-2.5 rounded-2xl bg-slate-50 p-3.5 border border-slate-200 text-xs">
            <label className="flex flex-col gap-1">
              <span className="text-slate-500">球員名稱</span>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="rounded-lg bg-white px-2.5 py-1.5 font-bold border border-slate-300"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-slate-500">守備位置</span>
              <input
                type="text"
                value={position}
                onChange={(e) => setPosition(e.target.value)}
                className="rounded-lg bg-white px-2.5 py-1.5 font-bold border border-slate-300"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-slate-500">投打習慣</span>
              <div className="flex gap-1">
                <select
                  value={throwsHand}
                  onChange={(e) => setThrowsHand(e.target.value)}
                  className="flex-1 rounded-lg bg-white px-2 py-1.5 font-bold border border-slate-300"
                >
                  <option value="投右">投右</option>
                  <option value="投左">投左</option>
                </select>
                <select
                  value={batsHand}
                  onChange={(e) => setBatsHand(e.target.value)}
                  className="flex-1 rounded-lg bg-white px-2 py-1.5 font-bold border border-slate-300"
                >
                  <option value="打右">打右</option>
                  <option value="打左">打左</option>
                  <option value="左右開弓">兩打</option>
                </select>
              </div>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-slate-500">年齡 / 身高 / 體重</span>
              <div className="flex gap-1">
                <input
                  type="number"
                  value={age}
                  onChange={(e) => setAge(Number(e.target.value) || 17)}
                  className="w-14 rounded-lg bg-white px-2 py-1.5 font-bold border border-slate-300"
                />
                <input
                  type="number"
                  value={heightCm}
                  onChange={(e) => setHeightCm(Number(e.target.value) || 170)}
                  className="w-16 rounded-lg bg-white px-2 py-1.5 font-bold border border-slate-300"
                />
                <input
                  type="number"
                  value={weightKg}
                  onChange={(e) => setWeightKg(Number(e.target.value) || 69)}
                  className="w-14 rounded-lg bg-white px-2 py-1.5 font-bold border border-slate-300"
                />
              </div>
            </label>
          </div>
        )}

        {/* 1個月 | 3個月 | 12個月 區間切換 */}
        <div className="grid grid-cols-3 rounded-xl bg-slate-200 p-1 text-xs font-bold">
          {(
            [
              { id: '1m', label: '1個月' },
              { id: '3m', label: '3個月' },
              { id: '12m', label: '12個月' },
            ] as const
          ).map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setPeriod(item.id)}
              className={`rounded-lg py-2 transition ${
                period === item.id
                  ? 'bg-[#0b72b9] text-white shadow'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>

        {/* 球員生涯核心指標卡片 (累計追蹤球數、最高球速、平均球速、最高轉速、平均垂直上竄量) */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pt-1">
          <div className="rounded-2xl bg-slate-100 p-3.5 border border-slate-200">
            <div className="text-xs text-slate-500">累計追蹤球數</div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-2xl font-black text-slate-900">{totalCount}</span>
              <span className="text-xs text-slate-500">球</span>
            </div>
          </div>

          <div className="rounded-2xl bg-slate-100 p-3.5 border border-slate-200">
            <div className="text-xs text-slate-500">最高球速 (MAX V)</div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-2xl font-black text-slate-900">{maxSpeed}</span>
              <span className="text-xs text-slate-500">km/h</span>
            </div>
          </div>

          <div className="rounded-2xl bg-slate-100 p-3.5 border border-slate-200">
            <div className="text-xs text-slate-500">平均球速 (AVG V)</div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-2xl font-black text-[#0b72b9]">{avgSpeed}</span>
              <span className="text-xs text-slate-500">km/h</span>
            </div>
          </div>

          <div className="rounded-2xl bg-slate-100 p-3.5 border border-slate-200">
            <div className="text-xs text-slate-500">最高總轉速 (MAX S)</div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-2xl font-black text-slate-900">{maxSpin}</span>
              <span className="text-xs text-slate-500">rpm</span>
            </div>
          </div>

          <div className="rounded-2xl bg-slate-100 p-3.5 border border-slate-200 sm:col-span-2">
            <div className="text-xs text-slate-500">平均垂直上竄量 (VB)</div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-2xl font-black text-rose-600">{avgVB}</span>
              <span className="text-xs text-slate-500">cm</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
