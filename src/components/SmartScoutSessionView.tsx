import React, { useMemo, useState } from 'react';
import {
  SMARTSCOUT_PITCH_TYPES,
  SmartScoutPitchRecord,
  SmartScoutSession,
  getPitchTypeColor,
} from '../data/smartScoutDefaultSessions';

interface SmartScoutSessionViewProps {
  sessions: SmartScoutSession[];
  selectedSessionId: string | null;
  onSelectSession: (sessionId: string | null) => void;
  onOpenPitchPlayer: (sessionId: string, pitchId: string) => void;
  onDeletePitch: (sessionId: string, pitchId: string) => void;
  onChangePitchType: (sessionId: string, pitchId: string, newType: string) => void;
  onStartNewRecording: (sessionId?: string) => void;
  onCreateNewSession: () => void;
}

export const SmartScoutSessionView: React.FC<SmartScoutSessionViewProps> = ({
  sessions,
  selectedSessionId,
  onSelectSession,
  onOpenPitchPlayer,
  onDeletePitch,
  onChangePitchType,
  onStartNewRecording,
  onCreateNewSession,
}) => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [disciplineFilter, setDisciplineFilter] = useState<'all' | 'pitching' | 'hitting'>('all');
  const [activeSubTab, setActiveSubTab] = useState<'list' | 'chart' | 'ai_coach'>('list');
  const [chartType, setChartType] = useState<'movement' | 'strike_zone'>('movement');
  const [selectedChartPitchId, setSelectedChartPitchId] = useState<string | null>(null);
  const [showCompareModal, setShowCompareModal] = useState<boolean>(false);
  const [comparePitchIds, setComparePitchIds] = useState<[string, string] | null>(null);

  const activeSession = useMemo(
    () => sessions.find((s) => s.id === selectedSessionId) ?? null,
    [sessions, selectedSessionId]
  );

  // 依球種彙整平均數據
  const pitchTypeSummaries = useMemo(() => {
    if (!activeSession) return [];
    const groups = new Map<
      string,
      {
        name: string;
        color: string;
        count: number;
        sumV: number;
        sumVB: number;
        sumHB: number;
        sumSpin: number;
        sumEff: number;
      }
    >();

    for (const p of activeSession.pitches) {
      const curr = groups.get(p.pitchType) ?? {
        name: p.pitchType,
        color: getPitchTypeColor(p.pitchType),
        count: 0,
        sumV: 0,
        sumVB: 0,
        sumHB: 0,
        sumSpin: 0,
        sumEff: 0,
      };
      curr.count += 1;
      curr.sumV += p.speedKmh;
      curr.sumVB += p.verticalBreakCm;
      curr.sumHB += p.horizontalBreakCm;
      curr.sumSpin += p.totalSpinRpm;
      curr.sumEff += p.efficiencyPct;
      groups.set(p.pitchType, curr);
    }

    return Array.from(groups.values()).map((g) => ({
      name: g.name,
      color: g.color,
      count: g.count,
      avgV: Number((g.sumV / g.count).toFixed(1)),
      avgVB: Number((g.sumVB / g.count).toFixed(1)),
      avgHB: Number((g.sumHB / g.count).toFixed(1)),
      avgSpin: Math.round(g.sumSpin / g.count),
      avgEff: Math.round(g.sumEff / g.count),
    }));
  }, [activeSession]);

  // 1. 若尚未點進特定場次 -> 顯示 SmartScout「場次列表」總覽畫面
  if (!activeSession) {
    const filtered = sessions.filter((s) => {
      if (disciplineFilter !== 'all' && s.discipline !== disciplineFilter) return false;
      if (searchQuery.trim() && !s.title.toLowerCase().includes(searchQuery.trim().toLowerCase())) {
        return false;
      }
      return true;
    });

    const groupedByMonth = new Map<string, SmartScoutSession[]>();
    for (const s of filtered) {
      const list = groupedByMonth.get(s.monthGroup) ?? [];
      list.push(s);
      groupedByMonth.set(s.monthGroup, list);
    }

    return (
      <div className="mx-auto max-w-2xl pb-24 space-y-4">
        {/* 頂部搜尋列與篩選按鈕 */}
        <div className="flex items-center gap-2.5">
          <div className="relative flex-1">
            <span className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400">
              🔍
            </span>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="依場次名稱搜尋"
              className="w-full rounded-2xl bg-white px-10 py-3 text-sm font-medium text-slate-800 placeholder-slate-400 shadow-sm border border-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-500"
            />
          </div>

          <button
            type="button"
            onClick={() =>
              setDisciplineFilter((prev) =>
                prev === 'all' ? 'pitching' : prev === 'pitching' ? 'hitting' : 'all'
              )
            }
            className="flex h-12 px-3.5 items-center justify-center rounded-2xl bg-slate-100 text-xs font-bold text-sky-700 border border-slate-200 hover:bg-slate-200 transition shrink-0"
          >
            {disciplineFilter === 'all'
              ? '全部類型'
              : disciplineFilter === 'pitching'
                ? '⚾ 僅看投球'
                : '🏏 僅看打擊'}
          </button>
        </div>

        {/* 依月份分組的場次卡片 */}
        {Array.from(groupedByMonth.entries()).map(([month, monthSessions]) => (
          <div key={month} className="space-y-3">
            <h3 className="px-1 text-sm font-bold text-slate-500">{month}</h3>

            {monthSessions.map((sess) => {
              const total = sess.pitches.length;
              const avgV =
                total > 0
                  ? (sess.pitches.reduce((a, p) => a + p.speedKmh, 0) / total).toFixed(1)
                  : '0.0';
              const avgVB =
                total > 0
                  ? (sess.pitches.reduce((a, p) => a + p.verticalBreakCm, 0) / total).toFixed(1)
                  : '0.0';
              const avgHB =
                total > 0
                  ? (sess.pitches.reduce((a, p) => a + p.horizontalBreakCm, 0) / total).toFixed(1)
                  : '0.0';
              const mainType = sess.pitches[0]?.pitchType ?? '四縫線快速球';
              const mainTypeCount = sess.pitches.filter((p) => p.pitchType === mainType).length;

              return (
                <div
                  key={sess.id}
                  onClick={() => {
                    onSelectSession(sess.id);
                    setActiveSubTab('list');
                  }}
                  className="cursor-pointer rounded-3xl bg-white p-5 text-slate-900 shadow-md border border-slate-200/90 hover:shadow-lg hover:border-sky-300 transition space-y-3"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="text-xl font-extrabold text-slate-900">{sess.title}</h4>
                      <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
                        <span>📅 {sess.date}</span>
                        <span>·</span>
                        <span>
                          {sess.cameraSetup === 'behind_pitcher'
                            ? '投手正後方視角'
                            : '投手正前方 / 捕手視角'}
                        </span>
                      </div>
                    </div>
                    <span className="text-slate-400 text-lg">›</span>
                  </div>

                  {total > 0 && (
                    <>
                      <div className="border-t border-slate-100 pt-3">
                        <span className="inline-flex items-center gap-1.5 rounded-xl bg-rose-100 px-3 py-1 text-xs font-bold text-rose-800">
                          <span>{sess.discipline === 'hitting' ? '🏏' : '⚾'}</span>
                          <span>
                            {mainType} {mainTypeCount}/{total}球
                          </span>
                        </span>
                      </div>

                      <div className="grid grid-cols-3 gap-2 pt-1">
                        <div>
                          <div className="text-xs text-slate-500">
                            {sess.discipline === 'hitting' ? '平均擊球初速' : '平均球速'}
                          </div>
                          <div className="mt-0.5 flex items-baseline gap-1">
                            <span className="text-2xl font-extrabold tabular-nums text-slate-900">
                              {avgV}
                            </span>
                            <span className="text-xs text-slate-500">km/h</span>
                          </div>
                        </div>

                        <div>
                          <div className="text-xs text-slate-500">
                            {sess.discipline === 'hitting' ? '平均擊球仰角' : '平均縱向位移'}
                          </div>
                          <div className="mt-0.5 flex items-baseline gap-1">
                            <span className="text-2xl font-extrabold tabular-nums text-slate-900">
                              {avgVB}
                            </span>
                            <span className="text-xs text-slate-500">
                              {sess.discipline === 'hitting' ? '°' : 'cm'}
                            </span>
                          </div>
                        </div>

                        <div>
                          <div className="text-xs text-slate-500">
                            {sess.discipline === 'hitting' ? '平均水平噴射角' : '平均橫向位移'}
                          </div>
                          <div className="mt-0.5 flex items-baseline gap-1">
                            <span className="text-2xl font-extrabold tabular-nums text-slate-900">
                              {avgHB}
                            </span>
                            <span className="text-xs text-slate-500">
                              {sess.discipline === 'hitting' ? '°' : 'cm'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        ))}

        {/* 底部懸浮按鈕列 (邁向下一個層級 + 新增場次) */}
        <div className="fixed bottom-18 inset-x-0 z-30 mx-auto flex max-w-md items-center gap-2.5 px-4">
          <button
            type="button"
            onClick={() => {
              const first = sessions[0];
              if (first) {
                onSelectSession(first.id);
                setActiveSubTab('ai_coach');
              }
            }}
            className="flex-1 flex items-center justify-center gap-2 rounded-2xl bg-[#0b72b9] py-3.5 text-sm font-bold text-white shadow-xl hover:bg-[#0962a0] transition"
          >
            <span>🏅</span>
            <span>邁向下一個層級（AI 教練完整分析）</span>
          </button>

          <button
            type="button"
            onClick={onCreateNewSession}
            title="新增投球或打擊場次"
            className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#0b72b9] text-white shadow-xl hover:bg-[#0962a0] transition text-xl font-bold"
          >
            ＋
          </button>
        </div>
      </div>
    );
  }

  // 2. 已點進特定場次 -> 顯示「列表 | 圖表 | AI 教練」三大分頁
  const pitchesReversed = [...activeSession.pitches].sort(
    (a, b) => b.pitchNumber - a.pitchNumber
  );

  const selectedChartPitch =
    activeSession.pitches.find((p) => p.id === selectedChartPitchId) ??
    activeSession.pitches[0] ??
    null;

  // 計算 AI 教練的 3 大優勢與 3 大重點改善
  const avgSpeed =
    activeSession.pitches.length > 0
      ? activeSession.pitches.reduce((a, p) => a + p.speedKmh, 0) / activeSession.pitches.length
      : 105;
  const avgVB =
    activeSession.pitches.length > 0
      ? activeSession.pitches.reduce((a, p) => a + p.verticalBreakCm, 0) /
        activeSession.pitches.length
      : 52;
  const avgHB =
    activeSession.pitches.length > 0
      ? activeSession.pitches.reduce((a, p) => a + p.horizontalBreakCm, 0) /
        activeSession.pitches.length
      : 23;
  const avgEff =
    activeSession.pitches.length > 0
      ? Math.round(
          activeSession.pitches.reduce((a, p) => a + p.efficiencyPct, 0) /
            activeSession.pitches.length
        )
      : 82;
  const avgSpin =
    activeSession.pitches.length > 0
      ? Math.round(
          activeSession.pitches.reduce((a, p) => a + p.totalSpinRpm, 0) /
            activeSession.pitches.length
        )
      : 2050;

  return (
    <div className="mx-auto max-w-2xl pb-24 space-y-4">
      {/* 頂部導覽列：< 2026/08/26 投球 ⓘ ⋮ */}
      <div className="rounded-3xl bg-white text-slate-900 shadow-md border border-slate-200 overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3.5">
          <button
            type="button"
            onClick={() => onSelectSession(null)}
            className="flex h-9 w-9 items-center justify-center rounded-full text--[#0b72b9] hover:bg-slate-100 transition text-xl font-bold text-[#0b72b9]"
          >
            ‹
          </button>

          <div className="flex items-center gap-1.5">
            <h2 className="text-base font-extrabold text-[#0b72b9] underline decoration-dotted underline-offset-4">
              {activeSession.title}
            </h2>
            <span className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-[#0b72b9] text-[11px] font-bold text-[#0b72b9]">
              i
            </span>
          </div>

          <button
            type="button"
            onClick={() => onStartNewRecording(activeSession.id)}
            className="rounded-xl bg-sky-50 px-2.5 py-1 text-xs font-bold text-[#0b72b9] hover:bg-sky-100"
          >
            ＋新增一球
          </button>
        </div>

        {/* 列表 | 圖表 | AI 教練 分頁切換列 */}
        <div className="grid grid-cols-3 border-t border-slate-200 text-center text-sm font-bold">
          {(
            [
              { id: 'list', label: '列表' },
              { id: 'chart', label: '圖表' },
              { id: 'ai_coach', label: 'AI 教練' },
            ] as const
          ).map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setActiveSubTab(t.id)}
              className={`py-3 border-b-[3px] transition ${
                activeSubTab === t.id
                  ? 'border-[#0b72b9] text-[#0b72b9] bg-sky-50/40'
                  : 'border-transparent text-slate-600 hover:text-slate-900'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* 子分頁 1：列表 (8 格數據卡片：V / VB / HB / E / SD / GD / AS / S) */}
      {activeSubTab === 'list' && (
        <div className="space-y-3.5">
          {pitchesReversed.map((p) => (
            <div
              key={p.id}
              onClick={() => onOpenPitchPlayer(activeSession.id, p.id)}
              className="cursor-pointer rounded-3xl bg-white p-4 text-slate-900 shadow-md border border-slate-200 hover:border-sky-400 hover:shadow-lg transition space-y-3"
            >
              <div className="flex items-center justify-between text-xs text-slate-500">
                <span className="font-mono">{p.timestamp}</span>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeletePitch(activeSession.id, p.id);
                  }}
                  className="rounded-lg p-1 text-slate-400 hover:bg-rose-50 hover:text-rose-600 transition"
                  title="刪除此球"
                >
                  🗑
                </button>
              </div>

              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-sky-100 text-xs">
                    {p.isHitting ? '🏏' : '⚾'}
                  </span>
                  <span className="text-base font-extrabold text-slate-900">
                    {p.pitchNumber}. {p.isHitting ? '打擊' : '投球'}
                  </span>
                </div>

                <div onClick={(e) => e.stopPropagation()}>
                  <select
                    value={p.pitchType}
                    onChange={(e) =>
                      onChangePitchType(activeSession.id, p.id, e.target.value)
                    }
                    style={{ backgroundColor: getPitchTypeColor(p.pitchType) }}
                    className="rounded-lg px-3 py-1 text-xs font-bold text-white shadow-sm focus:outline-none cursor-pointer"
                  >
                    {SMARTSCOUT_PITCH_TYPES.map((pt) => (
                      <option key={pt.name} value={pt.name} className="bg-slate-900 text-white">
                        {pt.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* SmartScout 經典 2x4 八格數據方塊 */}
              <div className="grid grid-cols-4 gap-2 text-center">
                <div className="rounded-xl bg-slate-100/90 py-2 px-1 border border-slate-200/80">
                  <div className="text-[11px] text-slate-500">V [km/h]</div>
                  <div className="mt-0.5 text-base font-extrabold tabular-nums text-slate-900">
                    {p.speedKmh.toFixed(1)}
                  </div>
                </div>

                <div className="rounded-xl bg-slate-100/90 py-2 px-1 border border-slate-200/80">
                  <div className="text-[11px] text-slate-500">
                    {p.isHitting ? 'LA [°]' : 'VB [cm]'}
                  </div>
                  <div className="mt-0.5 text-base font-extrabold tabular-nums text-slate-900">
                    {(p.isHitting ? p.launchAngleDeg ?? p.verticalBreakCm : p.verticalBreakCm).toFixed(1)}
                  </div>
                </div>

                <div className="rounded-xl bg-slate-100/90 py-2 px-1 border border-slate-200/80">
                  <div className="text-[11px] text-slate-500">
                    {p.isHitting ? 'SA [°]' : 'HB [cm]'}
                  </div>
                  <div className="mt-0.5 text-base font-extrabold tabular-nums text-slate-900">
                    {(p.isHitting ? p.sprayAngleDeg ?? p.horizontalBreakCm : p.horizontalBreakCm).toFixed(1)}
                  </div>
                </div>

                <div className="rounded-xl bg-slate-100/90 py-2 px-1 border border-slate-200/80">
                  <div className="text-[11px] text-slate-500">E [%]</div>
                  <div className="mt-0.5 text-base font-extrabold tabular-nums text-slate-900">
                    {p.efficiencyPct}
                  </div>
                </div>

                <div className="rounded-xl bg-slate-100/90 py-2 px-1 border border-slate-200/80">
                  <div className="text-[11px] text-slate-500">SD</div>
                  <div className="mt-0.5 text-base font-extrabold tabular-nums text-slate-900">
                    {p.spinDirection}
                  </div>
                </div>

                <div className="rounded-xl bg-slate-100/90 py-2 px-1 border border-slate-200/80">
                  <div className="text-[11px] text-slate-500">GD [°]</div>
                  <div className="mt-0.5 text-base font-extrabold tabular-nums text-slate-900">
                    {p.gyroDegree}
                  </div>
                </div>

                <div className="rounded-xl bg-slate-100/90 py-2 px-1 border border-slate-200/80">
                  <div className="text-[11px] text-slate-500">AS [rpm]</div>
                  <div className="mt-0.5 text-base font-extrabold tabular-nums text-slate-900">
                    {p.activeSpinRpm}
                  </div>
                </div>

                <div className="rounded-xl bg-slate-100/90 py-2 px-1 border border-slate-200/80">
                  <div className="text-[11px] text-slate-500">
                    {p.isHitting ? 'DIST [m]' : 'S [rpm]'}
                  </div>
                  <div className="mt-0.5 text-base font-extrabold tabular-nums text-slate-900">
                    {p.isHitting ? p.distanceM ?? 88 : p.totalSpinRpm}
                  </div>
                </div>
              </div>
            </div>
          ))}

          {/* 底部懸浮按鈕：比較結果 + 錄製新球 */}
          <div className="fixed bottom-18 inset-x-0 z-30 mx-auto flex max-w-md items-center gap-2.5 px-4">
            <button
              type="button"
              onClick={() => {
                if (activeSession.pitches.length >= 2) {
                  setComparePitchIds([
                    activeSession.pitches[0].id,
                    activeSession.pitches[1].id,
                  ]);
                }
                setShowCompareModal(true);
              }}
              className="flex-1 flex items-center justify-center gap-2 rounded-2xl bg-[#0b72b9] py-3.5 text-sm font-bold text-white shadow-xl hover:bg-[#0962a0] transition"
            >
              <span>◇</span>
              <span>比較結果</span>
            </button>

            <button
              type="button"
              onClick={() => onStartNewRecording(activeSession.id)}
              title="錄製或上傳新投球至此場次"
              className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#0b72b9] text-white shadow-xl hover:bg-[#0962a0] transition text-lg"
            >
              📹
            </button>
          </div>
        </div>
      )}

      {/* 子分頁 2：圖表 (投球變化量圖 -60~+60 cm & 九宮格進壘分布圖) */}
      {activeSubTab === 'chart' && (
        <div className="rounded-3xl bg-white p-5 text-slate-900 shadow-md border border-slate-200 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-lg font-extrabold text-slate-900">
              {chartType === 'movement' ? '投球變化量圖' : '九宮格好球帶進壘分布圖'}
            </h3>
            <div className="flex rounded-xl bg-slate-100 p-1 text-xs font-bold">
              <button
                type="button"
                onClick={() => setChartType('movement')}
                className={`rounded-lg px-2.5 py-1 transition ${
                  chartType === 'movement'
                    ? 'bg-[#0b72b9] text-white'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                投球變化量圖
              </button>
              <button
                type="button"
                onClick={() => setChartType('strike_zone')}
                className={`rounded-lg px-2.5 py-1 transition ${
                  chartType === 'strike_zone'
                    ? 'bg-[#0b72b9] text-white'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                九宮格進壘點
              </button>
            </div>
          </div>

          {chartType === 'movement' ? (
            /* SmartScout 1:1 投球變化量圖 (-60 ~ +60 cm 象限方格) */
            <div className="mx-auto max-w-[360px]">
              <svg viewBox="0 0 340 340" className="w-full h-auto select-none">
                {/* 縱軸標題：垂直變化量 [cm] */}
                <text
                  x="16"
                  y="160"
                  fill="#64748b"
                  fontSize="13"
                  fontWeight="600"
                  textAnchor="middle"
                  transform="rotate(-90 16 160)"
                >
                  垂直變化量 [cm]
                </text>

                {/* 橫軸標題：水平變化量 [cm] */}
                <text
                  x="185"
                  y="332"
                  fill="#64748b"
                  fontSize="13"
                  fontWeight="600"
                  textAnchor="middle"
                >
                  水平變化量 [cm]
                </text>

                {/* 座標外框 (x: 55..315, y: 20..280, 中心 185, 150, 每 20cm = 37.14px) */}
                <rect
                  x="55"
                  y="20"
                  width="260"
                  height="260"
                  fill="#f8fafc"
                  stroke="#94a3b8"
                  strokeWidth="1.5"
                />

                {/* 格線與刻度 (-60, -40, -20, 0, 20, 40, 60) */}
                {[-60, -40, -20, 0, 20, 40, 60].map((val) => {
                  const px = 185 + (val / 70) * 130;
                  const py = 150 - (val / 70) * 130;
                  const isZero = val === 0;
                  return (
                    <g key={val}>
                      <line
                        x1={px}
                        y1={20}
                        x2={px}
                        y2={280}
                        stroke={isZero ? '#94a3b8' : '#e2e8f0'}
                        strokeWidth={isZero ? '1.8' : '1'}
                      />
                      <line
                        x1={55}
                        y1={py}
                        x2={315}
                        y2={py}
                        stroke={isZero ? '#94a3b8' : '#e2e8f0'}
                        strokeWidth={isZero ? '1.8' : '1'}
                      />
                      {/* Y 軸刻度文字 */}
                      <text
                        x="47"
                        y={py + 4}
                        fill="#64748b"
                        fontSize="11"
                        textAnchor="end"
                      >
                        {val}
                      </text>
                      {/* X 軸刻度文字 */}
                      <text
                        x={px}
                        y="298"
                        fill="#64748b"
                        fontSize="11"
                        textAnchor="middle"
                      >
                        {val}
                      </text>
                    </g>
                  );
                })}

                {/* 每一球的散佈圓點 */}
                {activeSession.pitches.map((p) => {
                  const clampedHB = Math.max(-68, Math.min(68, p.horizontalBreakCm));
                  const clampedVB = Math.max(-68, Math.min(68, p.verticalBreakCm));
                  const cx = 185 + (clampedHB / 70) * 130;
                  const cy = 150 - (clampedVB / 70) * 130;
                  const isSelected = selectedChartPitch?.id === p.id;
                  const color = getPitchTypeColor(p.pitchType);
                  return (
                    <g
                      key={p.id}
                      onClick={() => {
                        if (selectedChartPitchId === p.id) {
                          onOpenPitchPlayer(activeSession.id, p.id);
                        } else {
                          setSelectedChartPitchId(p.id);
                        }
                      }}
                      className="cursor-pointer"
                    >
                      {isSelected && (
                        <circle
                          cx={cx}
                          cy={cy}
                          r="11"
                          fill="none"
                          stroke="#0284c7"
                          strokeWidth="2"
                        />
                      )}
                      <circle
                        cx={cx}
                        cy={cy}
                        r="6.5"
                        fill={color}
                        fillOpacity="0.82"
                        stroke={color}
                        strokeWidth="1.5"
                      />
                    </g>
                  );
                })}
              </svg>
            </div>
          ) : (
            /* 九宮格好球帶進壘點圖 */
            <div className="mx-auto max-w-[300px]">
              <svg viewBox="0 0 260 300" className="w-full h-auto select-none">
                {/* 外圍追打區 */}
                <rect
                  x="30"
                  y="20"
                  width="200"
                  height="220"
                  rx="8"
                  fill="#f1f5f9"
                  stroke="#cbd5e1"
                  strokeWidth="1.5"
                  strokeDasharray="4 4"
                />
                {/* 核心九宮格好球帶 (x: 65..195, y: 55..205) */}
                <rect
                  x="65"
                  y="55"
                  width="130"
                  height="150"
                  fill="#ffffff"
                  stroke="#334155"
                  strokeWidth="2.2"
                />
                {/* 九宮格內線 */}
                <line x1="108.3" y1="55" x2="108.3" y2="205" stroke="#94a3b8" strokeWidth="1.2" />
                <line x1="151.6" y1="55" x2="151.6" y2="205" stroke="#94a3b8" strokeWidth="1.2" />
                <line x1="65" y1="105" x2="195" y2="105" stroke="#94a3b8" strokeWidth="1.2" />
                <line x1="65" y1="155" x2="195" y2="155" stroke="#94a3b8" strokeWidth="1.2" />

                {/* 五角形本壘板示意 */}
                <polygon
                  points="65,255 195,255 195,270 130,290 65,270"
                  fill="#e2e8f0"
                  stroke="#475569"
                  strokeWidth="2"
                />

                {activeSession.pitches.map((p) => {
                  const cx = 130 + (p.plateXCm / 21.6) * 65;
                  const cy = 130 - ((p.plateYCm - 76) / 30) * 75;
                  const color = getPitchTypeColor(p.pitchType);
                  const isSelected = selectedChartPitch?.id === p.id;
                  return (
                    <g
                      key={p.id}
                      onClick={() => {
                        if (selectedChartPitchId === p.id) {
                          onOpenPitchPlayer(activeSession.id, p.id);
                        } else {
                          setSelectedChartPitchId(p.id);
                        }
                      }}
                      className="cursor-pointer"
                    >
                      <circle
                        cx={Math.max(25, Math.min(235, cx))}
                        cy={Math.max(20, Math.min(235, cy))}
                        r={isSelected ? '10' : '8'}
                        fill={color}
                        stroke="#ffffff"
                        strokeWidth="1.8"
                      />
                      <text
                        x={Math.max(25, Math.min(235, cx))}
                        y={Math.max(20, Math.min(235, cy)) + 3}
                        fill="#ffffff"
                        fontSize="8.5"
                        fontWeight="bold"
                        textAnchor="middle"
                      >
                        {p.pitchNumber}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          )}

          <p className="text-center text-xs text-slate-500">
            👆 點按圓點查看詳情，再次點按前往紅色彩帶軌跡影片
          </p>

          {selectedChartPitch && (
            <div
              onClick={() => onOpenPitchPlayer(activeSession.id, selectedChartPitch.id)}
              className="flex flex-wrap items-center justify-between gap-2 rounded-2xl bg-sky-50 px-4 py-2.5 text-xs border border-sky-200 cursor-pointer hover:bg-sky-100 transition"
            >
              <span className="font-bold text-slate-800">
                #{selectedChartPitch.pitchNumber} {selectedChartPitch.pitchType} · 初速{' '}
                <strong className="text-[#0b72b9]">{selectedChartPitch.speedKmh.toFixed(1)} km/h</strong>
              </span>
              <span className="text-slate-600">
                VB {selectedChartPitch.verticalBreakCm.toFixed(1)}cm · HB{' '}
                {selectedChartPitch.horizontalBreakCm.toFixed(1)}cm · {selectedChartPitch.totalSpinRpm} rpm
              </span>
              <span className="font-bold text-[#0b72b9]">開啟軌跡影片 ›</span>
            </div>
          )}

          {/* 球種平均值摘要表 */}
          <div className="overflow-x-auto border-t border-slate-200 pt-3">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-600">
                  <th className="py-2 pr-2">球種</th>
                  <th className="py-2 px-2 text-right">平均球速 [km/h]</th>
                  <th className="py-2 px-2 text-right">平均垂直變化量 [cm]</th>
                  <th className="py-2 pl-2 text-right">平均水平變化量 [cm]</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-semibold">
                {pitchTypeSummaries.map((s) => (
                  <tr key={s.name}>
                    <td className="py-2.5 pr-2 flex items-center gap-2">
                      <span
                        className="inline-block h-3 w-3 rounded-full"
                        style={{ backgroundColor: s.color }}
                      />
                      <span>{s.name}</span>
                    </td>
                    <td className="py-2.5 px-2 text-right font-mono text-sm">{s.avgV.toFixed(1)}</td>
                    <td className="py-2.5 px-2 text-right font-mono text-sm">{s.avgVB.toFixed(1)}</td>
                    <td className="py-2.5 pl-2 text-right font-mono text-sm">{s.avgHB.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 子分頁 3：AI 教練 (完全免費解鎖無馬賽克！含球種特徵 + 優勢 3 / 重點改善 3) */}
      {activeSubTab === 'ai_coach' && (
        <div className="space-y-4">
          <div className="text-center py-2">
            <h3 className="text-2xl font-black text-white">已分析您的投球</h3>
            <p className="mt-1 text-xs text-emerald-400 font-semibold">
              🔓 AI 教練專業版已完整解鎖（免訂閱、無次數限制）
            </p>
          </div>

          {/* 卡片 1：球種特徵 */}
          <div className="rounded-3xl bg-white p-5 text-slate-900 shadow-md border border-slate-200 space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-base font-extrabold text-slate-900">球種特徵</h4>
              <span className="rounded-full bg-emerald-100 px-2.5 py-0.5 text-[11px] font-bold text-emerald-800">
                🔓 已解鎖完整數據
              </span>
            </div>

            <div className="overflow-x-auto rounded-2xl border border-slate-200">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-100 text-slate-600">
                  <tr>
                    <th className="py-2.5 px-3">球種</th>
                    <th className="py-2.5 px-2 text-right">球速 [km/h]</th>
                    <th className="py-2.5 px-2 text-right">垂直變化量 [cm]</th>
                    <th className="py-2.5 px-2 text-right">水平變化量 [cm]</th>
                    <th className="py-2.5 px-3 text-right">平均轉速 / 效率</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 font-semibold">
                  {pitchTypeSummaries.map((s) => (
                    <tr key={s.name}>
                      <td className="py-3 px-3 flex items-center gap-2">
                        <span
                          className="inline-block h-2.5 w-2.5 rounded-full"
                          style={{ backgroundColor: s.color }}
                        />
                        <span>
                          {s.name} ({s.count}球)
                        </span>
                      </td>
                      <td className="py-3 px-2 text-right font-mono text-sm">{s.avgV.toFixed(1)}</td>
                      <td className="py-3 px-2 text-right font-mono text-sm">{s.avgVB.toFixed(1)}</td>
                      <td className="py-3 px-2 text-right font-mono text-sm">{s.avgHB.toFixed(1)}</td>
                      <td className="py-3 px-3 text-right font-mono text-slate-700">
                        {s.avgSpin} rpm ({s.avgEff}%)
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* 卡片 2：優勢與重點改善 (優勢 3 / 重點改善 3) */}
          <div className="rounded-3xl bg-white p-5 text-slate-900 shadow-md border border-slate-200 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-base font-extrabold text-slate-900">優勢與重點改善</h4>
                <p className="text-xs text-slate-500">優勢 3 / 重點改善 3</p>
              </div>
              <span className="rounded-full bg-sky-100 px-2.5 py-0.5 text-[11px] font-bold text-sky-800">
                AI 生物力學診斷
              </span>
            </div>

            <div className="space-y-2.5">
              <div className="rounded-2xl bg-emerald-50 p-3.5 border border-emerald-200">
                <div className="text-xs font-extrabold text-emerald-800">
                  ✓ 優勢 1：出色的垂直上竄尾勁（平均 VB {avgVB.toFixed(1)} cm）
                </div>
                <p className="mt-1 text-xs text-emerald-900/80 leading-relaxed">
                  您的主戰速球具備極佳的馬格努斯上竄位移（{avgVB.toFixed(1)} cm），代表手指在釋球瞬間能給予球心紮實的純背旋，進壘時視覺上竄感明顯，高角度進壘極易誘使打者揮空。
                </p>
              </div>

              <div className="rounded-2xl bg-emerald-50 p-3.5 border border-emerald-200">
                <div className="text-xs font-extrabold text-emerald-800">
                  ✓ 優勢 2：轉速與有效轉速穩定（平均 {avgSpin} rpm · 效率 {avgEff}%）
                </div>
                <p className="mt-1 text-xs text-emerald-900/80 leading-relaxed">
                  本場次平均總轉速達 {avgSpin} rpm，旋轉效率維持在 {avgEff}%，顯示手腕與手指扣球軸線穩定，能量傳遞損耗低。
                </p>
              </div>

              <div className="rounded-2xl bg-emerald-50 p-3.5 border border-emerald-200">
                <div className="text-xs font-extrabold text-emerald-800">
                  ✓ 優勢 3：自然的橫向尾勁延伸（平均 HB {avgHB.toFixed(1)} cm）
                </div>
                <p className="mt-1 text-xs text-emerald-900/80 leading-relaxed">
                  球路帶有 {avgHB.toFixed(1)} cm 的手臂側橫向位移，搭配 {avgSpeed.toFixed(1)} km/h 均速，能有效擠壓右打者內角與左打者外角好球帶邊緣。
                </p>
              </div>
            </div>

            <div className="space-y-2.5 pt-1">
              <div className="rounded-2xl bg-amber-50 p-3.5 border border-amber-200">
                <div className="text-xs font-extrabold text-amber-900">
                  ⚡ 重點改善 1：降低陀螺角度（目前約 {Math.round((Math.acos(Math.min(0.99, avgEff / 100)) * 180) / Math.PI)}°）以突破 90% 旋轉效率
                </div>
                <p className="mt-1 text-xs text-amber-900/80 leading-relaxed">
                  部分投球出手時食指與中指施力略偏球體側邊，產生約 {100 - avgEff}% 的陀螺旋轉耗損。建議進行近距離四縫線雙指正後方彈指練習，可再提升 5~8 cm 垂直上竄量。
                </p>
              </div>

              <div className="rounded-2xl bg-amber-50 p-3.5 border border-amber-200">
                <div className="text-xs font-extrabold text-amber-900">
                  ⚡ 重點改善 2：縮小連續投球間的旋轉方向離散度
                </div>
                <p className="mt-1 text-xs text-amber-900/80 leading-relaxed">
                  本場次旋轉軸介於 00:18 ~ 01:11 之間，代表放球點高度在不同球間略有浮動。固定前導腳落地後的軀幹傾角有助於穩定放球視窗。
                </p>
              </div>

              <div className="rounded-2xl bg-amber-50 p-3.5 border border-amber-200">
                <div className="text-xs font-extrabold text-amber-900">
                  ⚡ 重點改善 3：建立速差 12~15 km/h 的共軌變化球組合
                </div>
                <p className="mt-1 text-xs text-amber-900/80 leading-relaxed">
                  建議搭配一顆與四縫線相同出手角度（01:00 方向）的變速球或反向橫移滑球，利用共軌效應最大化打者識別延遲。
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 「◇ 比較結果」彈跳視窗 */}
      {showCompareModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4">
          <div className="w-full max-w-lg rounded-3xl bg-white p-5 text-slate-900 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-extrabold text-slate-900">◇ 投球數據並排比較</h3>
              <button
                type="button"
                onClick={() => setShowCompareModal(false)}
                className="rounded-full bg-slate-100 px-3 py-1 text-xs font-bold text-slate-600 hover:bg-slate-200"
              >
                ✕ 關閉
              </button>
            </div>

            {activeSession.pitches.length < 2 ? (
              <p className="text-xs text-slate-500 py-4 text-center">
                此場次至少需要 2 球以上才能進行並排比較。
              </p>
            ) : (
              (() => {
                const idA = comparePitchIds?.[0] ?? activeSession.pitches[0].id;
                const idB = comparePitchIds?.[1] ?? activeSession.pitches[1].id;
                const pA =
                  activeSession.pitches.find((p) => p.id === idA) ?? activeSession.pitches[0];
                const pB =
                  activeSession.pitches.find((p) => p.id === idB) ?? activeSession.pitches[1];

                return (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-3">
                      <select
                        value={pA.id}
                        onChange={(e) => setComparePitchIds([e.target.value, pB.id])}
                        className="rounded-xl bg-slate-100 px-3 py-2 text-xs font-bold text-slate-800 border border-slate-300"
                      >
                        {activeSession.pitches.map((p) => (
                          <option key={p.id} value={p.id}>
                            #{p.pitchNumber} {p.pitchType} ({p.speedKmh.toFixed(1)} km/h)
                          </option>
                        ))}
                      </select>

                      <select
                        value={pB.id}
                        onChange={(e) => setComparePitchIds([pA.id, e.target.value])}
                        className="rounded-xl bg-slate-100 px-3 py-2 text-xs font-bold text-slate-800 border border-slate-300"
                      >
                        {activeSession.pitches.map((p) => (
                          <option key={p.id} value={p.id}>
                            #{p.pitchNumber} {p.pitchType} ({p.speedKmh.toFixed(1)} km/h)
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="overflow-hidden rounded-2xl border border-slate-200 text-xs">
                      <table className="w-full text-center">
                        <thead className="bg-slate-100 font-bold text-slate-600">
                          <tr>
                            <th className="py-2 px-2">#{pA.pitchNumber} {pA.pitchType}</th>
                            <th className="py-2 px-2">指標項目</th>
                            <th className="py-2 px-2">#{pB.pitchNumber} {pB.pitchType}</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-200 font-mono font-bold">
                          <tr>
                            <td className="py-2 text-sm text-[#0b72b9]">{pA.speedKmh.toFixed(1)} km/h</td>
                            <td className="py-2 font-sans text-slate-500">初速 V</td>
                            <td className="py-2 text-sm text-rose-600">{pB.speedKmh.toFixed(1)} km/h</td>
                          </tr>
                          <tr>
                            <td className="py-2">{pA.verticalBreakCm.toFixed(1)} cm</td>
                            <td className="py-2 font-sans text-slate-500">垂直變化量 VB</td>
                            <td className="py-2">{pB.verticalBreakCm.toFixed(1)} cm</td>
                          </tr>
                          <tr>
                            <td className="py-2">{pA.horizontalBreakCm.toFixed(1)} cm</td>
                            <td className="py-2 font-sans text-slate-500">水平變化量 HB</td>
                            <td className="py-2">{pB.horizontalBreakCm.toFixed(1)} cm</td>
                          </tr>
                          <tr>
                            <td className="py-2">{pA.totalSpinRpm} rpm</td>
                            <td className="py-2 font-sans text-slate-500">總轉速 S</td>
                            <td className="py-2">{pB.totalSpinRpm} rpm</td>
                          </tr>
                          <tr>
                            <td className="py-2">{pA.activeSpinRpm} rpm</td>
                            <td className="py-2 font-sans text-slate-500">有效轉速 AS</td>
                            <td className="py-2">{pB.activeSpinRpm} rpm</td>
                          </tr>
                          <tr>
                            <td className="py-2">{pA.efficiencyPct}%</td>
                            <td className="py-2 font-sans text-slate-500">旋轉效率 E</td>
                            <td className="py-2">{pB.efficiencyPct}%</td>
                          </tr>
                          <tr>
                            <td className="py-2">{pA.spinDirection}</td>
                            <td className="py-2 font-sans text-slate-500">旋轉方向 SD</td>
                            <td className="py-2">{pB.spinDirection}</td>
                          </tr>
                          <tr>
                            <td className="py-2">{pA.gyroDegree}°</td>
                            <td className="py-2 font-sans text-slate-500">陀螺角度 GD</td>
                            <td className="py-2">{pB.gyroDegree}°</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                );
              })()
            )}
          </div>
        </div>
      )}
    </div>
  );
};
