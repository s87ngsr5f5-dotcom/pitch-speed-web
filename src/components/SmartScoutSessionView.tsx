import React, { useMemo, useState } from 'react';
import {
  SMARTSCOUT_PITCH_TYPES,
  SmartScoutPitchRecord,
  SmartScoutSession,
  buildTaperedRibbonPolygon,
  getPitchTypeColor,
  sampleQuadraticRibbon,
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

function averageMetric(
  pitches: SmartScoutPitchRecord[],
  key: 'speedKmh' | 'verticalBreakCm' | 'horizontalBreakCm' | 'totalSpinRpm' | 'efficiencyPct',
) {
  const values = pitches.flatMap((pitch) => pitch[key] ?? []);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : undefined;
}

const MiniRibbonAndZoneThumb: React.FC<{ pitch: SmartScoutPitchRecord }> = ({ pitch }) => {
  const samples = useMemo(
    () => sampleQuadraticRibbon(pitch.ribbonPath, 64, 64, 1, 16),
    [pitch.ribbonPath]
  );
  const poly = useMemo(() => buildTaperedRibbonPolygon(samples), [samples]);

  return (
    <div className="flex items-center gap-2 shrink-0">
      {/* 3D 彩帶縮圖 */}
      <div className="flex flex-col items-center">
        <svg
          width="46"
          height="46"
          viewBox="0 0 64 64"
          className="rounded-xl bg-slate-900 border border-slate-300"
        >
          <polygon points={poly} fill="#ef233c" fillOpacity="0.9" />
          {samples.length > 0 && (
            <circle
              cx={samples[samples.length - 1].x}
              cy={samples[samples.length - 1].y}
              r="3.5"
              fill="#ffffff"
              stroke="#dc2626"
              strokeWidth="1"
            />
          )}
        </svg>
        <span className="mt-0.5 text-[9px] text-slate-400">{pitch.measurement ? '球路' : '3D 軌跡'}</span>
      </div>

      {pitch.plateXCm !== undefined && pitch.plateYCm !== undefined && (
      <div className="flex flex-col items-center">
        <svg
          width="42"
          height="46"
          viewBox="0 0 36 40"
          className="rounded-xl bg-slate-50 border border-slate-200"
        >
          <rect x="6" y="5" width="24" height="28" fill="#ffffff" stroke="#475569" strokeWidth="1.4" />
          <line x1="14" y1="5" x2="14" y2="33" stroke="#cbd5e1" strokeWidth="1" />
          <line x1="22" y1="5" x2="22" y2="33" stroke="#cbd5e1" strokeWidth="1" />
          <line x1="6" y1="14.3" x2="30" y2="14.3" stroke="#cbd5e1" strokeWidth="1" />
          <line x1="6" y1="23.6" x2="30" y2="23.6" stroke="#cbd5e1" strokeWidth="1" />
          <circle
            cx={Math.max(3, Math.min(33, 18 + (pitch.plateXCm / 21.6) * 12))}
            cy={Math.max(3, Math.min(36, 19 - ((pitch.plateYCm - 76) / 30) * 14))}
            r="3.2"
            fill="#ef233c"
          />
        </svg>
        <span className="mt-0.5 text-[9px] text-slate-400">#{pitch.zoneId}</span>
      </div>
      )}
    </div>
  );
};

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
  const [activeSubTab, setActiveSubTab] = useState<'list' | 'chart' | 'summary'>('list');
  const [chartType, setChartType] = useState<'movement' | 'strike_zone'>('movement');
  const [selectedChartPitchId, setSelectedChartPitchId] = useState<string | null>(null);
  const [showCompareModal, setShowCompareModal] = useState<boolean>(false);
  const [comparePitchIds, setComparePitchIds] = useState<[string, string] | null>(null);

  const activeSession = useMemo(
    () => sessions.find((s) => s.id === selectedSessionId) ?? null,
    [sessions, selectedSessionId]
  );

  const pitchTypeSummaries = useMemo(() => {
    if (!activeSession) return [];
    const groups = new Map<string, SmartScoutPitchRecord[]>();
    for (const pitch of activeSession.pitches) {
      const group = groups.get(pitch.pitchType) ?? [];
      group.push(pitch);
      groups.set(pitch.pitchType, group);
    }

    return Array.from(groups, ([name, pitches]) => ({
      name,
      color: getPitchTypeColor(name),
      avgV: averageMetric(pitches, 'speedKmh')?.toFixed(1),
      maxV: Math.max(...pitches.map((p) => p.speedKmh)).toFixed(1),
      avgVB: averageMetric(pitches, 'verticalBreakCm')?.toFixed(1),
      avgHB: averageMetric(pitches, 'horizontalBreakCm')?.toFixed(1),
      avgSpin: averageMetric(pitches, 'totalSpinRpm')?.toFixed(0),
      avgEff: averageMetric(pitches, 'efficiencyPct')?.toFixed(0),
    }));
  }, [activeSession]);

  // 1. 場次總覽列表
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
        <div className="flex items-center gap-2">
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

          <div className="flex rounded-2xl bg-white p-1 border border-slate-200 shadow-sm text-xs font-bold">
            {(
              [
                { id: 'all', label: '全部' },
                { id: 'pitching', label: '投球' },
                { id: 'hitting', label: '打擊' },
              ] as const
            ).map((f) => (
              <button
                key={f.id}
                type="button"
                onClick={() => setDisciplineFilter(f.id)}
                className={`rounded-xl px-2.5 py-2 transition ${
                  disciplineFilter === f.id
                    ? 'bg-[#0b72b9] text-white'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {filtered.length === 0 && (
          <div className="rounded-3xl bg-white p-8 text-center text-slate-900 shadow-sm border border-slate-200 space-y-4">
            <div className="text-lg font-extrabold text-slate-900">尚無測速場次紀錄</div>
            <button
              type="button"
              onClick={() => onStartNewRecording()}
              className="rounded-2xl bg-[#0b72b9] px-6 py-3 text-sm font-extrabold text-white shadow hover:bg-[#0962a0] transition"
            >
              開啟相機開始測速
            </button>
          </div>
        )}

        {Array.from(groupedByMonth.entries()).map(([month, monthSessions]) => (
          <div key={month} className="space-y-3">
            <h3 className="px-1 text-sm font-bold text-slate-500">{month}</h3>

            {monthSessions.map((sess) => {
              const total = sess.pitches.length;
              const avgV =
                total > 0
                  ? (sess.pitches.reduce((a, p) => a + p.speedKmh, 0) / total).toFixed(1)
                  : '0.0';
              const avgVB = averageMetric(sess.pitches, 'verticalBreakCm')?.toFixed(1) ?? '—';
              const avgHB = averageMetric(sess.pitches, 'horizontalBreakCm')?.toFixed(1) ?? '—';
              const mainType = sess.pitches[0]?.pitchType ?? '四縫線快速球';
              const mainTypeCount = sess.pitches.filter((p) => p.pitchType === mainType).length;

              return (
                <div
                  key={sess.id}
                  onClick={() => {
                    onSelectSession(sess.id);
                    setActiveSubTab('list');
                  }}
                  className="cursor-pointer rounded-3xl bg-white p-5 text-slate-900 shadow-sm border border-slate-200 hover:shadow-md hover:border-sky-300 transition space-y-3"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="text-lg font-extrabold text-slate-900">{sess.title}</h4>
                      <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
                        <span>📅 {sess.date}</span>
                        <span>·</span>
                        <span>
                          {sess.cameraSetup === 'behind_pitcher'
                            ? '投手正後方'
                            : '投手正前方'}
                        </span>
                      </div>
                    </div>
                    <span className="text-slate-400 text-lg">›</span>
                  </div>

                  {total > 0 && (
                    <>
                      <div className="border-t border-slate-100 pt-3">
                        <span className="inline-flex items-center gap-1.5 rounded-xl bg-rose-100 px-3 py-1 text-xs font-bold text-rose-800">
                          <span>⚾</span>
                          <span>
                            {mainType} {mainTypeCount}/{total}球
                          </span>
                        </span>
                      </div>

                      <div className="grid grid-cols-3 gap-2 pt-1">
                        <div>
                          <div className="text-xs text-slate-500">平均球速</div>
                          <div className="mt-0.5 flex items-baseline gap-1">
                            <span className="text-2xl font-extrabold tabular-nums text-slate-900">
                              {avgV}
                            </span>
                            <span className="text-xs text-slate-500">km/h</span>
                          </div>
                        </div>
                        <div>
                          <div className="text-xs text-slate-500">平均縱向位移</div>
                          <div className="mt-0.5 flex items-baseline gap-1">
                            <span className="text-2xl font-extrabold tabular-nums text-slate-900">
                              {avgVB}
                            </span>
                            <span className="text-xs text-slate-500">cm</span>
                          </div>
                        </div>
                        <div>
                          <div className="text-xs text-slate-500">平均橫向位移</div>
                          <div className="mt-0.5 flex items-baseline gap-1">
                            <span className="text-2xl font-extrabold tabular-nums text-slate-900">
                              {avgHB}
                            </span>
                            <span className="text-xs text-slate-500">cm</span>
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

        <div className="fixed bottom-18 inset-x-0 z-30 mx-auto flex max-w-md items-center gap-2.5 px-4">
          <button
            type="button"
            onClick={() => onStartNewRecording()}
            className="flex-1 flex items-center justify-center gap-2 rounded-2xl bg-[#0b72b9] py-3.5 text-sm font-bold text-white shadow-xl hover:bg-[#0962a0] transition"
          >
            <span>📹</span>
            <span>開啟相機測速</span>
          </button>

          <button
            type="button"
            onClick={onCreateNewSession}
            title="新增場次"
            className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#0b72b9] text-white shadow-xl hover:bg-[#0962a0] transition text-xl font-bold"
          >
            ＋
          </button>
        </div>
      </div>
    );
  }

  // 2. 單一場次詳情：列表 | 圖表 | 球種摘要
  const pitchesReversed = [...activeSession.pitches].sort(
    (a, b) => b.pitchNumber - a.pitchNumber
  );

  const movementPitches = activeSession.pitches.filter(
    (p): p is SmartScoutPitchRecord & { horizontalBreakCm: number; verticalBreakCm: number } =>
      p.horizontalBreakCm !== undefined && p.verticalBreakCm !== undefined,
  );
  const locationPitches = activeSession.pitches.filter(
    (p): p is SmartScoutPitchRecord & { plateXCm: number; plateYCm: number } =>
      p.plateXCm !== undefined && p.plateYCm !== undefined,
  );
  const chartPitches = chartType === 'movement' ? movementPitches : locationPitches;
  const selectedChartPitch =
    chartPitches.find((p) => p.id === selectedChartPitchId) ?? chartPitches[0];

  return (
    <div className="mx-auto max-w-2xl pb-24 space-y-4">
      <div className="rounded-3xl bg-white text-slate-900 shadow-sm border border-slate-200 overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3.5">
          <button
            type="button"
            onClick={() => onSelectSession(null)}
            className="flex h-9 w-9 items-center justify-center rounded-full hover:bg-slate-100 transition text-xl font-bold text-[#0b72b9]"
          >
            ‹
          </button>

          <h2 className="text-base font-extrabold text-[#0b72b9]">
            {activeSession.title}
          </h2>

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
            className="rounded-xl bg-sky-50 px-3 py-1.5 text-xs font-extrabold text-[#0b72b9] hover:bg-sky-100 border border-sky-200"
          >
            ◇ 比較結果
          </button>
        </div>

        <div className="grid grid-cols-3 border-t border-slate-200 text-center text-sm font-bold">
          {(
            [
              { id: 'list', label: '列表' },
              { id: 'chart', label: '圖表' },
              { id: 'summary', label: '球種摘要' },
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

      {/* 子分頁 1：列表 (左側球號/時間/球種/3D彩帶縮圖/九宮格落點 + 右側 2x4 八格數據) */}
      {activeSubTab === 'list' && (
        <div className="space-y-3.5">
          {pitchesReversed.map((p) => (
            <div
              key={p.id}
              onClick={() => onOpenPitchPlayer(activeSession.id, p.id)}
              className="cursor-pointer rounded-3xl bg-white p-4 text-slate-900 shadow-sm border border-slate-200 hover:border-sky-400 hover:shadow-md transition space-y-3"
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-sky-100 text-xs">
                    ⚾
                  </span>
                  <div>
                    <div className="text-base font-extrabold text-slate-900">
                      {p.pitchNumber}. 投球
                    </div>
                    <div className="text-[11px] font-mono text-slate-400">{p.timestamp}</div>
                  </div>
                </div>

                <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
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

                  <button
                    type="button"
                    onClick={() => onDeletePitch(activeSession.id, p.id)}
                    className="rounded-lg p-1 text-slate-400 hover:bg-rose-50 hover:text-rose-600 transition"
                  >
                    🗑
                  </button>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center gap-3">
                {/* 左側：3D 彩帶縮圖 + 九宮格落點縮圖 */}
                <MiniRibbonAndZoneThumb pitch={p} />

                {p.measurement ? (
                  <div className="flex-1 text-sm">
                    <div className="font-bold">預估初速 {p.speedKmh.toFixed(1)} km/h</div>
                    <div className="mt-1 text-xs text-slate-500">
                      飛行 {Math.round(p.measurement.flightTimeMs)} ms · 偵測 {p.measurement.detectionCount} 幀
                    </div>
                  </div>
                ) : (
                <div className="flex-1 grid grid-cols-4 gap-1.5 text-center">
                  <div className="rounded-xl bg-slate-100/90 py-1.5 px-1 border border-slate-200/80">
                    <div className="text-[10px] text-slate-500">{p.measurement ? '預估球速' : 'V [km/h]'}</div>
                    <div className="mt-0.5 text-sm font-extrabold tabular-nums text-slate-900">
                      {p.speedKmh.toFixed(1)}
                    </div>
                  </div>
                  <div className="rounded-xl bg-slate-100/90 py-1.5 px-1 border border-slate-200/80">
                    <div className="text-[10px] text-slate-500">VB [cm]</div>
                    <div className="mt-0.5 text-sm font-extrabold tabular-nums text-slate-900">
                      {p.verticalBreakCm?.toFixed(1) ?? '—'}
                    </div>
                  </div>
                  <div className="rounded-xl bg-slate-100/90 py-1.5 px-1 border border-slate-200/80">
                    <div className="text-[10px] text-slate-500">HB [cm]</div>
                    <div className="mt-0.5 text-sm font-extrabold tabular-nums text-slate-900">
                      {p.horizontalBreakCm?.toFixed(1) ?? '—'}
                    </div>
                  </div>
                  <div className="rounded-xl bg-slate-100/90 py-1.5 px-1 border border-slate-200/80">
                    <div className="text-[10px] text-slate-500">E [%]</div>
                    <div className="mt-0.5 text-sm font-extrabold tabular-nums text-slate-900">
                      {p.efficiencyPct ?? '—'}
                    </div>
                  </div>
                  <div className="rounded-xl bg-slate-100/90 py-1.5 px-1 border border-slate-200/80">
                    <div className="text-[10px] text-slate-500">SD</div>
                    <div className="mt-0.5 text-sm font-extrabold tabular-nums text-slate-900">
                      {p.spinDirection ?? '—'}
                    </div>
                  </div>
                  <div className="rounded-xl bg-slate-100/90 py-1.5 px-1 border border-slate-200/80">
                    <div className="text-[10px] text-slate-500">GD [°]</div>
                    <div className="mt-0.5 text-sm font-extrabold tabular-nums text-slate-900">
                      {p.gyroDegree ?? '—'}
                    </div>
                  </div>
                  <div className="rounded-xl bg-slate-100/90 py-1.5 px-1 border border-slate-200/80">
                    <div className="text-[10px] text-slate-500">AS [rpm]</div>
                    <div className="mt-0.5 text-sm font-extrabold tabular-nums text-slate-900">
                      {p.activeSpinRpm ?? '—'}
                    </div>
                  </div>
                  <div className="rounded-xl bg-slate-100/90 py-1.5 px-1 border border-slate-200/80">
                    <div className="text-[10px] text-slate-500">S [rpm]</div>
                    <div className="mt-0.5 text-sm font-extrabold tabular-nums text-slate-900">
                      {p.totalSpinRpm ?? '—'}
                    </div>
                  </div>
                </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* 子分頁 2：圖表 (投球變化量圖 -60~+60 cm & 九宮格好球帶分布圖) */}
      {activeSubTab === 'chart' && (
        <div className="rounded-3xl bg-white p-5 text-slate-900 shadow-sm border border-slate-200 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-lg font-extrabold text-slate-900">
              {chartType === 'movement' ? '投球變化量圖' : '九宮格好球帶分布圖'}
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
                九宮格好球帶分布圖
              </button>
            </div>
          </div>

          {chartType === 'movement' ? (
            <div className="mx-auto max-w-[360px]">
              <svg viewBox="0 0 340 340" className="w-full h-auto select-none">
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

                <rect
                  x="55"
                  y="20"
                  width="260"
                  height="260"
                  fill="#f8fafc"
                  stroke="#94a3b8"
                  strokeWidth="1.5"
                />

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
                      <text x="47" y={py + 4} fill="#64748b" fontSize="11" textAnchor="end">
                        {val}
                      </text>
                      <text x={px} y="298" fill="#64748b" fontSize="11" textAnchor="middle">
                        {val}
                      </text>
                    </g>
                  );
                })}

                {movementPitches.map((p) => {
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
                        fillOpacity="0.85"
                        stroke={color}
                        strokeWidth="1.5"
                      />
                    </g>
                  );
                })}
              </svg>
            </div>
          ) : (
            <div className="mx-auto max-w-[300px]">
              <svg viewBox="0 0 260 300" className="w-full h-auto select-none">
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
                <rect
                  x="65"
                  y="55"
                  width="130"
                  height="150"
                  fill="#ffffff"
                  stroke="#334155"
                  strokeWidth="2.2"
                />
                <line x1="108.3" y1="55" x2="108.3" y2="205" stroke="#94a3b8" strokeWidth="1.2" />
                <line x1="151.6" y1="55" x2="151.6" y2="205" stroke="#94a3b8" strokeWidth="1.2" />
                <line x1="65" y1="105" x2="195" y2="105" stroke="#94a3b8" strokeWidth="1.2" />
                <line x1="65" y1="155" x2="195" y2="155" stroke="#94a3b8" strokeWidth="1.2" />
                <polygon
                  points="65,255 195,255 195,270 130,290 65,270"
                  fill="#e2e8f0"
                  stroke="#475569"
                  strokeWidth="2"
                />

                {locationPitches.map((p) => {
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
            👆 點按圓點查看完整數據預覽，再次點按前往影片分頁播放球路
          </p>

          {chartPitches.length === 0 && (
            <p className="text-sm text-slate-500">這場沒有可繪製的三維位移或進壘座標。</p>
          )}

          {selectedChartPitch && (
            <div
              onClick={() => onOpenPitchPlayer(activeSession.id, selectedChartPitch.id)}
              className="rounded-2xl bg-sky-50/90 p-3.5 text-xs border border-sky-200 cursor-pointer hover:bg-sky-100 transition space-y-2"
            >
              <div className="flex items-center justify-between">
                <span className="font-extrabold text-slate-900">
                  #{selectedChartPitch.pitchNumber} {selectedChartPitch.pitchType}{selectedChartPitch.zoneId ? `（進壘 #${selectedChartPitch.zoneId}）` : ''}
                </span>
                <span className="font-bold text-[#0b72b9]">再點一下播放球路影片 ›</span>
              </div>
              <div className="grid grid-cols-4 gap-1.5 text-center font-mono">
                <div className="rounded-lg bg-white py-1 border border-slate-200">
                  <div className="text-[10px] text-slate-400">V [km/h]</div>
                  <div className="font-bold text-slate-900">{selectedChartPitch.speedKmh.toFixed(1)}</div>
                </div>
                <div className="rounded-lg bg-white py-1 border border-slate-200">
                  <div className="text-[10px] text-slate-400">VB [cm]</div>
                  <div className="font-bold text-slate-900">{selectedChartPitch.verticalBreakCm?.toFixed(1) ?? '—'}</div>
                </div>
                <div className="rounded-lg bg-white py-1 border border-slate-200">
                  <div className="text-[10px] text-slate-400">HB [cm]</div>
                  <div className="font-bold text-slate-900">{selectedChartPitch.horizontalBreakCm?.toFixed(1) ?? '—'}</div>
                </div>
                <div className="rounded-lg bg-white py-1 border border-slate-200">
                  <div className="text-[10px] text-slate-400">S [rpm]</div>
                  <div className="font-bold text-slate-900">{selectedChartPitch.totalSpinRpm ?? '—'}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 子分頁 3：球種摘要 */}
      {activeSubTab === 'summary' && (
        <div className="space-y-4">
          <div className="rounded-3xl bg-white p-5 text-slate-900 shadow-sm border border-slate-200 space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-base font-extrabold text-slate-900">球種特徵</h4>
            </div>

            <div className="overflow-x-auto rounded-2xl border border-slate-200">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-100 text-slate-600">
                  <tr>
                    <th className="py-2.5 px-3">球種</th>
                    <th className="py-2.5 px-2 text-right">平均/最大球速</th>
                    <th className="py-2.5 px-2 text-right">平均轉速 / 效率</th>
                    <th className="py-2.5 px-2 text-right">垂直 VB</th>
                    <th className="py-2.5 px-3 text-right">水平 HB</th>
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
                        <span>{s.name}</span>
                      </td>
                      <td className="py-3 px-2 text-right font-mono">
                        {s.avgV} / <strong>{s.maxV}</strong> km/h
                      </td>
                      <td className="py-3 px-2 text-right font-mono">
                        {s.avgSpin ? `${s.avgSpin} rpm` : '—'}{s.avgEff ? ` (${s.avgEff}%)` : ''}
                      </td>
                      <td className="py-3 px-2 text-right font-mono">{s.avgVB ? `${s.avgVB} cm` : '—'}</td>
                      <td className="py-3 px-3 text-right font-mono">{s.avgHB ? `${s.avgHB} cm` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      )}

      {/* 「◇ 比較結果」並排比對視窗（含差值 Δ） */}
      {showCompareModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4">
          <div className="w-full max-w-lg rounded-3xl bg-white p-5 text-slate-900 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-extrabold text-slate-900">◇ 兩球數據並排比對</h3>
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

                const fmtDelta = (val: number, unit = '') =>
                  `${val >= 0 ? '+' : ''}${val.toFixed(1)}${unit}`;

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
                            <th className="py-2 px-2">差值 Δ</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-200 font-mono font-bold">
                          <tr>
                            <td className="py-2 text-[#0b72b9]">{pA.speedKmh.toFixed(1)}</td>
                            <td className="py-2 font-sans text-slate-500">預估初速 [km/h]</td>
                            <td className="py-2 text-rose-600">{pB.speedKmh.toFixed(1)}</td>
                            <td className="py-2 text-slate-700">
                              {fmtDelta(pA.speedKmh - pB.speedKmh)}
                            </td>
                          </tr>
                          {([
                            ['垂直 VB [cm]', 'verticalBreakCm'],
                            ['水平 HB [cm]', 'horizontalBreakCm'],
                            ['轉速 S [rpm]', 'totalSpinRpm'],
                            ['有效轉速 AS', 'activeSpinRpm'],
                            ['效率 E [%]', 'efficiencyPct'],
                          ] as const).map(([name, key]) => {
                            const a = pA[key];
                            const b = pB[key];
                            if (a === undefined && b === undefined) return null;
                            return (
                              <tr key={key}>
                                <td className="py-2">{a ?? '—'}</td>
                                <td className="py-2 font-sans text-slate-500">{name}</td>
                                <td className="py-2">{b ?? '—'}</td>
                                <td className="py-2 text-slate-700">
                                  {a !== undefined && b !== undefined ? fmtDelta(a - b) : '—'}
                                </td>
                              </tr>
                            );
                          })}
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
