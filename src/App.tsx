import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BallDetection,
  BenchmarkDataset,
  WorkerInMessage,
  WorkerOutMessage,
} from './types';
import { useModelCache } from './hooks/useModelCache';
import { SmartScoutLiveCamera } from './components/SmartScoutLiveCamera';
import { SmartScoutPitchPlayer } from './components/SmartScoutPitchPlayer';
import { SmartScoutSessionView } from './components/SmartScoutSessionView';
import { SmartScoutPlayerProfile } from './components/SmartScoutPlayerProfile';
import {
  INITIAL_SMARTSCOUT_SESSIONS,
  SmartScoutPitchRecord,
  SmartScoutSession,
  buildRibbonPathFromMetrics,
  convertBenchmark100ToSession,
  getPitchTypeColor,
  speakPitchResult,
} from './data/smartScoutDefaultSessions';

const SMARTSCOUT_SESSIONS_KEY = 'smartscout_sessions_v2';

export function App() {
  // 預設進入「追蹤」：一打開就直接開啟相機自動測速
  const [navTab, setNavTab] = useState<'track' | 'sessions' | 'video' | 'player' | 'settings'>(
    'track'
  );
  const [lightMode, setLightMode] = useState<boolean>(true);
  const [voiceEnabled, setVoiceEnabled] = useState<boolean>(true);
  const [cameraAngle, setCameraAngle] = useState<'behind_pitcher' | 'behind_catcher'>(
    'behind_pitcher'
  );
  const [discipline, setDiscipline] = useState<'pitching' | 'hitting'>('pitching');
  const [distanceMeters, setDistanceMeters] = useState<number>(18.44);

  const [sessions, setSessions] = useState<SmartScoutSession[]>(() => {
    try {
      const saved = window.localStorage.getItem(SMARTSCOUT_SESSIONS_KEY);
      if (saved) {
        const parsed = JSON.parse(saved) as SmartScoutSession[];
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch {
      // ignore
    }
    return INITIAL_SMARTSCOUT_SESSIONS;
  });

  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(
    INITIAL_SMARTSCOUT_SESSIONS[0].id
  );
  const [activePlayerPitch, setActivePlayerPitch] = useState<{
    sessionId: string;
    pitchId: string;
  }>({
    sessionId: INITIAL_SMARTSCOUT_SESSIONS[0].id,
    pitchId: INITIAL_SMARTSCOUT_SESSIONS[0].pitches[0].id,
  });

  const [workerStatus, setWorkerStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [backend, setBackend] = useState<string>('');
  const [inferTimeMs, setInferTimeMs] = useState<number>(0);
  const [latestBall, setLatestBall] = useState<BallDetection | null>(null);

  const workerRef = useRef<Worker | null>(null);
  const isBusyRef = useRef<boolean>(false);

  const modelUrl = useMemo(
    () => new URL('./models/ball_yolo11n.onnx', window.location.href).href,
    []
  );
  const {
    isModelCached,
    cachedSizeMb,
    isOnline,
    isUpdatingModel,
    forceUpdateModel,
  } = useModelCache(modelUrl, workerStatus);

  // 載入 100 球模型庫至場次列表
  useEffect(() => {
    const url = new URL('./data/benchmark_100.json', window.location.href).href;
    fetch(url)
      .then((r) => (r.ok ? r.json() : null))
      .then((data: BenchmarkDataset | null) => {
        if (!data?.pitches?.length) return;
        const benchSession = convertBenchmark100ToSession(data);
        setSessions((prev) => {
          if (prev.some((s) => s.id === benchSession.id)) return prev;
          return [...prev, benchSession];
        });
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    try {
      const toStore = sessions.filter((s) => s.id !== 'session-benchmark-100');
      window.localStorage.setItem(SMARTSCOUT_SESSIONS_KEY, JSON.stringify(toStore));
    } catch {
      // ignore
    }
  }, [sessions]);

  // 初始化 ONNX Web Worker (320x320)
  useEffect(() => {
    const worker = new Worker(new URL('./workers/inference.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent<WorkerOutMessage>) => {
      const msg = e.data;
      if (msg.type === 'READY') {
        setWorkerStatus('ready');
        setBackend(msg.backend);
      } else if (msg.type === 'RESULT') {
        isBusyRef.current = false;
        setInferTimeMs(msg.inferTimeMs);
        setLatestBall(msg.detection);
      } else if (msg.type === 'ERROR') {
        isBusyRef.current = false;
        setWorkerStatus('error');
      }
    };

    const initMsg: WorkerInMessage = {
      type: 'INIT',
      modelUrl,
      wasmPaths: new URL('./wasm/', window.location.href).href,
      inputSize: 320,
      confThreshold: 0.3,
    };
    worker.postMessage(initMsg);

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, [modelUrl]);

  const handleFrame = useCallback(
    (bitmap: ImageBitmap, timestampMs: number, origWidth: number, origHeight: number) => {
      const w = workerRef.current;
      if (!w) {
        bitmap.close();
        isBusyRef.current = false;
        return;
      }
      w.postMessage(
        { type: 'INFER', bitmap, timestampMs, origWidth, origHeight } satisfies WorkerInMessage,
        [bitmap]
      );
    },
    []
  );

  // 相機自動測得新球時寫入當前場次並語音播報
  const handleNewMeasuredPitch = useCallback(
    (newPitchData: Omit<SmartScoutPitchRecord, 'id' | 'pitchNumber' | 'timestamp'>) => {
      const now = new Date();
      const pad = (n: number) => String(n).padStart(2, '0');
      const dateStr = `${now.getFullYear()}/${pad(now.getMonth() + 1)}/${pad(now.getDate())}`;
      const timeStr = `${dateStr} ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
      const pitchId = `p-${Date.now()}`;

      const targetSessId =
        selectedSessionId && selectedSessionId !== 'session-benchmark-100'
          ? selectedSessionId
          : sessions[0]?.id ?? 'session-2026-08-26-pitch';

      let createdRecord: SmartScoutPitchRecord | null = null;

      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== targetSessId) return s;
          const nextNum = s.pitches.reduce((m, p) => Math.max(m, p.pitchNumber), 0) + 1;
          const record: SmartScoutPitchRecord = {
            ...newPitchData,
            id: pitchId,
            pitchNumber: nextNum,
            timestamp: timeStr,
          };
          createdRecord = record;
          return {
            ...s,
            pitches: [...s.pitches, record],
          };
        })
      );

      setActivePlayerPitch({ sessionId: targetSessId, pitchId });
      window.setTimeout(() => {
        if (createdRecord) {
          speakPitchResult(createdRecord, voiceEnabled);
        }
      }, 50);
    },
    [selectedSessionId, sessions, voiceEnabled]
  );

  const currentVideoSession =
    sessions.find((s) => s.id === activePlayerPitch.sessionId) ?? sessions[0];
  const currentVideoPitchIdx = Math.max(
    0,
    currentVideoSession.pitches.findIndex((p) => p.id === activePlayerPitch.pitchId)
  );
  const currentVideoPitch =
    currentVideoSession.pitches[currentVideoPitchIdx] ?? currentVideoSession.pitches[0];

  const handleDeletePitch = useCallback((sessionId: string, pitchId: string) => {
    setSessions((prev) =>
      prev.map((s) =>
        s.id === sessionId
          ? { ...s, pitches: s.pitches.filter((p) => p.id !== pitchId) }
          : s
      )
    );
  }, []);

  const handleChangePitchType = useCallback(
    (sessionId: string, pitchId: string, newType: string) => {
      setSessions((prev) =>
        prev.map((s) =>
          s.id === sessionId
            ? {
                ...s,
                pitches: s.pitches.map((p) =>
                  p.id === pitchId
                    ? { ...p, pitchType: newType, pitchColor: getPitchTypeColor(newType) }
                    : p
                ),
              }
            : s
        )
      );
    },
    []
  );

  const handleCreateNewSession = useCallback(() => {
    const now = new Date();
    const pad = (n: number) => String(n).padStart(2, '0');
    const dateStr = `${now.getFullYear()}/${pad(now.getMonth() + 1)}/${pad(now.getDate())}`;
    const newId = `session-${Date.now()}`;
    const samplePitch: SmartScoutPitchRecord = {
      id: `p-${Date.now()}`,
      pitchNumber: 1,
      timestamp: `${dateStr} ${pad(now.getHours())}:${pad(now.getMinutes())}:00`,
      pitchType: '四縫線快速球',
      pitchColor: '#be1e3c',
      cameraAngle,
      speedKmh: 108.0,
      verticalBreakCm: 53.5,
      horizontalBreakCm: 26.2,
      efficiencyPct: 83,
      spinDirection: '00:50',
      gyroDegree: 34,
      activeSpinRpm: 1740,
      totalSpinRpm: 2096,
      plateXCm: 5.0,
      plateYCm: 79.0,
      ribbonPath: buildRibbonPathFromMetrics(cameraAngle, 53.5, 26.2, 5.0, 79.0),
    };
    const newSess: SmartScoutSession = {
      id: newId,
      title: `${dateStr} 投球`,
      date: dateStr,
      monthGroup: `${now.getFullYear()}年${now.getMonth() + 1}月`,
      discipline: 'pitching',
      cameraSetup: cameraAngle,
      pitches: [samplePitch],
    };
    setSessions((prev) => [newSess, ...prev]);
    setSelectedSessionId(newId);
    setNavTab('sessions');
  }, [cameraAngle]);

  return (
    <div
      className={`min-h-screen pb-18 transition-colors ${
        lightMode ? 'bg-[#f4f6f9] text-slate-900' : 'bg-slate-950 text-slate-100'
      }`}
    >
      <main className="mx-auto max-w-2xl px-0 sm:px-4 sm:pt-3">
        {/* 1. 追蹤：直接開啟相機自動測速 + 紅色彩帶 + 底部數據卡 */}
        {navTab === 'track' && currentVideoPitch && (
          <SmartScoutLiveCamera
            currentPitch={currentVideoPitch}
            cameraAngle={cameraAngle}
            discipline={discipline}
            voiceEnabled={voiceEnabled}
            isWorkerReady={workerStatus === 'ready'}
            isBusyRef={isBusyRef}
            latestBall={latestBall}
            onChangeCameraAngle={setCameraAngle}
            onChangeDiscipline={setDiscipline}
            onToggleVoice={() => setVoiceEnabled((v) => !v)}
            onFrame={handleFrame}
            onNewMeasuredPitch={handleNewMeasuredPitch}
            onChangePitchType={(pitchId, newType) =>
              handleChangePitchType(currentVideoSession.id, pitchId, newType)
            }
          />
        )}

        {/* 2. 場次：淺色模式場次列表 + 列表 / 圖表 / AI 教練 */}
        {navTab === 'sessions' && (
          <div className="px-3 pt-3 sm:px-0">
            <SmartScoutSessionView
              sessions={sessions}
              selectedSessionId={selectedSessionId}
              onSelectSession={setSelectedSessionId}
              onOpenPitchPlayer={(sessionId, pitchId) => {
                setActivePlayerPitch({ sessionId, pitchId });
                setNavTab('video');
              }}
              onDeletePitch={handleDeletePitch}
              onChangePitchType={handleChangePitchType}
              onStartNewRecording={(sessionId) => {
                if (sessionId) setSelectedSessionId(sessionId);
                setNavTab('track');
              }}
              onCreateNewSession={handleCreateNewSession}
            />
          </div>
        )}

        {/* 3. 影片：投球 #1 紅色彩帶回放播放器 */}
        {navTab === 'video' && currentVideoPitch && (
          <SmartScoutPitchPlayer
            pitch={currentVideoPitch}
            totalInSession={currentVideoSession.pitches.length}
            onBack={() => {
              setSelectedSessionId(currentVideoSession.id);
              setNavTab('sessions');
            }}
            onDelete={(pitchId) =>
              handleDeletePitch(currentVideoSession.id, pitchId)
            }
            onChangePitchType={(pitchId, newType) =>
              handleChangePitchType(currentVideoSession.id, pitchId, newType)
            }
            onPrevPitch={
              currentVideoPitchIdx > 0
                ? () =>
                    setActivePlayerPitch({
                      sessionId: currentVideoSession.id,
                      pitchId: currentVideoSession.pitches[currentVideoPitchIdx - 1].id,
                    })
                : undefined
            }
            onNextPitch={
              currentVideoPitchIdx < currentVideoSession.pitches.length - 1
                ? () =>
                    setActivePlayerPitch({
                      sessionId: currentVideoSession.id,
                      pitchId: currentVideoSession.pitches[currentVideoPitchIdx + 1].id,
                    })
                : undefined
            }
          />
        )}

        {/* 4. 球員：個人檔案與熱力日曆 */}
        {navTab === 'player' && (
          <div className="px-3 pt-3 sm:px-0">
            <SmartScoutPlayerProfile sessions={sessions} />
          </div>
        )}

        {/* 5. 設定：極簡淺色設定列表 */}
        {navTab === 'settings' && (
          <div className="mx-auto max-w-md px-3 pt-3 sm:px-0 space-y-3">
            <div className="rounded-3xl bg-white p-5 text-slate-900 shadow-sm border border-slate-200 space-y-4">
              <h2 className="text-lg font-extrabold text-slate-900">設定</h2>

              {/* 淺色 / 深色模式切換 */}
              <div className="flex items-center justify-between py-2 border-b border-slate-100">
                <span className="text-sm font-bold text-slate-800">淺色模式</span>
                <button
                  type="button"
                  onClick={() => setLightMode((v) => !v)}
                  className={`relative h-7 w-13 rounded-full transition ${
                    lightMode ? 'bg-[#0b72b9]' : 'bg-slate-400'
                  }`}
                >
                  <span
                    className={`absolute top-0.5 h-6 w-6 rounded-full bg-white shadow transition ${
                      lightMode ? 'left-6.5' : 'left-0.5'
                    }`}
                  />
                </button>
              </div>

              {/* 鏡頭位置 */}
              <div className="flex items-center justify-between py-2 border-b border-slate-100">
                <span className="text-sm font-bold text-slate-800">拍攝方向</span>
                <div className="flex rounded-xl bg-slate-100 p-1 text-xs font-bold">
                  <button
                    type="button"
                    onClick={() => setCameraAngle('behind_pitcher')}
                    className={`rounded-lg px-3 py-1.5 transition ${
                      cameraAngle === 'behind_pitcher'
                        ? 'bg-[#0b72b9] text-white'
                        : 'text-slate-600'
                    }`}
                  >
                    投手正後方
                  </button>
                  <button
                    type="button"
                    onClick={() => setCameraAngle('behind_catcher')}
                    className={`rounded-lg px-3 py-1.5 transition ${
                      cameraAngle === 'behind_catcher'
                        ? 'bg-[#0b72b9] text-white'
                        : 'text-slate-600'
                    }`}
                  >
                    投手正前方
                  </button>
                </div>
              </div>

              {/* 語音朗讀結果 */}
              <div className="flex items-center justify-between py-2 border-b border-slate-100">
                <span className="text-sm font-bold text-slate-800">語音朗讀結果</span>
                <button
                  type="button"
                  onClick={() => setVoiceEnabled((v) => !v)}
                  className={`relative h-7 w-13 rounded-full transition ${
                    voiceEnabled ? 'bg-[#0b72b9]' : 'bg-slate-400'
                  }`}
                >
                  <span
                    className={`absolute top-0.5 h-6 w-6 rounded-full bg-white shadow transition ${
                      voiceEnabled ? 'left-6.5' : 'left-0.5'
                    }`}
                  />
                </button>
              </div>

              {/* 投球距離 */}
              <div className="flex items-center justify-between py-2 border-b border-slate-100">
                <span className="text-sm font-bold text-slate-800">投球距離</span>
                <div className="flex rounded-xl bg-slate-100 p-1 text-xs font-bold">
                  {[18.44, 16.46, 14.02].map((d) => (
                    <button
                      key={d}
                      type="button"
                      onClick={() => setDistanceMeters(d)}
                      className={`rounded-lg px-2.5 py-1.5 transition ${
                        Math.abs(distanceMeters - d) < 0.05
                          ? 'bg-[#0b72b9] text-white'
                          : 'text-slate-600'
                      }`}
                    >
                      {d}m
                    </button>
                  ))}
                </div>
              </div>

              {/* AI 引擎與離線狀態 */}
              <div className="flex items-center justify-between py-2 text-xs">
                <div className="flex items-center gap-2">
                  <span
                    className={`h-2.5 w-2.5 rounded-full ${
                      workerStatus === 'ready' ? 'bg-emerald-500' : 'bg-amber-400'
                    }`}
                  />
                  <span className="font-bold text-slate-700">
                    AI 引擎 ({backend || 'ONNX'})
                    {inferTimeMs > 0 ? ` · ${inferTimeMs}ms` : ''}
                  </span>
                  {isModelCached && (
                    <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-[11px] font-bold text-emerald-700">
                      離線可用{cachedSizeMb ? ` (${cachedSizeMb}MB)` : ''}
                    </span>
                  )}
                </div>

                <button
                  type="button"
                  disabled={!isOnline || isUpdatingModel}
                  onClick={forceUpdateModel}
                  className="rounded-xl bg-slate-100 px-3 py-1.5 font-bold text-[#0b72b9] hover:bg-slate-200 disabled:opacity-40"
                >
                  {isUpdatingModel ? '更新中' : '更新模型'}
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* SmartScout 底部 5 大導覽列 (淺色白底藍字) */}
      <nav
        className={`fixed bottom-0 inset-x-0 z-40 border-t backdrop-blur-md shadow-[0_-4px_16px_rgba(0,0,0,0.06)] ${
          lightMode
            ? 'border-slate-200 bg-white/95 text-slate-500'
            : 'border-slate-800 bg-slate-900/95 text-slate-400'
        }`}
      >
        <div className="mx-auto grid max-w-md grid-cols-5 py-1.5 text-center text-[11px] font-bold">
          <button
            type="button"
            onClick={() => setNavTab('track')}
            className={`flex flex-col items-center justify-center gap-0.5 py-1 transition ${
              navTab === 'track' ? 'text-[#0b72b9]' : 'hover:text-slate-800'
            }`}
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M23 7l-7 5 7 5V7z" />
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
            </svg>
            <span>追蹤</span>
          </button>

          <button
            type="button"
            onClick={() => setNavTab('sessions')}
            className={`flex flex-col items-center justify-center gap-0.5 py-1 transition ${
              navTab === 'sessions' ? 'text-[#0b72b9]' : 'hover:text-slate-800'
            }`}
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="8" y1="6" x2="21" y2="6" />
              <line x1="8" y1="12" x2="21" y2="12" />
              <line x1="8" y1="18" x2="21" y2="18" />
              <line x1="3" y1="6" x2="3.01" y2="6" />
              <line x1="3" y1="12" x2="3.01" y2="12" />
              <line x1="3" y1="18" x2="3.01" y2="18" />
            </svg>
            <span>場次</span>
          </button>

          <button
            type="button"
            onClick={() => setNavTab('video')}
            className={`flex flex-col items-center justify-center gap-0.5 py-1 transition ${
              navTab === 'video' ? 'text-[#0b72b9]' : 'hover:text-slate-800'
            }`}
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <polygon points="10 8 16 12 10 16 10 8" />
            </svg>
            <span>影片</span>
          </button>

          <button
            type="button"
            onClick={() => setNavTab('player')}
            className={`flex flex-col items-center justify-center gap-0.5 py-1 transition ${
              navTab === 'player' ? 'text-[#0b72b9]' : 'hover:text-slate-800'
            }`}
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
            <span>球員</span>
          </button>

          <button
            type="button"
            onClick={() => setNavTab('settings')}
            className={`flex flex-col items-center justify-center gap-0.5 py-1 transition ${
              navTab === 'settings' ? 'text-[#0b72b9]' : 'hover:text-slate-800'
            }`}
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
            <span>設定</span>
          </button>
        </div>
      </nav>
    </div>
  );
}

export default App;
