import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BallDetection,
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
  getPitchTypeColor,
  speakPitchResult,
} from './data/smartScoutDefaultSessions';

const SMARTSCOUT_REAL_SESSIONS_KEY = 'smartscout_real_sessions_v1';

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
  const [discipline] = useState<'pitching' | 'hitting'>('pitching');
  const [distanceMeters, setDistanceMeters] = useState<number>(18.44);

  const [sessions, setSessions] = useState<SmartScoutSession[]>(() => {
    try {
      // 清除舊版預載範例快取
      window.localStorage.removeItem('smartscout_sessions_v1');
      window.localStorage.removeItem('smartscout_sessions_v2');
      const saved = window.localStorage.getItem(SMARTSCOUT_REAL_SESSIONS_KEY);
      if (saved) {
        const parsed = JSON.parse(saved) as SmartScoutSession[];
        if (Array.isArray(parsed)) return parsed;
      }
    } catch {
      // ignore
    }
    return INITIAL_SMARTSCOUT_SESSIONS;
  });

  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [activePlayerPitch, setActivePlayerPitch] = useState<{
    sessionId: string;
    pitchId: string;
  } | null>(null);

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

  useEffect(() => {
    try {
      window.localStorage.setItem(SMARTSCOUT_REAL_SESSIONS_KEY, JSON.stringify(sessions));
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
      confThreshold: 0.22,
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

  // 相機或影片真實測得新球時自動寫入今日場次並語音播報
  const handleNewMeasuredPitch = useCallback(
    (newPitchData: Omit<SmartScoutPitchRecord, 'id' | 'pitchNumber' | 'timestamp'>) => {
      const now = new Date();
      const pad = (n: number) => String(n).padStart(2, '0');
      const dateStr = `${now.getFullYear()}/${pad(now.getMonth() + 1)}/${pad(now.getDate())}`;
      const timeStr = `${dateStr} ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
      const pitchId = `p-${Date.now()}`;

      let targetSessId = selectedSessionId ?? sessions[0]?.id ?? '';
      let createdRecord: SmartScoutPitchRecord | null = null;

      setSessions((prev) => {
        if (prev.length === 0 || !targetSessId) {
          targetSessId = `session-${Date.now()}`;
          const record: SmartScoutPitchRecord = {
            ...newPitchData,
            id: pitchId,
            pitchNumber: 1,
            timestamp: timeStr,
          };
          createdRecord = record;
          const newSess: SmartScoutSession = {
            id: targetSessId,
            title: `${dateStr} 投球`,
            date: dateStr,
            monthGroup: `${now.getFullYear()}年${now.getMonth() + 1}月`,
            discipline,
            cameraSetup: cameraAngle,
            pitches: [record],
          };
          return [newSess];
        }

        return prev.map((s, idx) => {
          if (s.id !== targetSessId && !(idx === 0 && !prev.some((x) => x.id === targetSessId))) {
            return s;
          }
          targetSessId = s.id;
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
        });
      });

      window.setTimeout(() => {
        setSelectedSessionId(targetSessId);
        setActivePlayerPitch({ sessionId: targetSessId, pitchId });
        if (createdRecord) {
          speakPitchResult(createdRecord, voiceEnabled);
        }
      }, 40);
    },
    [cameraAngle, discipline, selectedSessionId, sessions, voiceEnabled]
  );

  const currentVideoSession =
    sessions.find((s) => s.id === activePlayerPitch?.sessionId) ?? sessions[0] ?? null;
  const currentVideoPitchIdx = currentVideoSession
    ? Math.max(
        0,
        currentVideoSession.pitches.findIndex((p) => p.id === activePlayerPitch?.pitchId)
      )
    : -1;
  const currentVideoPitch =
    currentVideoSession && currentVideoSession.pitches.length > 0
      ? currentVideoSession.pitches[currentVideoPitchIdx] ??
        currentVideoSession.pitches[currentVideoSession.pitches.length - 1]
      : null;

  const handleDeletePitch = useCallback((sessionId: string, pitchId: string) => {
    setSessions((prev) =>
      prev
        .map((s) =>
          s.id === sessionId
            ? { ...s, pitches: s.pitches.filter((p) => p.id !== pitchId) }
            : s
        )
        .filter((s) => s.pitches.length > 0)
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

  return (
    <div
      className={`min-h-screen pb-18 transition-colors ${
        lightMode ? 'bg-[#f4f6f9] text-slate-900' : 'bg-slate-950 text-slate-100'
      }`}
    >
      <main className="mx-auto max-w-2xl px-0 sm:px-4 sm:pt-3">
        {/* 1. 追蹤：直接開啟真實相機自動測速 (無任何預載影片或假數據) */}
        {navTab === 'track' && (
          <SmartScoutLiveCamera
            currentPitch={currentVideoPitch}
            cameraAngle={cameraAngle}
            discipline={discipline}
            distanceMeters={distanceMeters}
            isWorkerReady={workerStatus === 'ready'}
            isBusyRef={isBusyRef}
            latestBall={latestBall}
            onBackToSession={() => setNavTab('sessions')}
            onDeletePitch={(pitchId) => {
              if (currentVideoSession) {
                handleDeletePitch(currentVideoSession.id, pitchId);
              }
            }}
            onFrame={handleFrame}
            onNewMeasuredPitch={handleNewMeasuredPitch}
            onChangePitchType={(pitchId, newType) => {
              if (currentVideoSession) {
                handleChangePitchType(currentVideoSession.id, pitchId, newType);
              }
            }}
          />
        )}

        {/* 2. 場次：僅顯示真實測得之投球紀錄 */}
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
              onCreateNewSession={() => setNavTab('track')}
            />
          </div>
        )}

        {/* 3. 影片：回放真實測得之投球與 3D 紅色彩帶 */}
        {navTab === 'video' && (
          <SmartScoutPitchPlayer
            pitch={currentVideoPitch}
            totalInSession={currentVideoSession?.pitches.length ?? 0}
            onBack={() => setNavTab('track')}
            onDelete={(pitchId) => {
              if (currentVideoSession) {
                handleDeletePitch(currentVideoSession.id, pitchId);
              }
            }}
            onChangePitchType={(pitchId, newType) => {
              if (currentVideoSession) {
                handleChangePitchType(currentVideoSession.id, pitchId, newType);
              }
            }}
            onPrevPitch={
              currentVideoSession && currentVideoPitchIdx > 0
                ? () =>
                    setActivePlayerPitch({
                      sessionId: currentVideoSession.id,
                      pitchId: currentVideoSession.pitches[currentVideoPitchIdx - 1].id,
                    })
                : undefined
            }
            onNextPitch={
              currentVideoSession &&
              currentVideoPitchIdx >= 0 &&
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

        {/* 4. 球員：個人檔案 */}
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

      {/* SmartScout 底部 5 大導覽列 */}
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
