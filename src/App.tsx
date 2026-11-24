import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BallDetection,
  PitchActionPhase,
  WorkerInMessage,
  WorkerOutMessage,
} from './types';
import { deletePitchVideoClip, loadPitchVideoUrl } from './utils/pitchVideoStore';
import { useModelCache } from './hooks/useModelCache';
import { SmartScoutLiveCamera } from './components/SmartScoutLiveCamera';
import { PitcherPoseGate } from './core/pitcherPoseGate';
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

const SMARTSCOUT_SESSIONS_KEY = 'smartscout_sessions_v2';
const LEGACY_TEMPLATE_IDS = new Set([
  'session-2026-08-26-pitch',
  'session-2026-08-26-hit',
  'session-2026-08-19-pitch',
  'session-ai-coach-demo',
  'session-benchmark-100',
]);

export function App() {
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
      const saved = window.localStorage.getItem(SMARTSCOUT_SESSIONS_KEY);
      if (saved) {
        const parsed = JSON.parse(saved) as SmartScoutSession[];
        if (Array.isArray(parsed)) {
          return parsed.filter((s) => !LEGACY_TEMPLATE_IDS.has(s.id));
        }
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
  const [workerError, setWorkerError] = useState<string>();
  const [backend, setBackend] = useState<string>('');
  const [inferTimeMs, setInferTimeMs] = useState<number>(0);
  const [latestBall, setLatestBall] = useState<BallDetection | null>(null);
  const [actionPhase, setActionPhase] = useState<PitchActionPhase>('idle');
  const [actionConfidence, setActionConfidence] = useState<number>(0);

  const workerRef = useRef<Worker | null>(null);
  const poseWorkerRef = useRef<Worker | null>(null);
  const poseReadyRef = useRef<boolean>(false);
  const poseBusyRef = useRef<boolean>(false);
  const isBusyRef = useRef<boolean>(false);
  const pendingScanResolveRef = useRef<((det: BallDetection | null) => void) | null>(null);
  // 投手骨架出手閘門（獨立 Worker 非阻塞執行，絕不拖慢球體偵測）
  const poseGateRef = useRef(new PitcherPoseGate());
  const lastPoseRequestTsRef = useRef(-1_000_000);
  const lastPoseIngestTsRef = useRef(-1);
  const shouldRequestPose = useCallback((timestampMs: number, minIntervalMs: number) => {
    if (timestampMs < lastPoseRequestTsRef.current - 100) {
      poseGateRef.current.reset();
      lastPoseIngestTsRef.current = -1;
      lastPoseRequestTsRef.current = -1_000_000;
    }
    if (timestampMs - lastPoseRequestTsRef.current < minIntervalMs) return false;
    lastPoseRequestTsRef.current = timestampMs;
    return true;
  }, []);

  const poseModelUrl = useMemo(
    () => new URL('./models/pitcher_pose_int8.onnx', window.location.href).href,
    []
  );

  const modelUrl = useMemo(
    () => new URL('./models/ball_yolo11n.onnx?v=v3fp32', window.location.href).href,
    []
  );
  const actionModelUrl = useMemo(
    () => new URL('./models/pitch_action_cls.onnx', window.location.href).href,
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
      // Strip transient blob: URLs before persisting to localStorage; videoClipId restores them from IndexedDB.
      const serializable = sessions.map((s) => ({
        ...s,
        pitches: s.pitches.map((p) =>
          p.videoUrl?.startsWith('blob:') ? { ...p, videoUrl: undefined } : p
        ),
      }));
      window.localStorage.setItem(SMARTSCOUT_SESSIONS_KEY, JSON.stringify(serializable));
    } catch {
      // ignore
    }
  }, [sessions]);

  // 從 IndexedDB 還原已儲存的各球投球影片
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const updates = new Map<string, string>();
      for (const s of sessions) {
        for (const p of s.pitches) {
          if (p.videoClipId && !p.videoUrl) {
            const url = await loadPitchVideoUrl(p.videoClipId);
            if (url) updates.set(p.id, url);
          }
        }
      }
      if (!cancelled && updates.size > 0) {
        setSessions((prev) =>
          prev.map((s) => ({
            ...s,
            pitches: s.pitches.map((p) =>
              updates.has(p.id) ? { ...p, videoUrl: updates.get(p.id) } : p
            ),
          }))
        );
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 初始化 ONNX Web Worker (320x320)
  useEffect(() => {
    const worker = new Worker(new URL('./workers/inference.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;

    const fail = (message: string) => {
      isBusyRef.current = false;
      setLatestBall(null);
      setWorkerError(message);
      setWorkerStatus('error');
    };
    worker.onerror = (event) => fail(event.message || '無法啟動模型工作程序');
    worker.onmessageerror = () => fail('無法讀取模型工作程序的回應');
    worker.onmessage = (e: MessageEvent<WorkerOutMessage>) => {
      const msg = e.data;
      if (msg.type === 'READY') {
        setWorkerStatus('ready');
        setBackend(msg.backend);
      } else if (msg.type === 'RESULT') {
        isBusyRef.current = false;
        setInferTimeMs(msg.inferTimeMs);
        if (pendingScanResolveRef.current) {
          const resolve = pendingScanResolveRef.current;
          pendingScanResolveRef.current = null;
          resolve(msg.detection);
        } else {
          setLatestBall(msg.detection);
          if (msg.actionPhase) {
            setActionPhase(msg.actionPhase);
            setActionConfidence(msg.actionConfidence ?? 0);
          }
        }
      } else if (msg.type === 'ERROR') {
        if (pendingScanResolveRef.current) {
          const resolve = pendingScanResolveRef.current;
          pendingScanResolveRef.current = null;
          resolve(null);
        }
        fail(msg.message);
      }
    };

    const initMsg: WorkerInMessage = {
      type: 'INIT',
      modelUrl,
      actionModelUrl,
      wasmPaths: new URL('./wasm/', window.location.href).href,
      inputSize: 320,
      confThreshold: 0.16,
    };
    worker.postMessage(initMsg);

    // 獨立背景骨架 Worker（不同 CPU 核心執行，完全不阻塞球體偵測）
    const poseWorker = new Worker(new URL('./workers/pose.worker.ts', import.meta.url), {
      type: 'module',
    });
    poseWorkerRef.current = poseWorker;
    poseReadyRef.current = false;
    poseBusyRef.current = false;
    poseWorker.onmessage = (ev: MessageEvent<{ type: string; timestampMs?: number; pose?: Array<[number, number, number]> | null }>) => {
      const m = ev.data;
      if (m.type === 'READY') {
        poseReadyRef.current = true;
      } else if (m.type === 'POSE_RESULT') {
        poseBusyRef.current = false;
        if (typeof m.timestampMs === 'number') {
          if (m.timestampMs < lastPoseIngestTsRef.current - 100) poseGateRef.current.reset();
          lastPoseIngestTsRef.current = m.timestampMs;
          poseGateRef.current.ingest(m.pose ?? null, m.timestampMs);
        }
      }
    };
    poseWorker.postMessage({
      type: 'INIT',
      poseModelUrl,
      wasmPaths: new URL('./wasm/', window.location.href).href,
    });

    return () => {
      worker.terminate();
      workerRef.current = null;
      poseWorker.terminate();
      poseWorkerRef.current = null;
      poseReadyRef.current = false;
      poseBusyRef.current = false;
    };
  }, [modelUrl, actionModelUrl, poseModelUrl]);

  const handleFrame = useCallback(
    (
      bitmap: ImageBitmap,
      timestampMs: number,
      origWidth: number,
      origHeight: number,
      predictedRoi?: { x: number; y: number; radius: number }
    ) => {
      const w = workerRef.current;
      if (!w) {
        bitmap.close();
        isBusyRef.current = false;
        return;
      }
      try {
        w.postMessage(
          {
            type: 'INFER',
            bitmap,
            timestampMs,
            origWidth,
            origHeight,
            detectAction: false,
            predictedRoi,
          } satisfies WorkerInMessage,
          [bitmap]
        );
      } catch (error) {
        bitmap.close();
        isBusyRef.current = false;
        setLatestBall(null);
        setWorkerError(error instanceof Error ? error.message : String(error));
        setWorkerStatus('error');
      }
    },
    []
  );

  // SmartScout 逐幀零掉幀掃描專用：Promise 形式等待 Worker 回傳該幀偵測結果
  const handleScanFrame = useCallback(
    (
      bitmap: ImageBitmap,
      timestampMs: number,
      origWidth: number,
      origHeight: number
    ): Promise<BallDetection | null> => {
      const w = workerRef.current;
      if (!w) {
        bitmap.close();
        return Promise.resolve(null);
      }
      return new Promise<BallDetection | null>((resolve) => {
        pendingScanResolveRef.current = resolve;
        isBusyRef.current = true;
        try {
          w.postMessage(
            {
              type: 'INFER',
              bitmap,
              timestampMs,
              origWidth,
              origHeight,
              detectAction: false,
            } satisfies WorkerInMessage,
            [bitmap]
          );
        } catch {
          bitmap.close();
          isBusyRef.current = false;
          pendingScanResolveRef.current = null;
          resolve(null);
        }
      });
    },
    []
  );

  const handlePoseFrame = useCallback(
    (bitmap: ImageBitmap, timestampMs: number, origWidth: number, origHeight: number) => {
      const pw = poseWorkerRef.current;
      if (!pw || !poseReadyRef.current || poseBusyRef.current || !shouldRequestPose(timestampMs, 100)) {
        bitmap.close();
        return;
      }
      poseBusyRef.current = true;
      try {
        pw.postMessage({ type: 'POSE', bitmap, timestampMs, origWidth, origHeight }, [bitmap]);
      } catch {
        bitmap.close();
        poseBusyRef.current = false;
      }
    },
    [shouldRequestPose]
  );

  const handleNewMeasuredPitch = useCallback(
    (newPitchData: Omit<SmartScoutPitchRecord, 'id' | 'pitchNumber' | 'timestamp'>) => {
      const now = new Date();
      const pad = (n: number) => String(n).padStart(2, '0');
      const dateStr = `${now.getFullYear()}/${pad(now.getMonth() + 1)}/${pad(now.getDate())}`;
      const timeStr = `${dateStr} ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
      const pitchId = `p-${Date.now()}`;

      let targetSessId =
        sessions.find((s) => s.date === dateStr && s.discipline === discipline)?.id ?? '';
      let createdRecord: SmartScoutPitchRecord | null = null;

      setSessions((prev) => {
        if (!prev.some((s) => s.id === targetSessId)) {
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
          return [newSess, ...prev];
        }

        return prev.map((s) => {
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

  // 相機頁面僅顯示使用者當次實測到的最新球，不預載舊模板球
  const activeUserSession =
    sessions.find((s) => s.id === activePlayerPitch?.sessionId) ??
    sessions[0] ??
    null;

  const liveMeasuredPitch =
    activeUserSession && activeUserSession.pitches.length > 0
      ? activeUserSession.pitches.find((p) => p.id === activePlayerPitch?.pitchId) ??
        activeUserSession.pitches[activeUserSession.pitches.length - 1]
      : null;

  const currentVideoSession =
    sessions.find((s) => s.id === activePlayerPitch?.sessionId) ??
    activeUserSession ??
    sessions[0] ??
    null;
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
        .map((s) => {
          if (s.id !== sessionId) return s;
          const target = s.pitches.find((p) => p.id === pitchId);
          if (target?.videoClipId) {
            void deletePitchVideoClip(target.videoClipId);
          }
          return { ...s, pitches: s.pitches.filter((p) => p.id !== pitchId) };
        })
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
        {/* 分頁 1：追蹤 (SmartScoutLiveCamera) */}
        {navTab === 'track' && (
          <SmartScoutLiveCamera
            currentPitch={liveMeasuredPitch}
            cameraAngle={cameraAngle}
            discipline={discipline}
            distanceMeters={distanceMeters}
            isWorkerReady={workerStatus === 'ready'}
            workerError={workerError}
            isBusyRef={isBusyRef}
            latestBall={latestBall}
            poseGate={poseGateRef.current}
            actionPhase={actionPhase}
            actionConfidence={actionConfidence}
            onBackToSession={() => setNavTab('sessions')}
            onDeletePitch={(pitchId) => {
              if (activeUserSession) {
                handleDeletePitch(activeUserSession.id, pitchId);
              }
            }}
            onFrame={handleFrame}
            onPoseFrame={handlePoseFrame}
            onScanFrame={handleScanFrame}
            onNewMeasuredPitch={handleNewMeasuredPitch}
            onChangePitchType={(pitchId, newType) => {
              if (activeUserSession) {
                handleChangePitchType(activeUserSession.id, pitchId, newType);
              }
            }}
          />
        )}

        {/* 分頁 2：場次 (SmartScoutSessionView) */}
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

        {/* 分頁 3：影片 (SmartScoutPitchPlayer) */}
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

        {/* 分頁 4：球員 (SmartScoutPlayerProfile) */}
        {navTab === 'player' && (
          <div className="px-3 pt-3 sm:px-0">
            <SmartScoutPlayerProfile sessions={sessions} />
          </div>
        )}

        {/* 分頁 5：設定 */}
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
                    ONNX AI 引擎 ({backend || 'WASM'})
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
                  {isUpdatingModel ? '更新中' : '強制更新模型'}
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* 五大底部導覽分頁：追蹤 | 場次 | 影片 | 球員 | 設定 */}
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
