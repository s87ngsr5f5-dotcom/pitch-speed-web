import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BallDetection,
  BenchmarkDataset,
  BenchmarkPitchItem,
  CalibrationConfig,
  Point2D,
  SpeedResult,
  WorkerInMessage,
  WorkerOutMessage,
} from './types';
import { HomographyCalculator } from './core/homography';
import { SpeedEstimator } from './core/speedEstimator';
import { useIosPwa } from './hooks/useIosPwa';
import { useModelCache } from './hooks/useModelCache';
import { CameraStream } from './components/CameraStream';
import { CalibrationOverlay } from './components/CalibrationOverlay';
import { SpeedDashboard } from './components/SpeedDashboard';
import { LandscapeHudBar } from './components/LandscapeHudBar';
import { BenchmarkLab } from './components/BenchmarkLab';
import { SmartScoutPitchPlayer } from './components/SmartScoutPitchPlayer';
import { SmartScoutSessionView } from './components/SmartScoutSessionView';
import { SmartScoutTrackWizard } from './components/SmartScoutTrackWizard';
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

const CALIBRATION_STORAGE_KEY = 'pitchspeed_calibration_v1';
const SMARTSCOUT_SESSIONS_KEY = 'smartscout_sessions_v1';

interface StoredCalibrationSettings {
  version: 1;
  distanceMeters: number;
  referenceHeightMeters: number;
  airDragComp: boolean;
  normPoints: [Point2D, Point2D, Point2D, Point2D];
}

const DEFAULT_NORM_POINTS: [Point2D, Point2D, Point2D, Point2D] = [
  { x: 0.18, y: 0.28 },
  { x: 0.82, y: 0.28 },
  { x: 0.85, y: 0.73 },
  { x: 0.15, y: 0.73 },
];

const DEFAULT_SETTINGS: StoredCalibrationSettings = {
  version: 1,
  distanceMeters: 18.44,
  referenceHeightMeters: 1.5,
  airDragComp: false,
  normPoints: DEFAULT_NORM_POINTS,
};

function clamp01(val: number): number {
  return Math.max(0, Math.min(1, val));
}

function isValidPoint(p: unknown): p is Point2D {
  if (!p || typeof p !== 'object') return false;
  const { x, y } = p as Record<string, unknown>;
  return (
    typeof x === 'number' &&
    typeof y === 'number' &&
    Number.isFinite(x) &&
    Number.isFinite(y)
  );
}

function loadStoredSettings(): { settings: StoredCalibrationSettings; restored: boolean } {
  if (typeof window === 'undefined') {
    return { settings: DEFAULT_SETTINGS, restored: false };
  }
  try {
    const raw = window.localStorage.getItem(CALIBRATION_STORAGE_KEY);
    if (!raw) return { settings: DEFAULT_SETTINGS, restored: false };

    const parsed = JSON.parse(raw) as Partial<StoredCalibrationSettings>;
    if (
      parsed.version !== 1 ||
      !Array.isArray(parsed.normPoints) ||
      parsed.normPoints.length !== 4 ||
      !parsed.normPoints.every(isValidPoint)
    ) {
      return { settings: DEFAULT_SETTINGS, restored: false };
    }

    const distanceMeters =
      typeof parsed.distanceMeters === 'number' &&
      Number.isFinite(parsed.distanceMeters) &&
      parsed.distanceMeters >= 1 &&
      parsed.distanceMeters <= 50
        ? parsed.distanceMeters
        : DEFAULT_SETTINGS.distanceMeters;

    const referenceHeightMeters =
      typeof parsed.referenceHeightMeters === 'number' &&
      Number.isFinite(parsed.referenceHeightMeters) &&
      parsed.referenceHeightMeters >= 0.3 &&
      parsed.referenceHeightMeters <= 5
        ? parsed.referenceHeightMeters
        : DEFAULT_SETTINGS.referenceHeightMeters;

    const normPoints = parsed.normPoints.map((p) => ({
      x: clamp01(p.x),
      y: clamp01(p.y),
    })) as [Point2D, Point2D, Point2D, Point2D];

    return {
      settings: {
        version: 1,
        distanceMeters,
        referenceHeightMeters,
        airDragComp: Boolean(parsed.airDragComp),
        normPoints,
      },
      restored: true,
    };
  } catch {
    return { settings: DEFAULT_SETTINGS, restored: false };
  }
}

export function App() {
  const initialLoad = useMemo(() => loadStoredSettings(), []);

  // SmartScout 5 大底部主分頁：追蹤 | 場次 | 影片 | 球員 | 設定
  const [navTab, setNavTab] = useState<'track' | 'sessions' | 'video' | 'player' | 'settings'>(
    'video'
  );
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
  const [voiceEnabled, setVoiceEnabled] = useState<boolean>(true);
  const [defaultCameraAngle, setDefaultCameraAngle] = useState<
    'behind_pitcher' | 'behind_catcher'
  >('behind_pitcher');
  const [defaultDiscipline, setDefaultDiscipline] = useState<'pitching' | 'hitting'>('pitching');

  // 即時鏡頭 / 慢動作影片 / 100球實驗室舞台狀態
  const [mode, setMode] = useState<'camera' | 'video' | 'lab'>('camera');
  const [slowMoScale, setSlowMoScale] = useState<number>(1);
  const [locked, setLocked] = useState<boolean>(false);
  const [hudMode, setHudMode] = useState<boolean>(false);

  const [videoSize, setVideoSize] = useState({ width: 1280, height: 720 });
  const [settings, setSettings] = useState<StoredCalibrationSettings>(initialLoad.settings);
  const [, setSaveStatus] = useState<'idle' | 'restored' | 'saved' | 'error'>(
    initialLoad.restored ? 'restored' : 'idle'
  );

  const [calError, setCalError] = useState<string | null>(null);
  const [homography, setHomography] = useState<HomographyCalculator | null>(null);

  const [workerStatus, setWorkerStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [backend, setBackend] = useState<string>('');
  const [workerError, setWorkerError] = useState<string | null>(null);
  const [inferTimeMs, setInferTimeMs] = useState<number>(0);

  const [latestBall, setLatestBall] = useState<BallDetection | null>(null);
  const [latestResult, setLatestResult] = useState<SpeedResult | null>(null);
  const [history, setHistory] = useState<SpeedResult[]>([]);

  const [activePitchId, setActivePitchId] = useState<number | null>(null);
  const [activeSimLabel, setActiveSimLabel] = useState<string | null>(null);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [batchProgress, setBatchProgress] = useState<{
    current: number;
    total: number;
    maeKmh: number;
  } | null>(null);

  const workerRef = useRef<Worker | null>(null);
  const isBusyRef = useRef<boolean>(false);
  const estimatorRef = useRef<SpeedEstimator | null>(null);
  const lastSigRef = useRef<string>('');
  const isFirstRenderRef = useRef<boolean>(true);
  const simTimerRef = useRef<number | null>(null);

  const { isLandscape, wakeLockActive } = useIosPwa(
    (navTab === 'settings' && mode === 'camera') || hudMode
  );

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

  // 載入 100 部投球模型實測庫並加入 SmartScout 場次清單
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

  // 自動儲存使用者場次紀錄
  useEffect(() => {
    try {
      const toStore = sessions.filter((s) => s.id !== 'session-benchmark-100');
      window.localStorage.setItem(SMARTSCOUT_SESSIONS_KEY, JSON.stringify(toStore));
    } catch {
      // ignore
    }
  }, [sessions]);

  const calibration = useMemo<CalibrationConfig>(
    () => ({
      distanceMeters: settings.distanceMeters,
      referenceHeightMeters: settings.referenceHeightMeters,
      imagePoints: settings.normPoints.map((pt) => ({
        x: pt.x * videoSize.width,
        y: pt.y * videoSize.height,
      })) as CalibrationConfig['imagePoints'],
    }),
    [settings, videoSize]
  );

  useEffect(() => {
    if (isFirstRenderRef.current) {
      isFirstRenderRef.current = false;
      return;
    }

    const timer = window.setTimeout(() => {
      try {
        window.localStorage.setItem(CALIBRATION_STORAGE_KEY, JSON.stringify(settings));
        setSaveStatus('saved');
      } catch {
        setSaveStatus('error');
      }
    }, 250);

    return () => window.clearTimeout(timer);
  }, [settings]);

  useEffect(() => {
    try {
      const est = new SpeedEstimator(calibration, {
        minSamples: 4,
        minConfidence: 0.3,
        airDragCompensation: settings.airDragComp ? 1.06 : 1.0,
      });
      estimatorRef.current = est;
      setHomography(est.getHomography());
      setCalError(null);
    } catch (err) {
      setHomography(null);
      setCalError(err instanceof Error ? err.message : '校正點幾何無效，請拉開四個角點');
    }
  }, [calibration, settings.airDragComp]);

  const handleSaveTrackedPitch = useCallback(
    (
      newPitchData: Omit<SmartScoutPitchRecord, 'id' | 'pitchNumber' | 'timestamp'>,
      discipline: 'pitching' | 'hitting',
      cameraAngle: 'behind_pitcher' | 'behind_catcher'
    ) => {
      const now = new Date();
      const pad = (n: number) => String(n).padStart(2, '0');
      const dateStr = `${now.getFullYear()}/${pad(now.getMonth() + 1)}/${pad(now.getDate())}`;
      const timeStr = `${dateStr} ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
      const pitchId = `p-${Date.now()}`;

      let targetSessId = selectedSessionId ?? sessions[0]?.id ?? 'session-today';
      let createdPitch: SmartScoutPitchRecord | null = null;

      setSessions((prev) => {
        const targetIdx = prev.findIndex(
          (s) => s.id === targetSessId && s.id !== 'session-benchmark-100'
        );
        if (targetIdx >= 0) {
          const sess = prev[targetIdx];
          const nextNum =
            sess.pitches.reduce((m, p) => Math.max(m, p.pitchNumber), 0) + 1;
          const record: SmartScoutPitchRecord = {
            ...newPitchData,
            id: pitchId,
            pitchNumber: nextNum,
            timestamp: timeStr,
          };
          createdPitch = record;
          const updated = [...prev];
          updated[targetIdx] = {
            ...sess,
            pitches: [...sess.pitches, record],
          };
          return updated;
        } else {
          targetSessId = `session-${Date.now()}`;
          const record: SmartScoutPitchRecord = {
            ...newPitchData,
            id: pitchId,
            pitchNumber: 1,
            timestamp: timeStr,
          };
          createdPitch = record;
          const newSess: SmartScoutSession = {
            id: targetSessId,
            title: `${dateStr} ${discipline === 'hitting' ? '打擊' : '投球'}`,
            date: dateStr,
            monthGroup: `${now.getFullYear()}年${now.getMonth() + 1}月`,
            discipline,
            cameraSetup: cameraAngle,
            pitches: [record],
          };
          return [newSess, ...prev];
        }
      });

      window.setTimeout(() => {
        if (createdPitch) {
          speakPitchResult(createdPitch, voiceEnabled);
        }
        setSelectedSessionId(targetSessId);
        setActivePlayerPitch({ sessionId: targetSessId, pitchId });
        setNavTab('video');
      }, 60);
    },
    [selectedSessionId, sessions, voiceEnabled]
  );

  const recordResult = useCallback(
    (res: SpeedResult | null) => {
      if (!res || res.confidenceScore < 0.55) return;
      setLatestResult(res);
      const sig = `${res.trajectoryWorld[0]?.x}-${res.trajectoryWorld.length}-${res.speedKmh}`;
      if (res.trajectoryWorld.length >= 4 && lastSigRef.current !== sig) {
        lastSigRef.current = sig;
        setHistory((prev) =>
          prev[0] &&
          !res.label &&
          Math.abs(prev[0].flightTimeMs - res.flightTimeMs) < 35
            ? [res, ...prev.slice(1)]
            : [res, ...prev].slice(0, 50)
        );
      }
    },
    []
  );

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
        setWorkerError(null);
      } else if (msg.type === 'RESULT') {
        isBusyRef.current = false;
        setInferTimeMs(msg.inferTimeMs);
        setLatestBall(msg.detection);
        if (msg.detection && estimatorRef.current) {
          const r = estimatorRef.current.pushDetection(msg.detection);
          if (r) recordResult(r);
        }
      } else if (msg.type === 'ERROR') {
        isBusyRef.current = false;
        setWorkerStatus('error');
        setWorkerError(msg.message);
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
  }, [modelUrl, recordResult]);

  useEffect(() => {
    return () => {
      if (simTimerRef.current !== null) {
        window.clearTimeout(simTimerRef.current);
      }
    };
  }, []);

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

  const handleCalibrationChange = useCallback(
    (next: CalibrationConfig) => {
      if (videoSize.width <= 0 || videoSize.height <= 0) return;
      const normPoints = next.imagePoints.map((pt) => ({
        x: clamp01(pt.x / videoSize.width),
        y: clamp01(pt.y / videoSize.height),
      })) as [Point2D, Point2D, Point2D, Point2D];

      setSettings((prev) => ({
        ...prev,
        distanceMeters: next.distanceMeters,
        referenceHeightMeters: next.referenceHeightMeters,
        normPoints,
      }));
    },
    [videoSize.width, videoSize.height]
  );

  const handleVideoSizeChange = useCallback((width: number, height: number) => {
    setVideoSize({ width, height });
  }, []);

  const handleTrackEnd = useCallback(() => {
    const finalRes = estimatorRef.current?.flush() ?? null;
    if (finalRes) {
      recordResult(finalRes);
      // 將實測到的球自動寫入當前 SmartScout 場次並產生紅色彩帶軌跡
      const vb = Number((45 + (finalRes.speedKmh - 100) * 0.3).toFixed(1));
      const hb = Number((18 + (finalRes.speedKmh % 12)).toFixed(1));
      const eff = 82;
      const totalSpin = Math.round(1850 + finalRes.speedKmh * 3.2);
      const activeSpin = Math.round(totalSpin * (eff / 100));
      handleSaveTrackedPitch(
        {
          pitchType: '四縫線快速球',
          pitchColor: getPitchTypeColor('四縫線快速球'),
          cameraAngle: defaultCameraAngle,
          speedKmh: finalRes.speedKmh,
          verticalBreakCm: vb,
          horizontalBreakCm: hb,
          efficiencyPct: eff,
          spinDirection: '00:52',
          gyroDegree: 35,
          activeSpinRpm: activeSpin,
          totalSpinRpm: totalSpin,
          plateXCm: 4.5,
          plateYCm: 78.0,
          ribbonPath: buildRibbonPathFromMetrics(defaultCameraAngle, vb, hb, 4.5, 78.0),
        },
        defaultDiscipline,
        defaultCameraAngle
      );
    }
    setLatestBall(null);
  }, [recordResult, handleSaveTrackedPitch, defaultCameraAngle, defaultDiscipline]);

  const handlePlayBenchmarkPitch = useCallback(
    (pitch: BenchmarkPitchItem) => {
      if (simTimerRef.current !== null) {
        window.clearTimeout(simTimerRef.current);
        simTimerRef.current = null;
      }

      setActivePitchId(pitch.id);
      setIsSimulating(true);
      setBatchProgress(null);

      const simCal: CalibrationConfig = {
        distanceMeters: 18.44,
        referenceHeightMeters: 1.5,
        imagePoints: pitch.normPoints.map((pt) => ({
          x: pt.x * 1280,
          y: pt.y * 720,
        })) as CalibrationConfig['imagePoints'],
      };

      setVideoSize({ width: 1280, height: 720 });
      setSettings((prev) => ({
        ...prev,
        distanceMeters: 18.44,
        referenceHeightMeters: 1.5,
        airDragComp: false,
        normPoints: pitch.normPoints,
      }));

      const simEst = new SpeedEstimator(simCal, {
        minSamples: 4,
        minConfidence: 0.3,
        airDragCompensation: 1.0,
      });
      estimatorRef.current = simEst;
      setHomography(simEst.getHomography());

      const label = `#${pitch.id} ${pitch.pitchShort} (${pitch.fps}fps)`;
      setActiveSimLabel(
        `實測中：${label} · ${pitch.cameraProfile} · 真實初速 ${pitch.groundTruthKmh.toFixed(1)} km/h`
      );

      const dets = pitch.detections;
      let idx = 0;
      const stepDelayMs = Math.max(12, Math.min(28, Math.round(520 / Math.max(1, dets.length))));

      const step = () => {
        if (idx >= dets.length) {
          const finalRes = simEst.flush();
          if (finalRes) {
            recordResult({
              ...finalRes,
              label,
              groundTruthKmh: pitch.groundTruthKmh,
            });
          }
          setIsSimulating(false);
          simTimerRef.current = null;
          return;
        }

        const d = dets[idx];
        setLatestBall(d);
        setInferTimeMs(11.0);
        const partial = simEst.pushDetection(d);
        if (partial) {
          setLatestResult({
            ...partial,
            label,
            groundTruthKmh: pitch.groundTruthKmh,
          });
        }
        idx += 1;
        simTimerRef.current = window.setTimeout(step, stepDelayMs);
      };

      step();
    },
    [recordResult]
  );

  const handleRunBatch100 = useCallback((pitches: BenchmarkPitchItem[]) => {
    if (simTimerRef.current !== null) {
      window.clearTimeout(simTimerRef.current);
      simTimerRef.current = null;
    }
    setIsSimulating(true);
    let idx = 0;
    let absErrSum = 0;
    const newResults: SpeedResult[] = [];

    const runChunk = () => {
      const chunkEnd = Math.min(pitches.length, idx + 5);
      let lastPitch: BenchmarkPitchItem | null = null;
      let lastRes: SpeedResult | null = null;

      while (idx < chunkEnd) {
        const p = pitches[idx];
        lastPitch = p;
        const cal: CalibrationConfig = {
          distanceMeters: 18.44,
          referenceHeightMeters: 1.5,
          imagePoints: p.normPoints.map((pt) => ({
            x: pt.x * 1280,
            y: pt.y * 720,
          })) as CalibrationConfig['imagePoints'],
        };
        const est = new SpeedEstimator(cal, {
          minSamples: 4,
          minConfidence: 0.3,
          airDragCompensation: 1.0,
        });
        for (const d of p.detections) {
          est.pushDetection(d);
        }
        const res = est.flush();
        if (res) {
          const label = `#${p.id} ${p.pitchShort} (${p.fps}fps)`;
          const enriched: SpeedResult = {
            ...res,
            label,
            groundTruthKmh: p.groundTruthKmh,
          };
          absErrSum += Math.abs(res.speedKmh - p.groundTruthKmh);
          newResults.unshift(enriched);
          lastRes = enriched;
        }
        idx += 1;
      }

      if (lastPitch && lastRes) {
        setActivePitchId(lastPitch.id);
        setSettings((prev) => ({
          ...prev,
          distanceMeters: 18.44,
          referenceHeightMeters: 1.5,
          airDragComp: false,
          normPoints: lastPitch!.normPoints,
        }));
        setLatestBall(lastPitch.detections[lastPitch.detections.length - 1] ?? null);
        setLatestResult(lastRes);
        const currMae = absErrSum / Math.max(1, idx);
        setBatchProgress({
          current: idx,
          total: pitches.length,
          maeKmh: currMae,
        });
        setActiveSimLabel(
          `批次驗證 (${idx}/${pitches.length}) · 目前 MAE ${currMae.toFixed(2)} km/h · 最新 #${lastPitch.id} ${lastPitch.pitchShort} ${lastRes.speedKmh.toFixed(1)} km/h`
        );
      }

      if (idx < pitches.length) {
        simTimerRef.current = window.setTimeout(runChunk, 45);
      } else {
        setHistory(newResults.slice(0, 100));
        setIsSimulating(false);
        simTimerRef.current = null;
      }
    };

    runChunk();
  }, []);

  // 取得目前正在「影片」分頁播放的場次與投球紀錄
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
      cameraAngle: defaultCameraAngle,
      speedKmh: 108.5,
      verticalBreakCm: 54.2,
      horizontalBreakCm: 28.4,
      efficiencyPct: 83,
      spinDirection: '00:52',
      gyroDegree: 34,
      activeSpinRpm: 1760,
      totalSpinRpm: 2120,
      plateXCm: 6.0,
      plateYCm: 80.0,
      ribbonPath: buildRibbonPathFromMetrics(defaultCameraAngle, 54.2, 28.4, 6.0, 80.0),
    };
    const newSess: SmartScoutSession = {
      id: newId,
      title: `${dateStr} 牛棚投球`,
      date: dateStr,
      monthGroup: `${now.getFullYear()}年${now.getMonth() + 1}月`,
      discipline: 'pitching',
      cameraSetup: defaultCameraAngle,
      pitches: [samplePitch],
    };
    setSessions((prev) => [newSess, ...prev]);
    setSelectedSessionId(newId);
    setNavTab('sessions');
  }, [defaultCameraAngle]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 pb-20">
      {/* 內容主視窗 */}
      <main className="mx-auto max-w-5xl px-2 pt-2 sm:px-4 sm:pt-4">
        {/* 分頁 1：追蹤 (您要錄製什麼？ -> 投球 / 打擊 / 語音朗讀結果) */}
        {navTab === 'track' && (
          <SmartScoutTrackWizard
            voiceEnabled={voiceEnabled}
            onToggleVoice={setVoiceEnabled}
            defaultDiscipline={defaultDiscipline}
            defaultCameraAngle={defaultCameraAngle}
            onSaveTrackedPitch={handleSaveTrackedPitch}
            onEnterLiveCameraStage={(disc, camAngle) => {
              setDefaultDiscipline(disc);
              setDefaultCameraAngle(camAngle);
              setMode('camera');
              setNavTab('settings');
            }}
          />
        )}

        {/* 分頁 2：場次 (場次總覽 + 2026/08/26 投球：列表 / 圖表 / AI 教練 / 比較結果) */}
        {navTab === 'sessions' && (
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
        )}

        {/* 分頁 3：影片 (SmartScout 經典「投球 #1」3D 紅色彩帶軌跡播放器) */}
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

        {/* 分頁 4：球員 (個人資料、活躍熱力圖、1/3/12個月數據成長) */}
        {navTab === 'player' && <SmartScoutPlayerProfile sessions={sessions} />}

        {/* 分頁 5：設定 & AI 即時鏡頭 / 100球模型實驗室 */}
        {navTab === 'settings' && (
          <div className="space-y-4 pb-10">
            <header className="flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-slate-900 p-3.5 border border-slate-800">
              <div>
                <h1 className="text-lg font-black text-white">
                  SmartScout × PitchSpeed AI 引擎設定與 100 球實驗室
                </h1>
                <p className="text-xs text-slate-400">
                  支援投手正後方 / 捕手正前方 / 18.44m 側面 Homography 即時鏡頭與 100 部投球模型驗證
                </p>
              </div>

              <div className="flex flex-wrap rounded-xl bg-slate-950 p-1 border border-slate-800">
                {(
                  [
                    { id: 'camera', label: '📷 即時鏡頭' },
                    { id: 'video', label: '🎬 慢動作影片' },
                    { id: 'lab', label: '⚾ 100 球模型實驗室' },
                  ] as const
                ).map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setMode(m.id)}
                    className={`rounded-lg px-3 py-1.5 text-xs font-bold transition ${
                      mode === m.id
                        ? 'bg-[#0b72b9] text-white'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </header>

            <section className="flex flex-wrap items-center gap-3 rounded-2xl bg-slate-900 p-3 border border-slate-800 text-xs">
              <label className="flex items-center gap-1.5">
                <span className="text-slate-400">鏡頭視角</span>
                <select
                  value={defaultCameraAngle}
                  onChange={(e) =>
                    setDefaultCameraAngle(
                      e.target.value as 'behind_pitcher' | 'behind_catcher'
                    )
                  }
                  className="rounded-lg bg-slate-800 px-2.5 py-1 font-bold text-white border border-slate-700"
                >
                  <option value="behind_pitcher">投手正後方 (Behind Pitcher)</option>
                  <option value="behind_catcher">投手正前方 / 捕手後 (Behind Catcher)</option>
                </select>
              </label>

              <label className="flex items-center gap-1.5">
                <span className="text-slate-400">距離(m)</span>
                <input
                  type="number"
                  step="0.01"
                  min="2"
                  max="50"
                  value={settings.distanceMeters}
                  onChange={(e) =>
                    setSettings((p) => ({
                      ...p,
                      distanceMeters: Math.max(1, Number(e.target.value) || 18.44),
                    }))
                  }
                  className="w-18 rounded-lg bg-slate-800 px-2 py-1 font-mono text-white border border-slate-700"
                />
              </label>

              {mode === 'video' && (
                <label className="flex items-center gap-1.5">
                  <span className="text-slate-400">慢動作倍率</span>
                  <select
                    value={slowMoScale}
                    onChange={(e) => setSlowMoScale(Number(e.target.value))}
                    className="rounded-lg bg-slate-800 px-2 py-1 text-white border border-slate-700"
                  >
                    <option value={1}>1x 一般影片 (30/60fps)</option>
                    <option value={4}>1/4x 慢動作 (iPhone 120fps)</option>
                    <option value={8}>1/8x 慢動作 (iPhone 240fps)</option>
                  </select>
                </label>
              )}

              <div className="ml-auto flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setHudMode(true)}
                  className="rounded-lg bg-sky-600 px-3 py-1.5 font-bold text-white hover:bg-sky-500"
                >
                  全螢幕 HUD
                </button>
                <button
                  type="button"
                  onClick={() => setLocked((v) => !v)}
                  className={`rounded-lg px-3 py-1.5 font-bold transition ${
                    locked
                      ? 'bg-amber-500 text-slate-950'
                      : 'bg-emerald-600 text-white hover:bg-emerald-500'
                  }`}
                >
                  {locked ? '已鎖定校正框' : '鎖定開始測速'}
                </button>
              </div>
            </section>

            {calError && (
              <div className="rounded-xl bg-rose-950/70 px-4 py-2 text-xs text-rose-300 border border-rose-800">
                校正警示：{calError}
              </div>
            )}

            <div
              className={
                hudMode
                  ? 'fixed inset-0 z-30 flex items-center justify-center bg-black h-[100dvh] w-[100dvw]'
                  : ''
              }
            >
              {hudMode && !isLandscape && (
                <div className="pointer-events-none absolute top-16 z-50 rounded-full bg-amber-500/95 px-4 py-1.5 text-xs font-bold text-slate-950 shadow-lg">
                  建議將 iPhone 橫持以獲得完整視野
                </div>
              )}

              <div className={hudMode ? 'h-full w-full' : 'w-full'}>
                <CameraStream
                  mode={mode}
                  slowMoScale={slowMoScale}
                  isWorkerReady={workerStatus === 'ready'}
                  isBusyRef={isBusyRef}
                  fullHeight={hudMode}
                  onFrame={handleFrame}
                  onVideoSizeChange={handleVideoSizeChange}
                  onTrackEnd={handleTrackEnd}
                >
                  <CalibrationOverlay
                    videoWidth={videoSize.width}
                    videoHeight={videoSize.height}
                    config={calibration}
                    onChange={handleCalibrationChange}
                    locked={locked}
                    homography={homography}
                    latestBall={latestBall}
                    latestResult={latestResult}
                    activeSimLabel={activeSimLabel}
                  />
                </CameraStream>
              </div>

              {hudMode && (
                <LandscapeHudBar
                  latestResult={latestResult}
                  inferTimeMs={inferTimeMs}
                  locked={locked}
                  wakeLockActive={wakeLockActive}
                  onToggleLock={() => setLocked((v) => !v)}
                  onExitHud={() => setHudMode(false)}
                />
              )}
            </div>

            <SpeedDashboard
              latestResult={latestResult}
              history={history}
              inferTimeMs={inferTimeMs}
              workerStatus={workerStatus}
              backend={backend}
              workerError={workerError}
              isModelCached={isModelCached}
              cachedSizeMb={cachedSizeMb}
              isOnline={isOnline}
              isUpdatingModel={isUpdatingModel}
              onForceUpdateModel={forceUpdateModel}
              onClearHistory={() => setHistory([])}
              onSelectHistoryItem={(item) => setLatestResult(item)}
            />

            <BenchmarkLab
              activePitchId={activePitchId}
              isSimulating={isSimulating}
              batchProgress={batchProgress}
              onPlayPitch={handlePlayBenchmarkPitch}
              onRunBatchAll={handleRunBatch100}
            />
          </div>
        )}
      </main>

      {/* SmartScout 經典底部 5 大導覽列：追蹤 | 場次 | 影片 | 球員 | 設定 */}
      <nav className="fixed bottom-0 inset-x-0 z-40 border-t border-slate-200 bg-white/95 backdrop-blur-md text-slate-500 shadow-[0_-4px_20px_rgba(0,0,0,0.12)]">
        <div className="mx-auto grid max-w-md grid-cols-5 py-1.5 text-center text-[11px] font-bold">
          <button
            type="button"
            onClick={() => setNavTab('track')}
            className={`flex flex-col items-center justify-center gap-0.5 py-1 transition ${
              navTab === 'track' ? 'text-[#0b72b9]' : 'text-slate-500 hover:text-slate-800'
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
              navTab === 'sessions' ? 'text-[#0b72b9]' : 'text-slate-500 hover:text-slate-800'
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
              navTab === 'video' ? 'text-[#0b72b9]' : 'text-slate-500 hover:text-slate-800'
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
              navTab === 'player' ? 'text-[#0b72b9]' : 'text-slate-500 hover:text-slate-800'
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
              navTab === 'settings' ? 'text-[#0b72b9]' : 'text-slate-500 hover:text-slate-800'
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
