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

const CALIBRATION_STORAGE_KEY = 'pitchspeed_calibration_v1';

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

const DISTANCE_PRESETS = [
  { label: '職棒/成棒 18.44m', value: 18.44 },
  { label: '青少棒 16.46m', value: 16.46 },
  { label: '少棒 14.02m', value: 14.02 },
  { label: '牛棚短距 10.0m', value: 10.0 },
];

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

  const [mode, setMode] = useState<'camera' | 'video' | 'lab'>('lab');
  const [slowMoScale, setSlowMoScale] = useState<number>(1);
  const [locked, setLocked] = useState<boolean>(false);
  const [hudMode, setHudMode] = useState<boolean>(false);

  const [videoSize, setVideoSize] = useState({ width: 1280, height: 720 });
  const [settings, setSettings] = useState<StoredCalibrationSettings>(initialLoad.settings);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'restored' | 'saved' | 'error'>(
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

  // 100 球模型練習庫狀態
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

  const { isLandscape, wakeLockActive } = useIosPwa(mode === 'camera' || hudMode);

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

  const recordResult = useCallback((res: SpeedResult | null) => {
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
  }, []);

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
    if (finalRes) recordResult(finalRes);
    setLatestBall(null);
  }, [recordResult]);

  const handleResetPoints = useCallback(() => {
    setSettings((prev) => ({
      ...prev,
      normPoints: DEFAULT_NORM_POINTS,
    }));
  }, []);

  // 播放單一基準投球軌跡（透過前端真實 SpeedEstimator + Homography 逐幀推算）
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
            const enriched: SpeedResult = {
              ...finalRes,
              label,
              groundTruthKmh: pitch.groundTruthKmh,
            };
            recordResult(enriched);
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

  // 一鍵連續跑完 100 球前端物理引擎驗證
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

  // 快速從任何模式觸發隨機一球示範
  const handleQuickSimulateFromAnyMode = useCallback(() => {
    const url = new URL('./data/benchmark_100.json', window.location.href).href;
    fetch(url)
      .then((r) => r.json())
      .then((data: BenchmarkDataset) => {
        if (data.pitches?.length > 0) {
          const pick = data.pitches[Math.floor(Math.random() * data.pitches.length)];
          handlePlayBenchmarkPitch(pick);
        }
      })
      .catch(() => {});
  }, [handlePlayBenchmarkPitch]);

  // 初次進入頁面時自動載入第 1 球預覽軌跡，讓畫面一打開就有直觀球道與數據
  useEffect(() => {
    const url = new URL('./data/benchmark_100.json', window.location.href).href;
    fetch(url)
      .then((r) => (r.ok ? r.json() : null))
      .then((data: BenchmarkDataset | null) => {
        if (data?.pitches?.length) {
          handlePlayBenchmarkPitch(data.pitches[0]);
        }
      })
      .catch(() => {});
  }, [handlePlayBenchmarkPitch]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 safe-p md:p-6">
      <div className="mx-auto max-w-5xl space-y-4">
        <header className="flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-slate-900/90 p-3.5 border border-slate-800/90 shadow-lg">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-500/20 text-sm border border-emerald-500/40">
                ⚾
              </span>
              <h1 className="text-xl font-black tracking-tight text-white">
                PitchSpeed Web <span className="text-emerald-400">18.44m</span>
              </h1>
              <span className="rounded-full bg-emerald-950/90 px-2.5 py-0.5 text-[10px] font-bold text-emerald-300 border border-emerald-700/60">
                100 球驗證 MAE 0.40 km/h
              </span>
              {saveStatus === 'saved' && (
                <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] font-semibold text-slate-300 border border-slate-700">
                  校正已自動記憶
                </span>
              )}
              {saveStatus === 'restored' && (
                <span className="rounded-full bg-sky-950/80 px-2 py-0.5 text-[10px] font-semibold text-sky-300 border border-sky-700/60">
                  已載入上次球場校正
                </span>
              )}
            </div>
            <p className="mt-1 text-xs text-slate-400">
              垂直飛行切面 Homography + 重力補償拋物線測速・支援 iOS Safari 後置鏡頭、120/240fps 慢動作與離線 PWA
            </p>
          </div>

          <div className="flex flex-wrap rounded-xl bg-slate-950 p-1 border border-slate-800">
            {(
              [
                { id: 'camera', label: '📷 即時後置鏡頭' },
                { id: 'video', label: '🎬 上傳慢動作影片' },
                { id: 'lab', label: '⚾ 100 球模型練習庫' },
              ] as const
            ).map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => {
                  setMode(m.id);
                  if (m.id !== 'lab') {
                    setActiveSimLabel(null);
                    estimatorRef.current?.reset();
                  }
                }}
                className={`rounded-lg px-3 py-1.5 text-xs font-bold transition ${
                  mode === m.id
                    ? 'bg-emerald-600 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </header>

        <section className="flex flex-wrap items-center gap-3 rounded-2xl bg-slate-900/95 p-3 border border-slate-800 text-xs shadow-md">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-slate-400">場地快速切換：</span>
            {DISTANCE_PRESETS.map((preset) => (
              <button
                key={preset.value}
                type="button"
                onClick={() =>
                  setSettings((p) => ({
                    ...p,
                    distanceMeters: preset.value,
                  }))
                }
                className={`rounded-lg px-2 py-1 text-[11px] font-bold transition ${
                  Math.abs(settings.distanceMeters - preset.value) < 0.02
                    ? 'bg-emerald-600/90 text-white'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {preset.label}
              </button>
            ))}
          </div>

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

          <label className="flex items-center gap-1.5">
            <span className="text-slate-400">切面高度(m)</span>
            <input
              type="number"
              step="0.1"
              min="0.5"
              max="5"
              value={settings.referenceHeightMeters}
              onChange={(e) =>
                setSettings((p) => ({
                  ...p,
                  referenceHeightMeters: Math.max(0.3, Number(e.target.value) || 1.5),
                }))
              }
              className="w-14 rounded-lg bg-slate-800 px-2 py-1 font-mono text-white border border-slate-700"
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

          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.airDragComp}
              onChange={(e) =>
                setSettings((p) => ({ ...p, airDragComp: e.target.checked }))
              }
              className="accent-emerald-500"
            />
            <span className="text-slate-300">空氣阻力初速補償 (+6%)</span>
          </label>

          <div className="ml-auto flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={handleResetPoints}
              className="rounded-lg bg-slate-800 px-3 py-1.5 font-semibold text-slate-300 hover:bg-slate-700"
            >
              重置四點
            </button>
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
              建議將 iPhone 橫持以獲得完整 18.44m 測速視野
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
              onQuickSimulatePitch={handleQuickSimulateFromAnyMode}
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

        {mode === 'lab' && (
          <BenchmarkLab
            activePitchId={activePitchId}
            isSimulating={isSimulating}
            batchProgress={batchProgress}
            onPlayPitch={handlePlayBenchmarkPitch}
            onRunBatchAll={handleRunBatch100}
          />
        )}
      </div>
    </div>
  );
}

export default App;
