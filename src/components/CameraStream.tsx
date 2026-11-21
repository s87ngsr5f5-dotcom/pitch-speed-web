import React, { useEffect, useRef, useState } from 'react';

interface VideoFrameMetadata {
  presentationTime: number;
  expectedDisplayTime: number;
  width: number;
  height: number;
  mediaTime: number;
  presentedFrames: number;
}

type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback?: (
    callback: (now: DOMHighResTimeStamp, metadata: VideoFrameMetadata) => void
  ) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

interface CameraStreamProps {
  mode: 'camera' | 'video' | 'lab';
  slowMoScale: number;
  isWorkerReady: boolean;
  isBusyRef: React.MutableRefObject<boolean>;
  fullHeight?: boolean;
  onFrame: (
    bitmap: ImageBitmap,
    timestampMs: number,
    width: number,
    height: number
  ) => void;
  onVideoSizeChange: (width: number, height: number) => void;
  onTrackEnd: () => void;
  onQuickSimulatePitch?: () => void;
  children?: React.ReactNode;
}

export const CameraStream: React.FC<CameraStreamProps> = ({
  mode,
  slowMoScale,
  isWorkerReady,
  isBusyRef,
  fullHeight = false,
  onFrame,
  onVideoSizeChange,
  onTrackEnd,
  onQuickSimulatePitch,
  children,
}) => {
  const videoRef = useRef<VideoWithRVFC | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoFileName, setVideoFileName] = useState<string | null>(null);
  const [videoSize, setVideoSize] = useState({ width: 1280, height: 720 });
  const [isPlaying, setIsPlaying] = useState(false);

  const latestPropsRef = useRef({
    mode,
    slowMoScale,
    isWorkerReady,
    onFrame,
  });
  latestPropsRef.current = { mode, slowMoScale, isWorkerReady, onFrame };

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let activeStream: MediaStream | null = null;
    let cancelled = false;

    const stopStream = () => {
      if (activeStream) {
        activeStream.getTracks().forEach((t) => t.stop());
        activeStream = null;
      }
      if (video.srcObject) {
        video.srcObject = null;
      }
    };

    if (mode !== 'camera') {
      stopStream();
      setCameraError(null);
      return;
    }

    const startCamera = async () => {
      setCameraError(null);
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraError(
          '無法存取相機：iOS Safari 必須在 HTTPS 網址（如 GitHub Pages）下才能開啟鏡頭。'
        );
        return;
      }

      const tryConstraints: MediaStreamConstraints[] = [
        {
          video: {
            facingMode: { exact: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 60 },
          },
          audio: false,
        },
        {
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 60 },
          },
          audio: false,
        },
        { video: true, audio: false },
      ];

      for (const constraints of tryConstraints) {
        try {
          activeStream = await navigator.mediaDevices.getUserMedia(constraints);
          break;
        } catch {
          // fallback
        }
      }

      if (cancelled) {
        stopStream();
        return;
      }

      if (!activeStream) {
        setCameraError('未偵測到可用相機或權限受拒。您可切換至「上傳慢動作影片」或「100 球模型練習庫」立即體驗測速。');
        return;
      }

      video.srcObject = activeStream;
      try {
        await video.play();
        setIsPlaying(true);
      } catch {
        setCameraError('請點擊畫面以啟動鏡頭串流。');
      }
    };

    startCamera();

    return () => {
      cancelled = true;
      stopStream();
    };
  }, [mode]);

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let stopped = false;
    let rvfcId: number | null = null;
    let rafId: number | null = null;
    let lastMediaTime = -1;

    const processFrame = (timestampMs: number) => {
      const { isWorkerReady: ready, onFrame: emitFrame, mode: currMode } = latestPropsRef.current;
      if (
        currMode === 'lab' ||
        !ready ||
        isBusyRef.current ||
        video.readyState < 2 ||
        video.paused ||
        video.ended ||
        video.videoWidth === 0
      ) {
        return;
      }

      isBusyRef.current = true;
      const w = video.videoWidth;
      const h = video.videoHeight;

      createImageBitmap(video)
        .then((bitmap) => {
          emitFrame(bitmap, timestampMs, w, h);
        })
        .catch(() => {
          isBusyRef.current = false;
        });
    };

    if (typeof video.requestVideoFrameCallback === 'function') {
      const onVideoFrame = (_now: DOMHighResTimeStamp, meta: VideoFrameMetadata) => {
        if (stopped) return;
        const { mode: currMode, slowMoScale: scale } = latestPropsRef.current;

        if (meta.mediaTime !== lastMediaTime) {
          lastMediaTime = meta.mediaTime;
          const realTimestampMs =
            currMode === 'video'
              ? (meta.mediaTime * 1000) / scale
              : meta.mediaTime * 1000;
          processFrame(realTimestampMs);
        }

        rvfcId = video.requestVideoFrameCallback!(onVideoFrame);
      };
      rvfcId = video.requestVideoFrameCallback(onVideoFrame);
    } else {
      const onRaf = (now: DOMHighResTimeStamp) => {
        if (stopped) return;
        const { mode: currMode, slowMoScale: scale } = latestPropsRef.current;
        const ts =
          currMode === 'video' ? (video.currentTime * 1000) / scale : now;
        processFrame(ts);
        rafId = requestAnimationFrame(onRaf);
      };
      rafId = requestAnimationFrame(onRaf);
    }

    return () => {
      stopped = true;
      if (rvfcId !== null && video.cancelVideoFrameCallback) {
        video.cancelVideoFrameCallback(rvfcId);
      }
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
      }
    };
  }, [isBusyRef]);

  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;
    setVideoSize({ width: video.videoWidth, height: video.videoHeight });
    onVideoSizeChange(video.videoWidth, video.videoHeight);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(URL.createObjectURL(file));
    setVideoFileName(file.name);
    onTrackEnd();
  };

  const toggleVideoPlay = async () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused || video.ended) {
      if (video.ended) video.currentTime = 0;
      await video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
      onTrackEnd();
    }
  };

  const aspectRatio = videoSize.width / videoSize.height;
  const stageStyle: React.CSSProperties = fullHeight
    ? {
        width: `min(100dvw, calc(100dvh * ${aspectRatio}))`,
        height: `min(100dvh, calc(100dvw / ${aspectRatio}))`,
      }
    : {
        width: `min(100%, calc(66vh * ${aspectRatio}))`,
        aspectRatio: `${videoSize.width} / ${videoSize.height}`,
      };

  return (
    <div className={fullHeight ? 'relative h-[100dvh] w-[100dvw] bg-black' : 'flex flex-col gap-3'}>
      {mode === 'video' && !fullHeight && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl bg-slate-900/95 p-3 border border-slate-800 shadow-lg">
          <div className="flex flex-wrap items-center gap-2.5">
            <label className="cursor-pointer inline-flex items-center gap-2 rounded-xl bg-emerald-600 px-3.5 py-2 text-xs font-bold text-white hover:bg-emerald-500 active:scale-95 transition shadow-md shadow-emerald-950/50">
              <span>📂 選擇投球影片（支援 iPhone 120/240fps 慢動作）</span>
              <input
                type="file"
                accept="video/*"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
            {videoFileName && (
              <span className="rounded-lg bg-slate-800 px-2.5 py-1 text-xs font-mono text-emerald-300 border border-slate-700">
                {videoFileName}
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            {onQuickSimulatePitch && (
              <button
                type="button"
                onClick={onQuickSimulatePitch}
                className="rounded-xl bg-slate-800 px-3 py-2 text-xs font-bold text-sky-300 border border-sky-800/60 hover:bg-slate-700 transition"
              >
                ⚡ 隨機實測一球示範
              </button>
            )}
            {videoUrl && (
              <button
                type="button"
                onClick={toggleVideoPlay}
                className="rounded-xl bg-emerald-500 px-4 py-2 text-xs font-black text-slate-950 hover:bg-emerald-400 transition"
              >
                {isPlaying ? '⏸ 暫停並結算球速' : '▶ 播放 / 重測'}
              </button>
            )}
          </div>
        </div>
      )}

      <div
        className={
          fullHeight
            ? 'relative flex h-full w-full items-center justify-center overflow-hidden bg-black'
            : 'relative flex w-full items-center justify-center overflow-hidden rounded-2xl bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 border border-slate-800 shadow-2xl min-h-[260px]'
        }
      >
        <div className="relative shrink-0" style={stageStyle}>
          <video
            ref={videoRef}
            src={mode === 'video' ? videoUrl ?? undefined : undefined}
            playsInline
            autoPlay={mode === 'camera'}
            muted
            onLoadedMetadata={handleLoadedMetadata}
            onPlay={() => setIsPlaying(true)}
            onPause={() => {
              setIsPlaying(false);
              onTrackEnd();
            }}
            onEnded={() => {
              setIsPlaying(false);
              onTrackEnd();
            }}
            className="h-full w-full object-fill bg-slate-950"
          />

          {/* 球場雷達網格背景（當未播放真實視訊時提供專業空間感，且不遮擋四點校正框） */}
          {(mode === 'lab' || (mode === 'video' && !videoUrl) || (mode === 'camera' && cameraError)) && (
            <div
              className="pointer-events-none absolute inset-0 opacity-35"
              style={{
                backgroundImage:
                  'radial-gradient(circle at 50% 50%, rgba(16, 185, 129, 0.14), transparent 70%), linear-gradient(to right, rgba(51, 65, 85, 0.25) 1px, transparent 1px), linear-gradient(to bottom, rgba(51, 65, 85, 0.25) 1px, transparent 1px)',
                backgroundSize: '100% 100%, 40px 40px, 40px 40px',
              }}
            />
          )}

          {children}
        </div>

        {mode === 'video' && fullHeight && (
          <div
            className="fixed z-50 flex items-center gap-2"
            style={{
              bottom: 'max(0.75rem, env(safe-area-inset-bottom))',
              left: 'max(0.75rem, env(safe-area-inset-left))',
            }}
          >
            <label className="cursor-pointer rounded-xl bg-emerald-600/90 px-3 py-1.5 text-xs font-bold text-white backdrop-blur-md shadow-lg">
              換片
              <input
                type="file"
                accept="video/*"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
            {videoUrl && (
              <button
                type="button"
                onClick={toggleVideoPlay}
                className="rounded-xl bg-slate-900/80 px-3 py-1.5 text-xs font-bold text-white backdrop-blur-md border border-slate-700 shadow-lg"
              >
                {isPlaying ? '暫停' : '播放 / 重測'}
              </button>
            )}
          </div>
        )}

        {mode === 'camera' && cameraError && (
          <div className="pointer-events-none absolute bottom-4 inset-x-4 flex justify-center">
            <div className="pointer-events-auto flex flex-wrap items-center justify-center gap-3 rounded-2xl bg-slate-950/90 px-4 py-2.5 text-center backdrop-blur-md border border-amber-500/40 shadow-xl">
              <p className="text-xs text-amber-300">{cameraError}</p>
              {onQuickSimulatePitch && (
                <button
                  type="button"
                  onClick={onQuickSimulatePitch}
                  className="rounded-lg bg-emerald-600 px-3 py-1 text-xs font-bold text-white hover:bg-emerald-500 transition"
                >
                  ⚡ 立即播放 100 球實測示範
                </button>
              )}
            </div>
          </div>
        )}

        {mode === 'video' && !videoUrl && (
          <div className="pointer-events-none absolute bottom-4 inset-x-4 flex justify-center">
            <div className="rounded-2xl bg-slate-950/85 px-4 py-2 text-center backdrop-blur-md border border-slate-700/80 shadow-xl">
              <p className="text-xs text-slate-300">
                可自由拖曳上方四個綠點校正 <strong className="text-emerald-400">18.44m 垂直飛行切面</strong>，或選擇慢動作影片開始測速
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
