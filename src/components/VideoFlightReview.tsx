import { useEffect, useRef, useState } from 'react';
import { measureVideoFlight } from '../core/videoFlightTime';

export function VideoFlightReview({ url, onClose }: { url: string; onClose: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const presentedTime = useRef<number | null>(null);
  const [duration, setDuration] = useState(0);
  const [time, setTime] = useState(0);
  const [seeking, setSeeking] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [release, setRelease] = useState<number | null>(null);
  const [arrival, setArrival] = useState<number | null>(null);
  const [distance, setDistance] = useState('');
  const [timeExpansion, setTimeExpansion] = useState(1);

  useEffect(() => {
    const video = videoRef.current;
    if (!video?.requestVideoFrameCallback) return;
    let id = video.requestVideoFrameCallback(function observe(_, frame) {
      presentedTime.current = frame.mediaTime;
      setTime(frame.mediaTime);
      id = video.requestVideoFrameCallback(observe);
    });
    return () => video.cancelVideoFrameCallback(id);
  }, []);

  function seek(seconds: number) {
    const video = videoRef.current;
    if (!video || !duration) return;
    video.pause();
    video.currentTime = Math.max(0, Math.min(seconds, duration - 0.001));
  }

  function visibleTime() {
    return presentedTime.current ?? videoRef.current?.currentTime ?? 0;
  }

  const result = release !== null && arrival !== null && distance
    ? measureVideoFlight(release, arrival, Number(distance), timeExpansion)
    : null;
  const canMark = !seeking && !playing && duration > 0;

  return (
    <section role="dialog" aria-modal="true" aria-label="手動核對影片時間"
      className="absolute inset-0 z-50 flex flex-col gap-3 overflow-y-auto bg-slate-950 p-4 text-sm text-white">
      <div className="flex items-center justify-between">
        <h2 className="font-bold">手動核對影片時間</h2>
        <button type="button" onClick={onClose} className="rounded bg-slate-700 px-3 py-1">關閉</button>
      </div>
      <p className="text-slate-300">暫停影片，先標記球離手，再標記同一球到達捕手的位置。</p>
      <video ref={videoRef} src={url} controls muted playsInline preload="metadata"
        className="min-h-40 w-full flex-1 object-contain"
        onLoadedMetadata={(e) => setDuration(Number.isFinite(e.currentTarget.duration) ? e.currentTarget.duration : 0)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onSeeking={() => { presentedTime.current = null; setSeeking(true); }}
        onSeeked={(e) => { setTime(presentedTime.current ?? e.currentTarget.currentTime); setSeeking(false); }}
        onTimeUpdate={(e) => { if (!e.currentTarget.requestVideoFrameCallback) setTime(e.currentTarget.currentTime); }}
      />
      <div className="flex items-center gap-2">
        <button type="button" disabled={!duration} className="rounded bg-slate-700 px-2 py-1"
          onClick={() => seek(visibleTime() - 1 / 30)}>−33 ms</button>
        <input type="range" aria-label="影片時間" min={0} max={duration || 1} step={0.001}
          value={Math.min(time, duration)} disabled={!duration} className="min-w-0 flex-1"
          onChange={(e) => seek(Number(e.currentTarget.value))} />
        <button type="button" disabled={!duration} className="rounded bg-slate-700 px-2 py-1"
          onClick={() => seek(visibleTime() + 1 / 30)}>+33 ms</button>
        <span className="tabular-nums">{time.toFixed(3)} s</span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <button type="button" disabled={!canMark} className="rounded bg-sky-700 py-2 disabled:opacity-50"
          onClick={() => { const t = visibleTime(); setRelease(t); if (arrival !== null && arrival <= t) setArrival(null); }}>
          標記出手{release === null ? '' : ` · ${release.toFixed(3)} s`}
        </button>
        <button type="button" disabled={!canMark || release === null || time <= release} className="rounded bg-sky-700 py-2 disabled:opacity-50"
          onClick={() => { const t = visibleTime(); if (release !== null && t > release) setArrival(t); }}>
          標記到達{arrival === null ? '' : ` · ${arrival.toFixed(3)} s`}
        </button>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <label>實際飛行距離（公尺）
          <input type="number" inputMode="decimal" min={0.1} step="any" value={distance} placeholder="請輸入量測距離"
            onChange={(e) => setDistance(e.currentTarget.value)} className="mt-1 w-full rounded bg-slate-800 px-2 py-1" />
        </label>
        <label>原始影片時間
          <select value={timeExpansion} onChange={(e) => setTimeExpansion(Number(e.currentTarget.value))}
            className="mt-1 w-full rounded bg-slate-800 px-2 py-1">
            <option value={1}>原速</option>
            <option value={2}>已放慢 2 倍</option>
            <option value={4}>已放慢 4 倍</option>
            <option value={8}>已放慢 8 倍</option>
          </select>
        </label>
      </div>
      <p role="status" className="text-slate-200">
        {result ? `平均球速約 ${result.averageKmh.toFixed(1)} km/h · 飛行 ${(result.flightSeconds * 1000).toFixed(0)} ms` :
          release !== null && arrival !== null && distance ? '請確認飛行距離、事件時間和影片倍率。' :
          '需要出手、到達時間和兩點間的實際距離。'}
      </p>
    </section>
  );
}
