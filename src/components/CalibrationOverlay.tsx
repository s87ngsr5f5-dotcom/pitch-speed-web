import React, { useRef, useState } from 'react';
import { BallDetection, CalibrationConfig, Point2D, SpeedResult } from '../types';
import { HomographyCalculator } from '../core/homography';

interface CalibrationOverlayProps {
  videoWidth: number;
  videoHeight: number;
  config: CalibrationConfig;
  onChange: (next: CalibrationConfig) => void;
  locked: boolean;
  homography: HomographyCalculator | null;
  latestBall: BallDetection | null;
  latestResult: SpeedResult | null;
  activeSimLabel?: string | null;
}

const CORNER_LABELS = ['投手頂 (0m)', '本壘頂', '本壘底', '投手底 (0m)'];

export const CalibrationOverlay: React.FC<CalibrationOverlayProps> = ({
  videoWidth,
  videoHeight,
  config,
  onChange,
  locked,
  homography,
  latestBall,
  latestResult,
  activeSimLabel,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [draggingIdx, setDraggingIdx] = useState<number | null>(null);

  if (videoWidth <= 0 || videoHeight <= 0) return null;

  const scaleUnit = Math.max(videoWidth, videoHeight) / 1000;
  const handleRadius = 21 * scaleUnit;
  const strokeW = 2.4 * scaleUnit;
  const fontSize = 16.5 * scaleUnit;

  const clientToVideoPoint = (clientX: number, clientY: number): Point2D => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const rect = svg.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * videoWidth;
    const y = ((clientY - rect.top) / rect.height) * videoHeight;
    return {
      x: Math.max(0, Math.min(videoWidth, x)),
      y: Math.max(0, Math.min(videoHeight, y)),
    };
  };

  const handlePointerDown = (idx: number, e: React.PointerEvent<SVGGElement>) => {
    if (locked) return;
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    setDraggingIdx(idx);
  };

  const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (locked || draggingIdx === null) return;
    const pt = clientToVideoPoint(e.clientX, e.clientY);
    const nextPoints = [...config.imagePoints] as CalibrationConfig['imagePoints'];
    nextPoints[draggingIdx] = pt;
    onChange({ ...config, imagePoints: nextPoints });
  };

  const handlePointerUp = () => {
    if (draggingIdx !== null) setDraggingIdx(null);
  };

  const gridLines: Array<{ x1: number; y1: number; x2: number; y2: number; label?: string }> = [];
  if (homography) {
    const { distanceMeters, referenceHeightMeters } = config;
    for (let i = 1; i < 4; i++) {
      const dist = (distanceMeters * i) / 4;
      const top = homography.worldToPixel({ x: dist, y: referenceHeightMeters });
      const bot = homography.worldToPixel({ x: dist, y: 0 });
      if (Number.isFinite(top.x) && Number.isFinite(bot.x)) {
        gridLines.push({
          x1: top.x,
          y1: top.y,
          x2: bot.x,
          y2: bot.y,
          label: `${dist.toFixed(1)}m`,
        });
      }
    }
    const midLeft = homography.worldToPixel({ x: 0, y: referenceHeightMeters * 0.5 });
    const midRight = homography.worldToPixel({
      x: distanceMeters,
      y: referenceHeightMeters * 0.5,
    });
    if (Number.isFinite(midLeft.x) && Number.isFinite(midRight.x)) {
      gridLines.push({
        x1: midLeft.x,
        y1: midLeft.y,
        x2: midRight.x,
        y2: midRight.y,
      });
    }
  }

  const trajectoryPixelPts =
    homography && latestResult
      ? latestResult.trajectoryWorld
          .map((wp) => homography.worldToPixel(wp))
          .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
      : [];

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${videoWidth} ${videoHeight}`}
      className="absolute inset-0 h-full w-full touch-none select-none"
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
    >
      <defs>
        <linearGradient id="planeFill" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#10b981" stopOpacity="0.16" />
          <stop offset="50%" stopColor="#0ea5e9" stopOpacity="0.10" />
          <stop offset="100%" stopColor="#10b981" stopOpacity="0.18" />
        </linearGradient>
        <linearGradient id="trajGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#38bdf8" />
          <stop offset="45%" stopColor="#facc15" />
          <stop offset="100%" stopColor="#f97316" />
        </linearGradient>
      </defs>

      <polygon
        points={config.imagePoints.map((p) => `${p.x},${p.y}`).join(' ')}
        fill="url(#planeFill)"
        stroke={locked ? 'rgba(52, 211, 153, 0.78)' : '#34d399'}
        strokeWidth={strokeW * 1.15}
        strokeDasharray={locked ? `${8 * scaleUnit} ${6 * scaleUnit}` : undefined}
      />

      {gridLines.map((line, idx) => (
        <g key={idx}>
          <line
            x1={line.x1}
            y1={line.y1}
            x2={line.x2}
            y2={line.y2}
            stroke="rgba(52, 211, 153, 0.55)"
            strokeWidth={strokeW * 0.75}
            strokeDasharray={`${5 * scaleUnit} ${5 * scaleUnit}`}
          />
          {line.label && (
            <text
              x={line.x2}
              y={line.y2 + fontSize * 1.35}
              fill="#6ee7b7"
              fontSize={fontSize * 0.85}
              textAnchor="middle"
              fontWeight="700"
              style={{ textShadow: '0 1px 4px rgba(0,0,0,0.95)' }}
            >
              {line.label}
            </text>
          )}
        </g>
      ))}

      {trajectoryPixelPts.length >= 2 && (
        <g>
          <polyline
            points={trajectoryPixelPts.map((p) => `${p.x},${p.y}`).join(' ')}
            fill="none"
            stroke="rgba(250, 204, 21, 0.28)"
            strokeWidth={strokeW * 4.2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <polyline
            points={trajectoryPixelPts.map((p) => `${p.x},${p.y}`).join(' ')}
            fill="none"
            stroke="url(#trajGrad)"
            strokeWidth={strokeW * 1.9}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {trajectoryPixelPts.map((pt, i) =>
            i % Math.max(1, Math.floor(trajectoryPixelPts.length / 18)) === 0 ||
            i === trajectoryPixelPts.length - 1 ? (
              <circle
                key={i}
                cx={pt.x}
                cy={pt.y}
                r={i === trajectoryPixelPts.length - 1 ? 5.5 * scaleUnit : 3.2 * scaleUnit}
                fill={i === trajectoryPixelPts.length - 1 ? '#f97316' : '#fef08a'}
                stroke="#020617"
                strokeWidth={1.2 * scaleUnit}
              />
            ) : null
          )}
        </g>
      )}

      {latestBall && (
        <g>
          <circle
            cx={latestBall.x}
            cy={latestBall.y}
            r={Math.max(latestBall.width, latestBall.height) * 0.9}
            fill="rgba(239, 68, 68, 0.18)"
            stroke="#f87171"
            strokeWidth={strokeW * 1.1}
            strokeDasharray={`${4 * scaleUnit} ${3 * scaleUnit}`}
          />
          <circle
            cx={latestBall.x}
            cy={latestBall.y}
            r={Math.max(latestBall.width, latestBall.height) * 0.55}
            fill="rgba(251, 191, 36, 0.32)"
            stroke="#ef4444"
            strokeWidth={strokeW * 1.5}
          />
          <circle cx={latestBall.x} cy={latestBall.y} r={3.8 * scaleUnit} fill="#ffffff" />
          <text
            x={latestBall.x}
            y={latestBall.y - Math.max(latestBall.width, latestBall.height) - 8 * scaleUnit}
            fill="#fde047"
            fontSize={fontSize * 0.8}
            fontWeight="700"
            textAnchor="middle"
            style={{ textShadow: '0 1px 4px rgba(0,0,0,0.95)' }}
          >
            ⚾ {Math.round(latestBall.confidence * 100)}%
          </text>
        </g>
      )}

      {activeSimLabel && (
        <g transform={`translate(${24 * scaleUnit}, ${34 * scaleUnit})`}>
          <rect
            x={0}
            y={0}
            width={430 * scaleUnit}
            height={38 * scaleUnit}
            rx={10 * scaleUnit}
            fill="rgba(2, 6, 23, 0.82)"
            stroke="rgba(16, 185, 129, 0.55)"
            strokeWidth={1.5 * scaleUnit}
          />
          <text
            x={14 * scaleUnit}
            y={24 * scaleUnit}
            fill="#a7f3d0"
            fontSize={fontSize * 0.88}
            fontWeight="700"
          >
            {activeSimLabel}
          </text>
        </g>
      )}

      {!locked &&
        config.imagePoints.map((pt, idx) => {
          const isActive = draggingIdx === idx;
          const labelText =
            idx === 1
              ? `本壘頂 (${config.distanceMeters}m)`
              : idx === 2
                ? `本壘底 (${config.distanceMeters}m)`
                : CORNER_LABELS[idx];
          return (
            <g
              key={idx}
              onPointerDown={(e) => handlePointerDown(idx, e)}
              className="cursor-grab active:cursor-grabbing"
            >
              <circle cx={pt.x} cy={pt.y} r={handleRadius * 1.7} fill="transparent" />
              <circle
                cx={pt.x}
                cy={pt.y}
                r={handleRadius}
                fill={isActive ? 'rgba(250, 204, 21, 0.92)' : 'rgba(16, 185, 129, 0.88)'}
                stroke="#ffffff"
                strokeWidth={strokeW}
              />
              <line
                x1={pt.x - 7 * scaleUnit}
                y1={pt.y}
                x2={pt.x + 7 * scaleUnit}
                y2={pt.y}
                stroke="#020617"
                strokeWidth={2 * scaleUnit}
              />
              <line
                x1={pt.x}
                y1={pt.y - 7 * scaleUnit}
                x2={pt.x}
                y2={pt.y + 7 * scaleUnit}
                stroke="#020617"
                strokeWidth={2 * scaleUnit}
              />
              <text
                x={pt.x}
                y={pt.y - handleRadius - 9 * scaleUnit}
                fill={isActive ? '#fde047' : '#ffffff'}
                fontSize={fontSize}
                fontWeight="700"
                textAnchor="middle"
                style={{ textShadow: '0 2px 6px rgba(0,0,0,0.95)' }}
              >
                {labelText}
              </text>
            </g>
          );
        })}
    </svg>
  );
};
