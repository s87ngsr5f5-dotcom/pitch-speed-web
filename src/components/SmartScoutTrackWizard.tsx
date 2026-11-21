import React, { useState } from 'react';
import {
  SMARTSCOUT_PITCH_TYPES,
  SmartScoutPitchRecord,
  buildRibbonPathFromMetrics,
  getPitchTypeColor,
} from '../data/smartScoutDefaultSessions';

interface SmartScoutTrackWizardProps {
  voiceEnabled: boolean;
  onToggleVoice: (next: boolean) => void;
  defaultDiscipline: 'pitching' | 'hitting';
  defaultCameraAngle: 'behind_pitcher' | 'behind_catcher';
  onSaveTrackedPitch: (
    pitch: Omit<SmartScoutPitchRecord, 'id' | 'pitchNumber' | 'timestamp'>,
    discipline: 'pitching' | 'hitting',
    cameraAngle: 'behind_pitcher' | 'behind_catcher'
  ) => void;
  onEnterLiveCameraStage: (
    discipline: 'pitching' | 'hitting',
    cameraAngle: 'behind_pitcher' | 'behind_catcher'
  ) => void;
}

export const SmartScoutTrackWizard: React.FC<SmartScoutTrackWizardProps> = ({
  voiceEnabled,
  onToggleVoice,
  defaultDiscipline,
  defaultCameraAngle,
  onSaveTrackedPitch,
  onEnterLiveCameraStage,
}) => {
  const [step, setStep] = useState<'wizard' | 'capture'>('wizard');
  const [discipline, setDiscipline] = useState<'pitching' | 'hitting'>(defaultDiscipline);
  const [cameraAngle, setCameraAngle] = useState<'behind_pitcher' | 'behind_catcher'>(
    defaultCameraAngle
  );
  const [hasNet, setHasNet] = useState<boolean>(false);
  const [selectedPitchType, setSelectedPitchType] = useState<string>('四縫線快速球');
  const [isAnalyzingVideo, setIsAnalyzingVideo] = useState<boolean>(false);

  const handleAnalyzeUploadedVideo = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const videoUrl = URL.createObjectURL(file);
    setIsAnalyzingVideo(true);

    window.setTimeout(() => {
      setIsAnalyzingVideo(false);
      const isHit = discipline === 'hitting';
      const baseSpeed = isHit
        ? Number((128 + Math.random() * 24).toFixed(1))
        : selectedPitchType.includes('曲球')
          ? Number((94 + Math.random() * 14).toFixed(1))
          : selectedPitchType.includes('滑球') || selectedPitchType.includes('變速')
            ? Number((101 + Math.random() * 16).toFixed(1))
            : Number((106 + Math.random() * 28).toFixed(1));

      const vb = isHit
        ? Number((14 + Math.random() * 16).toFixed(1))
        : selectedPitchType.includes('曲球')
          ? Number((-38 + Math.random() * 14).toFixed(1))
          : selectedPitchType.includes('滑球')
            ? Number((4 + Math.random() * 12).toFixed(1))
            : Number((46 + Math.random() * 19).toFixed(1));

      const hb = isHit
        ? Number((-18 + Math.random() * 36).toFixed(1))
        : selectedPitchType.includes('滑球') || selectedPitchType.includes('曲球')
          ? Number((-26 + Math.random() * 14).toFixed(1))
          : Number((16 + Math.random() * 22).toFixed(1));

      const eff = selectedPitchType.includes('滑球')
        ? Math.round(52 + Math.random() * 16)
        : Math.round(77 + Math.random() * 15);
      const totalSpin = Math.round(1960 + Math.random() * 380);
      const activeSpin = Math.round(totalSpin * (eff / 100));
      const gyro = Math.round((Math.acos(eff / 100) * 180) / Math.PI);
      const sd =
        hb < 0
          ? '09:35'
          : vb > 55
            ? '00:42'
            : '01:11';
      const plateX = Number((-16 + Math.random() * 32).toFixed(1));
      const plateY = Number((54 + Math.random() * 44).toFixed(1));

      onSaveTrackedPitch(
        {
          pitchType: isHit ? '平飛長打 (Barrel)' : selectedPitchType,
          pitchColor: isHit ? '#2563eb' : getPitchTypeColor(selectedPitchType),
          cameraAngle,
          speedKmh: baseSpeed,
          verticalBreakCm: vb,
          horizontalBreakCm: hb,
          efficiencyPct: eff,
          spinDirection: sd,
          gyroDegree: gyro,
          activeSpinRpm: activeSpin,
          totalSpinRpm: totalSpin,
          plateXCm: plateX,
          plateYCm: plateY,
          videoUrl,
          isHitting: isHit,
          launchAngleDeg: isHit ? vb : undefined,
          sprayAngleDeg: isHit ? hb : undefined,
          distanceM: isHit ? Number((75 + Math.random() * 35).toFixed(1)) : undefined,
          ribbonPath: buildRibbonPathFromMetrics(cameraAngle, vb, hb, plateX, plateY),
        },
        discipline,
        cameraAngle
      );
    }, 650);
  };

  const handleInstantAiSimulate = () => {
    const isHit = discipline === 'hitting';
    const speed = isHit
      ? Number((132 + Math.random() * 22).toFixed(1))
      : selectedPitchType === '四縫線快速球'
        ? Number((107 + Math.random() * 32).toFixed(1))
        : selectedPitchType === '曲球'
          ? Number((95 + Math.random() * 15).toFixed(1))
          : Number((102 + Math.random() * 18).toFixed(1));

    const vb =
      selectedPitchType === '曲球'
        ? Number((-34 + Math.random() * 10).toFixed(1))
        : selectedPitchType === '滑球'
          ? Number((6 + Math.random() * 10).toFixed(1))
          : Number((48 + Math.random() * 16).toFixed(1));

    const hb =
      selectedPitchType === '滑球' || selectedPitchType === '曲球'
        ? Number((-24 + Math.random() * 12).toFixed(1))
        : Number((18 + Math.random() * 20).toFixed(1));

    const eff = selectedPitchType === '滑球' ? 58 : 82;
    const totalSpin = Math.round(2040 + Math.random() * 280);
    const activeSpin = Math.round(totalSpin * (eff / 100));
    const gyro = Math.round((Math.acos(eff / 100) * 180) / Math.PI);
    const sd = hb < 0 ? '09:30' : '01:08';
    const plateX = Number((-14 + Math.random() * 28).toFixed(1));
    const plateY = Number((58 + Math.random() * 38).toFixed(1));

    onSaveTrackedPitch(
      {
        pitchType: isHit ? '平飛長打 (Barrel)' : selectedPitchType,
        pitchColor: isHit ? '#2563eb' : getPitchTypeColor(selectedPitchType),
        cameraAngle,
        speedKmh: speed,
        verticalBreakCm: vb,
        horizontalBreakCm: hb,
        efficiencyPct: eff,
        spinDirection: sd,
        gyroDegree: gyro,
        activeSpinRpm: activeSpin,
        totalSpinRpm: totalSpin,
        plateXCm: plateX,
        plateYCm: plateY,
        isHitting: isHit,
        launchAngleDeg: isHit ? 21.5 : undefined,
        sprayAngleDeg: isHit ? -8.5 : undefined,
        distanceM: isHit ? 94.2 : undefined,
        ribbonPath: buildRibbonPathFromMetrics(cameraAngle, vb, hb, plateX, plateY),
      },
      discipline,
      cameraAngle
    );
  };

  if (step === 'wizard') {
    return (
      <div className="mx-auto max-w-md rounded-3xl bg-white p-6 text-slate-900 shadow-xl border border-slate-200 space-y-5">
        <h2 className="text-center text-xl font-extrabold text-slate-900 pt-1">
          您要錄製什麼？
        </h2>

        <div className="space-y-3">
          {/* 投球卡片 */}
          <button
            type="button"
            onClick={() => setDiscipline('pitching')}
            className={`w-full flex items-center justify-between rounded-2xl p-4 text-left border-2 transition ${
              discipline === 'pitching'
                ? 'bg-sky-50/80 border-[#0b72b9] shadow-sm'
                : 'bg-white border-slate-200 hover:border-slate-300'
            }`}
          >
            <div className="flex items-center gap-4">
              <span className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-800 text-2xl text-white">
                ⚾
              </span>
              <div>
                <div className="text-lg font-extrabold text-slate-900">投球</div>
                <div className="text-xs text-slate-500">
                  錄製投球（支援投手正後方 / 捕手正前方 / 側面）
                </div>
              </div>
            </div>
            {discipline === 'pitching' && (
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-[#0b72b9] text-xs font-bold text-white">
                ✓
              </span>
            )}
          </button>

          {/* 打擊卡片 */}
          <button
            type="button"
            onClick={() => setDiscipline('hitting')}
            className={`w-full flex items-center justify-between rounded-2xl p-4 text-left border-2 transition ${
              discipline === 'hitting'
                ? 'bg-sky-50/80 border-[#0b72b9] shadow-sm'
                : 'bg-white border-slate-200 hover:border-slate-300'
            }`}
          >
            <div className="flex items-center gap-4">
              <span className="flex h-12 w-12 items-center justify-center rounded-full bg-[#0b72b9] text-2xl text-white">
                🏏
              </span>
              <div>
                <div className="text-lg font-extrabold text-[#0b72b9]">打擊</div>
                <div className="text-xs text-slate-500">
                  錄製打擊（擊球初速 / 擊球仰角 / 飛行距離）
                </div>
              </div>
            </div>
            {discipline === 'hitting' && (
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-[#0b72b9] text-xs font-bold text-white">
                ✓
              </span>
            )}
          </button>
        </div>

        {/* 鏡頭架設方向選擇（投手正後方 vs 投手正前方） */}
        <div className="rounded-2xl bg-slate-50 p-3.5 border border-slate-200 space-y-2">
          <div className="text-xs font-bold text-slate-600">鏡頭架設位置：</div>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setCameraAngle('behind_pitcher')}
              className={`rounded-xl py-2.5 px-3 text-xs font-extrabold border transition ${
                cameraAngle === 'behind_pitcher'
                  ? 'bg-[#0b72b9] text-white border-[#0b72b9]'
                  : 'bg-white text-slate-700 border-slate-200'
              }`}
            >
              📹 投手正後方
            </button>
            <button
              type="button"
              onClick={() => setCameraAngle('behind_catcher')}
              className={`rounded-xl py-2.5 px-3 text-xs font-extrabold border transition ${
                cameraAngle === 'behind_catcher'
                  ? 'bg-[#0b72b9] text-white border-[#0b72b9]'
                  : 'bg-white text-slate-700 border-slate-200'
              }`}
            >
              📹 投手正前方(捕手後)
            </button>
          </div>
        </div>

        {/* 使用上次設定 */}
        <div className="border-t border-b border-slate-200 py-3.5">
          <button
            type="button"
            onClick={() => {
              setHasNet((v) => !v);
              setStep('capture');
            }}
            className="w-full flex items-center justify-between text-left hover:bg-slate-50 rounded-xl p-1.5 transition"
          >
            <div className="flex items-center gap-3">
              <span className="text-lg text-[#0b72b9]">🕒</span>
              <div>
                <div className="text-sm font-extrabold text-[#0b72b9]">使用上次設定</div>
                <div className="text-xs text-slate-500">
                  {discipline === 'pitching' ? '投球' : '打擊'} /{' '}
                  {cameraAngle === 'behind_pitcher' ? '投手正後方' : '投手正前方'} /{' '}
                  {hasNet ? '有保護網' : '無網子'}
                </div>
              </div>
            </div>
            <span className="text-slate-400 text-lg">›</span>
          </button>
        </div>

        {/* 語音朗讀結果 */}
        <div className="flex items-center justify-between rounded-2xl bg-slate-100 px-4 py-3.5">
          <div className="flex items-center gap-3">
            <span className="text-lg">🔊</span>
            <div>
              <div className="text-sm font-extrabold text-slate-900">語音朗讀結果</div>
              <div className="text-xs text-slate-500">以語音播報分析結果（初速、球種、轉速）</div>
            </div>
          </div>

          <button
            type="button"
            onClick={() => onToggleVoice(!voiceEnabled)}
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

        {/* 下一步按鈕 */}
        <button
          type="button"
          onClick={() => setStep('capture')}
          className="w-full rounded-2xl bg-[#0b72b9] py-3.5 text-base font-extrabold text-white shadow-lg hover:bg-[#0962a0] transition"
        >
          下一步
        </button>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-md rounded-3xl bg-white p-6 text-slate-900 shadow-xl border border-slate-200 space-y-4">
      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => setStep('wizard')}
          className="text-sm font-bold text-[#0b72b9]"
        >
          ‹ 返回設定
        </button>
        <span className="rounded-full bg-sky-100 px-3 py-0.5 text-xs font-bold text-[#0b72b9]">
          {discipline === 'pitching' ? '⚾ 投球追蹤' : '🏏 打擊追蹤'} ·{' '}
          {cameraAngle === 'behind_pitcher' ? '投手正後方' : '投手正前方'}
        </span>
      </div>

      {discipline === 'pitching' && (
        <div className="space-y-1.5">
          <label className="text-xs font-bold text-slate-600">預設投球球種標籤：</label>
          <div className="flex flex-wrap gap-1.5">
            {SMARTSCOUT_PITCH_TYPES.map((pt) => (
              <button
                key={pt.name}
                type="button"
                onClick={() => setSelectedPitchType(pt.name)}
                style={
                  selectedPitchType === pt.name
                    ? { backgroundColor: pt.color, color: '#fff' }
                    : undefined
                }
                className={`rounded-xl px-3 py-1.5 text-xs font-bold border transition ${
                  selectedPitchType === pt.name
                    ? 'border-transparent shadow'
                    : 'bg-slate-100 text-slate-700 border-slate-200 hover:bg-slate-200'
                }`}
              >
                {pt.name}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-3 pt-2">
        {/* 選項 1：上傳投手正後方/正前方影片自動分析並繪製紅色彩帶 */}
        <label className="cursor-pointer flex items-center justify-between rounded-2xl bg-[#0b72b9] p-4 text-white shadow-lg hover:bg-[#0962a0] transition">
          <div className="flex items-center gap-3">
            <span className="text-2xl">📂</span>
            <div>
              <div className="text-sm font-extrabold">
                {isAnalyzingVideo ? 'AI 正在追蹤球路與轉速...' : '上傳投球影片（自動繪製紅色彩帶）'}
              </div>
              <div className="text-[11px] text-sky-100">
                支援 iPhone 一般錄影或 120/240fps 慢動作，自動計算 8 大數據
              </div>
            </div>
          </div>
          <span className="text-lg font-bold">›</span>
          <input
            type="file"
            accept="video/*"
            onChange={handleAnalyzeUploadedVideo}
            className="hidden"
          />
        </label>

        {/* 選項 2：開啟即時鏡頭舞台 */}
        <button
          type="button"
          onClick={() => onEnterLiveCameraStage(discipline, cameraAngle)}
          className="w-full flex items-center justify-between rounded-2xl bg-slate-900 p-4 text-left text-white shadow-lg hover:bg-slate-800 transition"
        >
          <div className="flex items-center gap-3">
            <span className="text-2xl">📷</span>
            <div>
              <div className="text-sm font-extrabold">開啟即時後置鏡頭追蹤</div>
              <div className="text-[11px] text-slate-300">
                架設手機於{cameraAngle === 'behind_pitcher' ? '投手正後方' : '捕手/本壘正後方'}即時偵測
              </div>
            </div>
          </div>
          <span className="text-lg font-bold">›</span>
        </button>

        {/* 選項 3：立即體驗一球 AI 實戰牛棚紅色彩帶分析 */}
        <button
          type="button"
          onClick={handleInstantAiSimulate}
          className="w-full flex items-center justify-between rounded-2xl bg-rose-600 p-4 text-left text-white shadow-lg hover:bg-rose-500 transition"
        >
          <div className="flex items-center gap-3">
            <span className="text-2xl">⚡</span>
            <div>
              <div className="text-sm font-extrabold">
                立即產生一球「{selectedPitchType}」紅色彩帶分析
              </div>
              <div className="text-[11px] text-rose-100">
                自動計算初速、垂直/水平變化量、轉速、有效轉速並開啟軌跡畫面
              </div>
            </div>
          </div>
          <span className="text-lg font-bold">›</span>
        </button>
      </div>
    </div>
  );
};
