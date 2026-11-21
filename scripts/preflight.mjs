import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const ROOT = process.cwd();
const SKIP_BUILD = process.argv.includes('--static-only');

const C = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  dim: '\x1b[2m',
};

let errorCount = 0;
let warnCount = 0;

function section(title) {
  console.log(`\n${C.cyan}${C.bold}=== ${title} ===${C.reset}`);
}
function pass(msg) {
  console.log(`  ${C.green}✓${C.reset} ${msg}`);
}
function warn(msg) {
  warnCount += 1;
  console.log(`  ${C.yellow}⚠${C.reset} ${msg}`);
}
function fail(msg) {
  errorCount += 1;
  console.log(`  ${C.red}✗${C.reset} ${msg}`);
}

function runNpm(args, label) {
  console.log(`\n  ${C.dim}> npm ${args.join(' ')}${C.reset}`);
  const res = spawnSync('npm', args, {
    cwd: ROOT,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  });
  if (res.status === 0) {
    pass(`${label} 通過`);
    return true;
  }
  fail(`${label} 失敗（退出碼 ${res.status ?? 'unknown'}）`);
  return false;
}

function checkGitIgnored(relativePath) {
  const res = spawnSync('git', ['check-ignore', '-q', relativePath], {
    cwd: ROOT,
    stdio: 'ignore',
  });
  return res.status === 0;
}

function readVarint(buf, offset, end) {
  let value = 0n;
  let shift = 0n;
  let pos = offset;
  while (pos < end) {
    const byte = buf[pos++];
    value |= BigInt(byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) {
      return { value, nextOffset: pos };
    }
    shift += 7n;
    if (shift > 70n) throw new Error('Protobuf varint 格式損毀');
  }
  throw new Error('讀取 Protobuf varint 時意外到達檔案結尾');
}

function iterateProtoFields(buf, start, end, onField) {
  let pos = start;
  while (pos < end) {
    const { value: tag, nextOffset } = readVarint(buf, pos, end);
    const fieldNumber = Number(tag >> 3n);
    const wireType = Number(tag & 0x7n);
    pos = nextOffset;

    if (wireType === 0) {
      const { value, nextOffset: after } = readVarint(buf, pos, end);
      onField(fieldNumber, wireType, pos, after, value);
      pos = after;
    } else if (wireType === 1) {
      pos += 8;
    } else if (wireType === 2) {
      const { value: len, nextOffset: dataStart } = readVarint(buf, pos, end);
      const dataEnd = dataStart + Number(len);
      if (dataEnd > end) throw new Error('Protobuf 區段長度超出檔案邊界');
      onField(fieldNumber, wireType, dataStart, dataEnd, null);
      pos = dataEnd;
    } else if (wireType === 5) {
      pos += 4;
    } else {
      throw new Error(`不支援的 Protobuf wireType: ${wireType}`);
    }
  }
}

function parseTensorShape(buf, start, end) {
  const dims = [];
  iterateProtoFields(buf, start, end, (field, wireType, s, e) => {
    if (field === 1 && wireType === 2) {
      let dim = null;
      iterateProtoFields(buf, s, e, (df, dw, ds, de, dval) => {
        if (df === 1 && dw === 0) dim = Number(dval);
        else if (df === 2 && dw === 2) dim = buf.toString('utf8', ds, de);
      });
      dims.push(dim);
    }
  });
  return dims;
}

function parseValueInfo(buf, start, end) {
  let name = '';
  let elemType = null;
  let dims = [];

  iterateProtoFields(buf, start, end, (field, wireType, s, e) => {
    if (field === 1 && wireType === 2) {
      name = buf.toString('utf8', s, e);
    } else if (field === 2 && wireType === 2) {
      iterateProtoFields(buf, s, e, (tf, tw, ts, te) => {
        if (tf === 1 && tw === 2) {
          iterateProtoFields(buf, ts, te, (ttf, ttw, tts, tte, ttval) => {
            if (ttf === 1 && ttw === 0) elemType = Number(ttval);
            else if (ttf === 2 && ttw === 2) dims = parseTensorShape(buf, tts, tte);
          });
        }
      });
    }
  });

  return { name, elemType, dims };
}

function inspectOnnxModel(filePath) {
  const buf = fs.readFileSync(filePath);
  const inputs = [];
  const outputs = [];

  iterateProtoFields(buf, 0, buf.length, (field, wireType, s, e) => {
    if (field === 7 && wireType === 2) {
      iterateProtoFields(buf, s, e, (gf, gw, gs, ge) => {
        if (gf === 11 && gw === 2) inputs.push(parseValueInfo(buf, gs, ge));
        else if (gf === 12 && gw === 2) outputs.push(parseValueInfo(buf, gs, ge));
      });
    }
  });

  const primaryInput =
    inputs.find((i) => i.name === 'images') ||
    inputs.find((i) => i.dims.length === 4 && i.dims[1] === 3) ||
    inputs[0] ||
    null;

  return { primaryInput, primaryOutput: outputs[0] || null };
}

function extractFrontendInputSize(appTsxPath, workerTsPath) {
  if (fs.existsSync(appTsxPath)) {
    const raw = fs.readFileSync(appTsxPath, 'utf8');
    const clean = raw.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
    const initBlockMatch = clean.match(/type\s*:\s*['"]INIT['"][\s\S]{0,400}?inputSize\s*:\s*(\d+)/);
    if (initBlockMatch) {
      return { size: Number(initBlockMatch[1]), source: 'src/App.tsx (Worker INIT)' };
    }
    const fallbackMatch = clean.match(/\binputSize\s*:\s*(\d+)/);
    if (fallbackMatch) {
      return { size: Number(fallbackMatch[1]), source: 'src/App.tsx' };
    }
  }

  if (fs.existsSync(workerTsPath)) {
    const workerRaw = fs.readFileSync(workerTsPath, 'utf8');
    const defaultMatch = workerRaw.match(/let\s+inputSize\s*=\s*(\d+)/);
    if (defaultMatch) {
      return { size: Number(defaultMatch[1]), source: 'src/workers/inference.worker.ts 預設值' };
    }
  }

  return null;
}

section('1. 檢查必備檔案與 PWA 資產');

const requiredFiles = [
  'package.json',
  'package-lock.json',
  'vite.config.ts',
  'index.html',
  '.github/workflows/deploy.yml',
  'public/manifest.json',
  'public/sw.js',
  'public/models/ball_yolo11n.onnx',
  'public/icons/apple-touch-icon.png',
  'public/icons/icon-192.png',
  'public/icons/icon-512.png',
];

for (const file of requiredFiles) {
  const full = path.join(ROOT, file);
  if (fs.existsSync(full) && fs.statSync(full).isFile()) {
    pass(`找到 ${file}`);
  } else {
    const hint =
      file === 'package-lock.json'
        ? '（請執行 npm install 產生，否則 GitHub Actions 的 npm ci 會失敗）'
        : file.includes('icons/')
          ? '（請執行 python ml_pipeline/generate_pwa_icons.py 產生）'
          : file.endsWith('.onnx')
            ? '（請執行 python ml_pipeline/export_model.py --weights yolo11n.pt 匯出）'
            : '';
    fail(`缺少必備檔案：${file} ${hint}`);
  }
}

for (const trackedFile of ['package-lock.json', 'public/models/ball_yolo11n.onnx']) {
  if (checkGitIgnored(trackedFile)) {
    fail(`${trackedFile} 正被 .gitignore 忽略！Push 上 GitHub 後會遺失此檔案`);
  }
}

section('2. 檢查 ONNX 模型體積與真實輸入/輸出維度');

const modelPath = path.join(ROOT, 'public/models/ball_yolo11n.onnx');
const appTsxPath = path.join(ROOT, 'src/App.tsx');
const workerPath = path.join(ROOT, 'src/workers/inference.worker.ts');

const frontendConfig = extractFrontendInputSize(appTsxPath, workerPath);
if (frontendConfig) {
  pass(`前端推論尺寸設定：inputSize = ${frontendConfig.size}（來源：${frontendConfig.source}）`);
} else {
  warn('無法從 src/App.tsx 或 inference.worker.ts 偵測到 inputSize 設定');
}

if (fs.existsSync(modelPath)) {
  const sizeBytes = fs.statSync(modelPath).size;
  const sizeMb = sizeBytes / (1024 * 1024);

  if (sizeBytes < 100 * 1024) {
    fail(`ONNX 模型檔案異常過小（${sizeBytes} bytes），疑似空檔或下載中斷`);
  } else if (sizeMb >= 100) {
    fail(`ONNX 模型為 ${sizeMb.toFixed(2)} MB，超過 GitHub 單檔 100 MB 硬上限`);
  } else if (sizeMb > 18) {
    warn(`ONNX 模型為 ${sizeMb.toFixed(2)} MB（建議壓在 15 MB 以下以加速手機首次載入）`);
  } else {
    pass(`ONNX 模型大小合規：${sizeMb.toFixed(2)} MB`);
  }

  try {
    const { primaryInput, primaryOutput } = inspectOnnxModel(modelPath);

    if (!primaryInput || primaryInput.dims.length !== 4) {
      fail(
        `無法從 ONNX 讀取 4D NCHW 輸入張量（讀到：${JSON.stringify(primaryInput?.dims ?? null)}）`
      );
    } else {
      const [batch, channels, modelH, modelW] = primaryInput.dims;
      const shapeStr = `[${primaryInput.dims.join(', ')}]`;

      if (primaryInput.elemType === 1) {
        pass(`ONNX 輸入張量 "${primaryInput.name}" 型別為 FLOAT32，維度 ${shapeStr}`);
      } else {
        fail(
          `ONNX 輸入型別代碼為 ${primaryInput.elemType}（非 FLOAT32=1），前端 inference.worker.ts 會推論失敗`
        );
      }

      if (channels !== 3) {
        fail(`ONNX 輸入通道數為 ${channels}（預期為 3 通道 RGB）`);
      }

      if (typeof modelH !== 'number' || typeof modelW !== 'number') {
        warn(
          `ONNX 輸入高寬為動態維度 (${modelH} x ${modelW})，建議匯出時使用 dynamic=False 以提升瀏覽器推論效能`
        );
      } else if (modelH !== modelW) {
        fail(`ONNX 模型輸入非正方形 (${modelH}x${modelW})，與前端正方形縮放假設不符`);
      } else if (frontendConfig && frontendConfig.size !== modelH) {
        fail(
          `輸入尺寸不匹配！ONNX 模型真實尺寸為 ${modelH}x${modelW}，但 ${frontendConfig.source} 設定為 inputSize: ${frontendConfig.size}。` +
            `請將 src/App.tsx 改為 inputSize: ${modelH}，或重新以 --imgsz ${frontendConfig.size} 匯出模型！`
        );
      } else if (frontendConfig) {
        pass(`ONNX 真實輸入尺寸 (${modelH}x${modelW}) 與前端 inputSize (${frontendConfig.size}) 完全一致`);
      }

      if (typeof batch === 'number' && batch !== 1) {
        warn(`ONNX batch 維度為 ${batch}（預期為 1）`);
      }
    }

    if (primaryOutput && primaryOutput.dims.length === 3 && typeof primaryOutput.dims[1] === 'number') {
      const numClasses = primaryOutput.dims[1] - 4;
      const outShapeStr = `[${primaryOutput.dims.join(', ')}]`;
      if (numClasses === 1) {
        pass(`ONNX 輸出維度 ${outShapeStr}（自訓單類別棒球模型，Worker 自動讀取 class 0）`);
      } else if (numClasses === 80) {
        pass(`ONNX 輸出維度 ${outShapeStr}（COCO 預訓練模型，Worker 自動讀取 class 32 sports ball）`);
      } else {
        fail(
          `ONNX 輸出類別數為 ${numClasses}（維度 ${outShapeStr}），非單類別(1)或 COCO(80)！請用 train_baseball.py 清洗為單類別`
        );
      }
    }
  } catch (err) {
    fail(`解析 ONNX 模型內部結構失敗（檔案可能已損毀）：${err.message}`);
  }
}

section('3. 檢查 PWA manifest.json 相對路徑');

const manifestPath = path.join(ROOT, 'public/manifest.json');
if (fs.existsSync(manifestPath)) {
  try {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

    for (const key of ['start_url', 'scope']) {
      const val = manifest[key];
      if (typeof val !== 'string' || !val.startsWith('./')) {
        fail(`manifest.json 的 "${key}" 為 "${val}"，GitHub Pages 子路徑必須使用 "./"`);
      } else {
        pass(`manifest.${key} = "${val}"`);
      }
    }

    if (!Array.isArray(manifest.icons) || manifest.icons.length === 0) {
      fail('manifest.json 缺少 icons 陣列');
    } else {
      for (const icon of manifest.icons) {
        if (typeof icon.src !== 'string' || icon.src.startsWith('/')) {
          fail(`manifest icon 路徑 "${icon.src}" 不可使用開頭斜線 "/"，請改為 "./icons/..."`);
        } else {
          const iconFull = path.join(ROOT, 'public', icon.src.replace(/^\.\//, ''));
          if (fs.existsSync(iconFull)) {
            pass(`manifest icon 路徑有效：${icon.src} (${icon.sizes ?? 'unknown'})`);
          } else {
            fail(`manifest 參照的圖示不存在：public/${icon.src.replace(/^\.\//, '')}`);
          }
        }
      }
    }
  } catch (err) {
    fail(`public/manifest.json JSON 格式解析失敗：${err.message}`);
  }
}

section('4. 檢查 Vite 與 iOS / GitHub Pages 防雷設定');

const viteConfigPath = path.join(ROOT, 'vite.config.ts');
if (fs.existsSync(viteConfigPath)) {
  const viteText = fs.readFileSync(viteConfigPath, 'utf8');
  if (/base\s*:\s*['"]\.\/['"]/.test(viteText)) {
    pass("vite.config.ts 已設定 base: './'");
  } else {
    fail("vite.config.ts 缺少 base: './'，部署至 GitHub Pages 子路徑時資源會 404");
  }
  if (/viteStaticCopy/.test(viteText) && /onnxruntime-web/.test(viteText)) {
    pass('vite.config.ts 已配置 onnxruntime-web WASM 複製外掛');
  } else {
    fail('vite.config.ts 缺少 viteStaticCopy 複製 onnxruntime-web WASM 設定');
  }
}

if (fs.existsSync(workerPath)) {
  const workerText = fs.readFileSync(workerPath, 'utf8');
  if (/numThreads\s*=\s*1/.test(workerText)) {
    pass('inference.worker.ts 已強制 ort.env.wasm.numThreads = 1（相容 GitHub Pages）');
  } else {
    fail('inference.worker.ts 未設定 numThreads = 1，在 GitHub Pages 會因缺少 SharedArrayBuffer 崩潰');
  }
  if (/\.bitmap\.close\(\)/.test(workerText)) {
    pass('inference.worker.ts 已釋放 ImageBitmap（防 iOS 記憶體閃退）');
  } else {
    warn('inference.worker.ts 未偵測到 bitmap.close()，長時間在 iOS 推論可能導致記憶體爆掉');
  }
}

const cameraPath = path.join(ROOT, 'src/components/CameraStream.tsx');
if (fs.existsSync(cameraPath)) {
  const camText = fs.readFileSync(cameraPath, 'utf8');
  if (/\bplaysInline\b/.test(camText) && /\bmuted\b/.test(camText)) {
    pass('CameraStream.tsx 已具備 iOS Safari 必需的 playsInline 與 muted 屬性');
  } else {
    fail('CameraStream.tsx 缺少 playsInline 或 muted，iOS Safari 將無法內嵌播放鏡頭串流');
  }
}

if (!SKIP_BUILD) {
  section('5. 執行 TypeScript、Vitest 與 Vite 正式打包');

  const tscOk = runNpm(['run', 'typecheck'], 'TypeScript 型別檢查 (tsc --noEmit)');
  const testOk = runNpm(['test'], 'Vitest 物理與單應性單元測試');
  const buildOk = tscOk ? runNpm(['run', 'build'], 'Vite 正式打包 (npm run build)') : false;

  if (buildOk) {
    section('6. 驗證 dist/ 打包產物與 WASM 檔案');

    const distMustExist = [
      'dist/index.html',
      'dist/manifest.json',
      'dist/sw.js',
      'dist/models/ball_yolo11n.onnx',
      'dist/icons/apple-touch-icon.png',
    ];

    for (const item of distMustExist) {
      if (fs.existsSync(path.join(ROOT, item))) {
        pass(`打包產物存在：${item}`);
      } else {
        fail(`打包產物遺失：${item}`);
      }
    }

    const wasmDir = path.join(ROOT, 'dist/wasm');
    if (!fs.existsSync(wasmDir) || !fs.statSync(wasmDir).isDirectory()) {
      fail('缺少 dist/wasm/ 目錄，請檢查 vite-plugin-static-copy 設定');
    } else {
      const filesInWasm = fs.readdirSync(wasmDir);
      const wasmFiles = filesInWasm.filter((f) => f.endsWith('.wasm'));
      const mjsFiles = filesInWasm.filter((f) => f.endsWith('.mjs'));

      if (wasmFiles.length > 0) {
        pass(`dist/wasm/ 包含 ${wasmFiles.length} 個 .wasm 檔案（如 ${wasmFiles[0]}）`);
      } else {
        fail('dist/wasm/ 中找不到任何 .wasm 執行檔');
      }

      if (mjsFiles.length > 0) {
        pass(`dist/wasm/ 包含 ${mjsFiles.length} 個 .mjs 載入檔（如 ${mjsFiles[0]}）`);
      } else {
        warn('dist/wasm/ 中未發現 .mjs 載入檔');
      }
    }
  } else if (!testOk || !tscOk) {
    warn('因前置檢查或建置未通過，略過 dist/ 產物檢查');
  }
} else {
  section('5. 已略過編譯測試（--static-only 模式）');
}

console.log(`\n${C.bold}========================================${C.reset}`);
if (errorCount === 0) {
  console.log(
    `${C.green}${C.bold}PREFLIGHT PASS${C.reset}（0 錯誤，${warnCount} 警告）— 可以安心 git push 上線 GitHub Pages！`
  );
  process.exit(0);
} else {
  console.log(
    `${C.red}${C.bold}PREFLIGHT FAILED${C.reset}（共 ${errorCount} 項錯誤，${warnCount} 項警告）— 請先修復上方 ✗ 項目再部署。`
  );
  process.exit(1);
}
