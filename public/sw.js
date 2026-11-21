// 版本號：更新前端 UI 時改 APP_CACHE；重新訓練替換 .onnx 模型時改 AI_CACHE
const APP_CACHE = 'pitchspeed-app-v3';
const AI_CACHE = 'pitchspeed-ai-v1';
const VALID_CACHES = [APP_CACHE, AI_CACHE];

// 透過當前 Service Worker 作用域動態解析子路徑（相容 GitHub Pages /<repo-name>/）
const scopeUrl = (relativePath) => new URL(relativePath, self.registration.scope).href;

const CORE_SHELL_FILES = ['./', './index.html', './manifest.json'];
const OPTIONAL_ICON_FILES = [
  './icons/apple-touch-icon.png',
  './icons/icon-192.png',
  './icons/icon-512.png',
];
const PRECACHE_MODEL_FILES = ['./models/ball_yolo11n.onnx'];

/**
 * 單檔安全快取：忽略 404 或 HTTP 206，避免單一選用檔案缺失導致整個 SW 安裝失敗
 */
async function safeCacheUrl(cache, url) {
  try {
    const req = new Request(url, { cache: 'reload' });
    const res = await fetch(req);
    if (res.ok && res.status === 200) {
      await cache.put(url, res);
    }
  } catch {
    // 離線或檔案尚未產出時靜默略過，改於首次實際請求時動態快取
  }
}

self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      const appCache = await caches.open(APP_CACHE);
      await Promise.all(
        [...CORE_SHELL_FILES, ...OPTIONAL_ICON_FILES].map((p) =>
          safeCacheUrl(appCache, scopeUrl(p))
        )
      );

      const aiCache = await caches.open(AI_CACHE);
      await Promise.all(
        PRECACHE_MODEL_FILES.map((p) => safeCacheUrl(aiCache, scopeUrl(p)))
      );

      await self.skipWaiting();
    })()
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((key) => !VALID_CACHES.includes(key))
          .map((oldKey) => caches.delete(oldKey))
      );
      await self.clients.claim();
    })()
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (!url.href.startsWith(self.registration.scope)) return;

  const pathname = url.pathname;

  // 策略 A：ONNX 模型與 WASM 執行檔 -> Cache-First
  if (
    pathname.endsWith('.onnx') ||
    pathname.endsWith('.wasm') ||
    pathname.includes('/wasm/') ||
    pathname.includes('/models/')
  ) {
    event.respondWith(cacheFirst(request, AI_CACHE));
    return;
  }

  // 策略 B：頁面導覽 (HTML) -> Network-First
  if (request.mode === 'navigate') {
    event.respondWith(networkFirstNavigation(request));
    return;
  }

  // 策略 C：Vite 帶雜湊打包資源 (/assets/*) -> Cache-First
  if (pathname.includes('/assets/')) {
    event.respondWith(cacheFirst(request, APP_CACHE));
    return;
  }

  // 策略 D：其餘靜態檔案 -> Stale-While-Revalidate
  event.respondWith(staleWhileRevalidate(request, APP_CACHE));
});

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) return cached;

  // 快取未命中時加上 cache: 'reload'，確保強制更新時繞過 GitHub Pages HTTP 快取
  const res = await fetch(new Request(request, { cache: 'reload' }));
  if (res.ok && res.status === 200) {
    cache.put(request, res.clone());
  }
  return res;
}

async function networkFirstNavigation(request) {
  const cache = await caches.open(APP_CACHE);
  try {
    const res = await fetch(request);
    if (res.ok && res.status === 200) {
      cache.put(scopeUrl('./index.html'), res.clone());
    }
    return res;
  } catch {
    return (
      (await cache.match(request)) ||
      (await cache.match(scopeUrl('./index.html'))) ||
      (await cache.match(scopeUrl('./')))
    );
  }
}

async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);

  const networkPromise = fetch(request)
    .then((res) => {
      if (res.ok && res.status === 200) {
        cache.put(request, res.clone());
      }
      return res;
    })
    .catch(() => cached);

  return cached || networkPromise;
}

self.addEventListener('message', (event) => {
  if (event.data?.type === 'SKIP_WAITING') {
    self.skipWaiting();
  } else if (event.data?.type === 'CLEAR_AI_CACHE') {
    event.waitUntil(
      caches.delete(AI_CACHE).then((deleted) => {
        event.ports[0]?.postMessage({ ok: true, deleted });
      })
    );
  }
});
