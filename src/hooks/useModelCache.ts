import { useCallback, useEffect, useState } from 'react';

export function useModelCache(
  modelUrl: string,
  workerStatus: 'loading' | 'ready' | 'error'
) {
  const [isModelCached, setIsModelCached] = useState<boolean | null>(null);
  const [cachedSizeMb, setCachedSizeMb] = useState<number | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(() =>
    typeof navigator !== 'undefined' ? navigator.onLine : true
  );
  const [isUpdatingModel, setIsUpdatingModel] = useState<boolean>(false);

  const checkCacheStatus = useCallback(async () => {
    if (typeof window === 'undefined' || !('caches' in window)) {
      setIsModelCached(null);
      return;
    }
    try {
      const matched = await caches.match(modelUrl);
      setIsModelCached(Boolean(matched));

      const contentLen = matched?.headers.get('content-length');
      if (contentLen) {
        const bytes = Number(contentLen);
        setCachedSizeMb(
          Number.isFinite(bytes) ? Number((bytes / (1024 * 1024)).toFixed(1)) : null
        );
      } else {
        setCachedSizeMb(null);
      }
    } catch {
      setIsModelCached(false);
    }
  }, [modelUrl]);

  useEffect(() => {
    const onOnline = () => setIsOnline(true);
    const onOffline = () => setIsOnline(false);
    window.addEventListener('online', onOnline);
    window.addEventListener('offline', onOffline);
    return () => {
      window.removeEventListener('online', onOnline);
      window.removeEventListener('offline', onOffline);
    };
  }, []);

  useEffect(() => {
    checkCacheStatus();
  }, [checkCacheStatus, workerStatus]);

  const forceUpdateModel = useCallback(async () => {
    if (!navigator.onLine || isUpdatingModel) return;
    setIsUpdatingModel(true);

    try {
      const controller = navigator.serviceWorker?.controller;
      if (controller) {
        await new Promise<void>((resolve) => {
          const channel = new MessageChannel();
          const timer = window.setTimeout(resolve, 1500);
          channel.port1.onmessage = () => {
            window.clearTimeout(timer);
            resolve();
          };
          controller.postMessage({ type: 'CLEAR_AI_CACHE' }, [channel.port2]);
        });
      }

      if ('caches' in window) {
        const keys = await caches.keys();
        await Promise.all(
          keys
            .filter((k) => k.startsWith('pitchspeed-ai-'))
            .map((k) => caches.delete(k))
        );
      }

      window.location.reload();
    } catch {
      setIsUpdatingModel(false);
    }
  }, [isUpdatingModel]);

  return {
    isModelCached,
    cachedSizeMb,
    isOnline,
    isUpdatingModel,
    forceUpdateModel,
  };
}
