import { useEffect, useState } from 'react';

interface WakeLockSentinelLike {
  release: () => Promise<void>;
  addEventListener: (type: 'release', listener: () => void) => void;
}

type NavigatorWithWakeLock = Omit<Navigator, 'wakeLock'> & {
  standalone?: boolean;
  wakeLock?: {
    request: (type: 'screen') => Promise<WakeLockSentinelLike>;
  };
};

export function useIosPwa(keepAwake: boolean) {
  const [isLandscape, setIsLandscape] = useState<boolean>(() =>
    typeof window !== 'undefined' ? window.innerWidth > window.innerHeight : false
  );
  const [isStandalone, setIsStandalone] = useState<boolean>(false);
  const [wakeLockActive, setWakeLockActive] = useState<boolean>(false);

  useEffect(() => {
    const nav = window.navigator as NavigatorWithWakeLock;
    const standalone =
      window.matchMedia('(display-mode: standalone)').matches || Boolean(nav.standalone);
    setIsStandalone(standalone);

    const updateOrientation = () => {
      setIsLandscape(window.innerWidth > window.innerHeight);
    };

    const mql = window.matchMedia('(orientation: landscape)');
    mql.addEventListener?.('change', updateOrientation);
    window.addEventListener('resize', updateOrientation);

    return () => {
      mql.removeEventListener?.('change', updateOrientation);
      window.removeEventListener('resize', updateOrientation);
    };
  }, []);

  useEffect(() => {
    if (!keepAwake) {
      setWakeLockActive(false);
      return;
    }

    const nav = window.navigator as NavigatorWithWakeLock;
    if (!nav.wakeLock) return;

    let sentinel: WakeLockSentinelLike | null = null;
    let cancelled = false;

    const acquire = async () => {
      try {
        sentinel = await nav.wakeLock!.request('screen');
        if (cancelled) {
          await sentinel.release();
          return;
        }
        setWakeLockActive(true);
        sentinel.addEventListener('release', () => setWakeLockActive(false));
      } catch {
        setWakeLockActive(false);
      }
    };

    const onVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        acquire();
      }
    };

    acquire();
    document.addEventListener('visibilitychange', onVisibilityChange);

    return () => {
      cancelled = true;
      document.removeEventListener('visibilitychange', onVisibilityChange);
      sentinel?.release().catch(() => {});
    };
  }, [keepAwake]);

  return { isLandscape, isStandalone, wakeLockActive };
}
