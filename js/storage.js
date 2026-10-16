'use strict';
/**
 * Safe browser-storage wrappers for Nova. Everything Nova persists (chat
 * history, vocabulary, model weights) stays on this device. Nothing is ever
 * sent to a third party. Unsupported storage and quota problems are handled
 * gracefully: Nova keeps working in memory for the current session.
 */
const NovaStorage = (() => {
  const available = (() => {
    try {
      const k = '__nova_test__';
      localStorage.setItem(k, '1');
      localStorage.removeItem(k);
      return true;
    } catch (e) {
      return false;
    }
  })();

  function get(key, fallback) {
    if (!available) return fallback;
    try {
      const raw = localStorage.getItem(key);
      if (raw === null) return fallback;
      return JSON.parse(raw);
    } catch (e) {
      console.warn('Nova storage: could not read', key, e);
      return fallback;
    }
  }

  /**
   * Returns { ok: true } or { ok: false, reason: 'unavailable' | 'quota' }.
   * Never throws.
   */
  function set(key, value) {
    if (!available) return { ok: false, reason: 'unavailable' };
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return { ok: true };
    } catch (e) {
      const quota = e &&
        ((e.name === 'QuotaExceededError') ||
         (e.code === 22) || (e.code === 1014) ||
         (e.name === 'NS_ERROR_DOM_QUOTA_REACHED'));
      return { ok: false, reason: quota ? 'quota' : 'error' };
    }
  }

  function remove(key) {
    if (!available) return;
    try { localStorage.removeItem(key); } catch (e) { /* ignore */ }
  }

  return {
    available,
    get,
    set,
    remove,
    KEYS: {
      chats: 'nova.chats.v1',
      model: 'nova.model.v1',
      activeChat: 'nova.activeChat.v1',
    },
  };
})();
