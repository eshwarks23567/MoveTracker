/**
 * HAR Activity Tracker — Service Worker
 * Caches static assets for offline / installed-app use.
 * The /infer endpoint is never cached (always hits the live server).
 */

const CACHE_NAME = 'har-app-v1';
const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/app.js',
    '/app.css',
    '/manifest.json',
    '/icon-192.png',
    '/icon-512.png',
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap',
];

// ── Install: cache all static assets ─────────────────────────
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(STATIC_ASSETS.filter(u => !u.startsWith('http'))))
            .then(() => self.skipWaiting())
    );
});

// ── Activate: remove old caches ───────────────────────────────
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys()
            .then(keys => Promise.all(
                keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
            ))
            .then(() => self.clients.claim())
    );
});

// ── Fetch: serve from cache, fall back to network ─────────────
self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);

    // Never cache inference requests — always go to network
    if (url.pathname === '/infer') return;

    // For everything else: cache-first strategy
    event.respondWith(
        caches.match(event.request).then(cached => {
            if (cached) return cached;

            return fetch(event.request).then(response => {
                // Only cache successful same-origin responses
                if (
                    response.ok &&
                    response.type === 'basic' &&
                    event.request.method === 'GET'
                ) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then(c => c.put(event.request, clone));
                }
                return response;
            });
        })
    );
});
