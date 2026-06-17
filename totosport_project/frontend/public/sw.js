/*
 * Service worker minimale per TotoSport (PWA installabile + shell offline).
 * Strategia:
 *  - navigazioni (caricamento pagine): network-first, fallback alla index in cache;
 *  - asset statici same-origin (js/css/img/font): stale-while-revalidate;
 *  - tutto il resto (API su /api, cross-origin, non-GET): nessuna intercettazione.
 * Niente precache di file con hash: la SW resta valida tra una build e l'altra.
 */
const CACHE = 'totosport-v1'
const APP_SHELL = ['/', '/index.html', '/manifest.webmanifest']

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(APP_SHELL)).then(() => self.skipWaiting()),
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  )
})

self.addEventListener('fetch', (event) => {
  const { request } = event
  const url = new URL(request.url)

  // Non gestire: metodi diversi da GET, altre origin, e le chiamate API (/api).
  if (request.method !== 'GET' || url.origin !== self.location.origin) return
  if (url.pathname.startsWith('/api')) return

  // Navigazioni → network-first con fallback alla shell in cache (offline).
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request).catch(() => caches.match('/index.html').then((r) => r || caches.match('/'))),
    )
    return
  }

  // Asset statici → stale-while-revalidate.
  event.respondWith(
    caches.match(request).then((cached) => {
      const network = fetch(request)
        .then((response) => {
          if (response && response.status === 200 && response.type === 'basic') {
            const copy = response.clone()
            caches.open(CACHE).then((cache) => cache.put(request, copy))
          }
          return response
        })
        .catch(() => cached)
      return cached || network
    }),
  )
})
