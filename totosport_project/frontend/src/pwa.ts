/**
 * Registrazione del service worker.
 * Solo in build di produzione: in dev la SW interferirebbe con l'HMR di Vite
 * servendo asset in cache. La SW vive in `public/sw.js`.
 */
export function registerServiceWorker() {
  if (!import.meta.env.PROD) return
  if (!('serviceWorker' in navigator)) return
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch((err) => {
      console.error('Registrazione service worker fallita:', err)
    })
  })
}
