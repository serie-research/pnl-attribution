// Only this lab is controlled. Runtime pin and cache revision move together.
const CACHE = 'signal-lab-0.29.5-v1';
self.addEventListener('install', event => event.waitUntil((async () => {
  const cache = await caches.open(CACHE);
  await cache.addAll(['./', './worker.mjs', './notebook.json']);
  const html = await (await cache.match('./')).text();
  const assets = [...html.matchAll(/(?:src|href)="([^"]+\.(?:js|css))"/g)]
    .map(match => new URL(match[1], self.registration.scope).href);
  await cache.addAll(assets);
  await self.skipWaiting();
})()));
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' ||
      !(url.origin === self.location.origin ||
        url.href.startsWith('https://cdn.jsdelivr.net/pyodide/v0.29.5/full/'))) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const saved = await cache.match(event.request);
    if (saved) return saved;
    const response = await fetch(event.request);
    if (response.ok) await cache.put(event.request, response.clone());
    return response;
  })());
});
