const CACHE_NAME = 'entangle-v1';
const ASSETS = [
  '.',
  'index.html',
  'style.css',
  'app.js',
  'manifest.json'
];

self.addEventListener('install', ev => {
  ev.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', ev => {
  ev.waitUntil(clients.claim());
});

self.addEventListener('fetch', ev => {
  ev.respondWith(
    caches.match(ev.request).then(r => r || fetch(ev.request))
  );
});