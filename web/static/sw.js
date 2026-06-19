/* PWA service worker — makes the OFFLINE query mode work with no network.
 *
 * Precaches the app shell + search index + the custom head (txt_h.onnx). Cross-origin
 * assets (transformers.js / onnxruntime-web from the CDN, and the stock e5 model shards
 * from the HF hub) are runtime-cached on first online use, so afterwards offline mode
 * runs fully offline. Thumbnails are runtime-cached as viewed. The exact/server mode
 * (/encode_query) is network-only and simply fails offline (app falls back to offline mode).
 */
const CACHE = "1bit-hybrid-v1";
const CORE = [
  "/", "/app.js", "/search.js",
  "/data/index_info.json", "/data/meta.json", "/data/index.bin",
  "/onnx/txt_h.onnx", "/manifest.webmanifest", "/icon.svg",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(CORE)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  const url = new URL(req.url);

  // exact/server encoding needs the backend — never cache; let it fail offline.
  if (url.pathname === "/encode_query") return;

  // cache-first for everything else (app shell, index, onnx, thumbs, CDN libs, HF model).
  e.respondWith(
    caches.match(req).then((hit) => {
      if (hit) return hit;
      return fetch(req).then((res) => {
        // runtime-cache successful GETs (same-origin assets + cross-origin CDN/HF model)
        if (req.method === "GET" && res && (res.ok || res.type === "opaque")) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
        }
        return res;
      });
    })
  );
});
