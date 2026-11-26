import type { ModelLoadProgress } from "../kernels/types";

const CACHE_NAME = "omarchy-mlx-webgpu-v1";

/**
 * Fetches a binary or JSON asset from Hugging Face with browser Cache API persistence
 * and streaming download progress reporting.
 */
export async function fetchCachedArrayBuffer(
  url: string,
  stage: ModelLoadProgress["stage"],
  label: string,
  expectedBytes: number,
  onProgress?: (p: ModelLoadProgress) => void,
): Promise<ArrayBuffer> {
  const hasCache = typeof caches !== "undefined";
  if (hasCache) {
    try {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(url);
      if (cached) {
        onProgress?.({
          stage,
          loadedBytes: expectedBytes,
          totalBytes: expectedBytes,
          message: `Loaded ${label} from browser cache`,
          fromCache: true,
        });
        return await cached.arrayBuffer();
      }
    } catch {
      // Cache API unavailable in private browsing; fall through to network
    }
  }

  const resp = await fetch(url, { mode: "cors" });
  if (!resp.ok) {
    throw new Error(`Failed to fetch ${label} (${resp.status} ${resp.statusText})`);
  }

  const totalBytes = Number(resp.headers.get("content-length")) || expectedBytes;
  if (!resp.body) {
    const buf = await resp.arrayBuffer();
    onProgress?.({ stage, loadedBytes: buf.byteLength, totalBytes: buf.byteLength, message: `Downloaded ${label}` });
    return buf;
  }

  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loadedBytes += value.byteLength;
      onProgress?.({
        stage,
        loadedBytes,
        totalBytes: Math.max(totalBytes, loadedBytes),
        message: `Downloading ${label} (${(loadedBytes / 1048576).toFixed(1)} / ${(Math.max(totalBytes, loadedBytes) / 1048576).toFixed(1)} MB)`,
      });
    }
  }

  const merged = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }

  if (hasCache) {
    try {
      const cache = await caches.open(CACHE_NAME);
      await cache.put(
        url,
        new Response(merged.buffer, {
          headers: { "Content-Type": "application/octet-stream" },
        }),
      );
    } catch {
      // Ignore quota exceeded; model still runs from memory
    }
  }

  return merged.buffer;
}

export async function fetchCachedText(url: string, label: string): Promise<string> {
  const buf = await fetchCachedArrayBuffer(url, "tokenizer", label, 0);
  return new TextDecoder().decode(buf);
}
