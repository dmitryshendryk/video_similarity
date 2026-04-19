const API_BASE = "/api";

// --- Types ---

export interface UploadResult {
  video_id: string;
  filename: string;
  path?: string;
  width?: number;
  height?: number;
  codec?: string;
  duration?: number;
  bitrate?: number;
  fps?: number;
}

export interface SearchMatch {
  video_id: string;
  score: number;
  path?: string;
  duration?: number;
  width?: number;
  height?: number;
  has_thumbnail?: boolean;
}

export interface SearchTiming {
  total_s: number;
  decode_s: number;
  descriptor_s: number;
  vector_search_s: number;
  visil_rerank_s: number;
  frames: number;
  candidates: number;
}

export interface SearchResult {
  query: string;
  total_results: number;
  threshold: number;
  timing?: SearchTiming;
  results: SearchMatch[];
}

export interface VideoInfo {
  video_id: string;
  path?: string;
  width?: number;
  height?: number;
  codec?: string;
  duration?: number;
  bitrate?: number;
  fps?: number;
  has_thumbnail?: boolean;
}

export interface VideoListResponse {
  total: number;
  videos: VideoInfo[];
  next_cursor: string | null;
}

export interface Config {
  index_backend: string;
  embedding_backend: string;
  pretrained: string;
  device: string;
  qdrant_url: string | null;
  collection_name: string;
  threshold: number;
  top_k: number;
}

export interface StatusInfo {
  total_videos: number;
  index_backend: string;
  embedding_backend: string;
  dimension: number;
}

export interface BatchUploadItem {
  video_id?: string;
  filename?: string;
  status: "ok" | "error";
  error?: string;
  path?: string;
  width?: number;
  height?: number;
  codec?: string;
  duration?: number;
  bitrate?: number;
  fps?: number;
}

export interface BatchUploadResult {
  total: number;
  succeeded: number;
  failed: number;
  results: BatchUploadItem[];
}

// --- API functions ---

export async function uploadVideo(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadBatch(files: File[]): Promise<BatchUploadResult> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  const res = await fetch(`${API_BASE}/upload-batch`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function searchDuplicates(
  file: File,
  topK = 50,
  threshold = 0.8,
): Promise<SearchResult> {
  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams({
    top_k: String(topK),
    threshold: String(threshold),
  });
  const res = await fetch(`${API_BASE}/search?${params}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listVideos(
  limit = 20,
  cursor?: string | null,
): Promise<VideoListResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (cursor) params.set("cursor", cursor);
  const res = await fetch(`${API_BASE}/videos?${params}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getVideo(videoId: string): Promise<VideoInfo> {
  const res = await fetch(`${API_BASE}/videos/${videoId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteVideo(videoId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/videos/${videoId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function deleteAllVideos(): Promise<{ deleted: number }> {
  const res = await fetch(`${API_BASE}/videos`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getConfig(): Promise<Config> {
  const res = await fetch(`${API_BASE}/config`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateConfig(
  config: Partial<Config>,
): Promise<Config> {
  const res = await fetch(`${API_BASE}/config`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getStatus(): Promise<StatusInfo> {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function thumbnailUrl(videoId: string): string {
  return `${API_BASE}/videos/${videoId}/thumbnail`;
}

export async function clearCache(): Promise<{ cleared: number }> {
  const res = await fetch(`${API_BASE}/cache`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
