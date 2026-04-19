import { useCallback, useEffect, useRef, useState } from "react";
import {
  listVideos,
  deleteVideo,
  deleteAllVideos,
  thumbnailUrl,
  type VideoInfo,
} from "../api";

const PAGE_SIZE = 20;

export default function VideoLibrary() {
  const [videos, setVideos] = useState<VideoInfo[]>([]);
  const [total, setTotal] = useState(0);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const sentinelRef = useRef<HTMLDivElement>(null);

  const loadPage = useCallback(
    async (nextCursor?: string | null, replace = false) => {
      if (replace) setLoading(true);
      else setLoadingMore(true);
      try {
        const data = await listVideos(PAGE_SIZE, nextCursor);
        setVideos((prev) => (replace ? data.videos : [...prev, ...data.videos]));
        setTotal(data.total);
        setCursor(data.next_cursor);
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load videos");
      } finally {
        setLoading(false);
        setLoadingMore(false);
      }
    },
    [],
  );

  const refresh = useCallback(() => {
    setCursor(null);
    loadPage(null, true);
  }, [loadPage]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Infinite scroll via IntersectionObserver
  useEffect(() => {
    if (!sentinelRef.current) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && cursor && !loadingMore) {
          loadPage(cursor);
        }
      },
      { threshold: 0.1 },
    );
    observer.observe(sentinelRef.current);
    return () => observer.disconnect();
  }, [cursor, loadingMore, loadPage]);

  const handleDelete = async (videoId: string) => {
    if (!confirm(`Delete ${videoId}?`)) return;
    try {
      await deleteVideo(videoId);
      setVideos((prev) => prev.filter((v) => v.video_id !== videoId));
      setTotal((prev) => prev - 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    }
  };

  const handleDeleteAll = async () => {
    if (!confirm(`Delete all ${total} videos? This cannot be undone.`)) return;
    try {
      await deleteAllVideos();
      setVideos([]);
      setTotal(0);
      setCursor(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete all failed");
    }
  };

  const filtered = filter
    ? videos.filter((v) =>
        v.video_id.toLowerCase().includes(filter.toLowerCase()),
      )
    : videos;

  if (loading) {
    return (
      <div className="py-12 text-center text-gray-500">Loading library...</div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">
          Library ({total} video{total !== 1 ? "s" : ""})
          {videos.length < total && !filter && (
            <span className="ml-1 text-sm text-gray-500">
              — showing {videos.length}
            </span>
          )}
        </h2>
        <div className="flex gap-3">
          <input
            type="text"
            placeholder="Filter by ID..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="rounded-lg bg-gray-900 px-4 py-2 text-sm text-gray-200 placeholder-gray-600 outline-none ring-1 ring-gray-700 focus:ring-blue-500"
          />
          <button
            onClick={refresh}
            className="rounded-lg bg-gray-800 px-4 py-2 text-sm text-gray-300 transition hover:bg-gray-700"
          >
            Refresh
          </button>
          {total > 0 && (
            <button
              onClick={handleDeleteAll}
              className="rounded-lg bg-red-900 px-4 py-2 text-sm text-red-300 transition hover:bg-red-800"
            >
              Delete All
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="rounded-lg bg-red-900/50 p-4 text-sm text-red-300">
          {error}
        </div>
      )}

      {filtered.length === 0 ? (
        <div className="py-12 text-center text-gray-500">
          {total === 0
            ? "No videos in the library yet. Upload one!"
            : "No videos match your filter."}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {filtered.map((video) => (
            <div
              key={video.video_id}
              className="overflow-hidden rounded-lg bg-gray-900 transition hover:bg-gray-800"
            >
              {video.has_thumbnail && (
                <img
                  src={thumbnailUrl(video.video_id)}
                  alt={video.video_id}
                  className="h-36 w-full bg-gray-800 object-cover"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
              )}
              <div className="p-3">
                <p className="truncate text-sm font-medium text-gray-200">
                  {video.video_id}
                </p>
                <div className="mt-1 space-y-0.5 text-xs text-gray-500">
                  {video.duration != null && (
                    <p>Duration: {Number(video.duration).toFixed(1)}s</p>
                  )}
                  {video.width != null && video.height != null && (
                    <p>
                      {video.width}x{video.height}
                    </p>
                  )}
                  {video.codec && <p>Codec: {video.codec}</p>}
                </div>
                <button
                  onClick={() => handleDelete(video.video_id)}
                  className="mt-2 text-xs text-red-400 transition hover:text-red-300"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Infinite scroll sentinel */}
      <div ref={sentinelRef} className="h-4">
        {loadingMore && (
          <p className="text-center text-sm text-gray-500">Loading more...</p>
        )}
      </div>
    </div>
  );
}
