import { useCallback, useEffect, useState } from "react";
import { clearCache, searchDuplicates, type SearchResult } from "../api";
import SearchResults from "./SearchResults";

function useVideoThumbnail(file: File | null): string | null {
  const [thumb, setThumb] = useState<string | null>(null);
  useEffect(() => {
    if (!file) {
      setThumb(null);
      return;
    }
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = true;
    video.playsInline = true;
    video.onloadeddata = () => {
      video.currentTime = Math.min(1, video.duration / 2);
    };
    video.onseeked = () => {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d")!.drawImage(video, 0, 0);
      setThumb(canvas.toDataURL("image/jpeg", 0.8));
      URL.revokeObjectURL(url);
    };
    video.onerror = () => {
      URL.revokeObjectURL(url);
    };
    video.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file]);
  return thumb;
}

export default function SearchTab() {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [searching, setSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.8);
  const [clearing, setClearing] = useState(false);
  const [cacheMsg, setCacheMsg] = useState<string | null>(null);
  const queryThumb = useVideoThumbnail(file);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setSearchResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  const handleSearch = async () => {
    if (!file) return;
    setSearching(true);
    setError(null);
    try {
      const result = await searchDuplicates(file, 50, threshold);
      setSearchResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setSearching(false);
    }
  };

  const handleClearCache = async () => {
    setClearing(true);
    setCacheMsg(null);
    try {
      const res = await clearCache();
      setCacheMsg(`Cache cleared (${res.cleared} files removed)`);
    } catch (e) {
      setCacheMsg(e instanceof Error ? e.message : "Failed to clear cache");
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header with clear cache */}
      <div className="flex items-center justify-between">
        <div />
        <div className="flex items-center gap-3">
          {cacheMsg && (
            <span className="text-xs text-gray-400">{cacheMsg}</span>
          )}
          <button
            onClick={handleClearCache}
            disabled={clearing}
            className="rounded-lg border border-gray-700 bg-gray-800 px-4 py-1.5 text-xs font-medium text-gray-300 transition hover:bg-gray-700 disabled:opacity-50"
          >
            {clearing ? "Clearing..." : "Clear Cache"}
          </button>
        </div>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-12 transition ${
          dragging
            ? "border-emerald-500 bg-emerald-500/10"
            : "border-gray-700 bg-gray-900 hover:border-gray-600"
        }`}
      >
        <p className="mb-4 text-gray-400">
          Drop a video file to search for duplicates
        </p>
        <input
          type="file"
          accept="video/*"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
          className="hidden"
          id="search-file-input"
        />
        <label
          htmlFor="search-file-input"
          className="cursor-pointer rounded-lg bg-gray-800 px-6 py-2 text-sm font-medium text-gray-300 transition hover:bg-gray-700"
        >
          Choose file
        </label>
      </div>

      {/* Selected file */}
      {file && (
        <div className="flex gap-4 rounded-lg bg-gray-900 p-4">
          {queryThumb && (
            <img
              src={queryThumb}
              alt="Query video"
              className="h-24 w-36 flex-shrink-0 rounded-lg bg-gray-800 object-cover"
            />
          )}
          <div className="flex-1">
            <p className="text-sm text-gray-300">
              <span className="font-medium text-white">{file.name}</span>
              {" — "}
              {(file.size / 1024 / 1024).toFixed(1)} MB
            </p>
          <div className="mt-3 flex items-center gap-4">
              <label className="text-sm text-gray-400">Threshold:</label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="h-2 flex-1 cursor-pointer appearance-none rounded-full bg-gray-700 accent-emerald-500"
              />
              <span className="w-12 text-right text-sm font-medium text-white">
                {threshold.toFixed(2)}
              </span>
            </div>
            <div className="mt-3">
              <button
                onClick={handleSearch}
                disabled={searching}
                className="rounded-lg bg-emerald-600 px-6 py-2 text-sm font-medium text-white transition hover:bg-emerald-500 disabled:opacity-50"
              >
                {searching ? "Searching..." : "Search for Duplicates"}
              </button>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-lg bg-red-900/50 p-4 text-sm text-red-300">
          {error}
        </div>
      )}

      {searchResult && <SearchResults result={searchResult} />}
    </div>
  );
}
