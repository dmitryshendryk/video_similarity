import { useCallback, useRef, useState } from "react";
import {
  uploadVideo,
  thumbnailUrl,
  type UploadResult,
  type BatchUploadItem,
} from "../api";

const VIDEO_EXTENSIONS = [".mp4", ".avi", ".webm", ".mov", ".mkv"];

function isVideoFile(file: File): boolean {
  return VIDEO_EXTENSIONS.some((ext) =>
    file.name.toLowerCase().endsWith(ext),
  );
}

export default function UploadTab() {
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [results, setResults] = useState<BatchUploadItem[]>([]);
  const [singleResult, setSingleResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const folderRef = useRef<HTMLInputElement>(null);

  const reset = useCallback(() => {
    setFiles([]);
    setResults([]);
    setSingleResult(null);
    setError(null);
    setProgress({ done: 0, total: 0 });
  }, []);

  const handleFiles = useCallback(
    (fileList: FileList | File[]) => {
      const arr = Array.from(fileList).filter(isVideoFile);
      if (arr.length === 0) {
        setError("No video files found. Supported: .mp4, .avi, .webm, .mov, .mkv");
        return;
      }
      setFiles(arr);
      setResults([]);
      setSingleResult(null);
      setError(null);
    },
    [],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles],
  );

  const handleUpload = async () => {
    if (files.length === 0) return;
    setUploading(true);
    setError(null);
    setResults([]);
    setSingleResult(null);
    setProgress({ done: 0, total: files.length });

    const batchResults: BatchUploadItem[] = new Array(files.length);
    let succeeded = 0;
    let failed = 0;
    const CONCURRENCY = 6;

    // Process files in concurrent chunks
    for (let i = 0; i < files.length; i += CONCURRENCY) {
      const chunk = files.slice(i, i + CONCURRENCY);
      const promises = chunk.map(async (file, j) => {
        const idx = i + j;
        try {
          const result = await uploadVideo(file);
          batchResults[idx] = { ...result, status: "ok" };
          succeeded++;
        } catch (e) {
          batchResults[idx] = {
            filename: file.name,
            status: "error",
            error: e instanceof Error ? e.message : "Upload failed",
          };
          failed++;
        }
        setProgress({ done: succeeded + failed, total: files.length });
      });
      await Promise.all(promises);
      setResults([...batchResults.filter(Boolean)]);
    }

    if (files.length === 1 && batchResults[0]?.status === "ok") {
      setSingleResult(batchResults[0] as unknown as UploadResult);
      setResults([]);
    }

    if (failed > 0) {
      setError(`${failed} of ${files.length} files failed`);
    }
    setUploading(false);
  };

  return (
    <div className="space-y-6">
      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-12 transition ${
          dragging
            ? "border-blue-500 bg-blue-500/10"
            : "border-gray-700 bg-gray-900 hover:border-gray-600"
        }`}
      >
        <p className="mb-4 text-gray-400">
          Drag & drop video files or a folder here
        </p>
        <div className="flex gap-3">
          <input
            type="file"
            accept="video/*"
            multiple
            onChange={(e) => {
              if (e.target.files) handleFiles(e.target.files);
            }}
            className="hidden"
            id="file-input"
          />
          <label
            htmlFor="file-input"
            className="cursor-pointer rounded-lg bg-gray-800 px-5 py-2 text-sm font-medium text-gray-300 transition hover:bg-gray-700"
          >
            Choose files
          </label>
          <input
            type="file"
            ref={folderRef}
            onChange={(e) => {
              if (e.target.files) handleFiles(e.target.files);
            }}
            className="hidden"
            id="folder-input"
            {...({ webkitdirectory: "", directory: "" } as React.InputHTMLAttributes<HTMLInputElement>)}
          />
          <label
            htmlFor="folder-input"
            className="cursor-pointer rounded-lg bg-gray-800 px-5 py-2 text-sm font-medium text-gray-300 transition hover:bg-gray-700"
          >
            Choose folder
          </label>
        </div>
      </div>

      {/* Selected files */}
      {files.length > 0 && (
        <div className="rounded-lg bg-gray-900 p-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-300">
              <span className="font-medium text-white">{files.length}</span>
              {" video"}{files.length !== 1 ? "s" : ""}{" selected"}
              {files.length === 1 && (
                <span className="text-gray-500">
                  {" — "}{files[0].name} ({(files[0].size / 1024 / 1024).toFixed(1)} MB)
                </span>
              )}
            </p>
            <button
              onClick={reset}
              className="text-xs text-gray-500 hover:text-gray-300"
            >
              Clear
            </button>
          </div>
          {files.length > 1 && (
            <div className="mt-2 max-h-32 overflow-y-auto text-xs text-gray-500">
              {files.map((f, i) => (
                <div key={i}>{f.name} — {(f.size / 1024 / 1024).toFixed(1)} MB</div>
              ))}
            </div>
          )}
          <div className="mt-3">
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="rounded-lg bg-blue-600 px-6 py-2 text-sm font-medium text-white transition hover:bg-blue-500 disabled:opacity-50"
            >
              {uploading
                ? `Uploading ${progress.done}/${progress.total}...`
                : `Upload ${files.length} video${files.length !== 1 ? "s" : ""}`}
            </button>
          </div>
          {uploading && progress.total > 0 && (
            <div className="mt-3 h-2 overflow-hidden rounded-full bg-gray-700">
              <div
                className="h-full rounded-full bg-blue-500 transition-all"
                style={{ width: `${(progress.done / progress.total) * 100}%` }}
              />
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="rounded-lg bg-red-900/50 p-4 text-sm text-red-300">{error}</div>
      )}

      {/* Single upload result */}
      {singleResult && (
        <div className="rounded-lg bg-gray-900 p-4">
          <h3 className="mb-2 font-medium text-green-400">Uploaded successfully</h3>
          <div className="flex gap-4">
            <img
              src={thumbnailUrl(singleResult.video_id)}
              alt="thumbnail"
              className="h-24 w-32 rounded bg-gray-800 object-cover"
              onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
            />
            <div className="text-sm text-gray-400">
              <p><span className="text-gray-300">ID:</span> {singleResult.video_id}</p>
              {singleResult.duration != null && (
                <p><span className="text-gray-300">Duration:</span> {singleResult.duration.toFixed(1)}s</p>
              )}
              {singleResult.width != null && singleResult.height != null && (
                <p><span className="text-gray-300">Resolution:</span> {singleResult.width}x{singleResult.height}</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Batch results */}
      {results.length > 0 && (
        <div className="space-y-2">
          <h3 className="font-medium text-gray-200">Upload Results</h3>
          <div className="max-h-96 space-y-1 overflow-y-auto">
            {results.map((r, i) => (
              <div
                key={i}
                className={`flex items-center gap-3 rounded-lg p-3 text-sm ${
                  r.status === "ok" ? "bg-gray-900" : "bg-red-900/30"
                }`}
              >
                {r.status === "ok" && r.video_id && (
                  <img
                    src={thumbnailUrl(r.video_id)}
                    alt=""
                    className="h-10 w-14 rounded bg-gray-800 object-cover"
                    onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                  />
                )}
                <div className="min-w-0 flex-1">
                  <p className="truncate text-gray-300">
                    {r.video_id ?? r.filename ?? `File ${i + 1}`}
                  </p>
                  {r.status === "error" && (
                    <p className="text-xs text-red-400">{r.error}</p>
                  )}
                </div>
                <span className={r.status === "ok" ? "text-green-400" : "text-red-400"}>
                  {r.status === "ok" ? "OK" : "Failed"}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
