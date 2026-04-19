import { useEffect, useState } from "react";
import {
  getConfig,
  getStatus,
  updateConfig,
  type Config,
  type StatusInfo,
} from "../api";

export default function Settings() {
  const [config, setConfig] = useState<Config | null>(null);
  const [status, setStatus] = useState<StatusInfo | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    Promise.all([getConfig(), getStatus()])
      .then(([c, s]) => {
        setConfig(c);
        setStatus(s);
      })
      .catch((e) =>
        setError(e instanceof Error ? e.message : "Failed to load config"),
      );
  }, []);

  const handleSave = async () => {
    if (!config) return;
    setSaving(true);
    setError(null);
    setSaved(false);
    try {
      const updated = await updateConfig(config);
      setConfig(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  if (!config) {
    return (
      <div className="py-12 text-center text-gray-500">
        Loading settings...
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-8">
      {/* Index stats */}
      {status && (
        <div className="rounded-lg bg-gray-900 p-6">
          <h3 className="mb-4 text-lg font-medium">Index Status</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-500">Total Videos</p>
              <p className="text-2xl font-bold text-white">
                {status.total_videos}
              </p>
            </div>
            <div>
              <p className="text-gray-500">Vector Dimension</p>
              <p className="text-2xl font-bold text-white">
                {status.dimension}
              </p>
            </div>
            <div>
              <p className="text-gray-500">Index Backend</p>
              <p className="font-medium text-gray-200">
                {status.index_backend}
              </p>
            </div>
            <div>
              <p className="text-gray-500">Embedding Backend</p>
              <p className="font-medium text-gray-200">
                {status.embedding_backend}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Configuration */}
      <div className="rounded-lg bg-gray-900 p-6">
        <h3 className="mb-4 text-lg font-medium">Configuration</h3>
        <div className="space-y-4">
          {/* Index backend */}
          <div>
            <label className="mb-1 block text-sm text-gray-400">
              Index Backend
            </label>
            <select
              value={config.index_backend}
              onChange={(e) =>
                setConfig({ ...config, index_backend: e.target.value })
              }
              className="w-full rounded-lg bg-gray-800 px-4 py-2 text-sm text-gray-200 outline-none ring-1 ring-gray-700 focus:ring-blue-500"
            >
              <option value="faiss">FAISS</option>
              <option value="qdrant">Qdrant</option>
            </select>
          </div>

          {/* Qdrant URL */}
          {config.index_backend === "qdrant" && (
            <div>
              <label className="mb-1 block text-sm text-gray-400">
                Qdrant URL (leave empty for local on-disk)
              </label>
              <input
                type="text"
                value={config.qdrant_url ?? ""}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    qdrant_url: e.target.value || null,
                  })
                }
                placeholder="http://localhost:6333"
                className="w-full rounded-lg bg-gray-800 px-4 py-2 text-sm text-gray-200 placeholder-gray-600 outline-none ring-1 ring-gray-700 focus:ring-blue-500"
              />
            </div>
          )}

          {/* Collection name */}
          {config.index_backend === "qdrant" && (
            <div>
              <label className="mb-1 block text-sm text-gray-400">
                Collection Name
              </label>
              <input
                type="text"
                value={config.collection_name}
                onChange={(e) =>
                  setConfig({ ...config, collection_name: e.target.value })
                }
                className="w-full rounded-lg bg-gray-800 px-4 py-2 text-sm text-gray-200 outline-none ring-1 ring-gray-700 focus:ring-blue-500"
              />
            </div>
          )}

          {/* Embedding backend */}
          <div>
            <label className="mb-1 block text-sm text-gray-400">
              Embedding Backend
            </label>
            <select
              value={config.embedding_backend}
              onChange={(e) =>
                setConfig({ ...config, embedding_backend: e.target.value })
              }
              className="w-full rounded-lg bg-gray-800 px-4 py-2 text-sm text-gray-200 outline-none ring-1 ring-gray-700 focus:ring-blue-500"
            >
              <option value="s2vs">S2VS</option>
              <option value="clip">CLIP</option>
            </select>
          </div>

          {/* Threshold */}
          <div>
            <label className="mb-1 block text-sm text-gray-400">
              Similarity Threshold: {config.threshold}
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={config.threshold}
              onChange={(e) =>
                setConfig({ ...config, threshold: Number(e.target.value) })
              }
              className="w-full"
            />
          </div>

          {/* Top K */}
          <div>
            <label className="mb-1 block text-sm text-gray-400">
              Top-K Candidates: {config.top_k}
            </label>
            <input
              type="range"
              min={10}
              max={200}
              step={10}
              value={config.top_k}
              onChange={(e) =>
                setConfig({ ...config, top_k: Number(e.target.value) })
              }
              className="w-full"
            />
          </div>
        </div>

        {error && (
          <div className="mt-4 rounded-lg bg-red-900/50 p-3 text-sm text-red-300">
            {error}
          </div>
        )}

        <div className="mt-6 flex items-center gap-3">
          <button
            onClick={handleSave}
            disabled={saving}
            className="rounded-lg bg-blue-600 px-6 py-2 text-sm font-medium text-white transition hover:bg-blue-500 disabled:opacity-50"
          >
            {saving ? "Saving..." : "Save"}
          </button>
          {saved && (
            <span className="text-sm text-green-400">Saved!</span>
          )}
          <p className="ml-auto text-xs text-gray-600">
            Backend changes require server restart.
          </p>
        </div>
      </div>
    </div>
  );
}
