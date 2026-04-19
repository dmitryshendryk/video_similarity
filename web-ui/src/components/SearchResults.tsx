import { thumbnailUrl, type SearchResult, type SearchTiming } from "../api";

function scoreColor(score: number): string {
  if (score >= 0.95) return "text-green-400";
  if (score >= 0.85) return "text-yellow-400";
  return "text-orange-400";
}

function scoreBar(score: number): string {
  if (score >= 0.95) return "bg-green-500";
  if (score >= 0.85) return "bg-yellow-500";
  return "bg-orange-500";
}

function fmt(seconds: number): string {
  if (seconds < 0.001) return "<1ms";
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  return `${seconds.toFixed(2)}s`;
}

function TimingBar({ timing }: { timing: SearchTiming }) {
  const total = timing.total_s || 1;
  const segments = [
    { label: "Decode", value: timing.decode_s, color: "bg-blue-500" },
    { label: "Descriptor", value: timing.descriptor_s, color: "bg-purple-500" },
    { label: "Vector search", value: timing.vector_search_s, color: "bg-cyan-500" },
    { label: "ViSiL rerank", value: timing.visil_rerank_s, color: "bg-amber-500" },
  ];

  return (
    <div className="rounded-lg bg-gray-900 p-4">
      <div className="mb-2 flex items-center justify-between text-sm">
        <span className="text-gray-400">
          Search completed in <span className="font-medium text-white">{fmt(timing.total_s)}</span>
        </span>
        <span className="text-xs text-gray-500">
          {timing.frames} frames, {timing.candidates} candidates
        </span>
      </div>
      <div className="flex h-3 overflow-hidden rounded-full bg-gray-700">
        {segments.map((seg) => (
          <div
            key={seg.label}
            className={`${seg.color} transition-all`}
            style={{ width: `${(seg.value / total) * 100}%` }}
            title={`${seg.label}: ${fmt(seg.value)}`}
          />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-400">
        {segments.map((seg) => (
          <span key={seg.label} className="flex items-center gap-1.5">
            <span className={`inline-block h-2 w-2 rounded-full ${seg.color}`} />
            {seg.label}: {fmt(seg.value)}
          </span>
        ))}
      </div>
    </div>
  );
}

interface Props {
  result: SearchResult;
}

export default function SearchResults({ result }: Props) {
  if (result.results.length === 0) {
    return (
      <div className="space-y-4">
        {result.timing && <TimingBar timing={result.timing} />}
        <div className="rounded-lg bg-gray-900 p-6 text-center text-gray-400">
          No duplicates found above threshold ({result.threshold}).
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {result.timing && <TimingBar timing={result.timing} />}

      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium">
          {result.total_results} match
          {result.total_results !== 1 ? "es" : ""} found
        </h3>
        <span className="text-sm text-gray-500">
          threshold: {result.threshold}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {result.results.map((match) => (
          <div
            key={match.video_id}
            className="overflow-hidden rounded-lg bg-gray-900 transition hover:bg-gray-800"
          >
            {match.has_thumbnail && (
              <img
                src={thumbnailUrl(match.video_id)}
                alt={match.video_id}
                className="h-40 w-full bg-gray-800 object-cover"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = "none";
                }}
              />
            )}
            <div className="p-4">
              <p className="truncate text-sm font-medium text-gray-200">
                {match.video_id}
              </p>

              {/* Score bar */}
              <div className="mt-2 flex items-center gap-2">
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-gray-700">
                  <div
                    className={`h-full rounded-full ${scoreBar(match.score)}`}
                    style={{ width: `${match.score * 100}%` }}
                  />
                </div>
                <span
                  className={`text-sm font-bold ${scoreColor(match.score)}`}
                >
                  {(match.score * 100).toFixed(1)}%
                </span>
              </div>

              <div className="mt-2 space-y-0.5 text-xs text-gray-500">
                {match.duration != null && (
                  <p>Duration: {Number(match.duration).toFixed(1)}s</p>
                )}
                {match.width != null && match.height != null && (
                  <p>
                    Resolution: {match.width}x{match.height}
                  </p>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
