import { useState } from "react";
import UploadTab from "./components/UploadTab";
import SearchTab from "./components/SearchTab";
import VideoLibrary from "./components/VideoLibrary";
import Settings from "./components/Settings";

type Tab = "upload" | "search" | "library" | "settings";

const TABS: { key: Tab; label: string }[] = [
  { key: "upload", label: "Upload" },
  { key: "search", label: "Search" },
  { key: "library", label: "Library" },
  { key: "settings", label: "Settings" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("upload");

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 bg-gray-900">
        <div className="mx-auto flex max-w-6xl items-center gap-8 px-6 py-4">
          <h1 className="text-xl font-bold tracking-tight">Video Dedup</h1>
          <nav className="flex gap-1">
            {TABS.map((t) => (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
                  tab === t.key
                    ? "bg-blue-600 text-white"
                    : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        {tab === "upload" && <UploadTab />}
        {tab === "search" && <SearchTab />}
        {tab === "library" && <VideoLibrary />}
        {tab === "settings" && <Settings />}
      </main>
    </div>
  );
}
