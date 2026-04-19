"""
Pluggable metadata store for video deduplication pipeline.

Maps video IDs to file paths and metadata (resolution, duration, etc.).
Default implementation: JSON file on disk. Abstract interface allows
swapping in PostgreSQL/S3 later without changing the pipeline.
"""

import json

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

# Metadata values are JSON-compatible: str, int, float, bool, None, or nested
MetadataValue = str | int | float | bool | None
MetadataDict = dict[str, MetadataValue]
VideoRecord = dict[str, MetadataValue]


class MetadataStore(ABC):
    """Abstract interface for video metadata storage."""

    @abstractmethod
    def add(
        self, video_id: str, path: str, metadata: MetadataDict | None = None
    ) -> None:
        """Add or update a video entry.

        Args:
            video_id: Unique identifier for the video.
            path: File path to the video.
            metadata: Optional dict of metadata (resolution, duration, etc.).
        """
        ...

    @abstractmethod
    def get(self, video_id: str) -> VideoRecord:
        """Get metadata for a video.

        Args:
            video_id: Unique identifier for the video.

        Returns:
            Dict with 'path' and any additional metadata fields.

        Raises:
            KeyError: If video_id is not found.
        """
        ...

    @abstractmethod
    def list_all(self) -> list[str]:
        """List all video IDs in the store.

        Returns:
            List of video ID strings.
        """
        ...

    @abstractmethod
    def remove(self, video_id: str) -> None:
        """Remove a video entry.

        Args:
            video_id: Unique identifier for the video.

        Raises:
            KeyError: If video_id is not found.
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the store to disk.

        Args:
            path: File path to save to.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "MetadataStore":
        """Load a store from disk.

        Args:
            path: File path to load from.

        Returns:
            Loaded MetadataStore instance.
        """
        ...

    def __len__(self) -> int:
        return len(self.list_all())

    def __contains__(self, video_id: str) -> bool:
        return video_id in self.list_all()


class JSONStore(MetadataStore):
    """JSON file-based metadata store."""

    def __init__(self) -> None:
        self._data: dict[str, VideoRecord] = {}

    def add(
        self, video_id: str, path: str, metadata: MetadataDict | None = None
    ) -> None:
        entry: VideoRecord = {"path": path}
        if metadata is not None:
            entry.update(metadata)
        entry["added_at"] = datetime.now(timezone.utc).isoformat()
        self._data[video_id] = entry

    def get(self, video_id: str) -> VideoRecord:
        if video_id not in self._data:
            raise KeyError(f"Video '{video_id}' not found in store")
        return self._data[video_id]

    def list_all(self) -> list[str]:
        return list(self._data.keys())

    def remove(self, video_id: str) -> None:
        if video_id not in self._data:
            raise KeyError(f"Video '{video_id}' not found in store")
        del self._data[video_id]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "JSONStore":
        store = cls()
        with open(path) as f:
            store._data = json.load(f)
        return store
