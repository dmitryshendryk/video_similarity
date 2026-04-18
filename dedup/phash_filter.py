"""
Optional pHash (perceptual hash) pre-filter for near-exact duplicate detection.

Uses a single representative frame at 10% duration to compute a 64-bit
perceptual hash. Hamming-distance comparison finds re-encodes and resolution
changes in O(N) time before any CNN inference.

Gracefully degrades if imagehash or Pillow is not installed — pipeline
skips this stage with a warning.
"""

import io
import json
import logging
import pickle
import subprocess

from pathlib import Path

from dedup.qdrant_index import QdrantIndex
from dedup.store import JSONStore

try:
    import imagehash
    from PIL import Image

    _IMAGEHASH_AVAILABLE = True
except ImportError:
    _IMAGEHASH_AVAILABLE = False

logger = logging.getLogger("api_server")


class PHashFilter:
    """In-memory perceptual hash database for near-exact duplicate detection.

    Stores one 64-bit pHash integer per video.  Hamming-distance queries run
    in O(N) time over the in-memory dict — fast enough for collections up to
    ~100 K videos on a single thread.
    """

    def __init__(self) -> None:
        self._hashes: dict[str, int] = {}

    @staticmethod
    def is_available() -> bool:
        """Return True if imagehash and Pillow are importable."""
        return _IMAGEHASH_AVAILABLE

    def compute_phash(self, video_path: str | Path) -> int:
        """Compute a 64-bit pHash from a representative frame at 10% duration.

        Extracts a single frame via FFmpeg/ffprobe at 10 % of the video's
        total duration, converts it to a PIL Image, and returns the integer
        representation of ``imagehash.phash(image)``.

        Args:
            video_path: Absolute or relative path to the video file.

        Returns:
            64-bit integer pHash value.

        Raises:
            RuntimeError: If imagehash / Pillow is not installed.
            ValueError: If the frame cannot be extracted (FFmpeg error).
        """
        if not _IMAGEHASH_AVAILABLE:
            raise RuntimeError(
                "imagehash is not installed. Install with: pip install imagehash"
            )

        video_path = Path(video_path)

        # Probe duration
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            probe = json.loads(result.stdout)
            duration = float(probe.get("format", {}).get("duration", 10))
        except Exception as exc:
            logger.warning("ffprobe failed for %s: %s", video_path, exc)
            duration = 10.0

        seek_time = max(0.0, duration * 0.1)

        # Extract frame as JPEG in memory and decode via PIL for pHash.
        try:
            jpeg_result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(seek_time),
                    "-i",
                    str(video_path),
                    "-vframes",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=30,
            )
        except Exception as exc:
            raise ValueError(
                f"FFmpeg JPEG extraction failed for {video_path}: {exc}"
            ) from exc

        if not jpeg_result.stdout:
            raise ValueError(f"FFmpeg JPEG produced no output for {video_path}")

        image = Image.open(io.BytesIO(jpeg_result.stdout)).convert("RGB")
        hash_value = imagehash.phash(image)
        return int(str(hash_value), 16)

    def add(self, video_id: str, phash_value: int) -> None:
        """Store a pHash for a video.

        Args:
            video_id: Unique identifier for the video.
            phash_value: 64-bit integer pHash.
        """
        self._hashes[video_id] = phash_value

    def remove(self, video_id: str) -> None:
        """Remove a video's pHash from the database.

        Args:
            video_id: Unique identifier for the video.
        """
        self._hashes.pop(video_id, None)

    def find_matches(
        self,
        query_hash: int,
        max_distance: int = 5,
    ) -> list[tuple[str, float]]:
        """Find videos whose pHash is within ``max_distance`` of the query.

        Hamming distance is computed as the bit count of XOR between the two
        64-bit integers.  Score is defined as ``1.0 - (distance / 64.0)``:

        - distance 0  → score 1.0    (identical hash)
        - distance 5  → score 0.9219 (near-duplicate)
        - distance 64 → score 0.0    (completely different)

        Args:
            query_hash: 64-bit integer pHash of the query video.
            max_distance: Maximum Hamming distance to consider a match.

        Returns:
            List of ``(video_id, score)`` tuples sorted by descending score.
        """
        results: list[tuple[str, float]] = []
        for video_id, stored_hash in self._hashes.items():
            distance = bin(query_hash ^ stored_hash).count("1")
            if distance <= max_distance:
                score = 1.0 - (distance / 64.0)
                results.append((video_id, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def load_from_index(self, idx: object) -> int:
        """Populate the in-memory hash dict from an existing index or store.

        Supports ``QdrantIndex`` (scrolls all Qdrant payloads) and
        ``JSONStore`` (iterates store metadata).  Entries without a ``phash``
        field are silently skipped.

        Args:
            idx: A ``QdrantIndex`` or ``JSONStore`` instance.

        Returns:
            Number of pHash values loaded.
        """
        loaded = 0

        if isinstance(idx, QdrantIndex):
            loaded = self._load_from_qdrant(idx)
        elif isinstance(idx, JSONStore):
            loaded = self._load_from_store(idx)
        else:
            logger.warning(
                "PHashFilter.load_from_index: unsupported index type %s",
                type(idx).__name__,
            )

        logger.info("PHashFilter: loaded %d pHash values", loaded)
        return loaded

    def _load_from_qdrant(self, idx: QdrantIndex) -> int:
        """Load pHash values from all Qdrant point payloads."""
        loaded = 0
        offset = None
        while True:
            try:
                results = idx._client.scroll(
                    collection_name=idx._collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=["video_id", "phash"],
                    with_vectors=False,
                )
            except Exception as exc:
                logger.warning("PHashFilter Qdrant scroll failed: %s", exc)
                break
            points, next_offset = results
            for point in points:
                vid = point.payload.get("video_id")
                phash_raw = point.payload.get("phash")
                if isinstance(vid, str) and isinstance(phash_raw, str):
                    try:
                        self._hashes[vid] = int(phash_raw, 16)
                        loaded += 1
                    except ValueError:
                        pass
            if next_offset is None:
                break
            offset = next_offset
        return loaded

    def _load_from_store(self, store: JSONStore) -> int:
        """Load pHash values from a JSONStore's metadata records."""
        loaded = 0
        for video_id in store.list_all():
            try:
                record = store.get(video_id)
            except KeyError:
                continue
            phash_raw = record.get("phash")
            if isinstance(phash_raw, str):
                try:
                    self._hashes[video_id] = int(phash_raw, 16)
                    loaded += 1
                except ValueError:
                    pass
        return loaded

    def save(self, path: str | Path) -> None:
        """Persist the hash database to a pickle file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._hashes, f)

    @classmethod
    def load(cls, path: str | Path) -> "PHashFilter":
        """Load a hash database from a pickle file.

        Args:
            path: Path to the saved pickle file.

        Returns:
            Loaded PHashFilter instance.
        """
        filt = cls()
        with open(path, "rb") as f:
            filt._hashes = pickle.load(f)  # noqa: S301
        return filt

    def __len__(self) -> int:
        return len(self._hashes)

    def __contains__(self, video_id: str) -> bool:
        return video_id in self._hashes
