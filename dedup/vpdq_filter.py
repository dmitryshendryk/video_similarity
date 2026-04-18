"""
Optional vpdq (Video PDQ) perceptual hash pre-filter.

Ultra-fast near-exact duplicate detection using Meta's Video PDQ hashes.
Catches re-encodes, resolution changes, minor crops, and watermarks.
Gracefully degrades if vpdq is not installed — pipeline skips this stage.
"""

import pickle

from pathlib import Path

try:
    import vpdq as _vpdq

    _VPDQ_AVAILABLE = True
except ImportError:
    _VPDQ_AVAILABLE = False


class VPDQFilter:
    """Perceptual hash database for fast near-exact video matching."""

    def __init__(self) -> None:
        self._hashes: dict[str, list[object]] = {}

    @staticmethod
    def is_available() -> bool:
        """Check if vpdq library is installed.

        Returns:
            True if vpdq is importable.
        """
        return _VPDQ_AVAILABLE

    def compute_hash(self, video_path: str) -> list[object]:
        """Compute perceptual hash for a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            List of vpdq hash features for the video.

        Raises:
            RuntimeError: If vpdq is not installed.
        """
        if not _VPDQ_AVAILABLE:
            raise RuntimeError("vpdq is not installed. Install with: pip install vpdq")
        hashes = _vpdq.computeHash(video_path)
        return hashes

    def add(self, video_id: str, video_hash: list[object]) -> None:
        """Add a video hash to the database.

        Args:
            video_id: Unique identifier for the video.
            video_hash: Hash features from compute_hash().
        """
        self._hashes[video_id] = video_hash

    def add_from_file(self, video_id: str, video_path: str) -> None:
        """Compute hash and add to database in one step.

        Args:
            video_id: Unique identifier for the video.
            video_path: Path to the video file.
        """
        video_hash = self.compute_hash(video_path)
        self.add(video_id, video_hash)

    def find_matches(
        self, query_hash: list[object], threshold: float = 0.9
    ) -> list[tuple[str, float]]:
        """Find videos matching a query hash above a similarity threshold.

        Args:
            query_hash: Hash features of the query video.
            threshold: Minimum similarity score (0.0 to 1.0).

        Returns:
            List of (video_id, similarity) tuples, sorted by descending similarity.

        Raises:
            RuntimeError: If vpdq is not installed.
        """
        if not _VPDQ_AVAILABLE:
            raise RuntimeError("vpdq is not installed. Install with: pip install vpdq")

        results: list[tuple[str, float]] = []
        for video_id, stored_hash in self._hashes.items():
            match_pct = _vpdq.matchTwoHash(query_hash, stored_hash)
            similarity = match_pct / 100.0
            if similarity >= threshold:
                results.append((video_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def save(self, path: str | Path) -> None:
        """Save hash database to disk.

        Args:
            path: File path for the pickle file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._hashes, f)

    @classmethod
    def load(cls, path: str | Path) -> "VPDQFilter":
        """Load hash database from disk.

        Args:
            path: Path to the saved pickle file.

        Returns:
            Loaded VPDQFilter instance.
        """
        filt = cls()
        with open(path, "rb") as f:
            filt._hashes = pickle.load(f)  # noqa: S301
        return filt

    def __len__(self) -> int:
        return len(self._hashes)

    def __contains__(self, video_id: str) -> bool:
        return video_id in self._hashes
