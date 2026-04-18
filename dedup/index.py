"""
FAISS-based video index for fast approximate nearest neighbor search.

Uses IndexFlatIP (exact inner product on L2-normalized vectors = cosine similarity).
At 10K scale, exact search is fast enough (<1ms) that IVF adds complexity with no benefit.
"""

import json
import faiss
import numpy as np

from pathlib import Path


class VideoIndex:
    """FAISS index mapping video IDs to L2-normalized descriptors."""

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._id_to_pos: dict[str, int] = {}

    def add(self, video_id: str, descriptor: np.ndarray) -> None:
        """Add a single video descriptor to the index.

        Args:
            video_id: Unique identifier for the video.
            descriptor: L2-normalized vector of shape (dim,).
        """
        if video_id in self._id_to_pos:
            self.remove(video_id)
        vec = descriptor.reshape(1, -1).astype(np.float32)
        self._index.add(vec)
        self._id_to_pos[video_id] = len(self._ids)
        self._ids.append(video_id)

    def add_batch(self, ids: list[str], descriptors: np.ndarray) -> None:
        """Add multiple video descriptors at once.

        Args:
            ids: List of unique video identifiers.
            descriptors: L2-normalized array of shape (N, dim).
        """
        vecs = descriptors.astype(np.float32)
        start_pos = len(self._ids)
        self._index.add(vecs)
        for i, video_id in enumerate(ids):
            self._id_to_pos[video_id] = start_pos + i
            self._ids.append(video_id)

    def search(self, query: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        """Find the top-K most similar videos to a query descriptor.

        Args:
            query: L2-normalized query vector of shape (dim,).
            top_k: Number of results to return.

        Returns:
            List of (video_id, score) tuples sorted by descending similarity.
        """
        if len(self._ids) == 0:
            return []
        k = min(top_k, len(self._ids))
        vec = query.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(vec, k)
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self._ids[idx], float(score)))
        return results

    def remove(self, video_id: str) -> None:
        """Remove a video from the index. Rebuilds the index internally.

        Args:
            video_id: ID of the video to remove.

        Raises:
            KeyError: If video_id is not in the index.
        """
        if video_id not in self._id_to_pos:
            raise KeyError(f"Video '{video_id}' not found in index")

        pos = self._id_to_pos[video_id]
        # Reconstruct all vectors except the one to remove
        n = self._index.ntotal
        all_vecs = np.zeros((n, self._dim), dtype=np.float32)
        for i in range(n):
            all_vecs[i] = self._index.reconstruct(i)

        keep_mask = np.ones(n, dtype=bool)
        keep_mask[pos] = False
        kept_vecs = all_vecs[keep_mask]
        kept_ids = [vid for vid in self._ids if vid != video_id]

        self._index = faiss.IndexFlatIP(self._dim)
        self._ids = []
        self._id_to_pos = {}

        if len(kept_ids) > 0:
            self._index.add(kept_vecs)
            for i, vid in enumerate(kept_ids):
                self._id_to_pos[vid] = i
                self._ids.append(vid)

    def save(self, path: str | Path) -> None:
        """Save the index and ID mapping to disk.

        Args:
            path: Base path (without extension). Creates .faiss and .json files.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".json"), "w") as f:
            json.dump({"ids": self._ids, "dim": self._dim}, f)

    @classmethod
    def load(cls, path: str | Path) -> "VideoIndex":
        """Load a saved index from disk.

        Args:
            path: Base path (without extension). Reads .faiss and .json files.

        Returns:
            Loaded VideoIndex instance.
        """
        path = Path(path)
        with open(path.with_suffix(".json")) as f:
            meta = json.load(f)

        idx = cls(dim=meta["dim"])
        idx._index = faiss.read_index(str(path.with_suffix(".faiss")))
        idx._ids = meta["ids"]
        idx._id_to_pos = {vid: i for i, vid in enumerate(idx._ids)}
        return idx

    def get_descriptor(self, video_id: str) -> np.ndarray:
        """Reconstruct the stored descriptor for a video.

        Args:
            video_id: Unique identifier for the video.

        Returns:
            Descriptor array of shape (dim,).

        Raises:
            KeyError: If video_id is not in the index.
        """
        if video_id not in self._id_to_pos:
            raise KeyError(f"Video '{video_id}' not found in index")
        pos = self._id_to_pos[video_id]
        return self._index.reconstruct(pos)

    def list_all(self) -> list[str]:
        """List all video IDs in the index.

        Returns:
            List of video ID strings.
        """
        return list(self._ids)

    def __len__(self) -> int:
        return self._index.ntotal

    def __contains__(self, video_id: str) -> bool:
        return video_id in self._id_to_pos
