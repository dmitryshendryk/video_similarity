"""
Qdrant-backed vector index for video deduplication.

Alternative to the FAISS-based VideoIndex. Qdrant provides:
- Built-in persistence (no separate save/load)
- Metadata filtering alongside vector search
- Concurrent access for production deployments
- Local on-disk mode (no server needed) or remote server mode

Requires: pip install qdrant-client
"""

import logging
import uuid

import numpy as np

from pathlib import Path

from dedup.qdrant_index._imports import (
    _QDRANT_AVAILABLE,
    _QDRANT_BQ_AVAILABLE,
)

# These names are conditionally available; referenced via string guards below.
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PayloadSchemaType,
        PointStruct,
        VectorParams,
    )
except ImportError:
    pass

try:
    from qdrant_client.models import (
        BinaryQuantization,
        BinaryQuantizationConfig,
        QuantizationSearchParams,
        SearchParams,
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)

MetadataValue = str | int | float | bool | None
MetadataDict = dict[str, MetadataValue]


class QdrantIndex:
    """Qdrant-backed vector index with integrated metadata storage.

    Supports two modes:
    - **Local**: ``QdrantIndex(dim=512, path="./qdrant_data")`` — on-disk, no server
    - **Remote**: ``QdrantIndex(dim=512, url="http://localhost:6333")`` — production server
    """

    def __init__(
        self,
        dim: int = 512,
        collection_name: str = "video_dedup",
        url: str | None = None,
        path: str | None = None,
        binary_quantization: bool = False,
    ) -> None:
        if not _QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant backend requires qdrant-client. "
                "Install with: pip install qdrant-client"
            ) from None

        self._dim = dim
        self._collection_name = collection_name
        self._binary_quantization = binary_quantization

        if url is not None:
            self._client = QdrantClient(url=url)
        elif path is not None:
            self._client = QdrantClient(path=path)
        else:
            # In-memory mode (for testing)
            self._client = QdrantClient(location=":memory:")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self._client.get_collections().collections
        existing = [c.name for c in collections]
        if self._collection_name not in existing:
            quantization_config = (
                BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=True))
                if self._binary_quantization and _QDRANT_BQ_AVAILABLE
                else None
            )
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._dim,
                    distance=Distance.COSINE,
                ),
                quantization_config=quantization_config,
            )
            for field, schema in [
                ("video_id", PayloadSchemaType.KEYWORD),
                ("duration_bucket", PayloadSchemaType.KEYWORD),
                ("aspect_ratio", PayloadSchemaType.KEYWORD),
                ("phash", PayloadSchemaType.KEYWORD),
            ]:
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field,
                    field_schema=schema,
                )

    def _video_id_filter(self, video_id: str) -> "Filter":
        """Build a Qdrant filter matching a specific video_id."""
        return Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        )

    def _get_point_id(self, video_id: str) -> str | None:
        """Find the Qdrant point ID for a given video_id."""
        results = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=self._video_id_filter(video_id),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        points = results[0]
        if points:
            return str(points[0].id)
        return None

    def add(
        self,
        video_id: str,
        descriptor: np.ndarray,
        metadata: MetadataDict | None = None,
    ) -> None:
        """Add a single video descriptor with optional metadata.

        Args:
            video_id: Unique identifier for the video.
            descriptor: L2-normalized vector of shape (dim,).
            metadata: Optional dict of metadata (path, resolution, etc.).
        """
        existing_id = self._get_point_id(video_id)
        if existing_id is not None:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=[existing_id],
            )

        payload: dict[str, MetadataValue] = {"video_id": video_id}
        if metadata is not None:
            payload.update(metadata)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=descriptor.flatten().tolist(),
            payload=payload,
        )
        self._client.upsert(collection_name=self._collection_name, points=[point])

    def add_batch(
        self,
        ids: list[str],
        descriptors: np.ndarray,
        metadata_list: list[MetadataDict] | None = None,
    ) -> None:
        """Add multiple video descriptors at once.

        Args:
            ids: List of unique video identifiers.
            descriptors: L2-normalized array of shape (N, dim).
            metadata_list: Optional list of metadata dicts, one per video.
        """
        points: list[PointStruct] = []
        for i, video_id in enumerate(ids):
            payload: dict[str, MetadataValue] = {"video_id": video_id}
            if metadata_list is not None and i < len(metadata_list):
                payload.update(metadata_list[i])
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=descriptors[i].flatten().tolist(),
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 50,
        query_filter: "Filter | None" = None,
    ) -> list[tuple[str, float]]:
        """Find the top-K most similar videos.

        Args:
            query: L2-normalized query vector of shape (dim,).
            top_k: Number of results to return.
            query_filter: Optional Qdrant Filter to narrow the search space.

        Returns:
            List of (video_id, score) tuples sorted by descending similarity.
        """
        search_params = (
            SearchParams(
                quantization=QuantizationSearchParams(rescore=True, oversampling=1.5)
            )
            if self._binary_quantization and _QDRANT_BQ_AVAILABLE
            else None
        )
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query.flatten().tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            search_params=search_params,
        )
        output: list[tuple[str, float]] = []
        for point in results.points:
            vid = point.payload.get("video_id", "")
            if isinstance(vid, str):
                output.append((vid, float(point.score)))
        return output

    def remove(self, video_id: str) -> None:
        """Remove a video from the index.

        Args:
            video_id: ID of the video to remove.

        Raises:
            KeyError: If video_id is not in the index.
        """
        point_id = self._get_point_id(video_id)
        if point_id is None:
            raise KeyError(f"Video '{video_id}' not found in index")
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=[point_id],
        )

    def get_metadata(self, video_id: str) -> MetadataDict:
        """Get stored metadata for a video.

        Args:
            video_id: Unique identifier for the video.

        Returns:
            Dict with all payload fields (video_id, path, metadata, etc.).

        Raises:
            KeyError: If video_id is not found.
        """
        results = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=self._video_id_filter(video_id),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        points = results[0]
        if not points:
            raise KeyError(f"Video '{video_id}' not found in index")
        return dict(points[0].payload)

    def get_descriptor(self, video_id: str) -> np.ndarray:
        """Retrieve the stored descriptor vector for a video.

        Args:
            video_id: Unique identifier for the video.

        Returns:
            Descriptor array of shape (dim,).

        Raises:
            KeyError: If video_id is not found.
        """
        results = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=self._video_id_filter(video_id),
            limit=1,
            with_payload=False,
            with_vectors=True,
        )
        points = results[0]
        if not points:
            raise KeyError(f"Video '{video_id}' not found in index")
        return np.array(points[0].vector, dtype=np.float32)

    def list_all(self) -> list[str]:
        """List all video IDs in the index.

        Returns:
            List of video ID strings.
        """
        video_ids: list[str] = []
        offset = None
        while True:
            results = self._client.scroll(
                collection_name=self._collection_name,
                limit=100,
                offset=offset,
                with_payload=["video_id"],
                with_vectors=False,
            )
            points, next_offset = results
            for point in points:
                vid = point.payload.get("video_id", "")
                if isinstance(vid, str):
                    video_ids.append(vid)
            if next_offset is None:
                break
            offset = next_offset
        return video_ids

    def scroll_all_payloads(self, fields: list[str]) -> list[dict[str, MetadataValue]]:
        """Scroll all points and return the requested payload fields.

        Provides a public alternative to accessing ``_client`` and
        ``_collection_name`` directly from external callers (e.g. PHashFilter).

        Args:
            fields: Payload field names to include in each returned dict.

        Returns:
            List of payload dicts, one per point.  Points without any of the
            requested fields are still included (with missing keys absent).
        """
        payloads: list[dict[str, MetadataValue]] = []
        offset = None
        while True:
            try:
                results = self._client.scroll(
                    collection_name=self._collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=fields,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.warning("QdrantIndex.scroll_all_payloads failed: %s", exc)
                break
            points, next_offset = results
            for point in points:
                payloads.append(dict(point.payload))
            if next_offset is None:
                break
            offset = next_offset
        return payloads

    def save(self, path: str | Path) -> None:
        """Create a snapshot of the collection.

        For local mode, data is already persisted on disk.
        For remote mode, triggers a server-side snapshot.

        Args:
            path: Base path (used for naming the snapshot).
        """
        self._client.create_snapshot(collection_name=self._collection_name)

    @classmethod
    def load(
        cls,
        path: str | Path,
        collection_name: str = "video_dedup",
        url: str | None = None,
        binary_quantization: bool = False,
    ) -> "QdrantIndex":
        """Connect to an existing Qdrant collection.

        For local mode, opens the on-disk database at path.
        For remote mode, connects to the server (data already persisted).

        Args:
            path: Path to the local Qdrant data directory.
            collection_name: Name of the collection.
            url: URL of the remote Qdrant server (overrides path).
            binary_quantization: Whether binary quantization is active on this
                collection. Used to enable rescoring search params consistently.

        Returns:
            QdrantIndex connected to the existing collection.
        """
        if url is not None:
            idx = cls(
                url=url,
                collection_name=collection_name,
                binary_quantization=binary_quantization,
            )
        else:
            idx = cls(
                path=str(path),
                collection_name=collection_name,
                binary_quantization=binary_quantization,
            )

        info = idx._client.get_collection(collection_name)
        vector_config = info.config.params.vectors
        if isinstance(vector_config, VectorParams):
            idx._dim = vector_config.size
        return idx

    def __len__(self) -> int:
        info = self._client.get_collection(self._collection_name)
        return info.points_count

    def __contains__(self, video_id: str) -> bool:
        return self._get_point_id(video_id) is not None
