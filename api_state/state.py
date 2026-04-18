"""
Module-level singletons for the API server: embed_backend, index, store, and phash_filter.
"""

import logging

from pathlib import Path

from api_state.config import INDEX_DIR, config
from dedup.helpers import create_index, load_index
from dedup.index import VideoIndex
from dedup.phash_filter import PHashFilter
from dedup.qdrant_index import QdrantIndex
from dedup.store import JSONStore
from model.video_descriptor import get_backend

logger = logging.getLogger("api_server")

embed_backend = get_backend(
    name=str(config.get("embedding_backend", "s2vs")),
    pretrained=str(config.get("pretrained", "s2vs_dns")),
    device=str(config.get("device", "cpu")),
)

index_path = INDEX_DIR / "dedup_index"

index: VideoIndex | QdrantIndex = (
    load_index(
        str(config["index_backend"]),
        index_path,
        qdrant_url=str(config["qdrant_url"]) if config.get("qdrant_url") else None,
        collection_name=str(config.get("collection_name", "video_dedup")),
    )
    if (
        index_path.with_suffix(".faiss").exists()
        or index_path.with_suffix(".json").exists()
    )
    else create_index(
        str(config["index_backend"]),
        embed_backend.dim,
        index_path,
        qdrant_url=str(config["qdrant_url"]) if config.get("qdrant_url") else None,
        collection_name=str(config.get("collection_name", "video_dedup")),
    )
)

store = JSONStore()
metadata_path = Path(str(index_path) + "_metadata.json")
if metadata_path.exists():
    store = JSONStore.load(metadata_path)

# pHash pre-filter: populated from existing index/store payloads at startup.
# Set to None if imagehash is not installed.
phash_filter: PHashFilter | None = None
if PHashFilter.is_available():
    phash_filter = PHashFilter()
    try:
        phash_filter.load_from_index(index if isinstance(index, QdrantIndex) else store)
    except Exception as exc:
        logger.warning("PHashFilter startup load failed: %s", exc)
        phash_filter = PHashFilter()
else:
    logger.warning(
        "imagehash not installed — pHash pre-filter disabled. "
        "Install with: pip install imagehash"
    )
