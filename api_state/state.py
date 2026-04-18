"""
Module-level singletons for the API server: embed_backend, index, and store.
"""

from pathlib import Path

from api_state.config import INDEX_DIR, config
from dedup.helpers import create_index, load_index
from dedup.index import VideoIndex
from dedup.qdrant_index import QdrantIndex
from dedup.store import JSONStore
from model.video_descriptor import get_backend

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
