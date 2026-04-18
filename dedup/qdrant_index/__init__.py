"""
Qdrant-backed vector index package.

Re-exports QdrantIndex and type aliases for backward-compatible imports:
    from dedup.qdrant_index import QdrantIndex
"""

from dedup.qdrant_index._index import MetadataDict, MetadataValue, QdrantIndex
