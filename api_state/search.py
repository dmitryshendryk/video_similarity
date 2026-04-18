"""
Search pipeline: two-stage vector search + ViSiL re-ranking with metadata filtering.
"""

import logging
import time

from dataclasses import dataclass
from pathlib import Path

from api_state.cache import get_cached_descriptors
from api_state.config import THUMBNAILS_DIR, config
from api_state.state import embed_backend, index, store
from api_state.visil import rerank_visil
from dedup.helpers import (
    aspect_ratio_class,
    duration_bucket,
    extract_video_metadata,
    load_video_tensor,
)
from dedup.qdrant_index import QdrantIndex

logger = logging.getLogger("api_server")

VISIL_MAX_CANDIDATES = 10
VISIL_SKIP_ABOVE = 0.95


@dataclass(frozen=True)
class QueryFilterResult:
    """Structured result from _build_query_filter, replacing bare tuple."""

    query_filter: object | None
    duration_bucket: str
    aspect_ratio: str


def _build_query_filter(video_path: Path) -> QueryFilterResult:
    """Build Qdrant metadata filter from query video properties."""
    if not isinstance(index, QdrantIndex):
        return QueryFilterResult(
            query_filter=None, duration_bucket="unknown", aspect_ratio="unknown"
        )

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    query_meta = extract_video_metadata(video_path)
    logger.info("Query metadata raw: %s", query_meta)
    q_bucket = duration_bucket(
        float(query_meta["duration"])
        if query_meta.get("duration") is not None
        else None
    )
    q_aspect = aspect_ratio_class(
        int(query_meta["width"]) if query_meta.get("width") is not None else None,
        int(query_meta["height"]) if query_meta.get("height") is not None else None,
    )
    conditions = []
    if q_bucket != "unknown":
        conditions.append(
            FieldCondition(key="duration_bucket", match=MatchValue(value=q_bucket))
        )
    if q_aspect != "unknown":
        conditions.append(
            FieldCondition(key="aspect_ratio", match=MatchValue(value=q_aspect))
        )
    qf = Filter(must=conditions) if conditions else None
    return QueryFilterResult(
        query_filter=qf, duration_bucket=q_bucket, aspect_ratio=q_aspect
    )


def get_video_metadata(vid: str) -> dict[str, str | float | bool | None]:
    """Get metadata for a video from Qdrant or JSONStore."""
    entry: dict[str, str | float | bool | None] = {"video_id": vid}
    if isinstance(index, QdrantIndex):
        try:
            entry.update(index.get_metadata(vid))
        except KeyError:
            pass
    elif vid in store:
        entry.update(store.get(vid))
    thumb = THUMBNAILS_DIR / f"{vid}.jpg"
    entry["has_thumbnail"] = thumb.exists()
    return entry


def run_search_pipeline(
    video_path: Path,
    top_k: int = 50,
    threshold: float = 0.8,
) -> dict[str, object]:
    """Full two-stage search: vector search + ViSiL re-ranking with timing."""
    t0 = time.perf_counter()

    video_tensor = load_video_tensor(video_path, keyframes_only=True)
    if video_tensor.shape[0] == 0:
        raise ValueError("Could not extract frames")
    t_decode = time.perf_counter() - t0
    logger.info("Decode: %.2fs (%d frames)", t_decode, video_tensor.shape[0])

    # Metadata filter
    qfr = _build_query_filter(video_path)
    logger.info(
        "Query filter: bucket=%s, aspect=%s", qfr.duration_bucket, qfr.aspect_ratio
    )

    # Extract features + descriptor (cached by file hash)
    t1 = time.perf_counter()
    cached = get_cached_descriptors(video_path, video_tensor)
    query_features = cached.features
    desc = cached.descriptor
    t_descriptor = time.perf_counter() - t1
    logger.info("Descriptor: %.2fs", t_descriptor)

    # Stage 1: vector search
    t2 = time.perf_counter()
    if isinstance(index, QdrantIndex) and qfr.query_filter is not None:
        stage1_results = index.search(desc, top_k=top_k, query_filter=qfr.query_filter)
    else:
        stage1_results = index.search(desc, top_k=top_k)
    t_search = time.perf_counter() - t2
    logger.info(
        "Stage 1: %.4fs, %d results, top=%s",
        t_search,
        len(stage1_results),
        [(v, round(s, 4)) for v, s in stage1_results[:5]],
    )

    pre_threshold = threshold * 0.5
    candidates = [(vid, s) for vid, s in stage1_results if s >= pre_threshold]

    # Stage 2: ViSiL re-ranking
    t_visil = 0.0
    if candidates and query_features is not None:
        high_conf = [(vid, s) for vid, s in candidates if s >= VISIL_SKIP_ABOVE]
        needs_rerank = [(vid, s) for vid, s in candidates if s < VISIL_SKIP_ABOVE]
        needs_rerank = needs_rerank[:VISIL_MAX_CANDIDATES]

        reranked: list[tuple[str, float]] = []
        if needs_rerank:
            t3 = time.perf_counter()
            rerank_ids = [vid for vid, _ in needs_rerank]
            reranked = rerank_visil(query_features, rerank_ids)
            t_visil = time.perf_counter() - t3
            logger.info(
                "Stage 2 ViSiL: %.2fs for %d candidates, top=%s",
                t_visil,
                len(rerank_ids),
                [(v, round(s, 4)) for v, s in reranked[:5]],
            )

        final_results = [(vid, s) for vid, s in high_conf if s >= threshold]
        final_results += [(vid, s) for vid, s in reranked if s >= threshold]
        final_results.sort(key=lambda x: x[1], reverse=True)
    else:
        final_results = [(vid, s) for vid, s in candidates if s >= threshold]

    t_total = time.perf_counter() - t0
    logger.info(
        "Total: %.2fs (decode=%.2f, desc=%.2f, search=%.4f, visil=%.2f)",
        t_total,
        t_decode,
        t_descriptor,
        t_search,
        t_visil,
    )

    matches: list[dict[str, str | float | None]] = []
    for vid, score in final_results:
        entry = get_video_metadata(vid)
        entry["score"] = round(float(score), 4)
        matches.append(entry)

    return {
        "total_results": len(matches),
        "threshold": threshold,
        "timing": {
            "total_s": round(t_total, 3),
            "decode_s": round(t_decode, 3),
            "descriptor_s": round(t_descriptor, 3),
            "vector_search_s": round(t_search, 4),
            "visil_rerank_s": round(t_visil, 3),
            "frames": video_tensor.shape[0],
            "candidates": len(candidates),
        },
        "results": matches,
    }
