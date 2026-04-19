"""
In-memory and disk-based descriptor cache for video feature extraction.
"""

import hashlib
import logging
import time

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from api_state.config import CACHE_DIR
from api_state.state import embed_backend
from model.video_descriptor import S2VSBackend

logger = logging.getLogger("api_server")

MAX_CACHE_SIZE = 100
CACHE_TTL_S = 86400  # 24 hours

_descriptor_cache: dict[str, "CachedDescriptor"] = {}


@dataclass
class CachedDescriptor:
    """Cached extraction result for a video file."""

    features: torch.Tensor | None
    descriptor: np.ndarray
    created_at: float


def _file_hash(path: Path) -> str:
    """Fast SHA-256 hash of file content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_cached_descriptors(
    video_path: Path,
    video_tensor: torch.Tensor,
) -> CachedDescriptor:
    """Get features + descriptor, using cache if available."""
    fhash = _file_hash(video_path)
    disk_path = CACHE_DIR / f"{fhash}.npz"

    if fhash in _descriptor_cache:
        cached = _descriptor_cache[fhash]
        if time.time() - cached.created_at > CACHE_TTL_S:
            logger.info("Descriptor cache EXPIRED (%s)", fhash[:12])
            del _descriptor_cache[fhash]
            disk_path.unlink(missing_ok=True)
        else:
            logger.info("Descriptor cache HIT (%s)", fhash[:12])
            return cached

    if disk_path.exists() and time.time() - disk_path.stat().st_mtime < CACHE_TTL_S:
        logger.info("Descriptor cache DISK HIT (%s)", fhash[:12])
        data = np.load(disk_path, allow_pickle=False)
        feats = torch.from_numpy(data["features"]) if "features" in data else None
        cached = CachedDescriptor(feats, data["descriptor"], disk_path.stat().st_mtime)
        _descriptor_cache[fhash] = cached
        return cached

    logger.info("Descriptor cache MISS (%s)", fhash[:12])
    if isinstance(embed_backend, S2VSBackend):
        result = embed_backend.extract_all(video_tensor)
        features = result.frame_features
        desc = result.descriptor
    else:
        features = None
        desc = embed_backend.extract_descriptor(video_tensor)

    cached = CachedDescriptor(
        features=features, descriptor=desc, created_at=time.time()
    )
    save_dict: dict[str, np.ndarray] = {"descriptor": desc}
    if features is not None:
        save_dict["features"] = features.numpy()
    np.savez(disk_path, **save_dict)
    if len(_descriptor_cache) >= MAX_CACHE_SIZE:
        _descriptor_cache.pop(next(iter(_descriptor_cache)))
    _descriptor_cache[fhash] = cached
    return cached


def clear_descriptor_cache() -> int:
    """Clear both in-memory and disk descriptor caches.

    Returns:
        Number of disk cache files removed.
    """
    _descriptor_cache.clear()
    removed = 0
    for f in CACHE_DIR.glob("*.npz"):
        f.unlink(missing_ok=True)
        removed += 1
    logger.info("Descriptor cache cleared (%d disk files removed)", removed)
    return removed
