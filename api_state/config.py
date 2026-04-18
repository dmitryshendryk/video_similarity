"""
Configuration constants, defaults, and persistence for the API server.
"""

import json

from pathlib import Path

import torch

# --- Directories ---
DATA_DIR = Path("api_data")
UPLOADS_DIR = DATA_DIR / "uploads"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
INDEX_DIR = DATA_DIR / "index"
CONFIG_PATH = DATA_DIR / "api_config.json"
FEATURES_HDF5 = DATA_DIR / "features.h5"
CACHE_DIR = DATA_DIR / "descriptor_cache"

for _d in [UPLOADS_DIR, THUMBNAILS_DIR, INDEX_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# --- Default config ---
DEFAULT_CONFIG: dict[str, str | int | float | bool | None] = {
    "index_backend": "faiss",
    "embedding_backend": "s2vs",
    "pretrained": "s2vs_dns",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "qdrant_url": None,
    "collection_name": "video_dedup",
    "threshold": 0.8,
    "top_k": 50,
    "qdrant_binary_quantization": False,
    "phash_skip_cnn": False,
}


def load_config() -> dict[str, str | int | float | bool | None]:
    """Load config from disk, merging with defaults for any missing keys."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            saved = json.load(f)
        merged = {**DEFAULT_CONFIG, **saved}
        return merged
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict[str, str | int | float | bool | None]) -> None:
    """Persist config dict to disk as JSON."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


config = load_config()
