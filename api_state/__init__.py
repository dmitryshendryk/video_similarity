"""
api_state package: re-exports all public names consumed by api_server.py.

The 15-name import contract:
    from api_state import (
        THUMBNAILS_DIR, UPLOADS_DIR, config, save_config,
        embed_backend, index, index_path, metadata_path, store, phash_filter,
        delete_features, save_features,
        generate_thumbnail,
        get_video_metadata, run_search_pipeline,
    )
"""

from api_state.config import THUMBNAILS_DIR, UPLOADS_DIR, config, save_config
from api_state.search import get_video_metadata, run_search_pipeline
from api_state.state import (
    embed_backend,
    index,
    index_path,
    metadata_path,
    phash_filter,
    store,
)
from api_state.cache import clear_descriptor_cache
from api_state.thumbnails import generate_thumbnail
from api_state.visil import delete_features, save_features
