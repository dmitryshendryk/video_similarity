"""
api_state package: re-exports all public names consumed by api_server.py.

The 14-name import contract:
    from api_state import (
        THUMBNAILS_DIR, UPLOADS_DIR, config, save_config,
        embed_backend, index, index_path, metadata_path, store,
        delete_features, save_features,
        generate_thumbnail,
        get_video_metadata, run_search_pipeline,
    )
"""

from api_state.config import THUMBNAILS_DIR, UPLOADS_DIR, config, save_config
from api_state.search import get_video_metadata, run_search_pipeline
from api_state.state import embed_backend, index, index_path, metadata_path, store
from api_state.thumbnails import generate_thumbnail
from api_state.visil import delete_features, save_features
