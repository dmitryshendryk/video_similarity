"""
ViSiL similarity network and per-frame feature persistence helpers.
"""

import logging

import numpy as np
import h5py
import torch

logger = logging.getLogger(__name__)

from api_state.config import FEATURES_HDF5, config
from api_state.state import embed_backend
from model.similarity_network import SimilarityNetwork

_sim_network: torch.nn.Module | None = None


def get_sim_network() -> torch.nn.Module:
    """Get or create the ViSiL similarity network for re-ranking."""
    global _sim_network
    if _sim_network is None:
        device = str(config.get("device", "cpu"))
        pretrained = str(config.get("pretrained", "s2vs_dns"))
        _sim_network = SimilarityNetwork["ViSiL"].get_model(
            dims=embed_backend.dim,
            attention=True,
            pretrained=pretrained,
        )
        _sim_network = _sim_network.to(device).eval()
    return _sim_network


def save_features(video_id: str, features: torch.Tensor) -> None:
    """Save per-frame region features to HDF5 for ViSiL re-ranking."""
    with h5py.File(FEATURES_HDF5, "a") as f:
        if video_id in f:
            del f[video_id]
        f.create_dataset(video_id, data=features.numpy())


def load_features(video_id: str) -> torch.Tensor | None:
    """Load per-frame region features from HDF5."""
    if not FEATURES_HDF5.exists():
        return None
    with h5py.File(FEATURES_HDF5, "r") as f:
        if video_id not in f:
            return None
        return torch.from_numpy(np.array(f[video_id]))


def delete_features(video_id: str) -> None:
    """Remove features for a video from HDF5."""
    if not FEATURES_HDF5.exists():
        return
    with h5py.File(FEATURES_HDF5, "a") as f:
        if video_id in f:
            del f[video_id]


def rerank_visil(
    query_features: torch.Tensor,
    candidate_ids: list[str],
) -> list[tuple[str, float]]:
    """Re-rank candidates using full ViSiL frame-to-frame similarity.

    Args:
        query_features: Query video features of shape (T, R, D).
        candidate_ids: Video IDs to re-rank.

    Returns:
        List of (video_id, score) sorted by descending ViSiL similarity.
    """
    sim_net = get_sim_network()
    device = str(config.get("device", "cpu"))

    query = query_features.unsqueeze(0).to(device)
    query_indexed = sim_net.index_video(query)

    results: list[tuple[str, float]] = []
    for vid in candidate_ids:
        target_features = load_features(vid)
        if target_features is None:
            continue
        target = target_features.unsqueeze(0).to(device)
        target_indexed = sim_net.index_video(target)

        with torch.no_grad():
            try:
                # Use similarity_matrix(normalize=True) + v2v_sim to match
                # ViSiL.forward() which normalizes to [0,1] BEFORE v2v_sim.
                # calculate_video_similarity() doesn't normalize, producing
                # systematically lower scores.
                sim_matrix, sim_mask = sim_net.similarity_matrix(
                    query_indexed, target_indexed, normalize=True
                )
                raw_score = sim_net.v2v_sim(sim_matrix, sim_mask)
            except RuntimeError:
                logger.warning(
                    "ViSiL re-rank skipped for %s (too few frames for pooling)", vid
                )
                continue
            score = float(raw_score.item())
        results.append((vid, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
