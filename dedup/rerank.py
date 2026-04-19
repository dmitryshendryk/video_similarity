"""
ViSiL re-ranking for FAISS top-K candidates.

Runs the full ViSiL frame-to-frame similarity computation on a small
candidate set (e.g., 50 videos) instead of the entire database.
This preserves ViSiL's high accuracy while keeping query time manageable.
"""

import h5py
import torch
import numpy as np

from model.similarity_network import SimilarityNetwork


def rerank_with_visil(
    query_features: torch.Tensor,
    candidate_ids: list[str],
    features_hdf5_path: str,
    pretrained: str = "s2vs_dns",
    device: str = "cuda",
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Re-rank FAISS candidates using full ViSiL similarity.

    Args:
        query_features: Query video features of shape (T, R, D).
        candidate_ids: List of candidate video IDs from FAISS search.
        features_hdf5_path: Path to HDF5 file with precomputed features.
        pretrained: Pretrained ViSiL model name.
        device: Torch device string.
        top_n: Number of top results to return after re-ranking.

    Returns:
        List of (video_id, score) tuples sorted by descending ViSiL similarity.
    """
    sim_network = SimilarityNetwork["ViSiL"].get_model(
        dims=query_features.shape[-1],
        attention=True,
        pretrained=pretrained,
    )
    sim_network = sim_network.to(device).eval()

    query = query_features.unsqueeze(0).to(device)  # (1, T, R, D)
    query_indexed = sim_network.index_video(query)

    results: list[tuple[str, float]] = []

    with h5py.File(features_hdf5_path, "r") as hdf5:
        for video_id in candidate_ids:
            if video_id not in hdf5:
                continue
            target_np = np.array(hdf5[video_id])
            target = (
                torch.from_numpy(target_np).unsqueeze(0).to(device)
            )  # (1, T2, R, D)
            target_indexed = sim_network.index_video(target)

            with torch.no_grad():
                score = sim_network.calculate_video_similarity(
                    query_indexed, target_indexed
                )
            results.append((video_id, float(score.item())))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def rerank_from_tensors(
    query_features: torch.Tensor,
    candidates: dict[str, torch.Tensor],
    pretrained: str = "s2vs_dns",
    device: str = "cuda",
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Re-rank candidates from in-memory feature tensors.

    Args:
        query_features: Query video features of shape (T, R, D).
        candidates: Dict mapping video_id -> features tensor of shape (T, R, D).
        pretrained: Pretrained ViSiL model name.
        device: Torch device string.
        top_n: Number of top results to return after re-ranking.

    Returns:
        List of (video_id, score) tuples sorted by descending ViSiL similarity.
    """
    sim_network = SimilarityNetwork["ViSiL"].get_model(
        dims=query_features.shape[-1],
        attention=True,
        pretrained=pretrained,
    )
    sim_network = sim_network.to(device).eval()

    query = query_features.unsqueeze(0).to(device)
    query_indexed = sim_network.index_video(query)

    results: list[tuple[str, float]] = []

    for video_id, target_features in candidates.items():
        target = target_features.unsqueeze(0).to(device)
        target_indexed = sim_network.index_video(target)

        with torch.no_grad():
            score = sim_network.calculate_video_similarity(
                query_indexed, target_indexed
            )
        results.append((video_id, float(score.item())))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]
