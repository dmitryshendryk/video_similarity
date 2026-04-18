"""
Shared helpers for the fast_dedup CLI commands.

Extracted from fast_dedup.py to keep the CLI module focused on command definitions.
"""

from pathlib import Path

import numpy as np
import torch

from dedup.index import VideoIndex
from dedup.qdrant_index import QdrantIndex


MetadataValue = str | int | float | bool | None


def create_index(
    index_backend: str,
    dim: int,
    index_path: Path,
    qdrant_url: str | None = None,
    collection_name: str = "video_dedup",
) -> VideoIndex | QdrantIndex:
    """Create a new index based on the selected backend.

    Args:
        index_backend: Backend type — 'faiss' or 'qdrant'.
        dim: Vector dimensionality.
        index_path: Base path for file-based persistence.
        qdrant_url: Qdrant server URL (remote mode). If None, uses local on-disk mode.
        collection_name: Qdrant collection name.

    Returns:
        New index instance.
    """
    if index_backend == "qdrant":
        if qdrant_url is not None:
            return QdrantIndex(dim=dim, url=qdrant_url, collection_name=collection_name)
        return QdrantIndex(
            dim=dim, path=str(index_path) + "_qdrant", collection_name=collection_name
        )
    return VideoIndex(dim=dim)


def load_index(
    index_backend: str,
    index_path: Path,
    qdrant_url: str | None = None,
    collection_name: str = "video_dedup",
) -> VideoIndex | QdrantIndex:
    """Load an existing index based on the selected backend.

    Args:
        index_backend: Backend type — 'faiss' or 'qdrant'.
        index_path: Base path for file-based persistence.
        qdrant_url: Qdrant server URL (remote mode). If None, uses local on-disk mode.
        collection_name: Qdrant collection name.

    Returns:
        Loaded index instance.
    """
    if index_backend == "qdrant":
        if qdrant_url is not None:
            return QdrantIndex.load(
                path=index_path, url=qdrant_url, collection_name=collection_name
            )
        return QdrantIndex.load(
            path=str(index_path) + "_qdrant", collection_name=collection_name
        )
    return VideoIndex.load(index_path)


def extract_video_metadata(
    video_path: Path,
) -> dict[str, MetadataValue]:
    """Extract basic metadata from a video file using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with path, width, height, codec, duration, bitrate, fps.
    """
    metadata: dict[str, MetadataValue] = {"path": str(video_path)}
    try:
        import ffmpeg

        probe = ffmpeg.probe(str(video_path))
        video_stream = next(
            (s for s in probe["streams"] if s["codec_type"] == "video"), None
        )
        if video_stream is not None:
            metadata["width"] = int(video_stream.get("width", 0))
            metadata["height"] = int(video_stream.get("height", 0))
            metadata["codec"] = video_stream.get("codec_name")
            metadata["duration"] = float(
                video_stream.get("duration", probe.get("format", {}).get("duration", 0))
            )
            bit_rate = video_stream.get(
                "bit_rate", probe.get("format", {}).get("bit_rate")
            )
            metadata["bitrate"] = int(bit_rate) if bit_rate is not None else None
            fps_str = video_stream.get("r_frame_rate", "0/1")
            num, den = fps_str.split("/")
            metadata["fps"] = round(int(num) / max(int(den), 1), 2)
    except Exception:
        pass
    return metadata


def duration_bucket(duration: float | None) -> str:
    """Classify a video duration into a bucket for Qdrant filtering.

    Buckets: 0-10s, 10-30s, 30-60s, 1-5m, 5-30m, 30m+, unknown.
    """
    if duration is None or duration <= 0:
        return "unknown"
    if duration <= 10:
        return "0-10s"
    if duration <= 30:
        return "10-30s"
    if duration <= 60:
        return "30-60s"
    if duration <= 300:
        return "1-5m"
    if duration <= 1800:
        return "5-30m"
    return "30m+"


def aspect_ratio_class(width: int | None, height: int | None) -> str:
    """Classify aspect ratio into a category for Qdrant filtering.

    Categories: landscape, portrait, square, unknown.
    """
    if width is None or height is None or width <= 0 or height <= 0:
        return "unknown"
    ratio = width / height
    if ratio > 1.2:
        return "landscape"
    if ratio < 0.8:
        return "portrait"
    return "square"


def discover_videos(dataset_path: Path, pattern: str) -> list[tuple[str, Path]]:
    """Discover video files in a dataset directory.

    Args:
        dataset_path: Root directory containing videos.
        pattern: Glob pattern for finding video files (e.g., '**/*.mp4').

    Returns:
        List of (video_id, video_path) tuples.
    """
    videos: list[tuple[str, Path]] = []
    for video_path in sorted(dataset_path.glob(pattern)):
        if video_path.is_file():
            video_id = video_path.stem
            videos.append((video_id, video_path))
    return videos


def load_video_tensor(
    video_path: Path,
    fps: int = 1,
    resize: int = 256,
    crop: int = 224,
    keyframes_only: bool = False,
    max_frames: int = 60,
) -> torch.Tensor:
    """Load a video file as a tensor of frames.

    Args:
        video_path: Path to the video file.
        fps: Frames per second to sample (ignored when keyframes_only=True).
        resize: Resize dimension.
        crop: Center crop dimension.
        keyframes_only: If True, extract only I-frames instead of uniform fps
            sampling.  Falls back to fps=1 if zero I-frames are found.
        max_frames: Maximum number of frames to return.

    Returns:
        Tensor of shape (T, H, W, C) as uint8.
    """
    from utils import load_video_ffmpeg

    effective_fps = None if keyframes_only else fps
    video_array = load_video_ffmpeg(
        str(video_path),
        fps=effective_fps,
        crop=crop,
        resize=resize,
        keyframes_only=keyframes_only,
        max_frames=max_frames,
    )
    if video_array is None or (
        isinstance(video_array, np.ndarray) and video_array.shape[0] == 0
    ):
        return torch.empty(0)
    if isinstance(video_array, np.ndarray):
        return torch.from_numpy(video_array)
    return video_array
