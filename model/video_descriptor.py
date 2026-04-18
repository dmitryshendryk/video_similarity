"""
Pluggable embedding backends for fast video deduplication.

Extracts a single compact descriptor (512-D) per video for FAISS indexing.
Two backends: S2VS (default, reuses existing model) and CLIP (optional).
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import Any

from model.feature_extractor import FeatureExtractor
from model.similarity_network import SimilarityNetwork
from model.pooling import gem


class EmbeddingBackend(ABC):
    """Abstract interface for video embedding backends."""

    @abstractmethod
    def extract_descriptor(self, video_tensor: torch.Tensor) -> np.ndarray:
        """Extract a single L2-normalized descriptor from a video tensor.

        Args:
            video_tensor: Video frames tensor, shape depends on backend.

        Returns:
            L2-normalized descriptor array of shape (dim,).
        """
        ...

    @abstractmethod
    def extract_batch(self, video_tensors: list[torch.Tensor]) -> np.ndarray:
        """Extract descriptors for a batch of videos.

        Args:
            video_tensors: List of video frame tensors.

        Returns:
            L2-normalized descriptor array of shape (B, dim).
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the output descriptor."""
        ...


class S2VSBackend(EmbeddingBackend):
    """S2VS-based embedding backend (default).

    Pipeline:
        Frames -> FeatureExtractor (frozen) -> (T, R, 512)
        -> Attention weighting (pretrained ViSiL) -> (T, R, 512)
        -> GeM pooling per frame -> (T, 512)
        -> Mean pooling across frames -> (512,)
        -> L2 normalize
    """

    def __init__(
        self,
        feat_extractor: nn.Module,
        attention: nn.Module,
        dims: int = 512,
        device: str = "cuda",
        batch_sz: int = 256,
    ) -> None:
        self._dims = dims
        self._device = device
        self._batch_sz = batch_sz
        self._feat_extractor = feat_extractor
        self._attention = attention

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dims: int = 512,
        device: str = "cuda",
        batch_sz: int = 256,
    ) -> "S2VSBackend":
        """Load from a saved ViSiL checkpoint file.

        Args:
            model_path: Path to the .pth checkpoint file.
            dims: Feature dimensionality.
            device: Torch device string.
            batch_sz: Batch size for frame processing.

        Returns:
            Configured S2VSBackend instance.
        """
        feat_extractor = FeatureExtractor["RESNET"].get_model(dims)
        feat_extractor = feat_extractor.to(device).eval()

        sim_network = SimilarityNetwork["ViSiL"].get_model(dims=dims, attention=True)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        sim_network.load_state_dict(checkpoint["model"])
        sim_network = sim_network.to(device).eval()

        # ViSiL with attention=True always has .attention attribute
        attention = sim_network.attention

        return cls(
            feat_extractor=feat_extractor,
            attention=attention,
            dims=dims,
            device=device,
            batch_sz=batch_sz,
        )

    @classmethod
    def from_hub(
        cls,
        pretrained: str = "s2vs_dns",
        dims: int = 512,
        device: str = "cuda",
        batch_sz: int = 256,
    ) -> "S2VSBackend":
        """Load using pretrained weights from PyTorch Hub.

        Args:
            pretrained: Pretrained model name ('s2vs_dns' or 's2vs_vcdb').
            dims: Feature dimensionality.
            device: Torch device string.
            batch_sz: Batch size for frame processing.

        Returns:
            Configured S2VSBackend instance.
        """
        feat_extractor = FeatureExtractor["RESNET"].get_model(dims)
        feat_extractor = feat_extractor.to(device).eval()

        sim_network = SimilarityNetwork["ViSiL"].get_model(
            dims=dims, attention=True, pretrained=pretrained
        )
        sim_network = sim_network.to(device).eval()

        # ViSiL with pretrained= always sets idx_type='att' and creates .attention
        attention = sim_network.attention

        return cls(
            feat_extractor=feat_extractor,
            attention=attention,
            dims=dims,
            device=device,
            batch_sz=batch_sz,
        )

    @property
    def dim(self) -> int:
        return self._dims

    @torch.no_grad()
    def extract_features(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Extract per-frame region features (before pooling).

        These features are needed for ViSiL re-ranking.

        Args:
            video_tensor: Tensor of shape (T, C, H, W) — raw video frames.

        Returns:
            Feature tensor of shape (T, R, D) on CPU.
        """
        features_list: list[torch.Tensor] = []
        for i in range(0, video_tensor.shape[0], self._batch_sz):
            batch = video_tensor[i : i + self._batch_sz].to(self._device)
            feats = self._feat_extractor(batch)  # (B, R, D)
            features_list.append(feats)
        return torch.cat(features_list, dim=0).cpu()  # (T, R, D)

    def _pool_features(self, region_features: torch.Tensor) -> np.ndarray:
        """Pool region features into a single L2-normalized descriptor.

        Args:
            region_features: Tensor of shape (T, R, D).

        Returns:
            L2-normalized descriptor of shape (dim,).
        """
        t, r, d = region_features.shape
        side = int(r**0.5)
        if side * side == r:
            spatial = region_features.permute(0, 2, 1).reshape(t, d, side, side)
            frame_descriptors = gem(spatial, dim=1)  # (T, D)
        else:
            frame_descriptors = region_features.mean(dim=1)  # (T, D)

        video_descriptor = frame_descriptors.mean(dim=0)  # (D,)
        video_descriptor = F.normalize(video_descriptor, p=2, dim=0)
        return video_descriptor.cpu().numpy()

    @torch.no_grad()
    def extract_descriptor(self, video_tensor: torch.Tensor) -> np.ndarray:
        """Extract descriptor from video frames.

        Args:
            video_tensor: Tensor of shape (T, C, H, W) — raw video frames.

        Returns:
            L2-normalized descriptor of shape (dim,).
        """
        _, descriptor = self.extract_all(video_tensor)
        return descriptor

    @torch.no_grad()
    def extract_all(
        self, video_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Extract both raw features and pooled descriptor in one pass.

        Args:
            video_tensor: Tensor of shape (T, C, H, W) — raw video frames.

        Returns:
            Tuple of (raw_features (T, R, D) on CPU, L2-normalized descriptor (dim,)).
        """
        region_features = self.extract_features(video_tensor)  # (T, R, D)
        attended, _weights = self._attention(region_features.to(self._device))
        descriptor = self._pool_features(attended)
        return region_features, descriptor

    @torch.no_grad()
    def extract_batch(self, video_tensors: list[torch.Tensor]) -> np.ndarray:
        """Extract descriptors for multiple videos.

        Args:
            video_tensors: List of tensors, each of shape (T_i, C, H, W).

        Returns:
            L2-normalized descriptors of shape (B, dim).
        """
        descriptors: list[np.ndarray] = []
        for video_tensor in video_tensors:
            desc = self.extract_descriptor(video_tensor)
            descriptors.append(desc)
        return np.stack(descriptors, axis=0)


class CLIPBackend(EmbeddingBackend):
    """CLIP-based embedding backend (optional).

    Pipeline:
        Frames -> CLIP ViT-B/32 image encoder -> (T, 512)
        -> Mean pooling across frames -> (512,)
        -> L2 normalize

    Requires: pip install open-clip-torch
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
        batch_sz: int = 256,
    ) -> None:
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "CLIP backend requires open-clip-torch. "
                "Install with: pip install open-clip-torch"
            ) from None

        self._device = device
        self._batch_sz = batch_sz

        model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model = model.to(device).eval()
        self._dims = model.visual.output_dim

    @property
    def dim(self) -> int:
        return self._dims

    @torch.no_grad()
    def extract_descriptor(self, video_tensor: torch.Tensor) -> np.ndarray:
        """Extract descriptor from video frames using CLIP.

        Args:
            video_tensor: Tensor of shape (T, C, H, W) — preprocessed video frames.

        Returns:
            L2-normalized descriptor of shape (dim,).
        """
        frame_features_list: list[torch.Tensor] = []
        for i in range(0, video_tensor.shape[0], self._batch_sz):
            batch = video_tensor[i : i + self._batch_sz].to(self._device)
            feats = self._model.encode_image(batch)  # (B, D)
            frame_features_list.append(feats)
        frame_features = torch.cat(frame_features_list, dim=0)  # (T, D)

        # Mean pooling across frames
        video_descriptor = frame_features.mean(dim=0)  # (D,)

        # L2 normalize
        video_descriptor = F.normalize(video_descriptor.float(), p=2, dim=0)

        return video_descriptor.cpu().numpy()

    @torch.no_grad()
    def extract_batch(self, video_tensors: list[torch.Tensor]) -> np.ndarray:
        """Extract descriptors for multiple videos.

        Args:
            video_tensors: List of tensors, each of shape (T_i, C, H, W).

        Returns:
            L2-normalized descriptors of shape (B, dim).
        """
        descriptors: list[np.ndarray] = []
        for video_tensor in video_tensors:
            desc = self.extract_descriptor(video_tensor)
            descriptors.append(desc)
        return np.stack(descriptors, axis=0)


def get_backend(
    name: str = "s2vs",
    model_path: str | None = None,
    pretrained: str = "s2vs_dns",
    device: str = "cuda",
    **kwargs: Any,
) -> EmbeddingBackend:
    """Factory function to create an embedding backend.

    Args:
        name: Backend name — 's2vs' (default) or 'clip'.
        model_path: Path to model checkpoint (S2VS only). If None, uses hub weights.
        pretrained: Pretrained model name for hub loading (S2VS: 's2vs_dns'/'s2vs_vcdb').
        device: Torch device string.
        **kwargs: Additional keyword arguments passed to the backend constructor.

    Returns:
        Configured EmbeddingBackend instance.

    Raises:
        ValueError: If backend name is not recognized.
    """
    if name == "s2vs":
        if model_path is not None:
            return S2VSBackend.from_pretrained(
                model_path=model_path, device=device, **kwargs
            )
        return S2VSBackend.from_hub(pretrained=pretrained, device=device, **kwargs)
    elif name == "clip":
        return CLIPBackend(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {name!r}. Choose 's2vs' or 'clip'.")
