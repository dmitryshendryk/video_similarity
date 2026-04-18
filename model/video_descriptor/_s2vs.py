"""S2VS-based embedding backend — the default pipeline."""

import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from model.feature_extractor import FeatureExtractor
from model.pooling import gem
from model.similarity_network import SimilarityNetwork
from model.video_descriptor._base import EmbeddingBackend
from model.video_descriptor._quantization import _quantize_with_quanto

logger = logging.getLogger(__name__)


@dataclass
class VideoFeatures:
    """Result of a full S2VS forward pass.

    Attributes:
        frame_features: Per-frame region features of shape (T, R, D) on CPU.
            Needed for ViSiL frame-level re-ranking.
        descriptor: L2-normalized global descriptor of shape (dim,).
    """

    frame_features: torch.Tensor
    descriptor: np.ndarray


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
        compile_model: bool = True,
        quantize_model: bool = False,
    ) -> None:
        self._dims = dims
        self._device = device
        self._batch_sz = batch_sz
        self._attention = attention

        if quantize_model:
            result = _quantize_with_quanto(feat_extractor)
            feat_extractor = result.model

        if compile_model and not quantize_model:
            try:
                feat_extractor = torch.compile(feat_extractor, mode="reduce-overhead")
            except Exception as exc:
                logger.warning("torch.compile failed, using eager mode: %s", exc)
        elif compile_model:
            logger.info(
                "Skipping torch.compile: mutually exclusive with quantize_model"
            )

        self._feat_extractor = feat_extractor
        self._warmup()

    def _warmup(self) -> None:
        """Run a dummy forward pass to trigger JIT compilation at boot."""
        t0 = time.monotonic()
        try:
            dummy = torch.randint(0, 256, (4, 224, 224, 3), dtype=torch.uint8).to(
                self._device
            )
            with torch.no_grad():
                self._feat_extractor(dummy)
            logger.info("Model warmup completed in %.2fs", time.monotonic() - t0)
        except Exception as exc:
            logger.warning(
                "Model warmup failed, will warm up on first request: %s", exc
            )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dims: int = 512,
        device: str = "cuda",
        batch_sz: int = 256,
        compile_model: bool = True,
        quantize_model: bool = False,
    ) -> "S2VSBackend":
        """Load from a saved ViSiL checkpoint file.

        Args:
            model_path: Path to the .pth checkpoint file.
            dims: Feature dimensionality.
            device: Torch device string.
            batch_sz: Batch size for frame processing.
            compile_model: Attempt torch.compile on the feature extractor.
            quantize_model: Attempt INT8 quantization via optimum-quanto.

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
            compile_model=compile_model,
            quantize_model=quantize_model,
        )

    @classmethod
    def from_hub(
        cls,
        pretrained: str = "s2vs_dns",
        dims: int = 512,
        device: str = "cuda",
        batch_sz: int = 256,
        compile_model: bool = True,
        quantize_model: bool = False,
    ) -> "S2VSBackend":
        """Load using pretrained weights from PyTorch Hub.

        Args:
            pretrained: Pretrained model name ('s2vs_dns' or 's2vs_vcdb').
            dims: Feature dimensionality.
            device: Torch device string.
            batch_sz: Batch size for frame processing.
            compile_model: Attempt torch.compile on the feature extractor.
            quantize_model: Attempt INT8 quantization via optimum-quanto.

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
            compile_model=compile_model,
            quantize_model=quantize_model,
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
        return self.extract_all(video_tensor).descriptor

    @torch.no_grad()
    def extract_all(self, video_tensor: torch.Tensor) -> VideoFeatures:
        """Extract both raw features and pooled descriptor in one pass.

        Args:
            video_tensor: Tensor of shape (T, C, H, W) — raw video frames.

        Returns:
            VideoFeatures with frame_features (T, R, D) on CPU and
            L2-normalized descriptor of shape (dim,).
        """
        region_features = self.extract_features(video_tensor)  # (T, R, D)
        attended, _weights = self._attention(region_features.to(self._device))
        descriptor = self._pool_features(attended)
        return VideoFeatures(frame_features=region_features, descriptor=descriptor)

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
