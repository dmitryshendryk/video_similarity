"""CLIP-based embedding backend (optional, requires open-clip-torch)."""

import numpy as np
import torch
import torch.nn.functional as F

from model.video_descriptor._base import EmbeddingBackend


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

        video_descriptor = frame_features.mean(dim=0)  # (D,)
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
