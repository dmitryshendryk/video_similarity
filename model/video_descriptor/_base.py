"""Abstract base class for all embedding backends."""

from abc import ABC, abstractmethod

import numpy as np
import torch


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
