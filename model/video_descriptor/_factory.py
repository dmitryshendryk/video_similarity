"""Factory function for creating embedding backends by name."""

import logging

from typing import Any

from model.video_descriptor._base import EmbeddingBackend
from model.video_descriptor._clip import CLIPBackend
from model.video_descriptor._s2vs import S2VSBackend

logger = logging.getLogger(__name__)


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
