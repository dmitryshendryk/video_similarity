"""INT8 quantization helpers using optimum-quanto (optional dependency)."""

import logging

import torch.nn as nn

from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from optimum.quanto import freeze, qint8, quantize as quanto_quantize

    _QUANTO_AVAILABLE = True
except ImportError:
    _QUANTO_AVAILABLE = False


@dataclass
class _QuantizeResult:
    """Result of an INT8 quantization attempt."""

    model: nn.Module
    success: bool


def _quantize_with_quanto(model: nn.Module) -> _QuantizeResult:
    """Attempt INT8 weight quantization via optimum-quanto.

    Args:
        model: PyTorch module to quantize in-place.

    Returns:
        _QuantizeResult with the (possibly quantized) model and a success flag.
        On failure returns the original unmodified model with success=False.
    """
    if not _QUANTO_AVAILABLE:
        logger.warning(
            "optimum-quanto not installed; skipping INT8 quantization. "
            "Install with: pip install optimum-quanto"
        )
        return _QuantizeResult(model=model, success=False)
    try:
        quanto_quantize(model, weights=qint8)
        freeze(model)
        return _QuantizeResult(model=model, success=True)
    except Exception as exc:
        logger.warning("quanto quantization failed, using FP32: %s", exc)
        return _QuantizeResult(model=model, success=False)
