"""
Pluggable embedding backends for fast video deduplication.

Extracts a single compact descriptor (512-D) per video for FAISS/Qdrant indexing.
Two backends: S2VS (default, reuses existing model) and CLIP (optional).
"""

from model.video_descriptor._base import EmbeddingBackend
from model.video_descriptor._clip import CLIPBackend
from model.video_descriptor._factory import get_backend
from model.video_descriptor._quantization import _QuantizeResult, _quantize_with_quanto
from model.video_descriptor._s2vs import S2VSBackend, VideoFeatures
