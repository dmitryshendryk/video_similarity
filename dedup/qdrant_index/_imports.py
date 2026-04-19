"""Optional qdrant-client imports with availability flags."""

import logging

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PayloadSchemaType,
        PointStruct,
        Range,
        VectorParams,
    )

    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

try:
    from qdrant_client.models import (
        BinaryQuantization,
        BinaryQuantizationConfig,
        QuantizationSearchParams,
        SearchParams,
    )

    _QDRANT_BQ_AVAILABLE = True
except ImportError:
    _QDRANT_BQ_AVAILABLE = False
    logger.warning(
        "qdrant-client does not support BinaryQuantization — "
        "binary_quantization option will be silently ignored. "
        "Upgrade with: pip install --upgrade qdrant-client"
    )
