# Fast Video Deduplication — Technical Approach

## Problem

Given a database of 10,000+ videos and a new incoming video, find duplicates in sub-second time. The existing S2VS/ViSiL pipeline computes full frame-to-frame similarity matrices per pair (~3.7B FLOPs each), making brute-force search over 10K videos take hours.

## Solution: Three-Stage Pipeline

```
                    New Video
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  Stage 0 │  │  Stage 1 │  │  Stage 2 │
    │   vpdq   │  │  FAISS   │  │  ViSiL   │
    │  (hash)  │  │ (vector) │  │ (rerank) │
    └────┬─────┘  └────┬─────┘  └────┬─────┘
         │             │             │
    Exact match   Top-K candidates  Top-N final
    in ~1ms       in <10ms          in ~5s
```

### Stage 0 — Perceptual Hash Pre-filter (optional, vpdq)

- **What**: Meta's Video PDQ generates per-frame perceptual hashes (256-bit)
- **How**: Hamming distance comparison against stored hashes
- **Speed**: ~1ms per query, O(N) but with trivial per-comparison cost
- **Catches**: Re-encodes, resolution changes, minor crops, watermarks
- **Misses**: Semantic duplicates, significant edits, different recordings of same event
- **Dependency**: `vpdq` (optional — pipeline works without it)

### Stage 1 — Vector Search with FAISS (core)

- **What**: Extract a single compact descriptor per video, index with FAISS for approximate nearest neighbor search
- **Embedding pipeline** (pluggable backend):

  **S2VS backend (default, 512-D):**
  ```
  Frames → ResNet50 (frozen) → PCA whitening → (T, 49, 512) region features
       → Attention weighting (pretrained) → (T, 49, 512)
       → GeM pooling per frame → (T, 512)
       → Mean pooling across time → (512,)
       → L2 normalize
  ```
  Reuses the existing trained feature extractor and ViSiL attention weights. No additional training needed.

  **CLIP backend (optional, 512-D):**
  ```
  Frames → CLIP ViT-B/32 image encoder → (T, 512)
       → Mean pooling across time → (512,)
       → L2 normalize
  ```
  Better robustness to color/brightness changes; requires `open-clip-torch`.

- **Index**: `faiss.IndexFlatIP` — exact inner product search on L2-normalized vectors (= cosine similarity)
- **Speed**: <1ms per query at 10K scale, ~20MB RAM
- **Why IndexFlatIP**: At 10K scale, exact search is fast enough that IVF clustering adds complexity with no meaningful speedup. 100% recall guaranteed.

### Stage 2 — ViSiL Re-ranking (optional)

- **What**: Run the full ViSiL frame-to-frame similarity on the top-K candidates from Stage 1
- **How**: For each of K candidates (default K=50), compute the T1×T2 similarity matrix → VideoComparator CNN → ChamferSimilarity → final score
- **Speed**: ~5s for 50 candidates (vs hours for 10K)
- **Accuracy**: The highest — spatial and temporal alignment, CNN-refined similarity
- **When to use**: When you need to distinguish near-duplicates from visually similar but different videos

## Performance Characteristics

| Metric | Brute-force ViSiL | This pipeline (FAISS only) | This pipeline (FAISS + rerank) |
|--------|-------------------|---------------------------|-------------------------------|
| Query time (10K DB) | ~30 min | <10ms | ~5s |
| Preprocessing | None | ~2s/video (one-time) | ~2s/video (one-time) |
| Index build (10K) | N/A | ~6 hours | ~6 hours |
| Index memory | N/A | ~20MB | ~20MB |
| Accuracy | Highest | Good (cosine on global descriptors) | Near-highest (ViSiL on top-K) |

## Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Embeddings | S2VS ResNet50+PCA+Attention (default) / CLIP (optional) | S2VS is video-specialized and already in the codebase; CLIP adds robustness to visual edits |
| Vector index | FAISS `IndexFlatIP` | 100% recall, <1ms at 10K, zero configuration |
| Hash pre-filter | vpdq (optional) | Ultra-fast exact-copy detection, graceful degradation if not installed |
| Re-ranking | ViSiL (existing model) | Frame-level spatial+temporal similarity for maximum accuracy |
| Metadata | JSON file (default) | Lightweight; abstract interface allows PostgreSQL/S3 swap later |
| Video decoding | pytorchvideo / ffmpeg-python (existing) | Already integrated; NVDEC GPU acceleration available as future optimization |

## Data Flow

```
build_index:
  videos/ → frame extraction → embedding backend → descriptors.h5
                                                  → faiss.index + ids.json
                                                  → metadata.json
                                                  → (optional) vpdq_hashes.pkl

query:
  new_video → embedding backend → descriptor (512-D)
           → [vpdq check] → exact match?
           → FAISS search → top-K candidates
           → [ViSiL rerank] → top-N results
           → threshold filter → duplicate report (JSON)
```

## Key Design Principles

1. **Reuse over reinvent** — The embedding pipeline reuses the existing S2VS feature extractor, attention layer, and pooling modules. Only the aggregation (mean pool across time) and indexing (FAISS) are new.
2. **Pluggable backends** — Abstract `EmbeddingBackend` interface allows swapping S2VS ↔ CLIP without changing the rest of the pipeline. Same for `MetadataStore` (JSON ↔ PostgreSQL).
3. **Graceful degradation** — vpdq and CLIP are optional imports. The core pipeline (S2VS + FAISS) works with only `faiss-cpu` as a new dependency.
4. **Accuracy when needed** — The optional ViSiL re-ranking stage preserves the original model's accuracy for the final ranking, applied only to a small candidate set.
