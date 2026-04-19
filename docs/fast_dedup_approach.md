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

## Lessons from czkawka (Rust video dedup tool)

The [czkawka](https://github.com/qarmin/czkawka) project is a mature Rust-based duplicate finder that includes video deduplication. Analyzing its approach reveals several ideas worth incorporating:

### Ideas adopted into our pipeline

1. **Parameter-keyed cache files** — czkawka names its cache files with the hashing parameters embedded: `cache_similar_videos_v3__skip_15__dur_10__cd_letterbox.bin`. This means changing parameters (e.g., skip amount) automatically invalidates the cache without manual cleanup. We should do the same for our descriptor cache — include the backend name, model checkpoint hash, and pooling strategy in the cache filename.

2. **Video metadata extraction alongside hashing** — czkawka extracts fps, codec, bitrate, resolution, and duration for every video during the hashing pass (`VideosEntry` struct). This metadata is used for post-filtering (exclude same-size, exclude same-resolution) and for choosing which duplicate to keep (highest bitrate/resolution wins). Our `MetadataStore` should capture the same fields during `build_index`.

3. **Crop detection modes** — czkawka supports three crop detection modes via `vid_dup_finder_lib`: `None`, `Letterbox` (detect and ignore black bars), and `Motion` (detect active region). Letterbox detection is valuable for video dedup since the same content can appear with different letterboxing. We should add a preprocessing step that optionally detects and crops letterbox bars before frame extraction.

4. **Configurable temporal sampling window** — Instead of processing the entire video, czkawka skips the first N seconds (default 15, range 0-300) then hashes only M seconds (default 10, range 2-60). This dramatically reduces processing time for long videos. Our pipeline should support `--skip_start` and `--max_duration` parameters for the same optimization.

5. **Parallelism with bounded concurrency** — czkawka uses `rayon::par_iter().with_max_len(2)` to limit parallel video hashing to 2 concurrent videos. This prevents FFmpeg/GPU memory exhaustion. Our batch descriptor extraction should similarly cap concurrent video processing.

6. **Reference folder mode** — czkawka supports a "reference folders" concept where one set of folders contains the "known good" originals and another contains potential duplicates. Results are grouped as (reference, [duplicates]) rather than flat groups. This is directly useful for our use case: the existing 10K videos are the reference set, new videos are checked against them.

7. **Thumbnail generation for review** — czkawka generates thumbnails (single frame or NxN grid) for each duplicate group to help users visually verify matches before deletion. Our pipeline should optionally generate comparison thumbnails for the final results.

### What czkawka does differently (and why we diverge)

| Aspect | czkawka | Our approach | Why we diverge |
|--------|---------|-------------|---------------|
| Hash type | DCT perceptual hash (via `vid_dup_finder_lib`) | S2VS deep features + FAISS | Deep features capture semantic similarity, not just perceptual; handles different recordings of same event |
| Matching | `vid_dup_finder_lib::search()` — connected components on hash similarity | FAISS ANN + optional ViSiL rerank | FAISS scales to 100K+ with sub-ms queries; ViSiL provides frame-level precision |
| Temporal scope | Fixed window (skip N, hash M seconds) | Full video by default, configurable window | Full video captures more context; window mode available as speed optimization |
| Storage | Binary cache on local filesystem | HDF5 descriptors + FAISS index + JSON metadata | HDF5 is standard for ML; FAISS index enables instant search |

### Updated pipeline parameters (inspired by czkawka)

```
fast_dedup.py
  # ... existing parameters ...
  --skip_start 0         Skip first N seconds of each video (default: 0, czkawka default: 15)
  --max_duration 0       Max seconds to process per video, 0=full (default: 0, czkawka default: 10)
  --crop_detect none     Crop detection: none|letterbox (default: none)
  --exclude_same_size    Skip pairs with identical file size
  --exclude_same_res     Skip pairs with identical resolution
  --reference_path PATH  Reference folder (known originals); only report new duplicates
  --thumbnails           Generate comparison thumbnails for results
```

## Key Design Principles

1. **Reuse over reinvent** — The embedding pipeline reuses the existing S2VS feature extractor, attention layer, and pooling modules. Only the aggregation (mean pool across time) and indexing (FAISS) are new.
2. **Pluggable backends** — Abstract `EmbeddingBackend` interface allows swapping S2VS ↔ CLIP without changing the rest of the pipeline. Same for `MetadataStore` (JSON ↔ PostgreSQL).
3. **Graceful degradation** — vpdq and CLIP are optional imports. The core pipeline (S2VS + FAISS) works with only `faiss-cpu` as a new dependency.
4. **Accuracy when needed** — The optional ViSiL re-ranking stage preserves the original model's accuracy for the final ranking, applied only to a small candidate set.
5. **Parameter-aware caching** — Cache filenames encode the parameters used to generate them, so changing settings auto-invalidates stale caches (learned from czkawka).
6. **Bounded parallelism** — Cap concurrent video processing to prevent memory exhaustion during batch extraction (learned from czkawka).
