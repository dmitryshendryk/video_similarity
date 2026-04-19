"""
Fast video deduplication CLI.

Three-stage pipeline:
  Stage 0: vpdq perceptual hash pre-filter (optional, ~1ms)
  Stage 1: Vector search on compact descriptors (<10ms)
  Stage 2: ViSiL re-ranking on top-K candidates (optional, ~5s)

Usage:
  python fast_dedup.py build-index --dataset-path data/videos/ --index-path /tmp/idx
  python fast_dedup.py query --query-video data/videos/001.mp4 --index-path /tmp/idx
  python fast_dedup.py batch-dedup --index-path /tmp/idx
"""

import json
import time

from pathlib import Path

import h5py
import numpy as np
import torch
import typer

from tqdm import tqdm

from dedup.helpers import (
    create_index,
    discover_videos,
    extract_video_metadata,
    load_index,
    load_video_tensor,
)
from dedup.qdrant_index import QdrantIndex
from dedup.store import JSONStore
from dedup.vpdq_filter import VPDQFilter
from model.video_descriptor import get_backend

app = typer.Typer(help="Fast video deduplication pipeline.")


@app.command()
def build_index(
    dataset_path: Path = typer.Option(..., help="Directory containing video files."),
    index_path: Path = typer.Option(..., help="Output path for the index."),
    pattern: str = typer.Option("**/*.mp4", help="Glob pattern to find video files."),
    backend: str = typer.Option("s2vs", help="Embedding backend: 's2vs' or 'clip'."),
    model_path: str | None = typer.Option(None, help="Model checkpoint path (S2VS)."),
    pretrained: str = typer.Option("s2vs_dns", help="Pretrained model name."),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
    fps: int = typer.Option(1, help="Frames per second for sampling."),
    resize: int = typer.Option(256, help="Resize dimension."),
    crop: int = typer.Option(224, help="Center crop dimension."),
    use_vpdq: bool = typer.Option(False, help="Compute and store vpdq hashes."),
    dataset_hdf5: str | None = typer.Option(None, help="Precomputed HDF5 features."),
    batch_sz: int = typer.Option(256, help="Batch size for frame processing."),
    index_backend: str = typer.Option(
        "faiss", help="Index backend: 'faiss' or 'qdrant'."
    ),
    qdrant_url: str | None = typer.Option(
        None, help="Qdrant server URL (remote mode)."
    ),
    collection_name: str = typer.Option("video_dedup", help="Qdrant collection name."),
) -> None:
    """Build a vector index from a video dataset."""
    embed = get_backend(
        name=backend,
        model_path=model_path,
        pretrained=pretrained,
        device=device,
        batch_sz=batch_sz,
    )

    index = create_index(
        index_backend, embed.dim, index_path, qdrant_url, collection_name
    )
    use_qdrant = index_backend == "qdrant"
    store = JSONStore() if not use_qdrant else None
    vpdq_filter = VPDQFilter() if use_vpdq and VPDQFilter.is_available() else None

    if use_vpdq and not VPDQFilter.is_available():
        typer.echo("Warning: --use-vpdq requested but vpdq not installed. Skipping.")

    if dataset_hdf5 is not None:
        typer.echo(f"Building index from HDF5: {dataset_hdf5}")
        with h5py.File(dataset_hdf5, "r") as hdf5:
            video_ids = list(hdf5.keys())
            typer.echo(f"Found {len(video_ids)} videos in HDF5.")
            for video_id in tqdm(video_ids, desc="Indexing"):
                features = torch.from_numpy(np.array(hdf5[video_id]))
                desc = embed.extract_descriptor(features)
                if use_qdrant:
                    index.add(video_id, desc, metadata={"path": video_id})
                else:
                    index.add(video_id, desc)
                    store.add(video_id, video_id)
    else:
        videos = discover_videos(dataset_path, pattern)
        typer.echo(f"Found {len(videos)} videos in {dataset_path}")
        for video_id, video_path in tqdm(videos, desc="Indexing"):
            vt = load_video_tensor(video_path, fps=fps, resize=resize, crop=crop)
            if vt.shape[0] == 0:
                typer.echo(f"  Skipping {video_id}: no frames extracted.")
                continue
            desc = embed.extract_descriptor(vt)
            metadata = extract_video_metadata(video_path)
            if use_qdrant:
                index.add(video_id, desc, metadata=metadata)
            else:
                index.add(video_id, desc)
                store.add(video_id, str(video_path), metadata)
            if vpdq_filter is not None:
                vpdq_filter.add_from_file(video_id, str(video_path))

    index.save(index_path)
    if store is not None:
        store.save(Path(str(index_path) + "_metadata.json"))
    if vpdq_filter is not None and len(vpdq_filter) > 0:
        vpdq_filter.save(Path(str(index_path) + "_vpdq.pkl"))
    typer.echo(f"Index built: {len(index)} videos, saved to {index_path}")


@app.command()
def query(
    query_video: Path = typer.Option(..., help="Path to the query video file."),
    index_path: Path = typer.Option(..., help="Path to the saved index."),
    top_k: int = typer.Option(50, help="Number of candidates to retrieve."),
    top_n: int = typer.Option(10, help="Final results after re-ranking."),
    threshold: float = typer.Option(0.8, help="Similarity threshold."),
    backend: str = typer.Option("s2vs", help="Embedding backend: 's2vs' or 'clip'."),
    model_path: str | None = typer.Option(None, help="Model checkpoint path (S2VS)."),
    pretrained: str = typer.Option("s2vs_dns", help="Pretrained model name."),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
    fps: int = typer.Option(1),
    resize: int = typer.Option(256),
    crop: int = typer.Option(224),
    batch_sz: int = typer.Option(256),
    rerank: bool = typer.Option(False, help="Enable ViSiL re-ranking."),
    features_hdf5: str | None = typer.Option(None, help="HDF5 features for reranking."),
    use_vpdq: bool = typer.Option(False, help="Check vpdq hashes first."),
    output: str = typer.Option("json", help="Output format: 'json' or 'table'."),
    index_backend: str = typer.Option(
        "faiss", help="Index backend: 'faiss' or 'qdrant'."
    ),
    qdrant_url: str | None = typer.Option(None, help="Qdrant server URL."),
    collection_name: str = typer.Option("video_dedup", help="Qdrant collection name."),
) -> None:
    """Find duplicates for a single query video."""
    t_start = time.time()
    use_qdrant = index_backend == "qdrant"

    index = load_index(index_backend, index_path, qdrant_url, collection_name)
    store: JSONStore | None = None
    if not use_qdrant:
        mp = Path(str(index_path) + "_metadata.json")
        store = JSONStore.load(mp) if mp.exists() else JSONStore()

    # Stage 0: vpdq pre-filter
    vpdq_results: list[tuple[str, float]] = []
    if use_vpdq and VPDQFilter.is_available():
        vpdq_path = Path(str(index_path) + "_vpdq.pkl")
        if vpdq_path.exists():
            vf = VPDQFilter.load(vpdq_path)
            qh = vf.compute_hash(str(query_video))
            vpdq_results = vf.find_matches(qh, threshold=0.9)
            if vpdq_results:
                typer.echo(f"vpdq exact matches: {len(vpdq_results)}")

    # Extract query descriptor
    embed = get_backend(
        name=backend,
        model_path=model_path,
        pretrained=pretrained,
        device=device,
        batch_sz=batch_sz,
    )
    vt = load_video_tensor(query_video, fps=fps, resize=resize, crop=crop)
    if vt.shape[0] == 0:
        typer.echo("Error: no frames extracted from query video.")
        raise typer.Exit(code=1)
    query_desc = embed.extract_descriptor(vt)

    # Stage 1: vector search
    search_results = index.search(query_desc, top_k=top_k)

    # Stage 2: ViSiL re-ranking (optional)
    if rerank and features_hdf5 is not None:
        from dedup.rerank import rerank_with_visil

        candidate_ids = [vid for vid, _ in search_results]
        final_results = rerank_with_visil(
            query_features=vt,
            candidate_ids=candidate_ids,
            features_hdf5_path=features_hdf5,
            pretrained=pretrained,
            device=device,
            top_n=top_n,
        )
    else:
        final_results = search_results[:top_n]

    duplicates = [(vid, score) for vid, score in final_results if score >= threshold]
    t_elapsed = time.time() - t_start

    # Merge vpdq matches
    seen_ids = {vid for vid, _ in duplicates}
    for vid, score in vpdq_results:
        if vid not in seen_ids:
            duplicates.insert(0, (vid, score))
            seen_ids.add(vid)

    # Output results
    qdrant_idx = index if isinstance(index, QdrantIndex) else None
    _print_results(
        duplicates,
        str(query_video),
        t_elapsed,
        threshold,
        output,
        store=store,
        qdrant_index=qdrant_idx,
    )


@app.command()
def batch_dedup(
    index_path: Path = typer.Option(..., help="Path to the saved index."),
    threshold: float = typer.Option(0.8, help="Similarity threshold."),
    top_k: int = typer.Option(50, help="Candidates per query."),
    output: str = typer.Option("json", help="Output format: 'json' or 'table'."),
    index_backend: str = typer.Option(
        "faiss", help="Index backend: 'faiss' or 'qdrant'."
    ),
    qdrant_url: str | None = typer.Option(None, help="Qdrant server URL."),
    collection_name: str = typer.Option("video_dedup", help="Qdrant collection name."),
) -> None:
    """Find all duplicate pairs within a dataset (index must already be built)."""
    index = load_index(index_backend, index_path, qdrant_url, collection_name)
    all_ids = index.list_all()
    typer.echo(f"Checking {len(all_ids)} videos for duplicates...")

    duplicate_groups: list[list[str]] = []
    seen: set[str] = set()

    for video_id in tqdm(all_ids, desc="Deduplicating"):
        if video_id in seen:
            continue
        desc = index.get_descriptor(video_id)
        results = index.search(desc, top_k=top_k)
        group = [
            vid for vid, score in results if score >= threshold and vid != video_id
        ]
        if group:
            full_group = [video_id] + group
            new_members = [v for v in full_group if v not in seen]
            if len(new_members) > 1:
                duplicate_groups.append(full_group)
                seen.update(full_group)

    if output == "json":
        typer.echo(
            json.dumps(
                {
                    "num_groups": len(duplicate_groups),
                    "num_duplicates": sum(len(g) - 1 for g in duplicate_groups),
                    "groups": duplicate_groups,
                },
                indent=2,
            )
        )
    else:
        typer.echo(f"\nFound {len(duplicate_groups)} duplicate groups:\n")
        for i, group in enumerate(duplicate_groups, 1):
            typer.echo(f"  Group {i}: {', '.join(group)}")


def _get_video_path(
    vid: str,
    store: JSONStore | None,
    qdrant_index: QdrantIndex | None,
) -> str:
    """Look up the file path for a video ID from store or Qdrant payload."""
    if store is not None and vid in store:
        val = store.get(vid).get("path")
        return str(val) if isinstance(val, str) else ""
    if qdrant_index is not None:
        try:
            val = qdrant_index.get_metadata(vid).get("path")
            return str(val) if isinstance(val, str) else ""
        except KeyError:
            pass
    return ""


def _print_results(
    duplicates: list[tuple[str, float]],
    query_path: str,
    elapsed: float,
    threshold: float,
    output: str,
    store: JSONStore | None = None,
    qdrant_index: QdrantIndex | None = None,
) -> None:
    """Format and print query results."""
    if output == "json":
        results_data: list[dict[str, str | float]] = []
        for vid, score in duplicates:
            entry: dict[str, str | float] = {"video_id": vid, "score": round(score, 4)}
            p = _get_video_path(vid, store, qdrant_index)
            if p:
                entry["path"] = p
            results_data.append(entry)
        typer.echo(
            json.dumps(
                {
                    "query": query_path,
                    "num_duplicates": len(duplicates),
                    "elapsed_seconds": round(elapsed, 3),
                    "results": results_data,
                },
                indent=2,
            )
        )
    else:
        typer.echo(f"\nQuery: {query_path}")
        typer.echo(
            f"Found {len(duplicates)} duplicates (threshold={threshold}) in {elapsed:.3f}s\n"
        )
        for i, (vid, score) in enumerate(duplicates, 1):
            p = _get_video_path(vid, store, qdrant_index)
            typer.echo(f"  {i:3d}. {vid}  score={score:.4f}  {p}")


if __name__ == "__main__":
    app()
