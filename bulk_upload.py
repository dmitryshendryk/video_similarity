#!/usr/bin/env python3
"""
Bulk video upload script for the S2VS video deduplication system.

Processes videos directly through the backend (no HTTP overhead),
designed for 20K-30K+ videos with progress tracking and resume support.

Usage:
    python bulk_upload.py /path/to/videos
    python bulk_upload.py /path/to/videos --workers 4 --extensions .mp4,.avi,.mov
    python bulk_upload.py /path/to/videos --resume  # skip already-indexed videos
"""

import argparse
import logging
import shutil
import sys
import time
import uuid

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".webm", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}


def discover_videos(source_dir: Path, extensions: set[str]) -> list[Path]:
    """Recursively find all video files in source directory.

    Handles: nested subdirectories, symlinks, hidden files, zero-byte files.
    """
    videos: list[Path] = []
    seen_resolved: set[Path] = set()
    for f in sorted(source_dir.rglob("*")):
        if not f.is_file():
            continue
        # Skip hidden files (e.g. .DS_Store, ._filename)
        if f.name.startswith("."):
            continue
        if f.suffix.lower() not in extensions:
            continue
        # Skip zero-byte files
        if f.stat().st_size == 0:
            continue
        # Resolve symlinks and skip duplicates pointing to same file
        resolved = f.resolve()
        if resolved in seen_resolved:
            continue
        seen_resolved.add(resolved)
        videos.append(f)
    return videos


def process_single_video(
    video_path_str: str,
    uploads_dir_str: str,
    copy_files: bool,
) -> dict[str, str | int | float | bool | None]:
    """Process a single video — runs in a worker process.

    Imports are done inside the function so each worker process
    initializes its own model/backend instance.
    """
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from pathlib import Path

    from api_state.config import UPLOADS_DIR
    from api_state.state import (
        embed_backend,
        index,
        phash_filter,
        store,
        index_path,
        metadata_path,
    )
    from api_state.thumbnails import generate_thumbnail
    from api_state.visil import save_features
    from dedup.helpers import (
        aspect_ratio_class,
        duration_bucket,
        extract_video_metadata,
        load_video_tensor,
    )
    from dedup.phash_filter import PHashFilter
    from dedup.qdrant_index import QdrantIndex
    from model.video_descriptor import S2VSBackend

    video_path = Path(video_path_str)
    uploads_dir = Path(uploads_dir_str)

    video_id = video_path.stem + "_" + uuid.uuid4().hex[:8]

    if copy_files:
        dest = uploads_dir / f"{video_id}{video_path.suffix}"
        shutil.copy2(video_path, dest)
        working_path = dest
    else:
        working_path = video_path

    # Extract metadata
    metadata = extract_video_metadata(working_path)
    metadata["duration_bucket"] = duration_bucket(
        float(metadata["duration"]) if metadata.get("duration") is not None else None
    )
    metadata["aspect_ratio"] = aspect_ratio_class(
        int(metadata["width"]) if metadata.get("width") is not None else None,
        int(metadata["height"]) if metadata.get("height") is not None else None,
    )
    metadata["path"] = str(working_path)

    # Thumbnail
    try:
        generate_thumbnail(working_path, video_id)
    except Exception:
        pass

    # pHash
    if phash_filter is not None:
        try:
            phash_value = phash_filter.compute_phash(str(working_path))
            metadata["phash"] = PHashFilter.serialize_phash(phash_value)
            phash_filter.add(video_id, phash_value)
        except Exception:
            pass

    # Decode + extract features
    video_tensor = load_video_tensor(working_path, keyframes_only=True)
    if video_tensor.shape[0] == 0:
        raise ValueError(f"No frames extracted from {video_path.name}")

    if isinstance(embed_backend, S2VSBackend):
        result = embed_backend.extract_all(video_tensor)
        save_features(video_id, result.frame_features)
        desc = result.descriptor
    else:
        desc = embed_backend.extract_descriptor(video_tensor)

    # Index
    if isinstance(index, QdrantIndex):
        index.add(video_id, desc, metadata=metadata)
    else:
        index.add(video_id, desc)
        store.add(video_id, str(working_path), metadata)

    return {"video_id": video_id, "filename": video_path.name}


def get_indexed_filenames() -> set[str]:
    """Get filenames already in the index (for resume support)."""
    try:
        from api_state.state import index
        from dedup.qdrant_index import QdrantIndex

        if isinstance(index, QdrantIndex):
            payloads = index.scroll_all_payloads(["path"])
            return {
                Path(str(p.get("path", ""))).stem.rsplit("_", 1)[0] for p in payloads
            }
        else:
            return {vid.rsplit("_", 1)[0] for vid in index.list_all()}
    except Exception:
        return set()


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "done"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def print_progress(
    done: int,
    total: int,
    succeeded: int,
    failed: int,
    skipped: int,
    elapsed: float,
    bar_width: int = 40,
) -> None:
    pct = done / total if total > 0 else 0
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)

    rate = done / elapsed if elapsed > 0 else 0
    remaining = total - done
    eta = remaining / rate if rate > 0 else 0

    sys.stdout.write(
        f"\r  [{bar}] {done}/{total} ({pct * 100:.1f}%) "
        f"| ✓ {succeeded} ✗ {failed} ⊘ {skipped} "
        f"| {rate:.1f} vid/s "
        f"| ETA {format_eta(eta)} "
        f"| elapsed {format_time(elapsed)}  "
    )
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk upload videos to S2VS deduplication system"
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Directory containing video files (searched recursively)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, safe for shared model)",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=None,
        help="Comma-separated extensions to include (default: all video types)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos whose stem is already in the index",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files to uploads dir (default: index from original location)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of videos to process (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover and count videos, don't process",
    )

    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"ERROR: {args.source} is not a directory")
        sys.exit(1)

    extensions = (
        {f".{e.strip('.')}" for e in args.extensions.split(",")}
        if args.extensions
        else VIDEO_EXTENSIONS
    )

    # Discover videos
    print(f"Scanning {args.source} ...")
    videos = discover_videos(args.source, extensions)
    print(f"Found {len(videos)} video files")

    if len(videos) == 0:
        print("Nothing to upload.")
        return

    # Resume: filter out already-indexed
    skipped = 0
    if args.resume:
        print("Checking index for already-uploaded videos...")
        indexed = get_indexed_filenames()
        before = len(videos)
        videos = [v for v in videos if v.stem not in indexed]
        skipped = before - len(videos)
        print(f"Skipping {skipped} already-indexed, {len(videos)} remaining")

    if args.limit:
        videos = videos[: args.limit]
        print(f"Limited to {len(videos)} videos")

    if args.dry_run:
        print("\nDry run — would process:")
        for v in videos[:20]:
            print(f"  {v}")
        if len(videos) > 20:
            print(f"  ... and {len(videos) - 20} more")
        return

    total = len(videos)
    if total == 0:
        print("All videos already indexed.")
        return

    uploads_dir = str(Path("api_data/uploads"))

    print(f"\nUploading {total} videos (workers={args.workers}, copy={args.copy})")
    print()

    t0 = time.perf_counter()
    succeeded = 0
    failed = 0
    errors: list[tuple[str, str]] = []

    if args.workers <= 1:
        # Sequential — single process, no overhead
        import os

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        for i, video_path in enumerate(videos):
            try:
                process_single_video(str(video_path), uploads_dir, args.copy)
                succeeded += 1
            except Exception as e:
                failed += 1
                errors.append((video_path.name, str(e)))

            elapsed = time.perf_counter() - t0
            print_progress(i + 1, total, succeeded, failed, skipped, elapsed)

    else:
        # Parallel — process pool
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_single_video, str(vp), uploads_dir, args.copy
                ): vp
                for vp in videos
            }
            done_count = 0
            for future in as_completed(futures):
                video_path = futures[future]
                done_count += 1
                try:
                    future.result()
                    succeeded += 1
                except Exception as e:
                    failed += 1
                    errors.append((video_path.name, str(e)))

                elapsed = time.perf_counter() - t0
                print_progress(done_count, total, succeeded, failed, skipped, elapsed)

    elapsed = time.perf_counter() - t0
    print()
    print()
    print("=" * 60)
    print(f"  Completed in {format_time(elapsed)}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Rate:      {succeeded / elapsed:.1f} videos/sec")
    print("=" * 60)

    if errors:
        print(f"\nFailed videos ({len(errors)}):")
        for name, err in errors[:50]:
            print(f"  {name}: {err}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")


if __name__ == "__main__":
    main()
