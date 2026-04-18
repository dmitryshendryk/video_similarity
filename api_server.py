"""
FastAPI backend for the video deduplication web UI.

Wraps the dedup pipeline as REST endpoints for uploading videos,
searching for duplicates, browsing the library, and managing config.

Usage:
    uvicorn api_server:app --reload --port 8000
"""

import logging
import uuid

from pathlib import Path

logger = logging.getLogger("api_server")
logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from dedup.helpers import (
    aspect_ratio_class,
    duration_bucket,
    extract_video_metadata,
    load_video_tensor,
)
from dedup.qdrant_index import QdrantIndex
from model.video_descriptor import S2VSBackend

from api_state import (
    THUMBNAILS_DIR,
    UPLOADS_DIR,
    config,
    delete_features,
    embed_backend,
    generate_thumbnail,
    get_video_metadata,
    index,
    index_path,
    metadata_path,
    run_search_pipeline,
    save_config,
    save_features,
    store,
)

# --- FastAPI app ---
app = FastAPI(title="Video Dedup API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def get_status() -> dict[str, str | int]:
    return {
        "total_videos": len(index),
        "index_backend": str(config["index_backend"]),
        "embedding_backend": str(config["embedding_backend"]),
        "dimension": embed_backend.dim,
    }


@app.get("/api/config")
def get_config() -> dict[str, str | int | float | bool | None]:
    return config


@app.post("/api/config")
def update_config(
    new_config: dict[str, str | int | float | bool | None],
) -> dict[str, str | int | float | bool | None]:
    config.update(new_config)
    save_config(config)
    return config


def _process_upload(
    video_path: Path,
    video_id: str,
    filename: str,
) -> dict[str, str | int | float | bool | None]:
    """Shared upload processing: metadata, thumbnail, features, index."""
    metadata = extract_video_metadata(video_path)
    metadata["duration_bucket"] = duration_bucket(
        float(metadata["duration"]) if metadata.get("duration") is not None else None
    )
    metadata["aspect_ratio"] = aspect_ratio_class(
        int(metadata["width"]) if metadata.get("width") is not None else None,
        int(metadata["height"]) if metadata.get("height") is not None else None,
    )
    generate_thumbnail(video_path, video_id)

    video_tensor = load_video_tensor(video_path)
    if video_tensor.shape[0] == 0:
        video_path.unlink(missing_ok=True)
        raise ValueError("Could not extract frames from video")

    if isinstance(embed_backend, S2VSBackend):
        features, desc = embed_backend.extract_all(video_tensor)
        save_features(video_id, features)
    else:
        desc = embed_backend.extract_descriptor(video_tensor)

    if isinstance(index, QdrantIndex):
        index.add(video_id, desc, metadata=metadata)
    else:
        index.add(video_id, desc)
        store.add(video_id, str(video_path), metadata)
        index.save(index_path)
        store.save(metadata_path)

    return {"video_id": video_id, "filename": filename, **metadata}


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
) -> dict[str, str | int | float | bool | None]:
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    video_id = Path(file.filename).stem + "_" + uuid.uuid4().hex[:8]
    video_path = UPLOADS_DIR / f"{video_id}{Path(file.filename).suffix}"

    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

    try:
        return _process_upload(video_path, video_id, file.filename)
    except (ValueError, Exception) as exc:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.post("/api/upload-batch")
async def upload_batch(
    files: list[UploadFile] = File(...),
) -> dict[str, object]:
    results: list[dict[str, str | int | float | bool | None]] = []
    succeeded = 0
    failed = 0

    for file in files:
        if file.filename is None:
            results.append({"status": "error", "error": "No filename"})
            failed += 1
            continue

        video_id = Path(file.filename).stem + "_" + uuid.uuid4().hex[:8]
        video_path = UPLOADS_DIR / f"{video_id}{Path(file.filename).suffix}"

        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)

        try:
            result = _process_upload(video_path, video_id, file.filename)
            results.append({**result, "status": "ok"})
            succeeded += 1
        except Exception as exc:
            video_path.unlink(missing_ok=True)
            results.append(
                {"status": "error", "filename": file.filename, "error": str(exc)}
            )
            failed += 1
            logger.warning("Batch upload failed for %s: %s", file.filename, exc)

    return {
        "total": len(files),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }


@app.post("/api/search")
async def search_duplicates(
    file: UploadFile = File(...),
    top_k: int = Query(default=50),
    threshold: float = Query(default=0.8),
) -> dict[str, object]:
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    temp_path = (
        UPLOADS_DIR / f"_query_{uuid.uuid4().hex[:8]}{Path(file.filename).suffix}"
    )
    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        result = run_search_pipeline(temp_path, top_k=top_k, threshold=threshold)
        result["query"] = file.filename
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/api/videos")
def list_videos() -> dict[str, object]:
    all_ids = index.list_all()
    videos = [get_video_metadata(vid) for vid in all_ids]
    return {"total": len(videos), "videos": videos}


@app.get("/api/videos/{video_id}")
def get_video(video_id: str) -> dict[str, str | float | bool | None]:
    if video_id not in index:
        raise HTTPException(status_code=404, detail="Video not found")
    return get_video_metadata(video_id)


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str) -> dict[str, str]:
    if video_id not in index:
        raise HTTPException(status_code=404, detail="Video not found")

    index.remove(video_id)
    delete_features(video_id)

    if not isinstance(index, QdrantIndex):
        if video_id in store:
            store.remove(video_id)
        index.save(index_path)
        store.save(metadata_path)

    thumb = THUMBNAILS_DIR / f"{video_id}.jpg"
    thumb.unlink(missing_ok=True)
    for f in UPLOADS_DIR.glob(f"{video_id}.*"):
        f.unlink(missing_ok=True)

    return {"status": "deleted", "video_id": video_id}


@app.delete("/api/videos")
def delete_all_videos() -> dict[str, int]:
    all_ids = index.list_all()
    deleted = 0
    for video_id in all_ids:
        index.remove(video_id)
        delete_features(video_id)
        if not isinstance(index, QdrantIndex):
            if video_id in store:
                store.remove(video_id)
        thumb = THUMBNAILS_DIR / f"{video_id}.jpg"
        thumb.unlink(missing_ok=True)
        for f in UPLOADS_DIR.glob(f"{video_id}.*"):
            f.unlink(missing_ok=True)
        deleted += 1

    if not isinstance(index, QdrantIndex):
        index.save(index_path)
        store.save(metadata_path)

    return {"deleted": deleted}


@app.get("/api/videos/{video_id}/thumbnail")
def get_thumbnail(video_id: str) -> FileResponse:
    thumb = THUMBNAILS_DIR / f"{video_id}.jpg"
    if not thumb.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(thumb, media_type="image/jpeg")
