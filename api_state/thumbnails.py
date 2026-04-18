"""
Thumbnail generation for uploaded videos via FFmpeg.
"""

import json
import subprocess

from pathlib import Path

from api_state.config import THUMBNAILS_DIR


def generate_thumbnail(video_path: Path, video_id: str) -> Path | None:
    """Extract a single frame at 10% of duration as JPEG thumbnail."""
    thumb_path = THUMBNAILS_DIR / f"{video_id}.jpg"
    if thumb_path.exists():
        return thumb_path
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        probe = json.loads(result.stdout)
        duration = float(probe.get("format", {}).get("duration", 10))
        seek_time = duration * 0.1

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(seek_time),
                "-i",
                str(video_path),
                "-vframes",
                "1",
                "-q:v",
                "2",
                "-vf",
                "scale=320:-1",
                str(thumb_path),
            ],
            capture_output=True,
            timeout=30,
        )
        if thumb_path.exists():
            return thumb_path
    except Exception:
        pass
    return None
