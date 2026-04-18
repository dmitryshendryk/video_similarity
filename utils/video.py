"""Video loading utilities: FFmpeg and OpenCV-based frame extraction."""

import glob
import logging
import os

import ffmpeg
import numpy as np

from utils.transforms import (
    center_crop,
    random_temporal_crop,
    repeat_tensor,
    resize_frame,
)

logger = logging.getLogger(__name__)


def load_video_ffmpeg(
    video: str,
    start: float | None = None,
    end: float | None = None,
    fps: int | None = None,
    crop: int | tuple[int, int] | None = None,
    resize: int | tuple[int, int] | None = None,
    keyframes_only: bool = False,
    max_frames: int = 60,
) -> np.ndarray:
    """Load a video file into a numpy array using FFmpeg.

    Args:
        video: Path to the video file.
        start: Start time in seconds (uniform-fps mode only).
        end: End time in seconds (uniform-fps mode only).
        fps: Frames per second to sample (ignored when keyframes_only=True).
        crop: Center-crop size as int or (H, W) tuple.
        resize: Resize shortest side to this value (int) or exact (H, W) tuple.
        keyframes_only: If True, extract only I-frames (keyframes) using
            ``select='eq(pict_type,I)'`` with ``vsync='vfr'``.
            Combining ``keyframes_only=True`` with ``start``/``end`` is not
            supported and raises ``ValueError``.
            If zero I-frames are extracted the function falls back to uniform
            1 FPS sampling and logs a warning.
        max_frames: Cap on the number of frames returned.  Frames are
            subsampled uniformly when the extracted count exceeds this value.
            Default 60.

    Returns:
        numpy.ndarray of shape (T, H, W, 3), dtype uint8.

    Raises:
        ValueError: When keyframes_only=True is combined with start or end.
    """
    if keyframes_only and (start is not None or end is not None):
        raise ValueError("keyframes_only=True with start/end is not supported")

    probe = ffmpeg.probe(video)
    video_info = next(x for x in probe["streams"] if x["codec_type"] == "video")
    width = int(video_info["width"])
    height = int(video_info["height"])

    if start is not None and end is not None:
        cap = ffmpeg.input(video, ss=start, to=end)
    else:
        cap = ffmpeg.input(video)

    if keyframes_only:
        cap = cap.filter("select", "eq(pict_type,I)")
    elif fps is not None:
        cap = cap.filter("fps", fps=fps)

    if isinstance(resize, int):
        min_size = np.min([width, height])
        ratio = resize / min_size
        height = int(np.ceil(height * ratio / 2) * 2)
        width = int(np.ceil(width * ratio / 2) * 2)
        cap = cap.filter("scale", width=width, height=height)
    elif isinstance(resize, tuple):
        height = resize[0]
        width = resize[1]
        cap = cap.filter("scale", width=resize[1], height=resize[0])

    if isinstance(crop, int):
        y = int(np.maximum(0, (height - crop) / 2))
        x = int(np.maximum(0, (width - crop) / 2))
        cap = cap.filter("crop", x=x, y=y, w=crop, h=crop)
        height = crop
        width = crop
    elif isinstance(crop, tuple):
        y = int(np.maximum(0, (height - crop[0]) / 2))
        x = int(np.maximum(0, (width - crop[1]) / 2))
        cap = cap.filter("crop", x=x, y=y, w=crop[1], h=crop[0])
        height = crop[0]
        width = crop[1]

    output_kwargs: dict[str, str | int] = {
        "format": "rawvideo",
        "pix_fmt": "rgb24",
        "crf": 0,
    }
    if keyframes_only:
        output_kwargs["vsync"] = "vfr"

    out, _ = (
        cap.output("pipe:", **output_kwargs)
        .global_args("-loglevel", "panic")
        .run(capture_stdout=True)
    )

    result = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

    if keyframes_only and result.shape[0] == 0:
        logger.warning(
            "Zero I-frames extracted from %s, falling back to uniform 1 FPS sampling",
            video,
        )
        return load_video_ffmpeg(
            video,
            fps=1,
            crop=crop,
            resize=resize,
            keyframes_only=False,
            max_frames=max_frames,
        )

    if result.shape[0] > max_frames:
        indices = np.linspace(0, result.shape[0] - 1, max_frames, dtype=int)
        result = result[indices]

    video_duration = float(
        video_info.get("duration", probe.get("format", {}).get("duration", 0) or 0)
    )
    if result.shape[0] < 3 and video_duration > 10.0:
        logger.warning(
            "Anomalously few frames (%d) extracted from %s (duration=%.1fs)",
            result.shape[0],
            video,
            video_duration,
        )

    return result


def load_video_opencv(
    video: str,
    all_frames: bool = False,
    fps: int = 1,
    crop: int | None = None,
    resize: int | tuple[int, int] | None = None,
) -> np.ndarray:
    """Load a video using OpenCV frame-by-frame decoding."""
    import cv2

    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video)
    fps_v = cap.get(cv2.CAP_PROP_FPS)
    if fps_v > 144 or fps_v is None:
        fps_v = 25
    frames = []
    count = 0
    while cap.isOpened():
        _ = cap.grab()
        if int(count % round(fps_v / fps)) == 0 or all_frames:
            ret, frame = cap.retrieve()
            if isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if resize is not None:
                    frame = resize_frame(frame, resize)
                frames.append(frame)
            else:
                break
        count += 1
    cap.release()
    frames_arr = np.array(frames)
    if crop is not None:
        frames_arr = center_crop(frames_arr, crop)
    return frames_arr


def load_frames_opencv(
    video_dir: str,
    start: int = 0,
    end: int | None = None,
    crop: int | None = None,
    resize: int | tuple[int, int] | None = None,
) -> np.ndarray:
    """Load pre-extracted JPEG frames from a directory."""
    import cv2

    cv2.setNumThreads(2)
    if end is None:
        end = len(os.listdir(video_dir))

    frames = []
    for frame_id in range(start, end):
        frame_file = os.path.join(video_dir, f"{frame_id:05d}.jpg")
        if os.path.exists(frame_file):
            frame = cv2.imread(frame_file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize is not None:
                frame = resize_frame(frame, resize)
            frames.append(frame)
    assert len(frames) > 0, "{} {} {}".format(video_dir, start, end)
    frames_arr = np.stack(frames)
    if crop is not None:
        frames_arr = center_crop(frames_arr, crop)
    return frames_arr


def get_video_length(video_dir: str) -> int:
    """Return the number of frames in a pre-extracted frame directory."""
    return len(os.listdir(video_dir))


def load_video(
    video_id: str,
    video_dir: str | None = None,
    fps: int = 1,
    start: int = 0,
    end: int | None = None,
    window: int | None = None,
    repeat_times: int = 1,
    resize: int | tuple[int, int] | None = None,
    crop: int | tuple[int, int] | None = None,
) -> np.ndarray:
    """Load a video by ID, optionally from a directory, with optional windowing."""
    if video_dir is not None:
        video_id = os.path.join(video_dir, video_id)

    video_file = glob.glob(os.path.join(video_id, "video.*"))[0]
    if repeat_times > 1:
        video = load_video_ffmpeg(video_file, fps=fps, resize=resize, crop=crop)
        result = repeat_tensor(video, repeat_times)
        video = result.tensor[start:end]
    else:
        if window is not None:
            video_len = get_video_length(video_file)
            if video_len > window:
                start = np.random.randint(max(video_len - window, 1))
                end = start + window
        video = load_video_ffmpeg(
            video_file, start=start, end=end, fps=fps, resize=resize, crop=crop
        )
        if window is not None:
            video = random_temporal_crop(video, window)
    return video


def load_frames(
    video_id: str = "",
    video_dir: str | None = None,
    start: int = 0,
    end: int | None = None,
    window: int | None = None,
    repeat_times: int = 1,
    resize: int = 256,
    crop: int | None = None,
) -> np.ndarray:
    """Load pre-extracted frames from a directory with optional windowing."""
    if video_dir is not None:
        video_id = os.path.join(video_dir, video_id)

    if repeat_times > 1:
        video = load_frames_opencv(video_id, resize=resize, crop=crop)
        result = repeat_tensor(video, repeat_times)
        video = result.tensor[start:end]
    else:
        if window is not None and start == 0 and end is None:
            video_len = get_video_length(video_id)
            if video_len > window:
                start = np.random.randint(max(video_len - window, 1))
                end = start + window
        video = load_frames_opencv(
            video_id, start=start, end=end, resize=resize, crop=crop
        )
        if window is not None:
            video = random_temporal_crop(video, window)
    return video


def load_features(
    feature_file: "h5py.File",
    video_id: str,
    start: int = 0,
    end: int | None = None,
    repeat_times: int = 1,
) -> np.ndarray:
    """Load features from an HDF5 file handle for a given video ID."""
    features = feature_file[video_id][:]
    result = repeat_tensor(features, repeat_times)
    feature = result.tensor[start:end]
    return feature
