"""Frame and tensor transform utilities: crop, resize, temporal ops, visualization."""

import io as BytesIO
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from IPython import display
    import imageio
except Exception:
    pass


@dataclass
class RepeatedTensor:
    """Result of repeat_tensor: the repeated array and the repeat count used."""

    tensor: np.ndarray
    repeat_times: int


def animate(
    frames: np.ndarray, fps: int = 1, save_file: str = "./animation.gif"
) -> None:
    """Save frames as an animated GIF and display inline in a notebook."""
    import os

    if frames.dtype == np.float32:
        frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    imageio.mimsave(save_file, frames, fps=fps, loop=65535)
    with open(save_file, "rb") as f:
        display.display(display.Image(data=f.read()))


def random_crop(video: np.ndarray, desired_size: int) -> np.ndarray:
    """Random spatial crop of a video array (T, H, W, C)."""
    H, W = video.shape[1:3]
    top = np.random.randint(np.maximum(1, (H - desired_size) / 2))
    left = np.random.randint(np.maximum(1, (W - desired_size) / 2))
    return video[:, top : top + desired_size, left : left + desired_size, :]


def center_crop(frame: np.ndarray, desired_size: int) -> np.ndarray:
    """Center crop a frame (H, W, C) or video (T, H, W, C)."""
    if frame.ndim == 3:
        old_size = frame.shape[:2]
        top = int(np.maximum(0, (old_size[0] - desired_size) / 2))
        left = int(np.maximum(0, (old_size[1] - desired_size) / 2))
        return frame[top : top + desired_size, left : left + desired_size, :]
    else:
        old_size = frame.shape[1:3]
        top = int(np.maximum(0, (old_size[0] - desired_size) / 2))
        left = int(np.maximum(0, (old_size[1] - desired_size) / 2))
        return frame[:, top : top + desired_size, left : left + desired_size, :]


def resize_frame(frame: np.ndarray, desired_size: int | tuple[int, int]) -> np.ndarray:
    """Resize a frame using bicubic interpolation."""
    if isinstance(desired_size, int):
        min_size = np.min(frame.shape[:2])
        ratio = desired_size / min_size
        frame = cv2.resize(
            frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC
        )
    elif isinstance(desired_size, tuple):
        frame = cv2.resize(
            frame,
            dsize=(desired_size[1], desired_size[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    return frame


def random_temporal_crop(tensor: np.ndarray, min_size: int) -> np.ndarray:
    """Randomly crop a temporal slice of at least min_size frames."""
    while tensor.shape[0] < min_size:
        tensor = np.concatenate([tensor, tensor], 0)
    offset = np.random.randint(max(len(tensor) - min_size, 1))
    return tensor[offset : offset + min_size]


def repeat_tensor(
    tensor: np.ndarray,
    repeat_times: int | None = None,
    min_size: int | None = None,
    axis: int = 0,
    segments: list | None = None,
) -> RepeatedTensor:
    """Repeat a tensor along an axis until it reaches min_size, or repeat_times times.

    Args:
        tensor: Input array to repeat.
        repeat_times: If given, repeat exactly this many times along axis.
        min_size: If given (with repeat_times=None), repeat until shape[axis] > min_size.
        axis: Axis along which to repeat.
        segments: Optional segment list to extend in sync with the repetition.

    Returns:
        RepeatedTensor with the repeated array and the repeat count used.
    """
    if repeat_times is None:
        repeat_times = 1
        while tensor.shape[axis] <= min_size:
            if segments is not None:
                if axis == 0:
                    q_len, r_len = tensor.shape[axis], 0
                elif axis == 1:
                    q_len, r_len = 0, tensor.shape[axis]
                for q_min, r_min, q_max, r_max in list(segments):
                    segments.append(
                        [q_min + q_len, r_min + r_len, q_max + q_len, r_max + r_len]
                    )
            tensor = np.concatenate([tensor, tensor], axis)
            repeat_times *= 2
    else:
        tensor = np.concatenate([tensor] * repeat_times, axis)
    return RepeatedTensor(tensor=tensor, repeat_times=repeat_times)


def heatmap(
    sim: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Render a similarity matrix as an HWC numpy image via seaborn."""
    ax = sns.heatmap(
        sim,
        cmap="jet",
        square=True,
        vmin=vmin,
        vmax=vmax,
        yticklabels=False,
        xticklabels=False,
    )
    plt.tight_layout()

    io_buf = BytesIO.BytesIO()
    ax.figure.savefig(io_buf, format="raw", pad_inches=0)
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(ax.figure.bbox.bounds[3]), int(ax.figure.bbox.bounds[2]), -1),
    )
    io_buf.close()
    plt.clf()
    return img_arr[:, :, :3]
