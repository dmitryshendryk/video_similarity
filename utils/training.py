"""Training utilities: distributed setup, model I/O, meters, logging helpers."""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Generator, Protocol

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class AttentionOutput:
    """Output of a model attention layer."""

    attended: torch.Tensor
    weights: torch.Tensor


@dataclass
class SimilarityMatrixOutput:
    """Output of a model similarity_matrix call."""

    output_sim: torch.Tensor
    input_sim: torch.Tensor
    extra: torch.Tensor


class _HasAttention(Protocol):
    """Protocol for models that expose an attention layer."""

    def attention(self, features: torch.Tensor) -> AttentionOutput: ...


class _HasSimilarityMatrix(Protocol):
    """Protocol for models that expose a similarity_matrix method."""

    def similarity_matrix(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        return_f2f: bool,
        normalize: bool,
        batched: bool,
    ) -> SimilarityMatrixOutput: ...


def init_distributed_mode(args: argparse.Namespace) -> None:
    """Initialise torch.distributed from environment or SLURM variables."""
    # launched with torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    # launched with submitit on a slurm cluster
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif torch.cuda.is_available():
        print("Will run the code on one GPU.")
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
    else:
        print("Does not support training without GPU.")
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    dist.barrier(device_ids=[int(args.rank)])
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master: bool) -> None:
    """Disable printing when not in master process."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def _print(*args, **kwargs) -> None:
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = _print


def is_parallel(model: nn.Module) -> bool:
    """Return True if model is wrapped in DataParallel or DistributedDataParallel."""
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


def collate_eval(batch: list) -> tuple:
    """Collate a batch of (video, video_id) pairs with zero-padding."""
    videos, video_ids = zip(*batch)
    num = len(videos)
    max_len = max([s.size(0) for s in videos])
    max_reg = max([s.size(1) for s in videos])
    dims = videos[0].size(2)

    padded_videos = videos[0].data.new(*(num, max_len, max_reg, dims)).fill_(0)
    masks = videos[0].data.new(*(num, max_len)).fill_(0)
    for i, tensor in enumerate(videos):
        length = tensor.size(0)
        padded_videos[i, :length] = tensor
        masks[i, :length] = 1

    return padded_videos, masks, video_ids


def batching(tensor: torch.Tensor, batch_sz: int) -> Generator:
    """Yield successive batch_sz-sized slices of tensor."""
    L = len(tensor)
    for i in range(L // batch_sz + 1):
        if i * batch_sz < L:
            yield tensor[i * batch_sz : (i + 1) * batch_sz]


def save_model(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    file_name: str = "model.pth",
) -> None:
    """Save model and optimiser state to disk."""
    save_dict = {
        "args": args,
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
    }
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
    torch.save(save_dict, os.path.join(args.experiment_path, file_name))


def load_model(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    file_name: str = "model.pth",
) -> int:
    """Load model and optimiser state from disk. Returns global_step."""
    print(">> loading network")
    d = torch.load(os.path.join(args.experiment_path, file_name), map_location="cpu")
    model.module.load_state_dict(d["model"])
    optimizer.load_state_dict(d["optimizer"])
    global_step = d.pop("global_step")
    return global_step


def bool_flag(s: str) -> bool:
    """Argparse type for boolean flags accepting on/off/true/false/1/0."""
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def pprint_args(args: argparse.Namespace) -> None:
    """Pretty-print argparse namespace."""
    print("\nInput Arguments")
    print("---------------")
    for k, v in sorted(dict(vars(args)).items()):
        print("%s: %s" % (k, str(v)))


@torch.no_grad()
def writer_log(
    writer: SummaryWriter,
    model: nn.Module,
    meters: "AverageMeterDict",
    total_values: int,
    lr: float,
    videos: torch.Tensor,
    features: torch.Tensor,
    global_step: int,
) -> None:
    """Log training metrics, histograms, and similarity matrix visualisations to TensorBoard."""
    from utils.transforms import heatmap

    model.eval()
    for k, v in meters.items():
        writer.add_scalar("training/{}".format(k), v.avg(total_values), global_step)
    for k, v in model.state_dict().items():
        writer.add_histogram(str(k).replace(".", "/"), v, global_step)

    writer.add_scalar("training/lr", lr, global_step)

    if isinstance(model, _HasAttention):
        attn_out = model.attention(features)
        writer.add_histogram("att/weights", attn_out.weights, global_step)

    features = model.index_video(features)
    writer.add_histogram("features", features, global_step)

    if isinstance(model, _HasSimilarityMatrix):
        idx = np.random.randint(videos.shape[0] // 2)

        anchors, positives = torch.chunk(features, 2, dim=0)
        sim_result = model.similarity_matrix(
            anchors[idx], positives[idx], return_f2f=True, normalize=True, batched=True
        )

        sim_out = sim_result.output_sim.cpu().numpy()
        sim_in = sim_result.input_sim.cpu().numpy()

        a, p = np.unravel_index(sim_in[0, 0].argmax(), sim_in[0, 0].shape)

        writer.add_image(
            "frames/anchor", videos[idx, a].cpu(), global_step, dataformats="HWC"
        )
        writer.add_image(
            "frames/positive",
            videos[idx + videos.shape[0] // 2, p].cpu(),
            global_step,
            dataformats="HWC",
        )

        writer.add_image(
            "similarity_matrices/input_matrix",
            heatmap(sim_in[0].mean(0)),
            global_step,
            dataformats="HWC",
        )
        writer.add_image(
            "similarity_matrices/output_matrix",
            heatmap(sim_out[0, 0], 0.0, 1.0),
            global_step,
            dataformats="HWC",
        )
    torch.cuda.empty_cache()
    model.train()


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name: str, fmt: str = ":.3f") -> None:
        self.name = name
        self.fmt = fmt
        self.values: list[float] = []

    def reset(self) -> None:
        self.values = []

    def update(self, val: float) -> None:
        self.values.append(val)

    def avg(self, n: int | None = None) -> float:
        avg = self.values[-n:] if n is not None else self.values
        return np.mean(avg)

    def last(self) -> float:
        return self.values[-1]

    def __len__(self) -> int:
        return len(self.values)

    def __str__(self) -> str:
        fmtstr = "{val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(val=self.last(), avg=self.avg())


class AverageMeterDict(object):
    """Dict of AverageMeter instances keyed by name."""

    def __init__(self) -> None:
        self.meter_dict: dict[str, AverageMeter] = dict()

    def reset(self) -> None:
        for k, v in self.meter_dict.items():
            v.reset()

    def add(self, name: str, fmt: str = ":.3f") -> None:
        self.meter_dict[name] = AverageMeter(name, fmt)

    def get(self, name: str) -> AverageMeter:
        return self.meter_dict[name]

    def update(self, name: str, val: float | torch.Tensor) -> None:
        if isinstance(val, torch.Tensor):
            val = val.clone().detach().cpu().numpy()
        if name not in self.meter_dict:
            self.add(name)
        self.meter_dict[name].update(val)

    def avg(self, n: int | None = None) -> dict[str, float]:
        return {k: v.avg(n) for k, v in self.meter_dict.items()}

    def last(self) -> dict[str, float]:
        return {k: v.last() for k, v in self.meter_dict.items()}

    def items(self) -> list:
        return list(self.meter_dict.items())

    def to_str(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.meter_dict.items()}

    def __len__(self) -> int:
        return min([len(v) for v in self.meter_dict.values()])
