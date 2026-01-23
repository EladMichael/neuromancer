"""Standalone PyTorch training script for TI-DON heat dynamics with YAML and CLI configuration.

Usage overview:
  - Default run uses values from DEFAULT_CONFIG.
  - YAML overrides defaults via --config_yaml.
  - CLI flags override YAML and defaults using --<key> for any DEFAULT_CONFIG key.
  - Resolved configuration can be written to result_dir/config_used.yaml with --dump_config 1.

Examples of common workflows:
  - Use a YAML file for experiment settings and keep CLI minimal.
  - Resume training by setting resume_path or passing --resume_path.
  - Save best checkpoints by setting save_best or passing --save_best 1.
  - Disable checkpoint outputs by setting save_best or save_last to 0 or using --save_best/--save_last.

Notes:
  - Supported YAML keys are the same as DEFAULT_CONFIG keys.
  - device accepts \"auto\", \"cuda\", or \"cpu\".
  - dtype currently supports \"float32\" only.
"""

import argparse
import datetime
import os
import sys
import time
import random
from contextlib import contextmanager
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

DEFAULT_CONFIG = {
    "data_path": "/home/pk222/projects/PDEControl_DPC/datasets/heat_smooth_f_dataset.npz",
    "samples_to_load": 100,
    "seed": 0,
    "train_frac": 0.8,
    "batch_size": 75,
    "num_steps": 100,
    "log_iter": 1000,
    "lr": 1e-3,
    "transition_steps": 2000,
    "decay_rate": 0.9,
    "sample_id_rollout": 5,
    "save_iter": 5000,
    "save_best": 1,
    "save_last": 1,
    "resume_path": "",
    "resume_strict": 1,
    "device": "auto",
    "dtype": "float32",
    "num_samples_to_evaluate": 10,
}


class Tee:
    """Duplicate stdout and stderr to multiple streams."""

    def __init__(self, *streams):
        """Initialize Tee with target streams.

        Args:
            *streams: Output streams with write and flush methods.
        """
        self.streams = streams

    def write(self, data):
        """Write data to all streams.

        Args:
            data: String payload for output.
        """
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        """Flush all streams."""
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        """Report TTY status for compatibility.

        Returns:
            bool: True if any stream reports TTY support.
        """
        return any(
            getattr(stream, "isatty", lambda: False)() for stream in self.streams
        )


@contextmanager
def setup_tee_logging(result_dir, seed):
    """Context manager to tee stdout and stderr to a log file.

    Args:
        result_dir: Directory for log file output.
        seed: Integer seed used in log file naming.

    Yields:
        str: Path to the log file.

    Invariants:
        stdout and stderr are restored after the context exits.
    """
    os.makedirs(result_dir, exist_ok=True)
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tee_path = os.path.join(result_dir, f"terminal_{run_ts}_seed{seed}.txt")
    tee_file = open(tee_path, "w", encoding="utf-8")
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    sys.stdout = Tee(stdout_orig, tee_file)
    sys.stderr = Tee(stderr_orig, tee_file)
    try:
        yield tee_path
    finally:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        tee_file.close()


def set_seed(seed):
    """Set global random seeds and deterministic behavior.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str):
    """Resolve computation device.

    Args:
        device_str: String selector "auto", "cuda", or "cpu".

    Returns:
        torch.device: Selected device.

    Raises:
        RuntimeError: If CUDA is requested but unavailable.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_data(config):
    """Load dataset tensors and perform temporal slicing.

    Args:
        config: Dict with keys data_path and samples_to_load.

    Returns:
        solutions: torch.Tensor of shape (samples, T+1, N).
        controls: torch.Tensor of shape (samples, T, 4).
        x_cord: torch.Tensor of shape (N, 1).
        dt: float scalar time step.
        u_t: torch.Tensor of shape (samples, T, N).
        u_next: torch.Tensor of shape (samples, T, N).
        f_t: torch.Tensor of shape (samples, T, 4).
    """
    data_path = config["data_path"]
    samples_to_load = config["samples_to_load"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataset = np.load(data_path)
    solutions = torch.from_numpy(dataset["solutions"][:samples_to_load])
    controls = torch.from_numpy(dataset["controls"][:samples_to_load])
    x_cord = torch.from_numpy(dataset["x"]).reshape(-1, 1)
    dt = float(dataset["dt"])

    print(f"Loaded {samples_to_load}/{dataset['solutions'].shape[0]} samples:")
    print(f"solutions: {tuple(solutions.shape)}")
    print(f"controls: {tuple(controls.shape)}")
    print(f"x: {tuple(x_cord.shape)}")
    print(f"dt: {dt}")

    u_t = solutions[:, :-1, :]
    u_next = solutions[:, 1:, :]
    f_t = controls[:, :, :]

    print(f"u_t: {tuple(u_t.shape)}")
    print(f"u_next: {tuple(u_next.shape)}")
    print(f"f_t: {tuple(f_t.shape)}")

    return solutions, controls, x_cord, dt, u_t, u_next, f_t


def make_train_test_flat(u_t, u_next, f_t, train_frac, seed):
    """Build flattened train/test arrays for supervised learning.

    Args:
        u_t: torch.Tensor of shape (samples, T, N).
        u_next: torch.Tensor of shape (samples, T, N).
        f_t: torch.Tensor of shape (samples, T, 4).
        train_frac: Fraction of samples for training.
        seed: Integer seed controlling shuffle order.

    Returns:
        X_train: torch.Tensor of shape (n_train, 104).
        Y_train: torch.Tensor of shape (n_train, 100).
        X_test: torch.Tensor of shape (n_test, 104).
        Y_test: torch.Tensor of shape (n_test, 100).
    """
    X = torch.cat([u_t, f_t], dim=-1)
    Y = u_next
    X = X.reshape(-1, X.shape[-1])
    Y = Y.reshape(-1, Y.shape[-1])

    print(f"X (input): {tuple(X.shape)}")
    print(f"Y (output): {tuple(Y.shape)}")

    np.random.seed(seed)
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    n_train = int(train_frac * num_samples)
    X_train = X_shuffled[:n_train]
    Y_train = Y_shuffled[:n_train]
    X_test = X_shuffled[n_train:]
    Y_test = Y_shuffled[n_train:]

    return X_train, Y_train, X_test, Y_test


class RandomBatchDataset(IterableDataset):
    """Iterable dataset that samples random batches without replacement.

    Args:
        u_t_f: torch.Tensor of shape (N_train, 104).
        x_cord: torch.Tensor of shape (N, 1).
        u_next: torch.Tensor of shape (N_train, 100).
        batch_size: Integer batch size.
        seed: Integer seed for random sampling.
    """

    def __init__(self, u_t_f, x_cord, u_next, batch_size, seed):
        """Initialize dataset with flat arrays and sampling parameters."""
        self.u_t_f = u_t_f
        self.x_cord = x_cord
        self.u_next = u_next
        self.batch_size = batch_size
        self.num_samples = u_t_f.shape[0]
        self.seed = seed

    def __iter__(self):
        """Yield random batches indefinitely.

        Yields:
            inputs: Tuple(u_t_f_batch, x_cord) with shapes (B, 104) and (N, 1).
            outputs: u_next_batch of shape (B, 100).
        """
        worker_info = torch.utils.data.get_worker_info()
        local_seed = self.seed if worker_info is None else self.seed + worker_info.id
        gen = torch.Generator(device="cpu")
        gen.manual_seed(local_seed)
        while True:
            idx = torch.randperm(self.num_samples, generator=gen, device="cpu")[
                : self.batch_size
            ]
            u_t_f_batch = self.u_t_f[idx]
            u_next_batch = self.u_next[idx]
            inputs = (u_t_f_batch, self.x_cord)
            outputs = u_next_batch
            yield inputs, outputs


def build_model():
    """Construct the DeepXDE MIONet Cartesian product model.

    Returns:
        net: dde.nn.MIONetCartesianProd model instance.
    """
    os.environ["DDE_BACKEND"] = "pytorch"
    import deepxde as dde

    activation = {"branch1": "gelu", "branch2": "gelu", "trunk": "tanh"}

    hidden_dim = 100
    layer_sizes_branch1 = [100, 128, 128, 128, hidden_dim]
    layer_sizes_branch2 = [4, 32, 32, 32, hidden_dim]
    layer_sizes_trunk = [1, 64, 64, 64, hidden_dim]

    net = dde.nn.MIONetCartesianProd(
        layer_sizes_branch1=layer_sizes_branch1,
        layer_sizes_branch2=layer_sizes_branch2,
        layer_sizes_trunk=layer_sizes_trunk,
        activation=activation,
        kernel_initializer="Glorot normal",
        trunk_last_activation=True,
        merge_operation="mul",
        layer_sizes_merger=None,
        layer_sizes_output_merger=None,
    )
    return net


def rk4_predict_one_step(net, u_state, c_t, x_loc, dt):
    """Perform one RK4 prediction using network RHS output.

    Args:
        net: Neural operator model.
        u_state: torch.Tensor of shape (B, 100).
        c_t: torch.Tensor of shape (B, 4).
        x_loc: torch.Tensor of shape (N, 1).
        dt: float time step.

    Returns:
        u_next_pred: torch.Tensor of shape (B, 100).
    """
    alpha1 = 1 / 6
    alpha2 = 2 / 6
    alpha3 = 2 / 6
    alpha4 = 1 / 6

    k1 = net((u_state, c_t, x_loc))
    k2 = net((u_state + 0.5 * dt * k1, c_t, x_loc))
    k3 = net((u_state + 0.5 * dt * k2, c_t, x_loc))
    k4 = net((u_state + dt * k3, c_t, x_loc))

    u_next_pred = u_state + dt * (alpha1 * k1 + alpha2 * k2 + alpha3 * k3 + alpha4 * k4)
    return u_next_pred


def rk4_batch_loss(net, batch, dt):
    """Compute RK4 one-step loss for a batch.

    Args:
        net: Neural operator model.
        batch: Tuple((u_t_f, x_cord), u_next) with shapes (B, 104), (N, 1), (B, 100).
        dt: float time step.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    inputs, outputs = batch
    u_t_f_batch, x_cord_full = inputs
    u_state = u_t_f_batch[:, :100]
    c_t = u_t_f_batch[:, 100:]
    u_next_pred = rk4_predict_one_step(net, u_state, c_t, x_cord_full, dt)
    loss = torch.mean((outputs - u_next_pred) ** 2)
    return loss


def evaluate_batch(net, batch, dt):
    """Evaluate loss and L2 error for a batch without gradients.

    Args:
        net: Neural operator model.
        batch: Tuple((u_t_f, x_cord), u_next) with shapes (B, 104), (N, 1), (B, 100).
        dt: float time step.

    Returns:
        loss_test: float scalar loss.
        l2err_test: float scalar sqrt(loss_test).
    """
    with torch.no_grad():
        loss_test = rk4_batch_loss(net, batch, dt)
        l2err_test = torch.sqrt(loss_test)
    return float(loss_test.item()), float(l2err_test.item())


def save_checkpoint(path, net, optimizer, scheduler, step, loss_test, config):
    """Save model checkpoint with optimizer and scheduler state.

    Args:
        path: Output checkpoint path.
        net: Neural operator model.
        optimizer: torch.optim.Optimizer instance.
        scheduler: torch.optim.lr_scheduler instance.
        step: Integer training step.
        loss_test: float loss value stored as loss_test.
        config: Dict of configuration values.
    """
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            if scheduler is not None
            else None,
            "step": step,
            "loss_test": loss_test,
            "seed": config["seed"],
            "config": config,
        },
        path,
    )


def load_checkpoint(path, net, optimizer, scheduler, strict):
    """Load model checkpoint and return resume state.

    Args:
        path: Input checkpoint path.
        net: Neural operator model.
        optimizer: torch.optim.Optimizer instance.
        scheduler: torch.optim.lr_scheduler instance.
        strict: Boolean strict flag for model state loading.

    Returns:
        start_step: Integer start step for resuming.
        best_loss_test: float best loss value from checkpoint.
    """
    ckpt = torch.load(path, map_location="cpu")
    if "model_state_dict" in ckpt:
        net.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    step = int(ckpt.get("step", -1))
    start_step = step + 1
    best_loss_test = float(ckpt.get("loss_test", float("inf")))
    return start_step, best_loss_test


def plot_train_test_losses(csv_path, result_dir):
    """Plot training and testing loss curves from CSV log.

    Args:
        csv_path: Path to log.csv file.
        result_dir: Output directory for plot file.
    """
    df = pd.read_csv(csv_path)
    fig, (ax1) = plt.subplots(1, 1, figsize=(18, 8), dpi=100)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    ax1.plot(
        df["epoch"], df["loss"], label="Training Loss", color=colors[0], linestyle="-"
    )
    ax1.plot(
        df["epoch"],
        df["loss_test"],
        label="Testing Loss",
        color=colors[1],
        linestyle="-",
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch", fontsize=24)
    ax1.set_ylabel("Train and Test Loss", fontsize=24)
    ax1.set_title("Training Loss and Testing Loss over Epochs", fontsize=26)
    ax1.legend(loc="best", fontsize=24)
    ax1.tick_params(axis="both", which="major", labelsize=24)
    plt.tight_layout()
    output_file = os.path.join(result_dir, "train_and_test_loss_plots.png")
    plt.savefig(output_file, dpi=100)
    plt.close(fig)


def autoregressive_rollout(net, solutions, controls, x_cord, dt, sample_id, device):
    """Run autoregressive RK4 rollout for one sample.

    Args:
        net: Neural operator model.
        solutions: torch.Tensor of shape (samples, T+1, N).
        controls: torch.Tensor of shape (samples, T, 4).
        x_cord: torch.Tensor of shape (N, 1).
        dt: float time step.
        sample_id: Integer sample index.
        device: torch.device for computation.

    Returns:
        mse_loss: torch.Tensor scalar.
        rel_l2_error: torch.Tensor scalar.
        rollout_pred: torch.Tensor of shape (T+1, N).
        true_states: torch.Tensor of shape (T+1, N).
    """
    net.eval()
    with torch.no_grad():
        u0 = solutions[sample_id, 0, :].to(device=device, dtype=torch.float32)
        true_states = solutions[sample_id, :, :].to(device=device, dtype=torch.float32)
        control_seq = controls[sample_id, :, :].to(device=device, dtype=torch.float32)
        x_loc = x_cord.to(device=device, dtype=torch.float32)
        T = control_seq.shape[0]

        u_t = u0.unsqueeze(0)
        rollout_pred = []

        a1 = 1 / 6
        a2 = 2 / 6
        a3 = 2 / 6
        a4 = 1 / 6

        for t in range(T):
            c_t = control_seq[t : t + 1, :]
            k1 = net((u_t, c_t, x_loc))
            k2 = net((u_t + 0.5 * dt * k1, c_t, x_loc))
            k3 = net((u_t + 0.5 * dt * k2, c_t, x_loc))
            k4 = net((u_t + dt * k3, c_t, x_loc))
            u_next = u_t + dt * (a1 * k1 + a2 * k2 + a3 * k3 + a4 * k4)
            rollout_pred.append(u_t.squeeze(0))
            u_t = u_next

        rollout_pred.append(u_t.squeeze(0))
        rollout_pred = torch.stack(rollout_pred, dim=0)

        mse_loss = torch.mean((true_states - rollout_pred) ** 2)
        rel_l2_error = torch.linalg.norm(
            true_states - rollout_pred
        ) / torch.linalg.norm(true_states)

    return mse_loss, rel_l2_error, rollout_pred, true_states


def plot_rollout_results(true_states, rollout_pred, x_cord, dt, sample_id, result_dir):
    """Plot rollout results and absolute error contours.

    Args:
        true_states: np.ndarray of shape (T+1, N).
        rollout_pred: np.ndarray of shape (T+1, N).
        x_cord: np.ndarray of shape (N, 1).
        dt: float time step.
        sample_id: Integer sample index.
        result_dir: Output directory for plot file.
    """
    x_axis = x_cord.squeeze()
    T_plus_1 = true_states.shape[0]
    time_axis = np.linspace(0, dt * (T_plus_1 - 1), T_plus_1)
    error = np.abs(true_states - rollout_pred)

    vmin = float(min(true_states.min(), rollout_pred.min()))
    vmax = float(max(true_states.max(), rollout_pred.max()))

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(
        3,
        3,
        height_ratios=[20, 0.3, 1.0],
        width_ratios=[1, 1, 1],
        hspace=0.15,
        wspace=0.2,
        bottom=0.15,
        top=0.88,
        left=0.08,
        right=0.95,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)

    im0 = ax0.contourf(
        x_axis, time_axis, true_states, levels=120, cmap="inferno", vmin=vmin, vmax=vmax
    )
    ax0.set_title("FDM Prediction", fontsize=22)
    ax0.set_xlabel("x", fontsize=20)
    ax0.set_ylabel("t", fontsize=20)
    ax0.tick_params(labelsize=18)

    im1 = ax1.contourf(
        x_axis,
        time_axis,
        rollout_pred,
        levels=120,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title("TI DON Prediction", fontsize=22)
    ax1.set_xlabel("x", fontsize=20)
    ax1.tick_params(labelsize=18, labelleft=False)

    im2 = ax2.contourf(x_axis, time_axis, error, levels=120, cmap="magma")
    ax2.set_title("Absolute Error", fontsize=22)
    ax2.set_xlabel("x", fontsize=20)
    ax2.tick_params(labelsize=18, labelleft=False)

    cbar_ax1 = fig.add_subplot(gs[2, 0:2])
    cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation="horizontal")
    cbar1.set_label("u", fontsize=20)
    cbar1.ax.tick_params(labelsize=18)

    cbar_ax2 = fig.add_subplot(gs[2, 2])
    cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation="horizontal")
    cbar2.set_label("Abs. Error", fontsize=20)
    cbar2.ax.tick_params(labelsize=18)

    output_path = os.path.join(result_dir, f"rollout_sample_{sample_id}.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Rollout plot saved to {output_path}")


def _dump_config_yaml(config, path):
    """Write configuration to a YAML file.

    Args:
        config: Dict of configuration values.
        path: Output YAML path.
    """
    try:
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    except ImportError:
        with open(path, "w", encoding="utf-8") as f:
            for key, value in config.items():
                if isinstance(value, str):
                    f.write(f'{key}: "{value}"\n')
                else:
                    f.write(f"{key}: {value}\n")


def main():
    """Entry point for training, logging, and rollout execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="")
    parser.add_argument("--dump_config", type=int, default=1)
    for key in DEFAULT_CONFIG:
        parser.add_argument(f"--{key}", default=None)
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.config_yaml:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for --config_yaml.") from exc
        with open(args.config_yaml, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}
        for key, value in yaml_config.items():
            if key in config:
                config[key] = value

    for key in DEFAULT_CONFIG:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    int_keys = [
        "samples_to_load",
        "seed",
        "batch_size",
        "num_steps",
        "log_iter",
        "transition_steps",
        "sample_id_rollout",
        "save_iter",
        "save_best",
        "save_last",
        "resume_strict",
        "num_samples_to_evaluate",
    ]
    float_keys = ["train_frac", "lr", "decay_rate"]
    for key in int_keys:
        config[key] = int(config[key])
    for key in float_keys:
        config[key] = float(config[key])

    dtype_str = str(config["dtype"]).lower()
    if dtype_str != "float32":
        raise ValueError("Only float32 dtype is supported.")
    dtype = torch.float32

    result_dir = os.path.join(os.getcwd(), f"result_{config['seed']}")
    os.makedirs(result_dir, exist_ok=True)
    if int(args.dump_config) == 1:
        _dump_config_yaml(config, os.path.join(result_dir, "config_used.yaml"))

    with setup_tee_logging(result_dir, config["seed"]):
        set_seed(config["seed"])
        device = get_device(str(config["device"]).lower())
        print(f"Using device: {device}")

        solutions, controls, x_cord, dt, u_t, u_next, f_t = load_data(config)
        X_train, Y_train, X_test, Y_test = make_train_test_flat(
            u_t, u_next, f_t, config["train_frac"], config["seed"]
        )

        train_dataset = RandomBatchDataset(
            X_train, x_cord, Y_train, config["batch_size"], config["seed"]
        )
        test_dataset = RandomBatchDataset(
            X_test, x_cord, Y_test, config["batch_size"], config["seed"] + 1
        )
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

        example_batch = next(iter(train_loader))
        (u_t_f_batch, x_cord_full), u_next_batch = example_batch
        print(f"u_t_f_batch: {tuple(u_t_f_batch.shape)}")
        print(f"x_cord: {tuple(x_cord_full.shape)}")
        print(f"u_next_batch: {tuple(u_next_batch.shape)}")

        net = build_model().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
        lr_lambda = lambda step: config["decay_rate"] ** (
            step / config["transition_steps"]
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        best_loss_test = float("inf")
        start_step = 0
        if config["resume_path"]:
            start_step, best_loss_test = load_checkpoint(
                config["resume_path"],
                net,
                optimizer,
                scheduler,
                bool(config["resume_strict"]),
            )

        log_file = os.path.join(result_dir, "log.csv")
        if config["resume_path"]:
            if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
                with open(log_file, "w") as f:
                    f.write("epoch,loss,l2err_test,loss_test,runtime\n")
        else:
            with open(log_file, "w") as f:
                f.write("epoch,loss,l2err_test,loss_test,runtime\n")
        event_log = os.path.join(result_dir, "checkpoint.log")

        start_time = time.time()
        train_iter = iter(train_loader)
        test_iter = iter(test_loader)
        debug_shapes = True
        last_step = start_step - 1
        last_loss_test_val = None

        for step in range(start_step, config["num_steps"]):
            net.train()
            inputs, outputs = next(train_iter)
            u_t_f_batch, x_cord_full = inputs
            u_t_f_batch = u_t_f_batch.to(device=device, dtype=dtype)
            x_cord_full = x_cord_full.to(device=device, dtype=dtype)
            u_next_batch = outputs.to(device=device, dtype=dtype)

            loss = rk4_batch_loss(net, ((u_t_f_batch, x_cord_full), u_next_batch), dt)
            if debug_shapes:
                with torch.no_grad():
                    u_state = u_t_f_batch[:, :100]
                    c_t = u_t_f_batch[:, 100:]
                    k1 = net((u_state, c_t, x_cord_full))
                    u_next_pred = rk4_predict_one_step(
                        net, u_state, c_t, x_cord_full, dt
                    )
                    print(f"u_state: {tuple(u_state.shape)}")
                    print(f"c_t: {tuple(c_t.shape)}")
                    print(f"x_loc: {tuple(x_cord_full.shape)}")
                    print(f"k1: {tuple(k1.shape)}")
                    print(f"u_next_pred: {tuple(u_next_pred.shape)}")
                    print(f"u_next_batch: {tuple(u_next_batch.shape)}")
                debug_shapes = False

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % config["log_iter"] == 0:
                net.eval()
                with torch.no_grad():
                    test_inputs, test_outputs = next(test_iter)
                    test_u_t_f_batch, test_x_cord_full = test_inputs
                    test_u_t_f_batch = test_u_t_f_batch.to(device=device, dtype=dtype)
                    test_x_cord_full = test_x_cord_full.to(device=device, dtype=dtype)
                    test_u_next_batch = test_outputs.to(device=device, dtype=dtype)
                    loss_test = rk4_batch_loss(
                        net,
                        ((test_u_t_f_batch, test_x_cord_full), test_u_next_batch),
                        dt,
                    )
                    l2err_test = torch.sqrt(loss_test)

                runtime = time.time() - start_time
                loss_val = loss.item()
                l2_val = l2err_test.item()
                loss_test_val = loss_test.item()
                last_loss_test_val = loss_test_val
                print(f"Iteration {step}/{config['num_steps']}")
                print(
                    f"Train_loss: {loss_val:.2e},l2_err_test: {l2_val:.2e}, test_loss: {loss_test_val:.2e} , runtime: {runtime:06.2f}"
                )
                with open(log_file, "a") as f:
                    f.write(f"{step},{loss_val},{l2_val},{loss_test_val},{runtime}\n")

                if config["save_best"] == 1 and loss_test_val < best_loss_test:
                    best_loss_test = loss_test_val
                    best_path = os.path.join(result_dir, "model_best.pt")
                    save_checkpoint(
                        best_path,
                        net,
                        optimizer,
                        scheduler,
                        step,
                        loss_test_val,
                        config,
                    )
                    msg = f"New best model saved at step {step} with loss_test = {loss_test_val}"
                    print(msg)
                    with open(event_log, "a") as f:
                        f.write(msg + "\n")

                plot_train_test_losses(log_file, result_dir)
            last_step = step

        if config["save_last"] == 1:
            last_path = os.path.join(result_dir, "model_last.pt")
            loss_test_for_last = (
                last_loss_test_val if last_loss_test_val is not None else best_loss_test
            )
            save_checkpoint(
                last_path,
                net,
                optimizer,
                scheduler,
                last_step,
                loss_test_for_last,
                config,
            )

        plot_train_test_losses(log_file, result_dir)

        mse_loss, rel_l2_error, rollout_pred, true_states = autoregressive_rollout(
            net,
            solutions,
            controls,
            x_cord,
            dt,
            config["sample_id_rollout"],
            device,
        )
        print(f"Test MSE Loss: {mse_loss.item()}")
        print(f"Relative L2 Error: {rel_l2_error.item()}")

        plot_rollout_results(
            true_states.detach().cpu().numpy(),
            rollout_pred.detach().cpu().numpy(),
            x_cord.detach().cpu().numpy(),
            dt,
            config["sample_id_rollout"],
            result_dir,
        )

        sample_indices = np.random.choice(
            solutions.shape[0],
            size=config.get("num_samples_to_evaluate", 10),
            replace=False,
        )
        all_losses = []
        all_l2_errors = []
        for idx in sample_indices:
            loss_val, l2_error, pred, true = autoregressive_rollout(
                net, solutions, controls, x_cord, dt, int(idx), device
            )
            loss_item = loss_val.item()
            l2_item = l2_error.item()
            all_losses.append(loss_item)
            all_l2_errors.append(l2_item)
            print(
                f"Sample {idx:4d} | MSE Loss: {loss_item:.4e} | Rel L2: {l2_item:.4e}"
            )
            plot_rollout_results(
                true.detach().cpu().numpy(),
                pred.detach().cpu().numpy(),
                x_cord.detach().cpu().numpy(),
                dt,
                int(idx),
                result_dir,
            )

        print(f"\n==== Summary over {len(sample_indices)} samples ====")
        print(f"Avg MSE Loss     : {np.mean(np.array(all_losses)):.4e}")
        print(f"Avg Rel L2 Error : {np.mean(np.array(all_l2_errors)):.4e}")


if __name__ == "__main__":
    main()
