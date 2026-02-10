"""
Standalone Python script equivalent to Part_7_TIDON_DPC_HEAT.ipynb.
"""

import os
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from neuromancer.callbacks import Callback
from neuromancer.constraint import Objective, variable
from neuromancer.dataset import DictDataset
from neuromancer.dynamics import integrators
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.modules.operators import DeepXDEIntegratorWrapper, DeepXDEWrapper
from neuromancer.problem import Problem
from neuromancer.system import Node, System
from neuromancer.trainer import Trainer

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def save_fig(name: str, tight_layout: bool = True):
    """Save the current matplotlib figure to the plots directory and close it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(PLOTS_DIR / name)
    plt.close()


def rbf_covariance_gpytorch(x_grid, lengthscale_eff, variance):
    """Compute RBF covariance using GPyTorch on CPU."""
    x = torch.tensor(x_grid, dtype=torch.float64).view(-1, 1)
    rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1)
    rbf_kernel.lengthscale = torch.tensor(float(lengthscale_eff), dtype=torch.float64)
    scale_kernel = gpytorch.kernels.ScaleKernel(rbf_kernel)
    scale_kernel.outputscale = torch.tensor(float(variance), dtype=torch.float64)
    with torch.no_grad():
        cov = scale_kernel(x).evaluate()
    return cov.cpu().numpy().astype(np.float64)


def create_1d_grf_zero_boundary(x_grid, lengthscale, variance, kernel_type, seed):
    """Generate a single 1D GRF sample with zero boundary conditions using an RBF kernel."""
    _ = kernel_type
    if seed is not None:
        np.random.seed(int(seed))

    x = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    n_points = x.shape[0]
    if n_points < 3:
        raise ValueError("x_grid must have at least 3 points for interior sampling.")

    boundary_indices = [0, n_points - 1]
    interior_indices = list(range(1, n_points - 1))

    lengthscale = float(lengthscale)
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be positive.")
    span = float(x[-1] - x[0])
    lengthscale_eff = lengthscale * span if span != 0.0 else lengthscale

    cov_full = rbf_covariance_gpytorch(x, lengthscale_eff, variance)
    cov_full = cov_full + 1e-10 * np.eye(n_points)

    k_ii = cov_full[np.ix_(interior_indices, interior_indices)]
    k_ib = cov_full[np.ix_(interior_indices, boundary_indices)]
    k_bb = cov_full[np.ix_(boundary_indices, boundary_indices)]
    k_bb_inv = np.linalg.inv(k_bb)
    k_conditional = k_ii - k_ib @ k_bb_inv @ k_ib.T
    k_conditional = 0.5 * (k_conditional + k_conditional.T)

    eigvals, eigvecs = np.linalg.eigh(k_conditional)
    eigvals = np.maximum(eigvals, 1e-10)
    l_mat = eigvecs @ np.diag(np.sqrt(eigvals))
    z = np.random.randn(len(interior_indices))
    interior_vals = l_mat @ z

    grf = np.zeros(n_points, dtype=np.float64)
    grf[interior_indices] = interior_vals
    return grf.astype(np.float32)


def generate_grf_pairs(
    num_samples,
    x_cord,
    length_scale,
    variance,
    num_modes,
    seed,
):
    """Generate GRF (u0, u_tf) pairs, no train/test split."""
    _ = num_modes
    rng = np.random.default_rng(None if seed is None else int(seed))
    x_grid = np.asarray(x_cord, dtype=np.float32).reshape(-1)

    grf_inits = []
    grf_finals = []
    for _ in range(int(num_samples)):
        s1 = int(rng.integers(0, 10_000))
        s2 = int(rng.integers(0, 10_000))
        grf_init = create_1d_grf_zero_boundary(
            x_grid, length_scale, variance, "rbf", seed=s1
        )
        grf_final = create_1d_grf_zero_boundary(x_grid, 0.4, variance, "rbf", seed=s2)
        grf_inits.append(grf_init)
        grf_finals.append(grf_final)

    grf_inits = np.array(grf_inits, dtype=np.float32)
    grf_finals = np.array(grf_finals, dtype=np.float32)

    inits = torch.from_numpy(grf_inits)
    finals = torch.from_numpy(grf_finals)
    return inits, finals


def build_dictdataset_policy(inits, finals, x_cord, nsteps, name="policy_grf"):
    """
    inits:    (B, nx)
    finals:   (B, nx)
    x_cord:   (nx, 1)
    """
    if x_cord.dim() == 1:
        x_cord = x_cord.reshape(-1, 1)

    trunk_inputs = x_cord.unsqueeze(0).expand(inits.shape[0], -1, -1)

    # repeat u_tf across time to match rollout horizon
    u_t_final = finals[:, None, :].repeat(1, nsteps + 1, 1)  # (B, nsteps+1, nx)

    return DictDataset(
        {
            "u_t": inits[:, None, :].float(),  # (B, 1, nx)
            "u_tf": u_t_final.float(),  # (B, nsteps+1, nx)
            "trunk_inputs": trunk_inputs.float(),  # (B, nx, 1)
        },
        name=name,
    )


class MLPControl(torch.nn.Module):
    """MLP control policy mapping (u_t, u_tf) to bounded controls."""

    def __init__(self, state_dim, control_dim, h_layers=None):
        super().__init__()
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        if h_layers is None:
            h_layers = [128, 128, 128, self.control_dim]
        self.h_layers = [int(v) for v in h_layers]
        self.output_dim = int(self.control_dim)
        if self.h_layers[-1] != self.output_dim:
            raise ValueError("Last h_layers entry must match output_dim.")

        input_dim = self.state_dim * 2
        layers = []
        prev_dim = input_dim
        for width in self.h_layers:
            layers.append(torch.nn.Linear(prev_dim, width))
            prev_dim = width
        self.layers = torch.nn.ModuleList(layers)
        self.output_bias = torch.nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, u_t, u_tf):
        if u_t.dim() != 2 or u_tf.dim() != 2:
            raise ValueError("u_t and u_tf must be 2D tensors.")
        if u_t.shape != u_tf.shape:
            raise ValueError("u_t and u_tf must have identical shapes.")
        if u_t.shape[1] != self.state_dim:
            raise ValueError("u_t second dimension must match state_dim.")
        inputs = torch.cat([u_t, u_tf], dim=1)
        if inputs.shape[1] != self.state_dim * 2:
            raise ValueError("Policy input dimension must be 2 * state_dim.")
        x = inputs
        for layer in self.layers:
            x = torch.nn.functional.silu(layer(x))
        if x.shape[1] != self.output_dim:
            raise ValueError("Policy output dimension must match output_dim.")
        x = x + self.output_bias
        return torch.tanh(x) * 40.0


class PolicyMLPBounds(torch.nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.nx = nx
        self.nu = nu
        self.net = blocks.MLP_bounds(
            insize=2 * nx,
            outsize=nu,
            hsizes=[128, 128, 128],
            nonlin=activations["gelu"],
            min=-40.0,
            max=40.0,
        )

    def forward(self, u_t, u_tf):
        x = torch.cat([u_t, u_tf], dim=1)
        return self.net(x)


class StepLRSchedulerCallback(Callback):
    """
    Scheduler that steps every batch end, and logs loss every `log_every`.

    Args:
        scheduler: Learning rate scheduler to step.
        log_every: Frequency of logging loss (in number of steps). If 0, no logging.
        global_step: Internal counter for total steps taken.
        epoch_step: Counter for steps within the current epoch.
        step_indices: List of global step indices where loss was logged.
        step_losses: List of losses corresponding to the logged step indices.
    """

    def __init__(self, scheduler, log_every=1):
        super().__init__()
        self.scheduler = scheduler
        self.log_every = log_every
        self.global_step = 0
        self._last_epoch = None
        self.epoch_step = 0
        self.step_indices = []
        self.step_losses = []

    def end_batch(self, trainer, output):
        if self._last_epoch != trainer.current_epoch:
            self._last_epoch = trainer.current_epoch
            self.epoch_step = 0

        self.scheduler.step()

        if self.log_every and self.epoch_step % self.log_every == 0:
            loss = output.get(trainer.train_metric)
            if loss is not None:
                loss_val = loss.detach().cpu().item()
                print(
                    f"epoch: {trainer.current_epoch} step: {self.epoch_step} "
                    f"{trainer.train_metric}: {loss_val:.6g}"
                )
                self.step_indices.append(self.global_step)
                self.step_losses.append(loss_val)

        self.epoch_step += 1
        self.global_step += 1


def predict_ct(u_t, u_tf, policy_node, key="u_tf"):
    # policy_node is a Neuromancer Node
    out = policy_node({"u_t": u_t, key: u_tf})
    return out["f_t"]


def loss_test_fn_physical(
    policy_node,
    x_cord,
    test_inits,
    test_finals,
    rollout_steps=400,
    policy_key="u_tf",  # or "outputs"
):
    """Roll out a Crank-Nicolson PDE solver with Gaussian actuator forcing."""
    policy_net = policy_node.callable
    policy_device = next(policy_net.parameters()).device
    policy_dtype = next(policy_net.parameters()).dtype

    u_t0 = test_inits.detach()
    u_tf = test_finals.detach()
    if u_t0.dim() == 1:
        u_t0 = u_t0.unsqueeze(0)
    if u_tf.dim() == 1:
        u_tf = u_tf.unsqueeze(0)

    u_t0_cpu = u_t0.detach().cpu().to(dtype=torch.float64)
    u_tf_cpu = u_tf.detach().cpu().to(dtype=torch.float64)

    x = x_cord.detach().cpu().to(dtype=torch.float64).reshape(-1)
    n_points = int(x.shape[0])
    dx = 1.0 / n_points

    nu = 0.1
    sigma = 0.1
    centers = torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device, dtype=torch.float64)
    fixed_dt = 0.001

    r = nu * fixed_dt / (2 * dx**2)
    main_diag = torch.ones(n_points, device=x.device, dtype=torch.float64) * (1 + 2 * r)
    off_diag = torch.ones(n_points - 1, device=x.device, dtype=torch.float64) * (-r)
    main_diag[0] = 1.0
    main_diag[-1] = 1.0
    off_diag[0] = 0.0
    if off_diag.numel() > 1:
        off_diag[-2] = 0.0

    A = torch.diag(main_diag)
    A = A + torch.diag(off_diag, diagonal=-1) + torch.diag(off_diag, diagonal=1)

    gaussian_basis = torch.exp(-0.5 * ((x[None, :] - centers[:, None]) / sigma) ** 2)

    def solve_step_implicit(u, f):
        rhs = u.clone()
        rhs[:, 1:-1] += r * (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2])
        rhs = rhs + fixed_dt * f
        rhs[:, 0] = 0.0
        rhs[:, -1] = 0.0
        u_next = torch.linalg.solve(A, rhs.T).T
        return u_next

    traj_pred = []
    controls_pred = []
    u_t_cpu = u_t0_cpu
    u_t_policy = u_t0.to(device=policy_device, dtype=policy_dtype)
    u_tf_policy = u_tf.to(device=policy_device, dtype=policy_dtype)

    with torch.no_grad():
        for _ in range(int(rollout_steps)):
            c_t = predict_ct(u_t_policy, u_tf_policy, policy_node, key=policy_key)
            c_t_cpu = c_t.detach().to(device="cpu", dtype=torch.float64)
            f = c_t_cpu @ gaussian_basis
            u_next_cpu = solve_step_implicit(u_t_cpu, f)
            traj_pred.append(u_next_cpu.clone())
            controls_pred.append(c_t_cpu.clone())
            u_t_cpu = u_next_cpu
            u_t_policy = u_next_cpu.to(device=policy_device, dtype=policy_dtype)

    traj_pred = torch.stack(traj_pred, dim=0)
    controls_pred = torch.stack(controls_pred, dim=0)
    u_pred_final = traj_pred[-1]
    loss_pred = torch.mean((u_pred_final - u_tf_cpu) ** 2)

    traj_zero = []
    u_t_cpu = u_t0_cpu
    with torch.no_grad():
        for _ in range(int(rollout_steps)):
            f = torch.zeros_like(u_t_cpu)
            u_next_cpu = solve_step_implicit(u_t_cpu, f)
            traj_zero.append(u_next_cpu.clone())
            u_t_cpu = u_next_cpu

    traj_zero = torch.stack(traj_zero, dim=0)
    return (
        traj_pred.to(dtype=torch.float32),
        traj_zero.to(dtype=torch.float32),
        controls_pred.to(dtype=torch.float32),
        u_t0_cpu.to(dtype=torch.float32),
        u_tf_cpu.to(dtype=torch.float32),
        loss_pred.to(dtype=torch.float32),
    )


def plot_rollout_comparison(
    x_cord, traj_pred, traj_zero, u_t0, u_tf, control_pred=None, step=20, save_path=None
):
    """Plot rollout comparison panels for zero vs predicted control trajectories."""
    x = np.array(x_cord).squeeze()
    traj_pred = np.array(traj_pred)
    traj_zero = np.array(traj_zero)
    u_t0 = np.array(u_t0)
    u_tf = np.array(u_tf)

    t_pred, _ = traj_pred.shape
    t_zero = traj_zero.shape[0]

    plot_indices_pred = list(range(0, t_pred, int(step)))
    if (t_pred - 1) not in plot_indices_pred:
        plot_indices_pred.append(t_pred - 1)

    plot_indices_zero = list(range(0, t_zero, int(step)))
    if (t_zero - 1) not in plot_indices_zero:
        plot_indices_zero.append(t_zero - 1)

    cmap = plt.get_cmap("RdBu_r")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    ax1, ax2, ax3 = axes

    label_fs = 20
    title_fs = 22
    tick_fs = 18
    legend_fs = 18

    for t in plot_indices_zero:
        color = cmap(t / (t_zero - 1))
        lw = 3 if t in [0, t_zero - 1] else 1.4
        alpha = 1.0 if t in [0, t_zero - 1] else 0.7
        ax1.plot(x, traj_zero[t], color=color, lw=lw, alpha=alpha)

    ax1.plot(x, u_t0, "k--", lw=2.5, label="Initial $u_{t0}$")
    ax1.plot(x, u_tf, "k:", lw=2.5, label="True $u_{tf}$")
    ax1.set_title("Evolution (Control = 0)", fontsize=title_fs)
    ax1.set_xlabel("x", fontsize=label_fs)
    ax1.set_ylabel("u(x)", fontsize=label_fs)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=legend_fs, loc="best")
    ax1.tick_params(axis="both", labelsize=tick_fs)

    for t in plot_indices_pred:
        color = cmap(t / (t_pred - 1))
        lw = 3 if t in [0, t_pred - 1] else 1.4
        alpha = 1.0 if t in [0, t_pred - 1] else 0.7
        ax2.plot(x, traj_pred[t], color=color, lw=lw, alpha=alpha)

    ax2.plot(x, u_t0, "k--", lw=2.5, label="Initial $u_{t0}$")
    ax2.plot(x, u_tf, "k:", lw=2.5, label="True $u_{tf}$")
    ax2.set_title("Evolution (Predicted Control)", fontsize=title_fs)
    ax2.set_xlabel("x", fontsize=label_fs)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=legend_fs, loc="best")
    ax2.tick_params(axis="both", labelsize=tick_fs)

    if control_pred is not None:
        control_pred = np.array(control_pred)
        t_ctrl, num_ctrl = control_pred.shape
        time = np.arange(t_ctrl)
        colors = plt.cm.tab10(np.linspace(0, 1, num_ctrl))
        for c in range(num_ctrl):
            ax3.plot(
                time,
                control_pred[:, c],
                lw=2.5,
                color=colors[c],
                label=f"f {c + 1}",
            )

        ax3.set_title("Predicted Control Signals", fontsize=title_fs, pad=15)
        ax3.set_xlabel("Time step", fontsize=label_fs)
        ax3.set_ylabel("Control amplitude", fontsize=label_fs, labelpad=10)
        ax3.grid(alpha=0.3)
        ax3.tick_params(axis="both", labelsize=tick_fs)

        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax3.legend(
            fontsize=legend_fs, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1
        )
    else:
        ax3.axis("off")
        ax3.text(
            0.5,
            0.5,
            "No control data provided",
            ha="center",
            va="center",
            fontsize=label_fs,
            color="gray",
        )

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    if save_path is not None:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot: {save_path}")
    plt.show()


def main():
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    # Set default dtype to float32
    torch.set_default_dtype(torch.float)
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Data loading for heat equation
    # -------------------------------------------------------------------------
    data_path = "pathto/datasets/heat_smooth_f_dataset.npz"
    data_path = "/home/pk222/projects/PDEControl_DPC/datasets/heat_smooth_f_dataset.npz"

    dataset = np.load(data_path)  # total 3000 samples available
    n_data_load = 100  # samples or number of trajectories to load, original has 500

    solutions = torch.from_numpy(
        dataset["solutions"][:n_data_load]
    )  # shape: (samples, T+1, N)
    controls = torch.from_numpy(dataset["controls"][:n_data_load])  # shape: (samples, T, 4)
    x_cord = torch.from_numpy(dataset["x"]).reshape(-1, 1)  # shape: (N, 1)
    dt = dataset["dt"]  # likely a scalar numpy value

    print(f"Loaded {n_data_load}/{dataset['solutions'].shape[0]} samples:")
    print("solutions:", solutions.shape)  # (samples, T+1, N)
    print("controls:", controls.shape)  # (samples, T, 4)
    print("x:", x_cord.shape)
    print("dt:", dt)

    # -------------------------------------------------------------------------
    # Data generation for control policy training
    # -------------------------------------------------------------------------
    train_frac = 0.8  # train/test split
    nsteps = 100  # rollout steps for DPC, original paper uses 400

    # GRF hyperparameters
    length_scale = 0.2
    variance = 1.0
    num_modes = 64

    # ---- prepare trunk grid ----
    trunk_grid = torch.as_tensor(x_cord, dtype=torch.float32).reshape(-1, 1)

    # Number of samples for generating GRF pairs, original paper uses 500, we use 100 for faster experimentation
    num_samples = 100

    # ---- generate GRF inits/finals (no split) ----
    inits, finals = generate_grf_pairs(
        num_samples=num_samples,
        x_cord=trunk_grid,  # same as trunk grid
        length_scale=0.2,
        variance=1.0,
        num_modes=64,
        seed=seed,
    )

    # ---- split indices ----
    num_train = int(num_samples * train_frac)
    train_idx = torch.arange(0, num_train, device="cpu")
    test_idx = torch.arange(num_train, num_samples, device="cpu")

    # ---- build train/test datasets ----
    train_ds_policy = build_dictdataset_policy(
        inits[train_idx],
        finals[train_idx],
        x_cord=trunk_grid,
        nsteps=nsteps,
        name="train",
    )

    test_ds_policy = build_dictdataset_policy(
        inits[test_idx],
        finals[test_idx],
        x_cord=trunk_grid,
        nsteps=nsteps,
        name="test",
    )

    # check dimensions
    print("Dimensions check train:")
    print("u_t (step):", train_ds_policy.datadict["u_t"].shape)
    print("u_tf (final state):", train_ds_policy.datadict["u_tf"].shape)
    print("trunk_inputs:", train_ds_policy.datadict["trunk_inputs"].shape)

    print("Dimensions check test:")
    print("u_t (step):", test_ds_policy.datadict["u_t"].shape)
    print("u_tf (final state):", test_ds_policy.datadict["u_tf"].shape)
    print("trunk_inputs:", test_ds_policy.datadict["trunk_inputs"].shape)

    # -------------------------------------------------------------------------
    # Create torch DataLoaders for Trainer
    # -------------------------------------------------------------------------
    batch_size = 75
    print(f"batch_size: {batch_size}")

    # Do this on device, as it needs to for Training
    g_device = torch.Generator(device=device).manual_seed(seed)

    steps_per_epoch = 100  # vary this to control how many samples are drawn per epoch, since we are sampling with replacement
    num_samples = steps_per_epoch * batch_size

    g_device = torch.Generator(device=device).manual_seed(seed)

    sampler = torch.utils.data.RandomSampler(
        train_ds_policy,
        replacement=True,
        num_samples=num_samples,
        generator=g_device,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds_policy,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=train_ds_policy.collate_fn,
        drop_last=True,
        shuffle=False,  # must be False when sampler is provided
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds_policy,
        batch_size=batch_size,
        collate_fn=test_ds_policy.collate_fn,
        shuffle=False,
    )

    # -------------------------------------------------------------------------
    # Load dynamics model TI-DeepONet
    # -------------------------------------------------------------------------
    os.environ["DDE_BACKEND"] = "pytorch"
    import deepxde as dde

    # ---- 1) rebuild TI-DON architecture ----
    activation = {"branch1": "gelu", "branch2": "gelu", "trunk": "tanh"}
    hidden_dim = 100

    layer_sizes_branch1 = [100, 128, 128, 128, hidden_dim]
    layer_sizes_branch2 = [4, 32, 32, 32, hidden_dim]
    layer_sizes_trunk = [1, 64, 64, 64, hidden_dim]

    tidon = dde.nn.MIONetCartesianProd(
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

    # ---- load saved weights from examples/neural_operators/Part_8_TIDON_HEAT.ipynb----
    path_to_model = (
        "/home/pk222/projects/PDEControl_DPC_PyTorch/HE_TT/result_32/model_best.pt"
    )

    ckpt = torch.load(path_to_model, map_location=device)
    tidon.load_state_dict(ckpt["model_state_dict"], strict=True)

    # ---- freeze AFTER load ----
    tidon.to(device)
    tidon.eval()

    for p in tidon.parameters():
        p.requires_grad_(False)

    # ---- wrap for Neuromancer ----
    deeponet_wrapped = DeepXDEWrapper(
        model=tidon,
        is_cartesian=True,
        branch_keys=["u_t", "f_t"],
    )

    deeponet_integrator = DeepXDEIntegratorWrapper(
        model=deeponet_wrapped,
        trunk_inputs=trunk_grid,  # (100,1) on correct device/dtype
    )

    # ---- build RK4 integrator + Node ----
    dt = float(dt)
    fxRK4 = integrators.RK4(deeponet_integrator, h=dt)

    node_rk4 = Node(
        fxRK4,
        ["u_t", "f_t"],
        ["u_t"],
        name="TI_DON + RK4",
    )

    # -------------------------------------------------------------------------
    # Initialize policy
    # -------------------------------------------------------------------------
    nx = 100  # state dimension u_t
    nu = 4  # control dimension f_t

    # policyMLP = MLPControl(state_dim=nx, control_dim=nu).to(device=device)
    policyMLP = PolicyMLPBounds(nx, nu).to(device=device)
    policy = Node(policyMLP, ["u_t", "u_tf"], ["f_t"], name="policy")

    # closed-loop system model
    system_dpc = System([policy, node_rk4], nsteps=nsteps)
    system_dpc.show()

    # -------------------------------------------------------------------------
    # Dimension checks
    # -------------------------------------------------------------------------
    ## dimnension checks for DeepONet inputs and outputs
    batch = next(iter(train_loader))
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k, type(v), v)

    tensor_batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "shape")}
    system_dpc = system_dpc.to(device)
    out = system_dpc(tensor_batch)
    print("System out keys:", out.keys())
    print("f_t out shape:", out["f_t"].shape)  # has u_t and u_t+1
    print("u_t out shape:", out["u_t"].shape)  # has u_t and u_t+1
    print("u_tf out shape:", out["u_tf"].shape)  # has u_t and u_t+1

    # -------------------------------------------------------------------------
    # Loss and problem
    # -------------------------------------------------------------------------
    # Terminal loss: MSE between predicted final state and true final state
    y = variable("u_t")[:, -1, :]  # (B, T+1, nx)
    r = variable("u_tf")[:, -1, :]  # (B, T+1, nx)

    track = (y - r) ** 2
    mse_obj = Objective(track, metric=torch.mean, name="mse_loss")

    loss = PenaltyLoss(objectives=[mse_obj], constraints=[])

    problem = Problem([system_dpc], loss)
    # plot computational graph
    problem.show()

    # -------------------------------------------------------------------------
    # Training setup
    # -------------------------------------------------------------------------
    # result_dir = './'
    epochs = int(10)  # 10 passes of 100 steps each
    log_every = 10
    lr = 1e-3
    transition_steps = 2000
    decay_rate = 0.9

    optimizer = torch.optim.Adam(problem.parameters(), lr=lr)

    def exponential_decay(step: int) -> float:
        """Exponential learning-rate decay applied per optimizer step."""
        return decay_rate ** (step / transition_steps)

    scheduler = LambdaLR(optimizer, lr_lambda=exponential_decay)

    lr_callback = StepLRSchedulerCallback(scheduler, log_every=log_every)  # set as needed

    trainer = Trainer(
        problem.to(device),
        train_data=train_loader,
        # dev_data=dev_loader,        # optional
        test_data=test_loader,
        optimizer=optimizer,
        callback=lr_callback,
        epochs=epochs,
        epoch_verbose=1,
        train_metric="train_loss",
        dev_metric="train_loss",  # or "dev_loss" if you use dev_data
        eval_metric="train_loss",
        test_metric="test_loss",
        warmup=epochs,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Sanity checks before training
    # -------------------------------------------------------------------------
    print(
        "policy params require grad:", any(p.requires_grad for p in policyMLP.parameters())
    )
    print("model params require grad:", any(p.requires_grad for p in tidon.parameters()))

    batch = next(iter(train_loader))

    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    # batch.pop("f_t", None)  # remove dataset controls
    print(batch.keys())

    out = problem(batch)
    print(out.keys())  # e.g. "train_loss" if name == "train"
    print("loss requires grad:", out["train_loss"].requires_grad)

    # -------------------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------------------
    best_model = trainer.train()

    # load best trained model
    trainer.dev_data = trainer.train_data  # workaround for replacing dev set to match keys
    # best_outputs = trainer.test(best_model) # optional test evaluation
    problem.load_state_dict(best_model)

    train_loss_history = [l.detach().cpu().numpy() for l in trainer.loss_history["train"]]
    # mean_test_loss = best_outputs["mean_test_loss"].detach().cpu().numpy()
    # print(f"mean_test_loss: {mean_test_loss}")
    print(f"len(train_loss_history): {len(train_loss_history)}")

    # -------------------------------------------------------------------------
    # Plot loss history
    # -------------------------------------------------------------------------
    epoch_steps = (np.arange(len(train_loss_history)) + 1) * steps_per_epoch

    plt.semilogy(
        lr_callback.step_indices, lr_callback.step_losses, label="Train loss (logged)"
    )
    plt.semilogy(epoch_steps, train_loss_history, "o-", label="Train loss (epoch)")

    # plt.scatter(
    #     epoch_steps[-1],
    #     mean_test_loss,
    #     label="Mean test loss",
    #     c="red",
    #     marker="x",
    # )

    plt.xlabel("# Steps")
    plt.legend()
    save_fig("part_7_training_loss.png", tight_layout=False)

    # -------------------------------------------------------------------------
    # Test-time predictions
    # -------------------------------------------------------------------------
    num_eval_samples = 5  # set to 2 for quick visualization; increase as needed
    rollout_steps = nsteps  # or horizon

    # test_idx is a 1D index tensor or list
    test_inits = inits[test_idx]
    test_finals = finals[test_idx]

    system_dpc = system_dpc.to(device)
    system_dpc.nsteps = nsteps

    u_t0 = test_inits[:, None, :].to(device)  # (B, 1, nx)
    u_tf_seq = test_finals[:, None, :].repeat(1, nsteps + 1, 1).to(device)  # (B, T+1, nx)

    test_data = {
        "u_t": u_t0,
        "u_tf": u_tf_seq,
    }

    trajectories = system_dpc(test_data)
    print(trajectories.keys())
    print(trajectories["u_t"].shape)  # (B, nsteps+1, nx)

    with torch.no_grad():
        num_eval = min(num_eval_samples, test_inits.shape[0])
        eval_indices = torch.randperm(test_inits.shape[0], device="cpu")[:num_eval].tolist()

        for i, idx in enumerate(eval_indices):
            print(f"\n=== Plotting sample {idx} ===")

            u0 = test_inits[idx : idx + 1].to(device)  # (1, nx)
            u_tf = test_finals[idx : idx + 1].to(device)  # (1, nx)

            # system inputs
            u_t0 = u0[:, None, :]  # (1, 1, nx)
            u_tf_seq = u_tf[:, None, :].repeat(1, rollout_steps + 1, 1)

            test_data = {"u_t": u_t0, "u_tf": u_tf_seq}

            # policy rollout (system_dpc)
            trajectories = system_dpc(test_data)
            traj_pred = trajectories["u_t"]  # (1, T+1, nx)
            controls_pred = trajectories.get("f_t")  # (1, T, nu)

            # # zero-control rollout (baseline)
            u_t = u0
            traj_zero = []
            for _ in range(int(rollout_steps)):
                c_t = torch.zeros_like(controls_pred[:, 0, :])
                u_t = node_rk4.callable(u_t, c_t)
                traj_zero.append(u_t)
            traj_zero = torch.stack(traj_zero, dim=0)  # (T, 1, nx)
            traj_zero = torch.cat([u0[None, ...], traj_zero], dim=0)  # (T+1, 1, nx)

            # terminal loss (extended)
            loss_extended = torch.mean((traj_pred[:, -1, :] - u_tf) ** 2)

            # physical PDE rollout
            (
                traj_pred_phys,
                traj_zero_phys,
                controls_pred_phys,
                u_t0_phys,
                u_tf_phys,
                loss_physical,
            ) = loss_test_fn_physical(policy, x_cord, u0, u_tf, rollout_steps=rollout_steps)

            # boundary checks (physical)
            if not torch.isclose(u_t0_phys[0, 0], torch.tensor(0.0), atol=1e-6):
                print("Warning: u_t0 boundary at x=0 is not zero (physical).")
            if not torch.isclose(u_t0_phys[0, -1], torch.tensor(0.0), atol=1e-6):
                print("Warning: u_t0 boundary at x=1 is not zero (physical).")
            if not torch.isclose(traj_pred_phys[-1, 0, 0], torch.tensor(0.0), atol=1e-6):
                print("Warning: physical rollout boundary at x=0 is not zero.")
            if not torch.isclose(traj_pred_phys[-1, 0, -1], torch.tensor(0.0), atol=1e-6):
                print("Warning: physical rollout boundary at x=1 is not zero.")

            print(
                "Shapes:",
                tuple(traj_pred.shape),
                tuple(traj_zero.shape),
                tuple(controls_pred.shape),
                tuple(u_t0[0].shape),
                tuple(u_tf.shape),
                "Loss (extended):",
                float(loss_extended),
                "Loss (physical):",
                float(loss_physical),
            )

            # plots
            plot_rollout_comparison(
                x_cord=x_cord.cpu().numpy().squeeze(),
                traj_pred=traj_pred[0].cpu().numpy(),
                traj_zero=traj_zero[:, 0, :].cpu().numpy(),
                u_t0=u0[0].cpu().numpy(),
                u_tf=u_tf[0].cpu().numpy(),
                control_pred=controls_pred[0].cpu().numpy(),
                step=max(1, nsteps // 10),
                save_path=PLOTS_DIR / f"part_7_rollout_comparison_sample_{idx}.png",
            )


if __name__ == "__main__":
    main()
