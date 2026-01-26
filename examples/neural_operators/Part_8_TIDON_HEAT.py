"""
Standalone Python script equivalent to Part_8_TIDON_HEAT.ipynb.

Mirrors the notebook workflow end-to-end, including plotting and training.
"""

import os
from pathlib import Path

os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from neuromancer.callbacks import Callback
from neuromancer.constraint import Objective
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.dynamics import integrators
from neuromancer.loss import PenaltyLoss
from neuromancer.modules.operators import DeepXDEIntegratorWrapper
from neuromancer.modules.operators import DeepXDEWrapper
from neuromancer.problem import Problem
from neuromancer.system import Node
from neuromancer.trainer import Trainer


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def save_fig(name: str, tight_layout: bool = True):
    """Save the current matplotlib figure to the plots directory and close it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(PLOTS_DIR / name)
    plt.close()


def build_dictdataset(b1, b2, y, name, trunk_grid):
    trunk_inputs = trunk_grid.expand(b1.shape[0], -1, -1)  # (N, 100, 1)
    return DictDataset(
        {
            "u_t": b1.float(),  # (N, 100)
            "f_t": b2.float(),  # (N, 4)
            "trunk_inputs": trunk_inputs.float(),  # (N, 100, 1)
            "outputs": y.float(),  # (N, 100)
        },
        name=name,
    )


def exponential_decay(step: int, decay_rate: float, transition_steps: int) -> float:
    """Exponential learning-rate decay applied per optimizer step."""
    return decay_rate ** (step / transition_steps)


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


def autoregressive_rollout(sample_id_value, problem, solutions, controls, device):
    node = problem.nodes[0].to(device)
    node.eval()

    with torch.no_grad():
        u_t = solutions[sample_id_value, 0, :].float().to(device)
        true_states = solutions[sample_id_value, :, :].float().to(device)
        control_seq = controls[sample_id_value, :, :].float().to(device)

        preds = [u_t]
        for t in range(control_seq.shape[0]):
            batch = {
                "u_t": u_t.unsqueeze(0),
                "f_t": control_seq[t : t + 1, :],
            }
            out = node(batch)
            u_t = out["u_t"].squeeze(0)
            preds.append(u_t)

        rollout_pred = torch.stack(preds, dim=0)
        mse_loss = torch.mean((true_states - rollout_pred) ** 2)
        rel_l2_error = torch.linalg.norm(
            true_states - rollout_pred
        ) / torch.linalg.norm(true_states)

    return mse_loss, rel_l2_error, rollout_pred, true_states


def plot_rollout_results(
    true_states,
    rollout_pred,
    x_cord,
    dt_value,
    sample_id_value,
):
    x_axis = x_cord.squeeze()
    T_plus_1 = true_states.shape[0]
    time_axis = np.linspace(0, dt_value * (T_plus_1 - 1), T_plus_1)
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

    save_fig(f"rollout_sample_{sample_id_value}.png", tight_layout=False)


def rollout_and_plot(sample_id_value, problem, solutions, controls, x_cord, dt, device):
    node = problem.nodes[0].to(device)
    node.eval()

    with torch.no_grad():
        u_t = solutions[sample_id_value, 0, :].float().to(device)
        true_states = solutions[sample_id_value, :, :].float().to(device)
        control_seq = controls[sample_id_value, :, :].float().to(device)

        preds = [u_t]
        for t in range(control_seq.shape[0]):
            batch = {
                "u_t": u_t.unsqueeze(0),
                "f_t": control_seq[t : t + 1, :],
            }
            out = node(batch)
            u_t = out["u_t"].squeeze(0)
            preds.append(u_t)

        rollout_pred = torch.stack(preds, dim=0)

    plot_rollout_results(
        true_states.detach().cpu().numpy(),
        rollout_pred.detach().cpu().numpy(),
        x_cord.detach().cpu().numpy(),
        dt,
        sample_id_value,
    )


def main():
    # Set default dtype to float32
    torch.set_default_dtype(torch.float)
    # PyTorch random seed
    seed = 1234
    torch.manual_seed(seed)
    # NumPy random seed
    np.random.seed(seed)

    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------
    data_path = "/home/pk222/projects/PDEControl_DPC/datasets/heat_smooth_f_dataset.npz"

    dataset = np.load(data_path)  # total 3000 samples available
    samples_to_load = 100

    solutions = torch.from_numpy(
        dataset["solutions"][:samples_to_load]
    )  # shape: (samples, T+1, N)
    controls = torch.from_numpy(
        dataset["controls"][:samples_to_load]
    )  # shape: (samples, T, 2)
    x_cord = torch.from_numpy(dataset["x"]).reshape(-1, 1)  # shape: (N, 1)
    dt = dataset["dt"]  # likely a scalar numpy value

    print(f"Loaded {samples_to_load}/{dataset['solutions'].shape[0]} samples:")
    print("solutions:", solutions.shape)
    print("controls:", controls.shape)
    print("x:", x_cord.shape)
    print("dt:", dt)

    # -------------------------------------------------------------------------
    # Prepare training pairs
    # -------------------------------------------------------------------------
    # Slice u_t and u_{t+1}
    u_t = solutions[:, :-1, :]  # (samples, T, N)
    u_next = solutions[:, 1:, :]  # (samples, T, N)
    f_t = controls[:, :, :]  # (samples, T, 2)

    print("u_t: ", str(u_t.shape))
    print("u_next: ", str(u_next.shape))
    print("f_t: ", str(f_t.shape))

    # -------------------------------------------------------------------------
    # Flatten dataset
    # -------------------------------------------------------------------------
    # flatten (samples × time)
    branch_x1 = u_t.reshape(-1, u_t.shape[-1])  # (N, 100)
    branch_x2 = f_t.reshape(-1, f_t.shape[-1])  # (N, 4)
    outputs = u_next.reshape(-1, u_next.shape[-1])  # (N, 100)

    print("branch_x1: ", str(branch_x1.shape))
    print("branch_x2: ", str(branch_x2.shape))
    print("outputs: ", str(outputs.shape))

    # -------------------------------------------------------------------------
    # Shuffle and split
    # -------------------------------------------------------------------------
    # Shuffle indices (do this on CPU for convenience)
    num_samples = branch_x1.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(num_samples, generator=g, device="cpu")

    print("Shuffled indices: ", str(idx))

    branch_x1 = branch_x1[idx]
    branch_x2 = branch_x2[idx]
    outputs = outputs[idx]

    # split
    train_frac = 0.8  # percentage
    n_train = int(train_frac * num_samples)

    print("Number of training samples: ", n_train)

    # -------------------------------------------------------------------------
    # Neuromancer datasets
    # -------------------------------------------------------------------------
    trunk_grid = torch.as_tensor(x_cord, dtype=torch.float32).reshape(-1, 1)

    train_ds = build_dictdataset(
        branch_x1[:n_train],
        branch_x2[:n_train],
        outputs[:n_train],
        "train",
        trunk_grid,
    )
    test_ds = build_dictdataset(
        branch_x1[n_train:],
        branch_x2[n_train:],
        outputs[n_train:],
        "test",
        trunk_grid,
    )

    # check dimensions
    print("Dimensions check train:")
    print(
        "branch_inputs_1 -> u_t:", train_ds.datadict["u_t"].shape
    )  # (N, 100) spatial field
    print(
        "branch_inputs_2 -> f_t :", train_ds.datadict["f_t"].shape
    )  # (N, 4) actuators
    print(
        "trunk_inputs -> x_coord:", train_ds.datadict["trunk_inputs"].shape
    )  # (N, 100, 1) grid., broadcasted N times
    print(
        "outputs- > u_t+1:", train_ds.datadict["outputs"].shape
    )  # (N, 100) output field

    print("Dimensions check test:")
    print(
        "branch_inputs_1 -> u_t:", test_ds.datadict["u_t"].shape
    )  # (N, 100) spatial field
    print("branch_inputs_2 - f_t :", test_ds.datadict["f_t"].shape)  # (N, 4) actuators
    print(
        "trunk_inputs-> x_coord:", test_ds.datadict["trunk_inputs"].shape
    )  # (N, 100, 1) grid., broadcasted N times
    print(
        "outputs -> u_t+1:", test_ds.datadict["outputs"].shape
    )  # (N, 100) output field

    # -------------------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------------------
    batch_size = 25
    print(f"batch_size: {batch_size}")

    # Do this on device, as it needs to for Training
    g_device = torch.Generator(device=device).manual_seed(seed)

    steps_per_epoch = 1000
    num_samples = steps_per_epoch * batch_size

    sampler = torch.utils.data.RandomSampler(
        train_ds,
        replacement=True,
        num_samples=num_samples,
        generator=g_device,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=train_ds.collate_fn,
        drop_last=True,
        shuffle=False,  # must be False when sampler is provided
    )

    # Only shuffle the training set
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=test_ds.collate_fn,
        shuffle=False,
    )

    # -------------------------------------------------------------------------
    # MIONet configuration
    # -------------------------------------------------------------------------
    activation = {
        "branch1": "gelu",
        "branch2": "gelu",
        "trunk": "tanh",
    }

    hidden_dim = 100

    layer_sizes_branch1 = [100, 128, 128, 128, hidden_dim]  # u_t: (B,100) -> (B,100)
    layer_sizes_branch2 = [4, 32, 32, 32, hidden_dim]  # f_t: (B,4)   -> (B,100)
    layer_sizes_trunk = [1, 64, 64, 64, hidden_dim]  # x:   (100,1) -> (100,100

    net = dde.nn.MIONetCartesianProd(
        layer_sizes_branch1=layer_sizes_branch1,
        layer_sizes_branch2=layer_sizes_branch2,
        layer_sizes_trunk=layer_sizes_trunk,
        activation=activation,
        kernel_initializer="Glorot normal",  # weight initialisation available in DeepXDE
        trunk_last_activation=True,  # matches the JAX trunk applying tanh after every Dense
        merge_operation="mul",  # y_func = y_func1 * y_func2
        layer_sizes_merger=None,  # no extra merger MLP
        layer_sizes_output_merger=None,  # keep dot-product einsum
    )

    # -------------------------------------------------------------------------
    # Wrap with Neuromancer operators
    # -------------------------------------------------------------------------
    deeponet_wrapped = DeepXDEWrapper(
        model=net,
        is_cartesian=True,
        branch_keys=["u_t", "f_t"],
    )
    # print(deeponet_wrapped)

    deeponet_integrator = DeepXDEIntegratorWrapper(
        model=deeponet_wrapped,
        trunk_inputs=trunk_grid,
    )

    # print(deeponet_integrator)

    fxRK4 = integrators.DiffEqIntegrator(deeponet_integrator, h=dt, method="rk4")

    node_rk4 = Node(
        fxRK4,
        ["u_t", "f_t"],
        ["u_t"],
        name="TI_DON + RK4",
    )

    # -------------------------------------------------------------------------
    # Loss and problem
    # -------------------------------------------------------------------------
    var_y_est = variable("u_t")
    var_y_true = variable("outputs")

    # MSE: mean((u_next - outputs)^2)
    mse_var = (var_y_est - var_y_true) ** 2
    mse_obj = Objective(mse_var, metric=torch.mean, name="mse_loss")

    loss = PenaltyLoss(objectives=[mse_obj], constraints=[])

    # Use RK4 node
    problem = Problem(
        nodes=[node_rk4],
        loss=loss,
    )

    # -------------------------------------------------------------------------
    # Training configuration
    # -------------------------------------------------------------------------
    # result_dir = './'
    epochs = int(1)  # 10 passes of 1000 steps each
    log_every = 100
    lr = 1e-3
    transition_steps = 2000
    decay_rate = 0.9

    optimizer = torch.optim.Adam(problem.parameters(), lr=lr)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: exponential_decay(step, decay_rate, transition_steps),
    )

    lr_callback = StepLRSchedulerCallback(
        scheduler, log_every=log_every
    )  # set as needed

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
    # Train and evaluate
    # -------------------------------------------------------------------------
    best_model = trainer.train()

    # load best trained model
    trainer.dev_data = (
        trainer.train_data
    )  # workaround for replacing dev set to match keys
    best_outputs = trainer.test(best_model)
    problem.load_state_dict(best_model)

    train_loss_history = [
        l.detach().cpu().numpy() for l in trainer.loss_history["train"]
    ]
    mean_test_loss = best_outputs["mean_test_loss"].detach().cpu().numpy()
    print(f"mean_test_loss: {mean_test_loss}")
    print(f"len(train_loss_history): {len(train_loss_history)}")

    # -------------------------------------------------------------------------
    # Plot training history
    # -------------------------------------------------------------------------
    epoch_steps = (np.arange(len(train_loss_history)) + 1) * steps_per_epoch

    plt.semilogy(
        lr_callback.step_indices, lr_callback.step_losses, label="Train loss (logged)"
    )
    plt.semilogy(epoch_steps, train_loss_history, "o-", label="Train loss (epoch)")

    plt.scatter(
        epoch_steps[-1],
        mean_test_loss,
        label="Mean test loss",
        c="red",
        marker="x",
    )

    plt.xlabel("# Steps")
    plt.legend()
    save_fig("training_history.png", tight_layout=False)

    # -------------------------------------------------------------------------
    # Autoregressive rollout
    # -------------------------------------------------------------------------
    sample_id = 12  # change this to test other samples
    mse_loss, rel_l2_error, rollout_pred, true_states = autoregressive_rollout(
        sample_id, problem, solutions, controls, device
    )
    print(mse_loss.item(), rel_l2_error.item())

    # assumes plot_rollout_results is already defined
    # run it
    rollout_and_plot(sample_id, problem, solutions, controls, x_cord, dt, device)


if __name__ == "__main__":
    main()
