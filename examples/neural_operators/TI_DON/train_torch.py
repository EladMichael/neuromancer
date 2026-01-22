# 1) Imports and configuration
import os
import time
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

data_path = "/home/pk222/projects/PDEControl_DPC/datasets/heat_smooth_f_dataset.npz"
samples_to_load = 100
seed = 32

# 2) Reproducibility and device
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required but not available.")
device = torch.device("cuda")
print(f"Using device: {device}")

# 3) Data loading
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found: {data_path}")

dataset = np.load(data_path)
solutions = torch.from_numpy(dataset["solutions"][:samples_to_load])
controls = torch.from_numpy(dataset["controls"][:samples_to_load])
x_cord = torch.from_numpy(dataset["x"]).reshape(-1, 1)
dt = dataset["dt"]

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

# 4) Train/test dataset and dataloaders
X = torch.cat([u_t, f_t], dim=-1)
Y = u_next
X = X.reshape(-1, X.shape[-1])
Y = Y.reshape(-1, Y.shape[-1])

print(f"X (input): {tuple(X.shape)}")
print(f"Y (output): {tuple(Y.shape)}")

train_frac = 0.8
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

batch_size = 75


class RandomBatchDataset(IterableDataset):
    def __init__(self, u_t_f, x_cord_full, u_next_batch, batch_size, seed):
        self.u_t_f = u_t_f
        self.x_cord = x_cord_full
        self.u_next = u_next_batch
        self.batch_size = batch_size
        self.num_samples = u_t_f.shape[0]
        self.seed = seed

    def __iter__(self):
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


train_dataset = RandomBatchDataset(X_train, x_cord, Y_train, batch_size, seed)
test_dataset = RandomBatchDataset(X_test, x_cord, Y_test, batch_size, seed + 1)
train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

example_batch = next(iter(train_loader))
(u_t_f_batch, x_cord_full), u_next_batch = example_batch
print(f"u_t_f_batch: {tuple(u_t_f_batch.shape)}")
print(f"x_cord: {tuple(x_cord_full.shape)}")
print(f"u_next_batch: {tuple(u_next_batch.shape)}")

# 5) Model setup
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

net = net.to(device)


# 6) Loss functions
def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def model_rhs(u_state, c_t, x_loc):
    x_func1 = u_state
    x_func2 = c_t
    pred_rhs = net((x_func1, x_func2, x_loc))
    return pred_rhs


def rk4_step(u_state, c_t, x_loc, dt_value):
    alpha1 = 1 / 6
    alpha2 = 2 / 6
    alpha3 = 2 / 6
    alpha4 = 1 / 6

    k1 = model_rhs(u_state, c_t, x_loc)
    k2 = model_rhs(u_state + 0.5 * dt_value * k1, c_t, x_loc)
    k3 = model_rhs(u_state + 0.5 * dt_value * k2, c_t, x_loc)
    k4 = model_rhs(u_state + dt_value * k3, c_t, x_loc)

    u_next_pred = u_state + dt_value * (
        alpha1 * k1 + alpha2 * k2 + alpha3 * k3 + alpha4 * k4
    )
    return u_next_pred, k1


def loss_fn(inputs, outputs, dt_value, debug_shapes=False):
    u_t_f_batch, x_cord_full = inputs
    u_state = u_t_f_batch[:, :100]
    c_t = u_t_f_batch[:, 100:]
    x_loc = x_cord_full

    u_next_pred, k1 = rk4_step(u_state, c_t, x_loc, dt_value)
    loss = mse(outputs, u_next_pred)

    if debug_shapes:
        print(f"u_state: {tuple(u_state.shape)}")
        print(f"c_t: {tuple(c_t.shape)}")
        print(f"x_loc: {tuple(x_loc.shape)}")
        print(f"k1: {tuple(k1.shape)}")
        print(f"u_next_pred: {tuple(u_next_pred.shape)}")
        print(f"u_next_batch: {tuple(outputs.shape)}")

    return loss


# 7) Optimizer and scheduler
batch_size = 75
num_steps = 500
log_iter = 10
lr = 1e-3
transition_steps = 2000
decay_rate = 0.9

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
lr_lambda = lambda step: decay_rate ** (step / transition_steps)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# 8) Training loop
result_dir = os.path.join(os.getcwd(), f"result_{seed}")
os.makedirs(result_dir, exist_ok=True)
log_file = os.path.join(result_dir, "log.csv")
with open(log_file, "w") as f:
    f.write("epoch,loss,l2err_test,loss_test,runtime\n")

start_time = time.time()
train_iter = iter(train_loader)
test_iter = iter(test_loader)
dt_value = float(dt)
debug_shapes = True

for step in range(num_steps):
    net.train()
    inputs, outputs = next(train_iter)
    u_t_f_batch, x_cord_full = inputs
    u_t_f_batch = u_t_f_batch.float().to(device)
    x_cord_full = x_cord_full.float().to(device)
    u_next_batch = outputs.float().to(device)

    loss = loss_fn((u_t_f_batch, x_cord_full), u_next_batch, dt_value, debug_shapes)
    debug_shapes = False

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % log_iter == 0:
        net.eval()
        with torch.no_grad():
            test_inputs, test_outputs = next(test_iter)
            test_u_t_f_batch, test_x_cord_full = test_inputs
            test_u_t_f_batch = test_u_t_f_batch.float().to(device)
            test_x_cord_full = test_x_cord_full.float().to(device)
            test_u_next_batch = test_outputs.float().to(device)
            loss_test = loss_fn(
                (test_u_t_f_batch, test_x_cord_full), test_u_next_batch, dt_value
            )
            l2err_test = torch.sqrt(loss_test)

        runtime = time.time() - start_time
        loss_val = loss.item()
        l2_val = l2err_test.item()
        loss_test_val = loss_test.item()
        print(f"Iteration {step + 1}/{num_steps}")
        print(
            f"Train_loss: {loss_val:.2e},"
            f"l2_err_test: {l2_val:.2e}, test_loss: {loss_test_val:.2e} , runtime: {runtime:06.2f}"
        )
        with open(log_file, "a") as f:
            f.write(f"{step},{loss_val},{l2_val},{loss_test_val},{runtime}\n")

# 9) Plot training progress
df = pd.read_csv(log_file)
fig, (ax1) = plt.subplots(1, 1, figsize=(18, 8), dpi=100)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
ax1.plot(df["epoch"], df["loss"], label="Training Loss", color=colors[0], linestyle="-")
ax1.plot(
    df["epoch"], df["loss_test"], label="Testing Loss", color=colors[1], linestyle="-"
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
plt.show()
plt.close()
print(f"Plots saved to {output_file}")

# 10) Autoregressive rollout
def autoregressive_rollout(net, solutions, controls, x_cord, dt_value, sample_id, device):
    net.eval()
    with torch.no_grad():
        u0 = solutions[sample_id, 0, :].float().to(device)
        true_states = solutions[sample_id, :, :].float().to(device)
        control_seq = controls[sample_id, :, :].float().to(device)
        x_loc = x_cord.float().to(device)
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
            k2 = net((u_t + 0.5 * dt_value * k1, c_t, x_loc))
            k3 = net((u_t + 0.5 * dt_value * k2, c_t, x_loc))
            k4 = net((u_t + dt_value * k3, c_t, x_loc))
            u_next = u_t + dt_value * (a1 * k1 + a2 * k2 + a3 * k3 + a4 * k4)
            rollout_pred.append(u_t.squeeze(0))
            u_t = u_next

        rollout_pred.append(u_t.squeeze(0))
        rollout_pred = torch.stack(rollout_pred, dim=0)

        mse_loss = torch.mean((true_states - rollout_pred) ** 2)
        rel_l2_error = torch.linalg.norm(true_states - rollout_pred) / torch.linalg.norm(
            true_states
        )

    return mse_loss, rel_l2_error, rollout_pred, true_states


def plot_rollout_results(
    true_states, rollout_pred, x_cord, dt_value, sample_id, result_dir
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

    im2 = ax2.contourf(
        x_axis, time_axis, error, levels=120, cmap="magma"
    )
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
    plt.show()
    plt.close(fig)
    print(f"Rollout plot saved to {output_path}")


sample_id = 5
mse_loss, rel_l2_error, rollout_pred, true_states = autoregressive_rollout(
    net, solutions, controls, x_cord, dt_value, sample_id, device
)

print(f"Test MSE Loss: {mse_loss.item()}")
print(f"Relative L2 Error: {rel_l2_error.item()}")

plot_rollout_results(
    true_states.detach().cpu().numpy(),
    rollout_pred.detach().cpu().numpy(),
    x_cord.detach().cpu().numpy(),
    dt_value,
    sample_id,
    result_dir,
)
