"""
RCAM System Identification using Neural ODE.

Predict state history x(t) given initial state x(0) and forcing history u(t).
Learns the dynamics dx/dt = f(t, x, u) using a neural network.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.append("/home/exurl/Projects/ENGR 520/project")
import preprocess_data

# Constants
method = "dopri5"
data_path = "RCAM_data.npy"
batch_time = 32
batch_size = 32
niters = 5000
test_freq = 100
lr = 1e-3
adjoint = False
seed = 0
STATE_DIM = 17
FORCING_DIM = 5
HIDDEN_DIM = 128
TIME_DIM = 1  # Time is scalar input

# Device Setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")
    torch.set_num_threads(12)
print(f"Using device: {device}")

# Random Seed
torch.manual_seed(seed)
np.random.seed(seed)

if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


# Batched Linear Interpolation for Forcing


def linear_interpolation_batch(t, xp, fp):
    """
    Performs batched 1D linear interpolation.
    Finds the value of `f` corresponding to time `t` using known points `(xp, fp)`.

    Args:
        t (torch.Tensor): Scalar time point for interpolation (float or 0-dim tensor).
        xp (torch.Tensor): 1D tensor of time points for known data (shared across batch).
                           Shape (num_points,). MUST be sorted.
        fp (torch.Tensor): Batched tensor of data values at xp.
                           Shape (batch_size, num_points, data_dim).

    Returns:
        torch.Tensor: Interpolated values. Shape (batch_size, data_dim).
    """
    t = t.to(xp.device)  # Ensure t is on the same device as xp, fp
    batch_size, _, data_dim = fp.shape

    # Find indices such that xp[idx] <= t < xp[idx+1]
    # searchsorted finds the index where t could be inserted maintaining order.
    # side='right' means we get the index *after* existing entries equal to t.
    # Subtracting 1 gives the index of the interval start (or -1 if t < xp[0]).
    idx = torch.searchsorted(xp, t, right=True) - 1

    # Clamp indices: 0 <= idx <= num_points - 2
    # This handles t < xp[0] (idx becomes 0) and t >= xp[-1] (idx becomes num_points-2)
    idx = torch.clamp(idx, 0, xp.shape[0] - 2)

    # Get interval points xp[idx] and xp[idx+1]
    x0 = xp[idx]
    x1 = xp[idx + 1]

    # Get corresponding function values fp[:, idx, :] and fp[:, idx+1, :]
    f0 = fp[:, idx, :]  # Shape: (batch_size, data_dim)
    f1 = fp[:, idx + 1, :]  # Shape: (batch_size, data_dim)

    # Calculate interpolation weight, handling potential division by zero if xp points are identical
    denom = x1 - x0
    # Avoid division by zero or NaN gradients if interval is zero length
    denom = torch.where(denom <= 0, torch.full_like(denom, 1e-6), denom)
    weight = (t - x0) / denom

    # Perform interpolation: f = f0 + weight * (f1 - f0)
    # Unsqueeze weight to allow broadcasting: (scalar) -> (1, 1)
    interpolated_val = f0 + weight.unsqueeze(-1).unsqueeze(-1) * (f1 - f0)

    # Handle cases exactly at the boundaries which clamping idx might miss
    # If t is exactly xp[0], ensure result is f0
    interpolated_val = torch.where(
        t == xp[0].unsqueeze(0), fp[:, 0, :], interpolated_val
    )
    # If t is exactly xp[-1], ensure result is f[-1] (use idx = num_points-2 and weight=1)
    interpolated_val = torch.where(
        t == xp[-1].unsqueeze(0), fp[:, -1, :], interpolated_val
    )

    return interpolated_val  # Shape (batch_size, data_dim)


# Neural Network Definitions


class ODEFunc(nn.Module):
    """
    Defines the parameterized dynamics function dx/dt = f(t, x, u(t)).
    It needs access to the forcing trajectory u(t) for the current batch
    to interpolate u at the required time t.
    """

    def __init__(self, state_dim, forcing_dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.state_dim = state_dim
        self.forcing_dim = forcing_dim

        # Network to approximate the dynamics: takes x, u -> outputs dx/dt
        # Input dim: 1 (time) + state_dim + forcing_dim
        # Output dim: state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + forcing_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Placeholders for the current batch's forcing trajectory data
        self._t_u = None  # Time points for the forcing data
        self._u_batch = None  # Forcing data corresponding to _t_u

    def set_u_trajectory(self, t_u, u_batch):
        """Stores the forcing trajectory for the current batch."""
        self._t_u = t_u.to(device)  # Shape: (num_u_points,)
        self._u_batch = u_batch.to(
            device
        )  # Shape: (batch_size, num_u_points, forcing_dim)

    def forward(self, t, x):
        """
        Calculates dx/dt at time t for state x.
        Args:
            t (torch.Tensor): Current time (scalar tensor).
            x (torch.Tensor): Current state. Shape (batch_size, state_dim).
        """
        if self._t_u is None or self._u_batch is None:
            raise RuntimeError(
                "set_u_trajectory must be called before forward."
            )

        # 1. Interpolate forcing u at the current time t for the entire batch
        u_interp = linear_interpolation_batch(t, self._t_u, self._u_batch)
        # u_interp shape: (batch_size, forcing_dim)

        # 2. Prepare input for the dynamics network
        # Expand t to match batch size: (scalar) -> (batch_size, 1)
        t_vec = t.expand(x.shape[0], 1)

        # Concatenate [x, u]
        nn_input = torch.cat([x, u_interp], dim=1)

        # 3. Calculate dx/dt using the network
        dxdt = self.net(nn_input)
        return dxdt


# Main Neural ODE Model Wrapper


class NeuralODE(nn.Module):
    def __init__(self, ode_func):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func

    def forward(self, x0, t_eval, t_u, u_batch):
        """
        Solves the ODE for the given initial state and forcing.
        Args:
            x0 (torch.Tensor): Initial states. Shape (batch_size, state_dim).
            t_eval (torch.Tensor): Time points to evaluate the solution at. Shape (num_eval_points,).
            t_u (torch.Tensor): Time points corresponding to the forcing data u_batch. Shape (num_u_points,).
            u_batch (torch.Tensor): Forcing trajectory data. Shape (batch_size, num_u_points, forcing_dim).
        Returns:
            torch.Tensor: Predicted state trajectory. Shape (batch_size, num_eval_points, state_dim).
        """
        # Set the forcing trajectory data in the ODE function for this batch
        self.ode_func.set_u_trajectory(t_u, u_batch)

        # Solve the ODE
        # odeint expects func, y0, t
        # output shape: (num_eval_points, batch_size, state_dim)
        x_pred_batch_first = odeint(self.ode_func, x0, t_eval, method=method)

        # Permute to (batch_size, num_eval_points, state_dim)
        x_pred = x_pred_batch_first.permute(1, 0, 2)

        return x_pred


# Data Loading and Batching


def load_data(data_path):
    """Loads and preprocesses the RCAM data."""
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    RCAM_data = np.load(data_path, allow_pickle=True).item()
    all_conditions = []

    for trim_c in RCAM_data:
        for force_c in RCAM_data[trim_c]:
            data = RCAM_data[trim_c][force_c]
            t = data["time"].astype(np.float64)
            u = preprocess_data.preprocess(data["u"]).astype(np.float64)
            x = preprocess_data.preprocess(data["x"]).astype(np.float64)

            assert (
                t.shape[0] == u.shape[0] == x.shape[0]
            ), f"Mismatching shapes in {trim_c}/{force_c}"
            assert t.shape[0] > 1, f"<= 1 time steps in {trim_c}/{force_c}"
            assert not (
                np.isnan(t).any()
                or np.isnan(u).any()
                or np.isnan(x).any()
                or np.isinf(t).any()
                or np.isinf(u).any()
                or np.isinf(x).any()
            ), f"NaN/Inf in {trim_c}/{force_c}"

            # Ensure time is monotonically increasing
            assert np.all(
                np.diff(t) > 0
            ), f"Non-monotonic time in {trim_c}/{force_c}"

            all_conditions.append(
                {
                    "t": torch.from_numpy(t).float(),
                    "u": torch.from_numpy(u).float(),
                    "x": torch.from_numpy(x).float(),
                    "id": f"{trim_c}/{force_c}",  # For potential debugging
                }
            )
    if not all_conditions:
        raise ValueError("No valid conditions found in the data file.")
    print(f"Loaded {len(all_conditions)} valid conditions.")
    return all_conditions


def get_batch(all_data, batch_size, batch_time_len, device):
    """Generates a batch of random trajectory segments."""
    num_trajectories = len(all_data)
    batch_x0 = []
    batch_t_eval = []  # Time points for ODE solver evaluation (relative time)
    batch_t_u = []  # Time points for the forcing data (relative time)
    batch_u_segment = []  # Forcing data for the segment
    batch_x_true = []  # Ground truth state trajectory for the segment

    successful_segments = 0
    while successful_segments < batch_size:
        traj_idx = np.random.randint(num_trajectories)
        trajectory = all_data[traj_idx]
        total_len = len(trajectory["t"])

        # Need at least 2 points for a segment
        if total_len <= 1:
            continue

        # Determine segment length dynamically if batch_time_len is invalid
        if batch_time_len < 1:
            current_segment_len = total_len
        else:
            current_segment_len = min(batch_time_len, total_len)
        if current_segment_len <= 1:  # Still need at least 2 points
            continue

        # Select a random starting time index
        if total_len <= current_segment_len:
            start_idx = 0
        else:
            start_idx = np.random.randint(total_len - current_segment_len + 1)
        end_idx = (
            start_idx + current_segment_len
        )  # Exclusive index for slicing

        # Extract the segment
        t_abs = trajectory["t"][start_idx:end_idx]  # Absolute time
        u = trajectory["u"][start_idx:end_idx]
        x = trajectory["x"][start_idx:end_idx]

        # Ensure segment has at least 2 points after slicing
        if len(t_abs) < 2:
            continue

        # Create relative time vectors starting from 0 for this segment
        t_relative = t_abs - t_abs[0]

        batch_x0.append(x[0])  # Initial state x(0)
        batch_t_eval.append(
            t_relative
        )  # Times to evaluate solution: [0, t1-t0, ..., tN-t0]
        batch_t_u.append(
            t_relative
        )  # Times for u interpolation (same as eval here)
        batch_u_segment.append(u)  # Forcing data u(t)
        batch_x_true.append(x)  # True state trajectory x(t)
        successful_segments += 1

    # Pad sequences in the batch to the maximum length in the batch
    max_len = max(len(t) for t in batch_t_eval)

    # Reference time evaluation points (from the longest sequence)
    ref_t_eval = next(t for t in batch_t_eval if len(t) == max_len).to(device)
    # Reference time points for u (usually same as eval)
    ref_t_u = next(t for t in batch_t_u if len(t) == max_len).to(device)

    # Create padded tensors
    padded_u = torch.zeros(batch_size, max_len, FORCING_DIM, device=device)
    padded_x_true = torch.zeros(batch_size, max_len, STATE_DIM, device=device)

    for i in range(batch_size):
        seq_len = len(batch_u_segment[i])
        padded_u[i, :seq_len, :] = batch_u_segment[i].to(device)
        padded_x_true[i, :seq_len, :] = batch_x_true[i].to(device)
        # Optional: Handle padding for sequences shorter than max_len in x_true if needed by loss
        # Often, loss calculation might need a mask or only consider valid parts.
        # For simple MSE over the whole padded sequence, this is okay if padding is 0.

    # Stack initial conditions
    batch_x0 = torch.stack(batch_x0).to(device)

    # Return: eval times, initial states, u times, u data, true states
    return ref_t_eval, batch_x0, ref_t_u, padded_u, padded_x_true


# Training Loop


if __name__ == "__main__":

    # Load Data
    all_data = load_data(data_path)

    # Build Model
    ode_func = ODEFunc(STATE_DIM, FORCING_DIM, hidden_dim).to(device)
    model = NeuralODE(ode_func).to(device)

    print("\nModel Architecture (ODEFunc Dynamics Net):")
    print(ode_func.net)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss Function
    loss_func = nn.MSELoss()

    print(f"\nStarting Training for {niters} iterations...")
    start_time = time.time()
    loss_hist = []

    for itr in range(1, niters + 1):
        model.train()
        optimizer.zero_grad()

        # Get Batch
        # t_eval: Times to evaluate solution at (rel, shape [max_len])
        # x0: Initial states (shape [batch_size, state_dim])
        # t_u: Times corresponding to u data (rel, shape [max_len])
        # u_batch: Forcing data (shape [batch_size, max_len, forcing_dim])
        # x_true: True state trajectories (shape [batch_size, max_len, state_dim])
        t_eval, x0, t_u, u_batch, x_true = get_batch(
            all_data, batch_size, batch_time, device
        )

        # Forward Pass
        # Predict state trajectory using the Neural ODE model
        x_pred = model(x0, t_eval, t_u, u_batch)

        # Calculate Loss
        # Compare predicted state trajectory with the true state trajectory
        # Note: If using padding, may need a mask here for more accurate loss,
        # but MSE on zero-padded sequences is often acceptable as a start.
        loss = loss_func(x_pred, x_true)

        # Backward Pass & Optimization
        loss.backward()
        optimizer.step()

        # Logging
        loss_hist.append(loss.item())
        if itr % test_freq == 0 or itr == 1:
            elapsed_time = time.time() - start_time
            t_remain = elapsed_time * ((niters - itr) / itr)
            print(
                f"Iter {itr:6d}/{niters} | Loss: {loss.item():.6f} | Estimate: {t_remain / 3600:02.0f}:{t_remain % 3600 / 60:02.0f}:{t_remain % 60:02.0f}"
            )

    # End Training
    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")
    print(f"Final Loss: {loss_hist[-1]:.6f}")

    # Save Final Model
    torch.save(model.state_dict(), "forward_NODE/final_rcam_forward_node.pth")
    print("\nFinal model saved to final_rcam_forward_node.pth")

    # Plot convergence
    plt.semilogy(np.arange(len(loss_hist)), loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("forward_NODE/convergence.png", dpi=256)
    plt.show()

    # Example: Using the trained model for prediction
    print("\nExample Prediction on a sample segment:")
    model.eval()
    with torch.no_grad():
        # Get a sample trajectory segment (use batch_size=1 for simplicity)
        t_eval_test, x0_test, t_u_test, u_test, x_true_test = get_batch(
            all_data, 1, 100, device
        )  # Length 100

        # Predict states using the trained model
        x_pred_test = model(x0_test, t_eval_test, t_u_test, u_test)

        # Compare first few steps (example)
        print("Time steps (relative):", t_eval_test[:5].cpu().numpy())
        print("\nInitial State (x0) [Batch 0]:")
        print(x0_test[0, :].cpu().numpy())
        print("\nForcing (u) sample [Batch 0, First 5 steps]:")
        print(u_test[0, :5, :].cpu().numpy())

        print("\nTrue State (x) sample [Batch 0, First 5 steps]:")
        print(x_true_test[0, :5, :].cpu().numpy())
        print("\nPredicted State (x) sample [Batch 0, First 5 steps]:")
        print(x_pred_test[0, :5, :].cpu().numpy())

        # Calculate MSE on this test sample
        test_loss = loss_func(x_pred_test, x_true_test).item()
        print(f"\nTest Sample MSE: {test_loss:.6f}")
