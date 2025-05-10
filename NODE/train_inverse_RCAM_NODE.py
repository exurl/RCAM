"""
RCAM System Identification using a Neural Network for Inverse Dynamics.

Predict control input u(t) given state x(t) and state derivative dx/dt(t).
Learns the mapping u = g(x, dx/dt) using a neural network.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Ensure the preprocess_data module can be found
# Please adjust this path if your preprocess_data.py is located elsewhere
try:
    sys.path.append(
        "/home/exurl/Projects/ENGR 520/project"
    )  # User's original path
    import preprocess_data
except ImportError:
    print(
        "Warning: preprocess_data module not found at the specified sys.path."
    )
    print(
        "Attempting to import preprocess_data from the current directory or PYTHONPATH."
    )
    try:
        import preprocess_data
    except ImportError as e:
        print(
            f"Fatal: Could not import preprocess_data. Please ensure it's in the Python path. Error: {e}"
        )
        sys.exit(1)


# Constants
data_path = "RCAM_data.npy"
BATCH_SEQ_LEN = 32  # Number of time steps in each training sequence segment
BATCH_SIZE = 32  # Number of trajectory segments per batch
NITERS = 5000  # Number of training iterations (increased for better training)
TEST_FREQ = 100  # Frequency of logging and testing
LR = 1e-3  # Learning rate
SEED = 0
STATE_DIM = 17  # Dimension of the augmented state vector x
FORCING_DIM = 5  # Dimension of the control vector u
HIDDEN_DIMS = [64, 128, 128, 64]  # Hidden dimension for the neural network
OUTPUT_DIR = "NODE"  # Directory for saving results

# Device Setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")
    # Consider adjusting thread count based on your CPU
    # torch.set_num_threads(12)
print(f"Using device: {device}")

# Random Seed
torch.manual_seed(SEED)
np.random.seed(SEED)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Neural Network Definition for Inverse Dynamics
class InverseDynamicsNet(nn.Module):
    """
    Predicts control input u given state x and state derivative dx/dt.
    Network: (x, dx/dt) -> u
    """

    def __init__(self, state_dim, forcing_dim, hidden_dims):
        super(InverseDynamicsNet, self).__init__()
        self.state_dim = state_dim
        self.forcing_dim = forcing_dim

        # Input dimension: state_dim (for x) + state_dim (for dx/dt)
        # Output dimension: forcing_dim (for u)
        layers = []
        layers.append(nn.Linear(state_dim * 2, hidden_dims[0]))
        layers.append(nn.ReLU())
        for dim1, dim2 in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(dim1, dim2))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], forcing_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, dxdt):
        """
        Calculates u_pred from x and dx/dt.
        Args:
            x (torch.Tensor): Current states. Shape (num_samples, state_dim).
            dxdt (torch.Tensor): Current state derivatives. Shape (num_samples, state_dim).
        Returns:
            torch.Tensor: Predicted control inputs. Shape (num_samples, forcing_dim).
        """
        # Concatenate [x, dx/dt]
        nn_input = torch.cat([x, dxdt], dim=1)
        u_pred = self.net(nn_input)
        return u_pred


# Data Loading and Batching
def load_data(data_path):
    """Loads and preprocesses the RCAM data for inverse dynamics."""
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    RCAM_data = np.load(data_path, allow_pickle=True).item()
    all_conditions = []

    for trim_c in RCAM_data:
        for force_c in RCAM_data[trim_c]:
            data = RCAM_data[trim_c][force_c]
            t_np = data["time"].astype(np.float64)
            u_raw_np = data["u"]  # Shape (num_points, 5 original)
            x_raw_np = data["x"]  # Shape (num_points, 12 original)

            # Preprocess u to get scaled u
            u_scaled_np = preprocess_data.preprocess(u_raw_np).astype(
                np.float64
            )

            # Preprocess x: augment_dxdt=True will create augmented x (17-dim),
            # calculate its derivative (17-dim), concatenate them (34-dim), and scale.
            x_processed_full_np = preprocess_data.preprocess(
                x_raw_np, augment_dxdt=True
            ).astype(
                np.float64
            )  # Shape (num_points, 34)

            x_scaled_np = x_processed_full_np[
                :, :STATE_DIM
            ]  # Shape (num_points, 17)
            dxdt_scaled_np = x_processed_full_np[
                :, STATE_DIM : STATE_DIM * 2
            ]  # Shape (num_points, 17)

            assert (
                t_np.shape[0]
                == u_scaled_np.shape[0]
                == x_scaled_np.shape[0]
                == dxdt_scaled_np.shape[0]
            ), f"Mismatching time steps in {trim_c}/{force_c} after processing"
            assert (
                t_np.shape[0] > 1
            ), f"Trajectory too short (<= 1 time steps) in {trim_c}/{force_c}"
            assert not (
                np.isnan(t_np).any()
                or np.isnan(u_scaled_np).any()
                or np.isnan(x_scaled_np).any()
                or np.isnan(dxdt_scaled_np).any()
                or np.isinf(t_np).any()
                or np.isinf(u_scaled_np).any()
                or np.isinf(x_scaled_np).any()
                or np.isinf(dxdt_scaled_np).any()
            ), f"NaN/Inf in processed data for {trim_c}/{force_c}"
            assert np.all(
                np.diff(t_np) > 0
            ), f"Non-monotonic time in {trim_c}/{force_c}"

            all_conditions.append(
                {
                    "t": torch.from_numpy(t_np).float(),
                    "u_scaled": torch.from_numpy(u_scaled_np).float(),
                    "x_scaled": torch.from_numpy(x_scaled_np).float(),
                    "dxdt_scaled": torch.from_numpy(dxdt_scaled_np).float(),
                    "id": f"{trim_c}/{force_c}",
                }
            )
    if not all_conditions:
        raise ValueError("No valid conditions found in the data file.")
    print(f"Loaded {len(all_conditions)} valid conditions.")
    return all_conditions


def get_batch_inverse(all_data, batch_size, segment_len, device):
    """
    Generates a batch of random trajectory segments for the inverse model.
    Returns padded tensors for x_scaled, dxdt_scaled, u_scaled, and the actual max sequence length in the batch.
    """
    num_trajectories = len(all_data)
    batch_x_segments = []
    batch_dxdt_segments = []
    batch_u_segments = []

    current_max_len = 0

    for _ in range(batch_size):
        traj_idx = np.random.randint(num_trajectories)
        trajectory = all_data[traj_idx]
        total_len = len(trajectory["t"])

        if (
            total_len <= 1
        ):  # Need at least one data point. If segment_len is 1, this is fine.
            # Fallback: if a trajectory is too short, try another one or pad significantly
            # For simplicity, we'll pick segments that are at least 1 point long.
            # A segment of length 1 means we take one (x, dxdt, u) tuple.
            start_idx = 0
            current_segment_len = 1
            if total_len == 0:
                continue  # Should not happen with previous checks
        else:
            # Determine actual segment length for this draw
            current_segment_len = min(segment_len, total_len)

            # Select a random starting time index
            if total_len <= current_segment_len:
                start_idx = 0
            else:
                start_idx = np.random.randint(
                    total_len - current_segment_len + 1
                )

        end_idx = start_idx + current_segment_len

        # Extract the segment
        x_s = trajectory["x_scaled"][start_idx:end_idx]
        dxdt_s = trajectory["dxdt_scaled"][start_idx:end_idx]
        u_s = trajectory["u_scaled"][start_idx:end_idx]

        if (
            len(x_s) == 0
        ):  # Should not happen if total_len > 0 and current_segment_len > 0
            continue

        batch_x_segments.append(x_s)
        batch_dxdt_segments.append(dxdt_s)
        batch_u_segments.append(u_s)
        if len(x_s) > current_max_len:
            current_max_len = len(x_s)

    if not batch_x_segments:  # If no valid segments were found
        # This might happen if all trajectories are too short or segment_len is 0.
        # Fallback: return empty tensors or raise an error.
        # For now, let's try to make sure current_max_len is at least 1 if possible.
        if current_max_len == 0 and segment_len > 0:
            current_max_len = segment_len  # Default to desired segment_len
        elif current_max_len == 0 and segment_len == 0:
            raise ValueError(
                "Segment length is 0 and no data could be batched."
            )

    # Pad sequences in the batch to current_max_len
    padded_x = torch.zeros(
        batch_size, current_max_len, STATE_DIM, device=device
    )
    padded_dxdt = torch.zeros(
        batch_size, current_max_len, STATE_DIM, device=device
    )
    padded_u_true = torch.zeros(
        batch_size, current_max_len, FORCING_DIM, device=device
    )

    for i in range(
        len(batch_x_segments)
    ):  # Use len(batch_x_segments) in case some were skipped
        seq_actual_len = len(batch_x_segments[i])
        padded_x[i, :seq_actual_len, :] = batch_x_segments[i].to(device)
        padded_dxdt[i, :seq_actual_len, :] = batch_dxdt_segments[i].to(device)
        padded_u_true[i, :seq_actual_len, :] = batch_u_segments[i].to(device)

    # If batch_x_segments is empty, batch_size might be greater than len(batch_x_segments)
    # The padded tensors will be zeros. Training loop should handle this (e.g. if current_max_len is 0).

    return padded_x, padded_dxdt, padded_u_true, current_max_len


# Training Loop
if __name__ == "__main__":
    # Load Data
    all_data = load_data(data_path)

    # Build Model
    model = InverseDynamicsNet(STATE_DIM, FORCING_DIM, HIDDEN_DIMS).to(device)

    print("\nInverse Dynamics Model Architecture:")
    print(model.net)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Loss Function
    loss_func = nn.MSELoss()

    print(f"\nStarting Training for {NITERS} iterations...")
    start_time = time.time()
    loss_hist = []

    for itr in range(1, NITERS + 1):
        model.train()
        optimizer.zero_grad()

        # Get Batch
        # x_batch_padded: (BATCH_SIZE, current_max_len, STATE_DIM)
        # dxdt_batch_padded: (BATCH_SIZE, current_max_len, STATE_DIM)
        # u_true_batch_padded: (BATCH_SIZE, current_max_len, FORCING_DIM)
        # current_max_len: int, max sequence length in this batch
        (
            x_batch_padded,
            dxdt_batch_padded,
            u_true_batch_padded,
            current_max_len,
        ) = get_batch_inverse(all_data, BATCH_SIZE, BATCH_SEQ_LEN, device)

        if current_max_len == 0:  # Skip if batch is empty
            print(
                f"Warning: Iter {itr}, get_batch_inverse returned empty batch. Skipping."
            )
            continue

        # Reshape for NN: (BATCH_SIZE * current_max_len, feature_dim)
        # This flattens all time steps from all segments into one large batch.
        num_samples_in_batch = BATCH_SIZE * current_max_len

        input_x = x_batch_padded.reshape(num_samples_in_batch, STATE_DIM)
        input_dxdt = dxdt_batch_padded.reshape(num_samples_in_batch, STATE_DIM)
        target_u = u_true_batch_padded.reshape(
            num_samples_in_batch, FORCING_DIM
        )

        # Forward Pass: Predict u_scaled
        u_pred = model(input_x, input_dxdt)

        # Calculate Loss
        # Compares predicted u with true u for all time steps in the batch
        loss = loss_func(u_pred, target_u)

        # Backward Pass & Optimization
        loss.backward()
        optimizer.step()

        # Logging
        loss_hist.append(loss.item())
        if itr % TEST_FREQ == 0 or itr == 1 or itr == NITERS:
            elapsed_time = time.time() - start_time
            if (
                itr > 1
            ):  # Avoid division by zero for ETA on first iteration if TEST_FREQ is 1
                t_remain = elapsed_time * (
                    (NITERS - itr) / (itr - 1)
                )  # More accurate ETA after some iters
            else:
                t_remain = elapsed_time * (
                    NITERS - 1
                )  # Rough estimate for first iteration

            print(
                f"Iter {itr:6d}/{NITERS} | Loss: {loss.item():.6f} | "
                f"Elapsed: {elapsed_time/60:.1f}m | "
                f"ETA: {t_remain / 3600:02.0f}:{t_remain % 3600 / 60:02.0f}:{t_remain % 60:02.0f}"
            )

    # End Training
    end_time = time.time()
    total_training_time = end_time - start_time
    print(
        f"\nTraining finished in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)."
    )
    if loss_hist:
        print(f"Final Loss: {loss_hist[-1]:.6f}")
    else:
        print("No training iterations completed.")

    # Save Final Model
    model_save_path = os.path.join(OUTPUT_DIR, "final_rcam_inverse_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\nFinal model saved to {model_save_path}")

    # Plot convergence
    if loss_hist:
        plt.figure(figsize=(10, 6))
        plt.semilogy(np.arange(len(loss_hist)), loss_hist)
        plt.xlabel("Iteration")
        plt.ylabel("Loss (MSE)")
        plt.title("Inverse Model Training Convergence")
        plt.grid(True, which="both", ls="-")
        convergence_plot_path = os.path.join(
            OUTPUT_DIR, "inverse_convergence.png"
        )
        plt.savefig(convergence_plot_path, dpi=256)
        print(f"Convergence plot saved to {convergence_plot_path}")
        # plt.show() # Optionally display plot

    # Example: Using the trained model for prediction on a sample segment
    print("\nExample Prediction on a sample segment:")
    model.eval()
    with torch.no_grad():
        # Get a sample trajectory segment (use BATCH_SIZE=1 for simplicity)
        x_test_padded, dxdt_test_padded, u_true_test_padded, test_seq_len = (
            get_batch_inverse(
                all_data,
                1,
                BATCH_SEQ_LEN,
                device,  # Get one segment of up to BATCH_SEQ_LEN
            )
        )

        if test_seq_len > 0:
            # Reshape for model input
            input_x_test = x_test_padded.reshape(-1, STATE_DIM)
            input_dxdt_test = dxdt_test_padded.reshape(-1, STATE_DIM)

            # Predict u_scaled using the trained model
            u_pred_test_scaled = model(
                input_x_test, input_dxdt_test
            )  # Shape (test_seq_len, FORCING_DIM)
            u_true_test_scaled_flat = u_true_test_padded.reshape(
                -1, FORCING_DIM
            )  # Shape (test_seq_len, FORCING_DIM)

            # Compare first few steps (example)
            print(f"\nTest segment length: {test_seq_len}")
            print("\nSample States (x_scaled) [First 5 steps, if available]:")
            print(x_test_padded[0, : min(5, test_seq_len), :].cpu().numpy())
            print(
                "\nSample State Derivatives (dxdt_scaled) [First 5 steps, if available]:"
            )
            print(dxdt_test_padded[0, : min(5, test_seq_len), :].cpu().numpy())

            print("\nTrue Controls (u_scaled) [First 5 steps, if available]:")
            print(
                u_true_test_scaled_flat[: min(5, test_seq_len), :]
                .cpu()
                .numpy()
            )
            print(
                "\nPredicted Controls (u_scaled) [First 5 steps, if available]:"
            )
            print(u_pred_test_scaled[: min(5, test_seq_len), :].cpu().numpy())

            # Calculate MSE on this test sample
            # Note: preprocess_data.postprocess can be used to unscale u if needed for interpretable error.
            # Here, we calculate MSE on the scaled values, consistent with training.
            test_loss = loss_func(
                u_pred_test_scaled, u_true_test_scaled_flat
            ).item()
            print(f"\nTest Sample Scaled MSE: {test_loss:.6f}")

            # Example of unscaling for one predicted control vector (optional)
            if FORCING_DIM == u_pred_test_scaled.shape[1] and test_seq_len > 0:
                sample_pred_u_scaled = (
                    u_pred_test_scaled[0:1, :].cpu().numpy()
                )  # Take first predicted step
                sample_pred_u_unscaled = preprocess_data.postprocess(
                    sample_pred_u_scaled
                )
                print(
                    f"\nExample Unscaled Predicted Control (first step): {sample_pred_u_unscaled}"
                )

                sample_true_u_scaled = (
                    u_true_test_scaled_flat[0:1, :].cpu().numpy()
                )
                sample_true_u_unscaled = preprocess_data.postprocess(
                    sample_true_u_scaled
                )
                print(
                    f"Example Unscaled True Control (first step):     {sample_true_u_unscaled}"
                )

        else:
            print(
                "Could not retrieve a valid test segment for example prediction."
            )
