import sys
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from train_forward_RCAM_NODE import (
    load_data,
    get_batch,
    linear_interpolation_batch,
)
from train_forward_RCAM_NODE import (
    ODEFunc,
    NeuralODE,
    STATE_DIM,
    FORCING_DIM,
    TIME_DIM,
    HIDDEN_DIM,
)

sys.path.append("/home/exurl/Projects/ENGR 520/project")
from preprocess_data import preprocess, postprocess

# Device Configuration (match your training device if possible)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device for loading: {device}")

# --- 1. Define the Model Architecture (MUST MATCH TRAINING) ---
# see imports

# --- 2. Instantiate the Model ---
loaded_ode_func = ODEFunc(STATE_DIM, FORCING_DIM, HIDDEN_DIM).to(device)
loaded_model = NeuralODE(loaded_ode_func).to(device)

# --- 3. Load the State Dictionary ---
checkpoint = torch.load(
    "NODE/final_rcam_forward_node.pth", map_location=device
)

# --- 4. Load the State Dictionary into the Model ---
loaded_model.load_state_dict(checkpoint)
loaded_model.eval()  # Set the model to evaluation mode for inference

print("Trained model loaded successfully!")

# --- Example of making a prediction ---
if __name__ == "__main__":
    all_conditions = load_data("RCAM_data.npy")
    t_eval, x0, t_u, u_batch, x_true = get_batch(all_conditions, 1, 0, device)

    with torch.no_grad():
        predicted_trajectory = loaded_model(x0, t_eval, t_u, u_batch)

    print("\nPredicted Trajectory Shape:", predicted_trajectory.shape)
    print(
        "First few predicted states:\n",
        predicted_trajectory[0, :5, :].cpu().numpy(),
    )
    x_true = postprocess(x_true.squeeze().numpy())
    x = postprocess(predicted_trajectory.squeeze().numpy())
    u = postprocess(u_batch.squeeze().numpy())

    # Plotting

    x_labels = [
        "u",
        "v",
        "w",
        "p",
        "q",
        "r",
        "phi",
        "theta",
        "psi",
        "North",
        "East",
        "Down",
    ]
    u_labels = ["aileron", "horiz. tail", "rudder", "thrust 1", "thrust 2"]

    x_types = (
        ["velocity (m/s)"] * 3
        + ["angular rate (rad/s)"] * 3
        + ["Euler angle (rad)"] * 3
        + ["position (m)"] * 3
    )
    u_types = ["deflection (rad)"] * 5

    fig = plt.figure(figsize=(4, 6), constrained_layout=True)
    axs = fig.subplots(5, 1, sharex=True)

    axs[0].plot(t_eval, u)
    axs[0].legend(u_labels, loc="right", fontsize=8)
    axs[0].set_title("Inputs", fontsize=10)
    axs[0].set_ylabel(u_types[0], fontsize=7)
    axs[0].tick_params(axis="both", which="major", labelsize=6)
    axs[0].tick_params(axis="both", which="minor", labelsize=6)

    axs[1].set_title("Outputs", fontsize=10)
    axs[-1].set_xlabel("Time (s)", fontsize=7)
    for i in range(4):
        idx_1 = i * 3
        idx_2 = (i + 1) * 3

        for j in range(3):
            axs[i + 1].plot(
                t_eval,
                x_true[:, idx_1 + j],
                "-",
                c=tuple(mcolors.TABLEAU_COLORS)[j],
                label=f"truth {x_labels[j]}",
            )
            axs[i + 1].plot(
                t_eval,
                x[:, idx_1 + j],
                ":",
                c=tuple(mcolors.TABLEAU_COLORS)[j],
                label=f"model {x_labels[j]}",
            )
        axs[i + 1].legend(loc="right", fontsize=8)
        axs[i + 1].set_ylabel(x_types[idx_1], fontsize=7)
        axs[i + 1].tick_params(axis="both", which="major", labelsize=6)
        axs[i + 1].tick_params(axis="both", which="minor", labelsize=6)

    fig.suptitle(
        "RCAM Neural ODE Response Comparison",
        fontweight="bold",
        fontsize=10,
    )
    plt.show()
