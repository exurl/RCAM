import sys
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # For color consistency if desired

# --- Imports from your Inverse Model Training Script ---
# Assuming your inverse model training script is named 'rcam_inverse_model.py'
# and is in a location Python can find (e.g., same directory or PYTHONPATH)
# Adjust the import path if your script is named or located differently.
try:
    from train_inverse_RCAM_NODE import (
        InverseDynamicsNet,
        STATE_DIM,
        FORCING_DIM,
        HIDDEN_DIMS,
        load_data as load_data_inverse,  # Alias to avoid name collision if testing both
        OUTPUT_DIR as INVERSE_MODEL_OUTPUT_DIR,  # Get the output directory used during training
    )
except ImportError as e:
    print(f"Error importing from rcam_inverse_model.py: {e}")
    print(
        "Please ensure 'rcam_inverse_model.py' is in the Python path and contains the necessary definitions."
    )
    sys.exit(1)

# --- Import for preprocessing ---
# Ensure the preprocess_data module can be found
# Please adjust this path if your preprocess_data.py is located elsewhere
try:
    # Using the user's original path structure
    sys.path.append("/home/exurl/Projects/ENGR 520/project")
    from preprocess_data import postprocess
except ImportError:
    print(
        "Warning: preprocess_data module not found at the specified sys.path."
    )
    print(
        "Attempting to import preprocess_data from current directory or PYTHONPATH."
    )
    try:
        from preprocess_data import postprocess
    except ImportError as e_local:
        print(
            f"Fatal: Could not import preprocess_data. Please ensure it's in the Python path. Error: {e_local}"
        )
        sys.exit(1)


# Device Configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device for loading and inference: {device}")

# --- 1. Define the Model Architecture (Imported as InverseDynamicsNet) ---

# --- 2. Instantiate the Model ---
# Dimensions must match the trained model
loaded_inverse_model = InverseDynamicsNet(
    STATE_DIM, FORCING_DIM, HIDDEN_DIMS
).to(device)

# --- 3. Load the State Dictionary ---
# Path to the trained inverse model
model_filename = "final_rcam_inverse_model.pth"
model_path = os.path.join(INVERSE_MODEL_OUTPUT_DIR, model_filename)

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print(
        f"Please ensure the INVERSE_MODEL_OUTPUT_DIR ('{INVERSE_MODEL_OUTPUT_DIR}') and model filename ('{model_filename}') are correct."
    )
    sys.exit(1)

checkpoint = torch.load(model_path, map_location=device)

# --- 4. Load the State Dictionary into the Model ---
loaded_inverse_model.load_state_dict(checkpoint)
loaded_inverse_model.eval()  # Set the model to evaluation mode

print(f"Trained inverse model loaded successfully from {model_path}!")

# --- Example of making a prediction and plotting ---
if __name__ == "__main__":
    # 1. Load data using the inverse model's data loader
    # This data contains x_scaled, dxdt_scaled, and u_scaled
    all_conditions_inverse = load_data_inverse("RCAM_data.npy")

    # 2. Select a single trajectory for testing
    # Try to pick a reasonably long one for better visualization

    test_trajectory = all_conditions_inverse[-1]
    print(
        f"Using trajectory ID: {test_trajectory['id']} with {len(test_trajectory['t'])} time steps for testing."
    )

    # Extract data for the selected trajectory
    t_full_np = test_trajectory["t"].cpu().numpy()
    x_scaled_traj = test_trajectory["x_scaled"].to(
        device
    )  # Shape: (seq_len, STATE_DIM)
    dxdt_scaled_traj = test_trajectory["dxdt_scaled"].to(
        device
    )  # Shape: (seq_len, STATE_DIM)
    u_true_scaled_traj = test_trajectory["u_scaled"].to(
        device
    )  # Shape: (seq_len, FORCING_DIM)

    # Create a relative time vector for plotting
    t_relative_for_plot = t_full_np - t_full_np[0]
    seq_len = t_relative_for_plot.shape[0]

    # 3. Make predictions using the loaded inverse model
    with torch.no_grad():
        # The model expects inputs of shape (num_samples, feature_dim)
        # For a single trajectory, num_samples is seq_len
        predicted_u_scaled = loaded_inverse_model(
            x_scaled_traj, dxdt_scaled_traj
        )

    # 4. Postprocess data (unscale) for plotting
    # Unscale predicted controls
    u_pred_unscaled = postprocess(predicted_u_scaled.cpu().numpy())
    # Unscale true controls
    u_true_unscaled = postprocess(u_true_scaled_traj.cpu().numpy())
    # Unscale states (x) to show as inputs to the inverse model
    # x_scaled_traj is (seq_len, 17). postprocess will unaugment to (seq_len, 12)
    x_unscaled_inputs = postprocess(x_scaled_traj.cpu().numpy())

    print("\nPredicted (scaled) Controls Shape:", predicted_u_scaled.shape)
    print("First few predicted (unscaled) controls:\n", u_pred_unscaled[:5, :])
    print("\nFirst few true (unscaled) controls:\n", u_true_unscaled[:5, :])

    # 5. Plotting
    # Labels for states (12 after unscaling/unaugmenting state vector x)
    x_plot_labels = [
        "Body u (m/s)",
        "Body v (m/s)",
        "Body w (m/s)",
        "p (rad/s)",
        "q (rad/s)",
        "r (rad/s)",
        "$\phi$ (rad)",
        "$\\theta$ (rad)",
        "$\psi$ (rad)",
        "North Pos (m)",
        "East Pos (m)",
        "Down Pos (m)",
    ]
    # Labels for controls (5)
    u_plot_labels = [
        "Aileron (rad)",
        "Elevator (rad)",
        "Rudder (rad)",
        "Throttle 1",
        "Throttle 2",
    ]

    # Choose some states from x_unscaled_inputs to display as context
    # Indices correspond to x_plot_labels after postprocessing
    states_to_plot_indices = [3, 4, 5, 6, 7, 8]  # e.g., phi, theta, p, q
    selected_state_labels_for_plot = [
        x_plot_labels[i] for i in states_to_plot_indices
    ]

    num_control_plots = FORCING_DIM

    fig, axs = plt.subplots(
        1 + num_control_plots,
        1,
        sharex=True,
        figsize=(12, 2.5 * (1 + num_control_plots)),
        constrained_layout=False,
    )
    fig.suptitle(
        "RCAM Inverse Dynamics Model: Test Trajectory Prediction",
        fontweight="bold",
        fontsize=14,
    )

    # Plot selected input states (x) that were fed (after scaling) into the inverse model
    ax_states = axs[0]
    ax_states.set_title(
        "Example Input States to Inverse Model (Unscaled)", fontsize=10
    )
    for i, state_idx in enumerate(states_to_plot_indices):
        if state_idx < x_unscaled_inputs.shape[1]:
            ax_states.plot(
                t_relative_for_plot,
                x_unscaled_inputs[:, state_idx],
                label=selected_state_labels_for_plot[i],
            )
    ax_states.legend(loc="upper right", fontsize=8)
    ax_states.set_ylabel("State Values", fontsize=9)
    ax_states.grid(True, linestyle=":", alpha=0.7)
    ax_states.tick_params(axis="y", labelsize=8)

    # Plot true vs. predicted controls
    for i in range(num_control_plots):
        ax_ctrl = axs[i + 1]
        ax_ctrl.plot(
            t_relative_for_plot,
            u_true_unscaled[:, i],
            "-",
            label=f"True {u_plot_labels[i]}",
            lw=1.5,
        )
        ax_ctrl.plot(
            t_relative_for_plot,
            u_pred_unscaled[:, i],
            ":",
            label=f"Predicted {u_plot_labels[i]}",
            lw=2,
        )
        ax_ctrl.legend(loc="upper right", fontsize=8)
        ax_ctrl.set_ylabel(u_plot_labels[i], fontsize=9)
        ax_ctrl.grid(True, linestyle=":", alpha=0.7)
        ax_ctrl.tick_params(axis="y", labelsize=8)

    axs[-1].set_xlabel("Time (s)", fontsize=10)
    axs[-1].tick_params(axis="x", labelsize=8)

    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )  # Adjust layout to make space for suptitle and xlabel

    plot_save_path = os.path.join(
        INVERSE_MODEL_OUTPUT_DIR, "inverse_model_test_prediction.png"
    )
    try:
        plt.savefig(plot_save_path, dpi=300)
        print(f"\nPlot saved to {plot_save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.show()
