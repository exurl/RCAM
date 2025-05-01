import numpy as np
import matplotlib.pyplot as plt
import torch
from create_model_LSTM import InverseDynamicsLSTM

# Import training data
RCAM_data = np.load("RCAM_data.npy", allow_pickle=True).item()

# Define the model
state_dim = 12
control_dim = 5
hidden_dim = 64
model = InverseDynamicsLSTM(state_dim, hidden_dim, control_dim)

# Load the trained weights
model.load_state_dict(torch.load("model_parameters_lstm.pth"))

# Set model to evaluation mode
model.eval()

# --- Utility function for prediction ---
def predict_u_from_x(x_seq, model):
    """
    Predicts control input trajectory u given a single unpadded state trajectory x.
    Args:
        x_seq (Tensor): shape [T, state_dim]
        model (nn.Module): trained LSTM model
    Returns:
        u_pred (Tensor): shape [T, control_dim]
    """
    with torch.no_grad():
        x_seq = x_seq.unsqueeze(0)  # add batch dimension -> [1, T, state_dim]
        lstm_out, _ = model.lstm(x_seq)
        u_pred = model.fc(lstm_out)
        u_pred = u_pred.squeeze(0)  # remove batch dimension -> [T, control_dim]
    return u_pred

# --- Plotting true vs. predicted u ---
fig_idx = 1
for trim_key_idx, (trim_key, trim_profiles) in enumerate(RCAM_data.items()):
    if trim_key_idx >= 2:
        break  # only first 2 trim cases

    for prof_idx, (prof_key, profile) in enumerate(trim_profiles.items()):
        if prof_idx >= 3:
            break  # only first 3 profiles per trim

        x = torch.tensor(profile["x"], dtype=torch.float32)
        u_true = torch.tensor(profile["u"], dtype=torch.float32)
        t = torch.tensor(profile["time"], dtype=torch.float32)

        u_pred = predict_u_from_x(x, model)

        # --- Plot ---
        fig, axes = plt.subplots(control_dim, 1, figsize=(8, 2 * control_dim), sharex=True)
        fig.suptitle(f"{trim_key} - {prof_key}")

        for i in range(control_dim):
            axes[i].plot(t, u_true[:, i], label="True", linewidth=2)
            axes[i].plot(t, u_pred[:, i], label="Predicted", linestyle="--")
            axes[i].set_ylabel(f"u[{i}]")
            axes[i].grid(True)

        axes[-1].set_xlabel("Time (s)")
        axes[0].legend()
        plt.tight_layout()
        plt.show()
