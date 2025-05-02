import numpy as np
import matplotlib.pyplot as plt
from preprocess_data import *
from common_plotting import plot_predicted_vs_true
import torch
from create_model_LSTM import InverseDynamicsLSTM

# Import training data
RCAM_data = np.load("RCAM_data.npy", allow_pickle=True).item()

# Define the model
state_dim = 16  # 12 state variables + 4 augmented variables
control_dim = 5
hidden_dim = 64
model = InverseDynamicsLSTM(state_dim, hidden_dim, control_dim)

# Load the trained weights
model.load_state_dict(torch.load("model_parameters_lstm.pth"))

# Set model to evaluation mode
model.eval()

# --- Utility function for prediction ---
def predict_u_from_x(x, model):
    """
    Predicts control input trajectory u given a state trajectory x.
    Args:
        x (Numpy array): shape [T, state_dim]
        model (nn.Module): trained LSTM model
    Returns:
        u_pred (Numpy array): shape [T, control_dim]
    """
    x_processed = preprocess(x)  # Preprocess state variables
    x = torch.tensor(x_processed, dtype=torch.float32)
    
    with torch.no_grad():
        x = x.unsqueeze(0)  # add batch dimension -> [1, T, state_dim]
        lstm_out, _ = model.lstm(x)
        u_pred = model.fc(lstm_out)
        u_pred = u_pred.squeeze(0)  # remove batch dimension -> [T, control_dim]
    
    # Convert u_pred to Numpy array and unscale
    u_pred = u_pred.numpy()
    u_pred = unscale_u_data(u_pred)  # Unscale control inputs

    return u_pred

# --- Plotting true vs. predicted u ---
fig_idx = 1
for trim_key_idx, (trim_key, trim_profiles) in enumerate(RCAM_data.items()):
    if trim_key_idx >= 2:
        break  # only first 2 trim cases

    for prof_idx, (prof_key, profile) in enumerate(trim_profiles.items()):
        if prof_idx >= 3:
            break  # only first 3 profiles per trim

        if prof_idx >= 120:
            x = profile["x"]
            u_true = torch.tensor(profile["u"], dtype=torch.float32)
            t = profile["time"]

            # Compute model prediction
            u_pred = predict_u_from_x(x, model)

            # --- Plot ---
            title = f"Comparison of Model Prediction to True Control Data for Trim: {trim_key}, Profile: {prof_key}"
            plot_predicted_vs_true(t, u_pred, u_true, title)
