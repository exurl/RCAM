import time
import numpy as np
from compute_trajectory import simulate_trajectory
from rcam import rcam_system, rcam_plant
from common_plotting import *

# ==== LOAD PREVIOUSLY SAVED CONTROL PROFILES AND TRIM POINTS ====

# Load the saved control profile data and convert to a regular dictionary
control_profiles_raw = np.load("control_profiles_train.npz")  # For training data
# control_profiles_raw = np.load("control_profiles_test.npz")  # For test data
control_profiles = {key: control_profiles_raw[key] for key in control_profiles_raw}

# Load the saved trim points and convert to a regular dictionary
trim_results_raw = np.load("trim_points_train.npz", allow_pickle=True)                # For training data
# trim_results_raw = np.load("trim_points_test_interpolated.npz", allow_pickle=True)   # For test data - interpolated
# trim_results_raw = np.load("trim_points_test_extrapolated.npz", allow_pickle=True)   # For test data - extrapolated
trim_results = {key: trim_results_raw[key].item() if isinstance(trim_results_raw[key], np.ndarray) and trim_results_raw[key].dtype == 'object' else trim_results_raw[key] 
                for key in trim_results_raw}

# ==== RUN RCAM MODEL TO GENERATE DATA ON COMBINATIONS OF CONTROL PROFILES & TRIM POINTS ====

# Saturation limits for each control variable (in radians)
saturation_limits = {
    "d_A": np.radians([-25, 25]),  # Aileron
    "d_T": np.radians([-25, 10]),  # Tailplane
    "d_R": np.radians([-30, 30]),  # Rudder
    "d_th1": np.radians([0.5, 10]),  # Throttle 1
    "d_th2": np.radians([0.5, 10])   # Throttle 2
}

RCAM_data = {}

# Start timer
start_time = time.time()

for trim_case, trim_data in trim_results.items():
    RCAM_data[trim_case] = {}

    print(f"Running simulation for Trim Case: {trim_case}")

    # Extract trim data
    x0 = trim_data['x_trim']
    u0 = trim_data['u_trim']

    for profile_ID, profile_data in control_profiles.items():
        
        # print(f"Running simulation for Trim Case: {trim_case}, Control Profile: {profile_ID}")

        # Extract control profile
        t_vec = profile_data[:, 0]      # time vector
        u_profile = profile_data[:, 1:]     # control profile time history

        # Build control input array and apply saturation limits
        u_unlimited = u_profile + u0    # Add trim control inputs to profile
        mins = np.array([limit[0] for limit in saturation_limits.values()])     # Build array of min. saturation limits
        maxs = np.array([limit[1] for limit in saturation_limits.values()])     # Build array of max. saturation limits
        u = np.clip(u_unlimited, mins, maxs)

        # # Simulate the RCAM aircraft model
        # x0_full = np.concatenate((x0, np.zeros(9)))
        # t_out, x_out = simulate_trajectory(
        #     f=rcam_system, x0=x0_full, t_in=t_vec, u_in=u, t_out=t_vec
        # )

        # Simulate the RCAM plant model
        t_out, x_out = simulate_trajectory(
            f=rcam_plant, x0=x0, t_in=t_vec, u_in=u, t_out=t_vec
        )

        # Store results
        RCAM_data[trim_case][profile_ID] = {
            "time": t_vec,
            "u": u, 
            "x": x_out
        }

# print(f"RCAM_data: {RCAM_data}")

# End timer
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Simulations completed in {elapsed_time:.2f} seconds (total)")

# ==== SAVE RESULTS TO FILE ====

# Save all results to one file
# output_file_path = "RCAM_data.npy"
# output_file_path = "RCAM_data_test_interpolation.npy"
output_file_path = "RCAM_data_test_extrapolation.npy"
np.save(output_file_path, RCAM_data)
print(f"Saved output data as {output_file_path}")

# ==== EXAMPLE FILE LOADING ====

# # Load the results
# RCAM_data = np.load("RCAM_data.npy", allow_pickle=True).item()
# loaded = np.load("simulation_results.npz", allow_pickle=True)

# ==== PLOT DATA ====

# Plot response for each trim case and control profile
for trim_case, profile_data in RCAM_data.items():
    for profile_ID, data in profile_data.items():
        t = data["time"]
        x = data["x"]
        u = data["u"]

        # Plot state variables
        plot_state_variables(t, x, f"State Variables for {trim_case} with {profile_ID}")

        # Plot trajectory
        plot_trajectory(t, x, f"Trajectory for {trim_case} with {profile_ID}")

        # Plot roll axis trajectories
        plot_roll(t, x, u, f"Roll Axis Trajectories for {trim_case} with {profile_ID}")

        # Plot pitch axis & throttle trajectories
        plot_pitch_throttle(t, x, u, f"Pitch Axis & Throttle Trajectories for {trim_case} with {profile_ID}")

        # PLot yaw axis trajectories
        plot_yaw(t, x, u, f"Yaw Axis Trajectories for {trim_case} with {profile_ID}")
