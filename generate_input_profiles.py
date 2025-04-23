import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== USER-SPECIFIED VALUES ====
# Path to the Excel file containing control input profiles
excel_file = "Control Input Profiles.xlsx"

# Time step for generating profiles
delta_t = 0.1  

# Trim values for each control variable (default values)
trim_values = { 
    "d_A": 0.0,  # Aileron
    "d_T": 0.0,  # Tailplane
    "d_R": 0.0,  # Rudder
    "d_th1": np.deg2rad(4),  # Throttle 1
    "d_th2": np.deg2rad(4)   # Throttle 2
}

# Saturation limits for each control variable (in radians)
saturation_limits = {
    "d_A": np.radians([-25, 25]),  # Aileron
    "d_T": np.radians([-25, 10]),  # Tailplane
    "d_R": np.radians([-30, 30]),  # Rudder
    "d_th1": np.radians([0.5, 10]),  # Throttle 1
    "d_th2": np.radians([0.5, 10])   # Throttle 2
}

# ==== LOAD EXCEL ====
# Load the Excel file into a pandas DataFrame
df = pd.read_excel(excel_file)

# ==== HELPER FUNCTIONS ====
# Parse duration strings (e.g., "10s") into float values (seconds)
def parse_duration(duration_str):
    return float(duration_str.strip().replace("s", ""))

# Parse start time strings (e.g., "t=5s") into float values (seconds)
def parse_start_time(start_str):
    if isinstance(start_str, str) and "t=" in start_str:
        return float(start_str.strip().replace("t=", "").replace("s", ""))
    return 0.0

# Extract amplitude percentage from description strings (e.g., "50%")
def extract_amplitude(description):
    if isinstance(description, str) and "%" in description:
        match = pd.Series(description).str.extract(r"(\d+)%")[0]
        if not match.empty:
            return int(match[0]) / 100.0
    return 0.0

# ==== MAIN PROCESSING ====
# Dictionary to store generated control profiles
control_profiles = {}

# Iterate through each row in the DataFrame
for _, row in df.iterrows():
    profile_id = row["Profile ID"]  # Unique identifier for the profile
    duration = parse_duration(row["Duration"])  # Duration of the profile
    start_time = parse_start_time(row["Feature Start Time"])  # Start time of the profile
    n = int(duration / delta_t) + 1  # Number of time steps
    time_vector = np.linspace(0, duration, n)  # Time vector for the profile

    # Initialize profile array with time and control variables
    profile_array = np.zeros((n, 6))  # Columns: [t, d_A, d_T, d_R, d_th1, d_th2]
    profile_array[:, 0] = time_vector  # Set time column

    # Determine which control variable this profile applies to
    for var in ["d_A", "d_T", "d_R", "d_th1", "d_th2"]:
        for col in df.columns:
            if isinstance(col, str) and var in col:
                amp = extract_amplitude(row[col])  # Extract amplitude percentage
                if amp > 0:
                    # Calculate amplitude based on saturation limits
                    min_val, max_val = saturation_limits[var]
                    full_scale = max(abs(min_val), abs(max_val))
                    amplitude = amp * full_scale

                    # Determine the start index for the profile
                    idx_start = int(start_time / delta_t)

                    # Generate the profile based on the description
                    if "Step" in str(row["Description"]):
                        # Step input: Apply constant amplitude
                        if "negative" in str(row["Description"]):
                            profile_array[idx_start:, ["d_A", "d_T", "d_R", "d_th1", "d_th2"].index(var) + 1] = -amplitude
                        else:
                            profile_array[idx_start:, ["d_A", "d_T", "d_R", "d_th1", "d_th2"].index(var) + 1] = amplitude
                    elif "Ramp" in str(row["Description"]):
                        # Ramp input: Linearly increase amplitude
                        ramp_duration = duration - start_time
                        ramp_time = np.linspace(0, ramp_duration, n - idx_start)
                        ramp_profile = amplitude * (ramp_time / ramp_duration)
                        if "negative" in str(row["Description"]):
                            profile_array[:, ["d_A", "d_T", "d_R", "d_th1", "d_th2"].index(var) + 1] = -ramp_profile
                        else:
                            profile_array[:, ["d_A", "d_T", "d_R", "d_th1", "d_th2"].index(var) + 1] = ramp_profile
                    elif "Sine" in str(row["Description"]):
                        # TODO: Implement sine wave input
                        pass
                    elif "Doub" in str(row["Description"]):
                        # TODO: Implement doublet input
                        pass
                    elif "Rand" in str(row["Description"]):
                        # TODO: Implement random input
                        pass

    # Add trim values and clip to saturation limits
    for i, var in enumerate(["d_A", "d_T", "d_R", "d_th1", "d_th2"]):
        profile_array[:, i+1] += trim_values[var]  # Add trim value
        min_val, max_val = saturation_limits[var]
        profile_array[:, i+1] = np.clip(profile_array[:, i+1], min_val, max_val)  # Clip to limits

    # Store the generated profile in the dictionary
    control_profiles[profile_id] = profile_array

# ==== USAGE EXAMPLE ====
# Example: Access a specific profile
# profile = control_profiles["Rol_Step+_Amp4"]

# Example: Print all dictionary keys in the control_profiles dictionary
# print("Control Profiles Dictionary Keys:")
# for key in control_profiles.keys():
#     print(key)

# Example: Print type and shape of a profile
# print(f"type(profile): {type(profile)}")
# print(f"shape(profile): {profile.shape}")

# ==== PLOT CONTROL INPUT PROFILES ====

# Function to plot a control input profile
def plot_profile(profile_id, profile):
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    # Set overall figure title
    fig.suptitle(f"Control Input Profile: {profile_id}", fontsize=16)

    # Define input names and titles for the plots
    input_names = ["d_A", "d_T", "d_R", "d_th1", "d_th2"]
    titles = ["d_A (aileron)", "d_T (tailplane)", "d_R (rudder)", "d_th1 (throttle 1)", "d_th2 (throttle 2)"]

    # Plot each control variable
    for i, ax in enumerate(axes):
        input_col = i + 1  # Column index for the control variable
        var = input_names[i]  # Variable name

        # Plot the profile and saturation limits
        ax.plot(profile[:, 0], profile[:, input_col], label=titles[i])
        ax.axhline(saturation_limits[var][0], color='k', linestyle='--')  # Lower limit
        ax.axhline(saturation_limits[var][1], color='k', linestyle='--')  # Upper limit

        ax.legend(loc="upper right")
        ax.grid(True)

    # Common x and y labels
    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.text(0.05, 0.5, "Deflection from trim (rad)", va='center', rotation='vertical', fontsize=12)

    # Adjust layout and show the plot
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust for space around the y-label and suptitle
    plt.xlim([0, profile[-1, 0]])
    plt.show()

    return

# Customize how many profiles you want to plot
profile_keys = list(control_profiles.keys())[34:35]  # Plot specific profiles

# Plot each selected profile
for profile_id in profile_keys:
    profile = control_profiles[profile_id]
    plot_profile(profile_id, profile)
