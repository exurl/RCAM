import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ==== USER-SPECIFIED VALUES ====
# Path to the Excel file containing control input profiles
excel_file = "Control Input Profiles.xlsx"

# Time step for generating profiles
delta_t = 0.05  # seconds

# Random number generator seed
np.random.seed(42)

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

# Extract amplitude from description strings
def extract_amplitude(description):
    # If description is not a string, return 0.0
    if not isinstance(description, str):
        return 0.0
    
    # If description contains an amplitude percentage, extract it
    if isinstance(description, str) and "%" in description:
        match = pd.Series(description).str.extract(r"(\d+)%")[0]
        if not match.empty:
            return int(match[0]) / 100.0

    # If description contains ramp rate information, extract it
    elif isinstance(description, str):
        rate_match = re.search(r"Amplitude: full scale/([\d.]+)\s*s", description)
        if rate_match:
            return float(rate_match.group(1))
        
    # If the amplitude is random, return -1.0
    if "random" in description:
        return -1.0

    return 0.0

# Extract frequency from description strings
def extract_frequency(description):
    # If description is not a string, return 0.0
    if not isinstance(description, str):
        return 0.0

    # Match something like "Freq.: 0.1 Hz"
    freq_match = re.search(r"Freq.:\s*[:.]?\s*([\d.]+)\s*Hz", description)
    if freq_match:
        return float(freq_match.group(1))

    # If it's random, return -1.0
    if "random" in description.lower():
        return -1.0

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
                amp = extract_amplitude(row[col])  # Extract amplitude
                if amp != 0:
                    # Calculate amplitude based on saturation limits
                    min_val, max_val = saturation_limits[var]
                    if var == "d_th1" or var == "d_th2":
                        # For throttle, compute full scale as (max-min)
                        full_scale = max_val - min_val
                    else:
                        # For other controls, use max absolute value
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
                        ramp_duration = amp   # Note that "amplitude" is actually the ramp duration here
                        ramp_end_time = start_time + ramp_duration
                        idx_end = int(ramp_end_time / delta_t) + 1
                        ramp_n = int(ramp_duration / delta_t) + 1
                        ramp_time = np.linspace(0, ramp_duration, ramp_n)
                        ramp_profile = full_scale * (ramp_time / ramp_duration)
                        if "negative" in str(row["Description"]):
                            profile_array[idx_start:idx_end, ["d_A", "d_T", "d_R", "d_th1", "d_th2"].index(var) + 1] = -ramp_profile
                            profile_array[idx_end:, ["d_A", "d_T", "d_R", "d_th1", "d_th2"].index(var) + 1] = -full_scale
                        else:
                            profile_array[idx_start:idx_end, ["d_A", "d_T", "d_R", "d_th1", "d_th2"].index(var) + 1] = ramp_profile
                            profile_array[idx_end:, ["d_A", "d_T", "d_R", "d_th1", "d_th2"].index(var) + 1] = full_scale
                    
                    elif "Sinusoidal" in str(row["Description"]):
                        # Sinusoidal input: Generate sine wave profile
                        freq = extract_frequency(row[col])   # Frequency in Hz
                        if freq > 0:
                            sine_duration = duration - start_time
                            sine_n = int(sine_duration / delta_t) + 1
                            sine_time = np.linspace(0, sine_duration, sine_n)
                            sine_profile = amplitude * np.sin(2 * np.pi * freq * sine_time)
                            profile_array[idx_start:, ["d_A", "d_T", "d_R", "d_th1", "d_th2"].index(var) + 1] = sine_profile
                    
                    elif "Doublet" in str(row["Description"]):
                        # Doublet input: Generate doublet profile
                        doublet_dwell_duration = 1.0    # Duration of each dwell in seconds
                        dwell1_end_idx = idx_start + int(doublet_dwell_duration / delta_t) + 1
                        dwell2_end_idx = dwell1_end_idx + int(doublet_dwell_duration / delta_t) + 1
                        idx_end = dwell1_end_idx + int(doublet_dwell_duration / delta_t) + 1

                        if "negative" in str(row["Description"]):
                            profile_array[idx_start:dwell1_end_idx, ["d_A", "d_T", "d_R", "d_th1", "d_th2"].index(var) + 1] = -amplitude
                            profile_array[dwell1_end_idx:idx_end, ["d_A", "d_T", "d_R", "d_th1", "d_th2"].index(var) + 1] = amplitude
                        else:
                            profile_array[idx_start:dwell1_end_idx, ["d_A", "d_T", "d_R", "d_th1", "d_th2"].index(var) + 1] = amplitude
                            profile_array[dwell1_end_idx:idx_end, ["d_A", "d_T", "d_R", "d_th1", "d_th2"].index(var) + 1] = -amplitude
                    
                    elif "Random" in str(row["Description"]):
                        # Random input: Generate random profile by summing three random sine waves
                        # Set throttle profiles equal to one another
                        if var == "d_th2":
                            d_th1_profile = profile_array[:, 4]
                            profile_array[:, 5] = d_th1_profile
                        else:
                            freq1 = np.random.uniform(0.0, 2.0)  # Random frequency 1 in Hz (max: 2 Hz)
                            freq2 = np.random.uniform(0.0, 2.0)  # Random frequency 2 in Hz (max: 2 Hz)
                            freq3 = np.random.uniform(0.0, 2.0)  # Random frequency 3 in Hz (max: 2 Hz)
                            ampl1 = np.random.uniform(0.0, 0.2) * full_scale  # Random amplitude 1 (max: 20% of full scale)
                            ampl2 = np.random.uniform(0.0, 0.2) * full_scale  # Random amplitude 2 (max: 20% of full scale)
                            ampl3 = np.random.uniform(0.0, 0.2) * full_scale  # Random amplitude 3 (max: 20% of full scale)
                            rand_profile = ampl1 * np.sin(2 * np.pi * freq1 * time_vector) + \
                                            ampl2 * np.sin(2 * np.pi * freq2 * time_vector) + \
                                            ampl3 * np.sin(2 * np.pi * freq3 * time_vector)
                            profile_array[:, ["d_A", "d_T", "d_R", "d_th1", "d_th2"].index(var) + 1] = rand_profile

    # # Add trim values and clip to saturation limits
    # for i, var in enumerate(["d_A", "d_T", "d_R", "d_th1", "d_th2"]):
    #     profile_array[:, i+1] += trim_values[var]  # Add trim value
    #     min_val, max_val = saturation_limits[var]
    #     profile_array[:, i+1] = np.clip(profile_array[:, i+1], min_val, max_val)  # Clip to limits

    # Store the generated profile in the dictionary
    control_profiles[profile_id] = profile_array

# ==== USAGE EXAMPLE ====
# Example: Access a specific profile
# profile = control_profiles["Rol_Ramp+_Amp1"]

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

# # Customize how many profiles you want to plot
# profile_keys = list(control_profiles.keys())[132:]  # Plot specific profiles

# # Plot each selected profile
# for profile_id in profile_keys:
#     profile = control_profiles[profile_id]
#     plot_profile(profile_id, profile)
