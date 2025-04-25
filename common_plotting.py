import numpy as np
import matplotlib.pyplot as plt

# Function to plot time history of control inputs
def plot_control_inputs(profile, title):
    """
    Plots the time history of five aircraft control inputs along with their saturation limits.

    Parameters:
    -----------
    profile : numpy.ndarray
        A 2D array where the first column represents time (in seconds), and the following columns 
        represent control inputs in the following order:
        [d_A (aileron), d_T (tailplane), d_R (rudder), d_th1 (throttle 1), d_th2 (throttle 2)].
        The control inputs should be in radians.

    title : str
        The title for the overall figure.

    Returns:
    --------
    None
        Displays a matplotlib figure with five subplots, each showing one control input over time 
        with dashed lines indicating the corresponding saturation limits.
    """
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    # Set overall figure title
    fig.suptitle(title, fontsize=16)

    # Saturation limits for each control variable (in radians)
    saturation_limits = {
    "d_A": np.radians([-25, 25]),  # Aileron
    "d_T": np.radians([-25, 10]),  # Tailplane
    "d_R": np.radians([-30, 30]),  # Rudder
    "d_th1": np.radians([0.5, 10]),  # Throttle 1
    "d_th2": np.radians([0.5, 10])   # Throttle 2
    }

    # Define input names and titles for the plots
    input_names = ["d_A", "d_T", "d_R", "d_th1", "d_th2"]
    titles = ["d_A (aileron)", "d_T (tailplane)", "d_R (rudder)", "d_th1 (throttle 1)", "d_th2 (throttle 2)"]

    # Plot each control variable
    for i, ax in enumerate(axes):
        input_col = i + 1  # Column index for the control variable
        var = input_names[i]  # Variable name

        # Plot the control time history and saturation limits
        ax.plot(profile[:, 0], profile[:, input_col], label=titles[i])
        ax.axhline(saturation_limits[var][0], color='k', linestyle='--')  # Lower limit
        ax.axhline(saturation_limits[var][1], color='k', linestyle='--')  # Upper limit

        ax.legend(loc="upper right")
        ax.grid(True)

    # Common x and y labels
    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.text(0.05, 0.5, "Deflection (rad)", va='center', rotation='vertical', fontsize=12)

    # Adjust layout and show the plot
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust for space around the y-label and suptitle
    plt.xlim([0, profile[-1, 0]])
    plt.show()

    return
