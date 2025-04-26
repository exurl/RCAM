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

# Function to plot time history of RCAM state variables
def plot_state_variables(t, x, title):
    """
    Plot the 9 aircraft state variables over time in a 3x3 grid of subplots.

    Parameters:
    -----------
    t : array-like
        Time vector (in seconds).
    x : array-like
        State variable matrix of shape (n_samples, 12), where columns represent:
        [u, v, w, p, q, r, phi, theta, psi, PN, PE, PD]
        Velocities are in m/s, angular rates in rad/s, and angles in radians.
    title : str
        Title for the entire figure.
    
    Returns:
    --------
    None
        Displays a matplotlib figure with nine subplots.

    Notes:
    ------
    - Angular rate variables (p, q, r) and angles (phi, theta, psi) are converted to degrees.
    - The plot is arranged in a 3x3 grid with gridlines and axis labels.
    """
    fig, axs = plt.subplots(3, 3, figsize=(10, 6))
    fig.suptitle(title, fontsize=16)

    labels = [
        ("u (forward velocity)", "Velocity (m/s)", x[:, 0]),
        ("v (lateral velocity)", "Velocity (m/s)", x[:, 1]),
        ("w (vertical velocity)", "Velocity (m/s)", x[:, 2]),
        ("p (roll rate)", "Angular Velocity (deg/s)", np.rad2deg(x[:, 3])),
        ("q (pitch rate)", "Angular Velocity (deg/s)", np.rad2deg(x[:, 4])),
        ("r (yaw rate)", "Angular Velocity (deg/s)", np.rad2deg(x[:, 5])),
        ("phi (roll angle)", "Angle (deg)", np.rad2deg(x[:, 6])),
        ("theta (pitch angle)", "Angle (deg)", np.rad2deg(x[:, 7])),
        ("psi (yaw angle)", "Angle (deg)", np.rad2deg(x[:, 8])),
    ]

    for i, ax in enumerate(axs.flat):
        title_i, ylabel_i, data_i = labels[i]
        ax.plot(t, data_i)
        ax.set_title(title_i)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel_i)
        ax.grid(True)

    fig.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5, rect=[0, 0, 1, 0.95])
    plt.show()

# Function to plot top-down and side views of aircraft trajectory
def plot_trajectory(t, x, title):
    """
    Plots the aircraft trajectory in two subplots: a top-down view of the trajectory
    and the altitude over time.

    Parameters:
    -----------
    t : array-like
        A 1D array representing the time values (in seconds).
    x : array-like
        State variable matrix of shape (n_samples, 12), where columns represent:
        [u, v, w, p, q, r, phi, theta, psi, PN (north position), PE (east position), PD (down position)]
        Positions are in meters from the origin.
    title : str
        The title of the plot.

    Returns:
    --------
    None
        Displays a matplotlib figure with two subplots.

    Notes:
    ------
    - The first subplot shows the top-down view of the trajectory (PE vs PN) with equal axis scaling.
    - The second subplot shows the altitude (in meters) as a function of time.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)

    # Top-down view
    ax[0].plot(x[:, 9], x[:, 10], label='Trajectory (PE vs PN)')
    ax[0].set_title("Top-down View of Trajectory")
    ax[0].set_xlabel("PN (North Position)")
    ax[0].set_ylabel("PE (East Position)")
    ax[0].axis('equal')
    ax[0].grid(True)

    # Altitude over time
    ax[1].plot(t, -x[:, 11], label='Altitude (m)')
    ax[1].set_title("Altitude")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Altitude (m)")
    ax[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Function to plot comparison of roll input and response
def plot_roll(t, x, u, title):
    """
    Plots the roll input and response of the aircraft over time.

    Parameters:
    -----------
    t : array-like
        A 1D array representing the time values (in seconds).
    x : array-like
        State variable matrix of shape (n_samples, 12), where columns represent:
        [u, v, w, p, q, r, phi, theta, psi, PN (north position), PE (east position), PD (down position)]
        Positions are in meters from the origin.
    u : array-like
        Control input matrix of shape (n_samples, 5), where columns represent:
        [d_A (aileron), d_T (tailplane), d_R (rudder), d_th1 (throttle 1), d_th2 (throttle 2)]
        Control inputs are in radians.
    title : str
        The title of the plot.

    Returns:
    --------
    None
        Displays a matplotlib figure with two subplots.

    Notes:
    ------
    - The first subplot shows the roll input (d_A) and roll rate response (p) over time.
    - The second subplot shows the roll angle response (phi) over time.
    """
    # Saturation limits for each control variable (in radians)
    saturation_limits = {
    "d_A": np.radians([-25, 25]),  # Aileron
    "d_T": np.radians([-25, 10]),  # Tailplane
    "d_R": np.radians([-30, 30]),  # Rudder
    "d_th1": np.radians([0.5, 10]),  # Throttle 1
    "d_th2": np.radians([0.5, 10])   # Throttle 2
    }
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=16)

    # Roll input and response
    ax[0].plot(t, np.rad2deg(u[:, 0]), label='Roll Input (d_A)', color='blue')
    ax[0].plot(t, np.rad2deg(x[:, 3]), label='Roll Rate Response (p)', color='orange')
    ax[0].axhline(np.rad2deg(saturation_limits["d_A"][0]), color='k', linestyle='--')  # Lower limit
    ax[0].axhline(np.rad2deg(saturation_limits["d_A"][1]), color='k', linestyle='--')  # Upper limit
    ax[0].set_title("Roll Input and Response")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Roll Input (d_A) [deg] / Roll Rate (p) [deg/s]")
    ax[0].legend()
    ax[0].grid(True)

    # Roll angle response
    ax[1].plot(t, np.rad2deg(x[:, 6]), label='Roll Angle Response (phi)', color='green')
    ax[1].set_title("Roll Angle Response")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Roll Angle (phi) [deg]")
    ax[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return

# Function to plot comparison of pitch & throttle inputs and response
def plot_pitch_throttle(t, x, u, title):
    """
    Plots the pitch and throttle inputs and responses of the aircraft over time.

    Parameters:
    -----------
    t : array-like
        A 1D array representing the time values (in seconds).
    x : array-like
        State variable matrix of shape (n_samples, 12), where columns represent:
        [u, v, w, p, q, r, phi, theta, psi, PN (north position), PE (east position), PD (down position)]
        Positions are in meters from the origin.
    u : array-like
        Control input matrix of shape (n_samples, 5), where columns represent:
        [d_A (aileron), d_T (tailplane), d_R (rudder), d_th1 (throttle 1), d_th2 (throttle 2)]
        Control inputs are in radians.
    title : str
        The title of the plot.

    Returns:
    --------
    None
        Displays a matplotlib figure with two subplots.

    Notes:
    ------
    - The first subplot shows the pitch input (d_T) and pitch rate response (q) over time.
    - The second subplot shows the throttle inputs (d_th1, d_th2) over time.
    - The third subplot shows the pitch angle response (theta) over time.
    """
    # Saturation limits for each control variable (in radians)
    saturation_limits = {
    "d_A": np.radians([-25, 25]),  # Aileron
    "d_T": np.radians([-25, 10]),  # Tailplane
    "d_R": np.radians([-30, 30]),  # Rudder
    "d_th1": np.radians([0.5, 10]),  # Throttle 1
    "d_th2": np.radians([0.5, 10])   # Throttle 2
    }
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=16)

    # Pitch input and response
    ax[0].plot(t, np.rad2deg(u[:, 1]), label='Pitch Input (d_T)', color='blue')
    ax[0].plot(t, np.rad2deg(x[:, 4]), label='Pitch Rate Response (q)', color='orange')
    ax[0].axhline(np.rad2deg(saturation_limits["d_T"][0]), color='k', linestyle='--')  # Lower limit
    ax[0].axhline(np.rad2deg(saturation_limits["d_T"][1]), color='k', linestyle='--')  # Upper limit
    ax[0].set_title("Pitch Input and Response")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Pitch Input (d_T) [deg] / Pitch Rate (q) [deg/s]")
    ax[0].legend()
    ax[0].grid(True)

    # Throttle inputs
    ax[1].plot(t, np.rad2deg(u[:, 3]), label='Throttle 1 (d_th1)', color='green')
    ax[1].plot(t, np.rad2deg(u[:, 4]), label='Throttle 2 (d_th2)', color='purple')
    ax[1].axhline(np.rad2deg(saturation_limits["d_th1"][0]), color='k', linestyle='--')  # Lower limit
    ax[1].axhline(np.rad2deg(saturation_limits["d_th1"][1]), color='k', linestyle='--')  # Upper limit
    ax[1].set_title("Throttle Inputs")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Throttle Input (d_th) [deg]")
    ax[1].legend()
    ax[1].grid(True)

    # Pitch angle response
    ax[2].plot(t, np.rad2deg(x[:, 7]), label='Pitch Angle Response (theta)', color='red')
    ax[2].set_title("Pitch Angle Response")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Pitch Angle (theta) [deg]")
    ax[2].grid(True)
    ax[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return

# Function to plot comparison of yaw input and response
def plot_yaw(t, x, u, title):
    """
    Plots the yaw input and response of the aircraft over time.

    Parameters:
    -----------
    t : array-like
        A 1D array representing the time values (in seconds).
    x : array-like
        State variable matrix of shape (n_samples, 12), where columns represent:
        [u, v, w, p, q, r, phi, theta, psi, PN (north position), PE (east position), PD (down position)]
        Positions are in meters from the origin.
    u : array-like
        Control input matrix of shape (n_samples, 5), where columns represent:
        [d_A (aileron), d_T (tailplane), d_R (rudder), d_th1 (throttle 1), d_th2 (throttle 2)]
        Control inputs are in radians.
    title : str
        The title of the plot.

    Returns:
    --------
    None
        Displays a matplotlib figure with two subplots.

    Notes:
    ------
    - The first subplot shows the yaw input (d_R) and yaw rate response (r) over time.
    - The second subplot shows the yaw angle response (psi) over time.
    """
    # Saturation limits for each control variable (in radians)
    saturation_limits = {
    "d_A": np.radians([-25, 25]),  # Aileron
    "d_T": np.radians([-25, 10]),  # Tailplane
    "d_R": np.radians([-30, 30]),  # Rudder
    "d_th1": np.radians([0.5, 10]),  # Throttle 1
    "d_th2": np.radians([0.5, 10])   # Throttle 2
    }
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=16)

    # Yaw input and response
    ax[0].plot(t, np.rad2deg(u[:, 2]), label='Yaw Input (d_R)', color='blue')
    ax[0].plot(t, np.rad2deg(x[:, 5]), label='Yaw Rate Response (r)', color='orange')
    ax[0].axhline(np.rad2deg(saturation_limits["d_R"][0]), color='k', linestyle='--')  # Lower limit
    ax[0].axhline(np.rad2deg(saturation_limits["d_R"][1]), color='k', linestyle='--')  # Upper limit
    ax[0].set_title("Yaw Input and Response")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Yaw Input (d_R) [deg] / Yaw Rate (r) [deg/s]")
    ax[0].legend()
    ax[0].grid(True)

    # Yaw angle response
    ax[1].plot(t, np.rad2deg(x[:, 8]), label='Yaw Angle Response (psi)', color='green')
    ax[1].set_title("Yaw Angle Response")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Yaw Angle (psi) [deg]")
    ax[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return
