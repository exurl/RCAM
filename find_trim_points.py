import numpy as np
from trim import *
from compute_trajectory import simulate_trajectory
import matplotlib.pyplot as plt

# Initial guess for the state and control vectors (rough estimate)
x0_guess = np.zeros(12)
u0_guess = np.zeros(5)
u0_guess[3] = deg2rad(5.25)  # throttle position of engine 1 setting
u0_guess[4] = deg2rad(5.25)  # throttle position of engine 2 setting

trim_cases = [
    # Straight and level at 80 m/s and 1000 m altitude
    {
        "name": "StraightLevel_80ms_1000m",
        "airspeed": 80.0,
        "altitude": 1000.0
    },
    # Straight and level at 120 m/s and 10000 m altitude
    {
        "name": "StraightLevel_120ms_10000m",
        "airspeed": 120.0,
        "altitude": 10000.0
    },
    # Coordinated turn at 70 m/s with 10 degree bank
    {
        "name": "BankedTurn_70ms_10deg",
        "airspeed": 70.0,
        "altitude": 1000.0,
        "bank_angle": np.deg2rad(10)
    },
    # Coordinated turn at 80 m/s with 30 degree bank
    {
        "name": "BankedTurn_80ms_30deg",
        "airspeed": 80.0,
        "altitude": 2000.0,
        "bank_angle": np.deg2rad(30)
    },
    # Constant descent with 3 deg flight path angle
    {
        "name": "Descent_80ms_3deg_fltpath",
        "airspeed": 80.0,
        "altitude": 2000.0,
        "flight_path_angle": -np.deg2rad(3)
    }
]

for case in trim_cases:
    print(f"\nTrimming case: {case['name']}")

    # Update intial guess based on case constraint parameters
    x0_guess[0] = case["airspeed"]      # u (forward velocity)
    x0_guess[11] = -case["altitude"]    # pd (altitude)

    result = trim_rcam(x0_guess, u0_guess, constraints=case)

    if result['success']:
        print(f"Trim successful for {case['name']}")
        print(f"x_trim = {result['x_trim']}")
        print(f"u_trim = {result['u_trim']}")
        print(f"Cost = {result['cost']:.4e}")
        
        x_dot = rcam_plant(result['x_trim'], result['u_trim'])
        print(f"x_dot = {x_dot}")

        # Store trim results in the case dictionary
        case["x_trim"] = result["x_trim"]
        case["u_trim"] = result["u_trim"]
    else:
        print(f"Trim failed for {case['name']}")
        print(result['result'].message)

# Evaluate trim points

# Define time vector
tmin = 0
tmax = 10
dt = 0.1    # step size of time vector
N = int(((tmax - tmin) / dt) + 1)
t_span = (tmin, tmax)  # time from 0 to 10
t_vec = np.linspace(tmin, tmax, N)  # evaluation points from 0 to 10 with step size 0.1

for case in trim_cases:
    if "x_trim" in case and "u_trim" in case:
        print(f"\nEvaluating trim point for {case['name']}")
        x_trim = case["x_trim"]
        u_trim = case["u_trim"]

        # Define array of control inputs
        u_trim = np.tile(u_trim, (N, 1))    # Repeat the trim control inputs for each time step

        # Simulate the trajectory with the trim point
        t_out, x_out = simulate_trajectory(
            f=rcam_plant, x0=x_trim, t_in=t_vec, u_in=u_trim, t_out=t_vec
        )
        
        # Plot array of state variables
        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(f"Trajectory for {case['name']}", fontsize=16)
        plt.subplot(3, 3, 1)
        plt.plot(t_out, x_out[:, 0], label='u (forward velocity)')
        plt.title('u (forward velocity)')
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.grid()
        plt.subplot(3, 3, 2)
        plt.plot(t_out, x_out[:, 1], label='v (lateral velocity)')
        plt.title('v (lateral velocity)')
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.grid()
        plt.subplot(3, 3, 3)
        plt.plot(t_out, x_out[:, 2], label='w (vertical velocity)')
        plt.title('w (vertical velocity)')
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.grid()
        plt.subplot(3, 3, 4)
        plt.plot(t_out, np.rad2deg(x_out[:, 3]), label='p (roll rate)')
        plt.title('p (roll rate)')
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (deg/s)")
        plt.grid()
        plt.subplot(3, 3, 5)
        plt.plot(t_out, np.rad2deg(x_out[:, 4]), label='q (pitch rate)')
        plt.title('q (pitch rate)')
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (deg/s)")
        plt.grid()
        plt.subplot(3, 3, 6)
        plt.plot(t_out, np.rad2deg(x_out[:, 5]), label='r (yaw rate)')
        plt.title('r (yaw rate)')
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (deg/s)")
        plt.grid()
        plt.subplot(3, 3, 7)
        plt.plot(t_out, np.rad2deg(x_out[:, 6]), label='phi (roll angle)')
        plt.title('phi (roll angle)')
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.grid()
        plt.subplot(3, 3, 8)
        plt.plot(t_out, x_out[:, 7], label='theta (pitch angle)')
        plt.title('theta (pitch angle)')
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.grid()
        plt.subplot(3, 3, 9)
        plt.plot(t_out, x_out[:, 8], label='psi (yaw angle)')
        plt.title('psi (yaw angle)')
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.grid()
        plt.tight_layout()
        plt.show()

        # Cross-plot showing top-down view of trajectory
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(x_out[:, 9], x_out[:, 10], label='Trajectory (PE vs PN)')
        plt.title(f"Top-down view of trajectory for {case['name']}")
        plt.xlabel("PN (North Position)")
        plt.ylabel("PE (East Position)")
        plt.axis('equal')
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(-x_out[:, 11], label='Altitude (m)')
        plt.title(f"Altitude for {case['name']}")
        plt.xlabel("Time (s)")
        plt.ylabel("Altitude (m)")
        plt.grid()
        plt.show()

    else:
        print(f"No trim results available for {case['name']}")
