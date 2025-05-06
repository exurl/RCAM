import numpy as np
from trim import *
from compute_trajectory import simulate_trajectory
from common_plotting import plot_state_variables, plot_trajectory


# Initial guess for the state and control vectors (rough estimate)
x0_guess = np.zeros(12)
u0_guess = np.zeros(5)
u0_guess[3] = deg2rad(5.25)  # throttle position of engine 1 setting
u0_guess[4] = deg2rad(5.25)  # throttle position of engine 2 setting

# ==== Define trim cases ====

# # Define trim cases for training data generation
# trim_cases = [
#     # Straight and level at 70 m/s and 1000 m altitude
#     {
#         "name": "StraightLevel_70ms_1000m",
#         "airspeed": 70.0,
#         "altitude": 1000.0
#     },
#     # Straight and level at 120 m/s and 10000 m altitude
#     {
#         "name": "StraightLevel_120ms_10000m",
#         "airspeed": 120.0,
#         "altitude": 10000.0
#     },
#     # Coordinated turn at 70 m/s with +10 degree bank
#     {
#         "name": "BankedTurn_70ms_1000m_+10deg",
#         "airspeed": 70.0,
#         "altitude": 1000.0,
#         "bank_angle": np.deg2rad(10)
#     },
#     # Coordinated turn at 70 m/s with -10 degree bank
#     {
#         "name": "BankedTurn_70ms_1000m_-10deg",
#         "airspeed": 70.0,
#         "altitude": 1000.0,
#         "bank_angle": np.deg2rad(-10)
#     },
#     # Coordinated turn at 80 m/s with +30 degree bank
#     {
#         "name": "BankedTurn_80ms_2000m_+30deg",
#         "airspeed": 80.0,
#         "altitude": 2000.0,
#         "bank_angle": np.deg2rad(30)
#     },
#     # Coordinated turn at 80 m/s with -30 degree bank
#     {
#         "name": "BankedTurn_80ms_2000m_-30deg",
#         "airspeed": 80.0,
#         "altitude": 2000.0,
#         "bank_angle": np.deg2rad(-30)
#     },
#     # Coordinated turn at 120 m/s with +10 degree bank
#     {
#         "name": "BankedTurn_120ms_10000m_+10deg",
#         "airspeed": 120.0,
#         "altitude": 10000.0,
#         "bank_angle": np.deg2rad(10)
#     },
#     # Coordinated turn at 120 m/s with -10 degree bank
#     {
#         "name": "BankedTurn_120ms_10000m_-10deg",
#         "airspeed": 120.0,
#         "altitude": 10000.0,
#         "bank_angle": np.deg2rad(-10)
#     },
#     # Constant descent with -3 deg flight path angle
#     {
#         "name": "Descent_80ms_2000m_-3deg_fltpath",
#         "airspeed": 80.0,
#         "altitude": 2000.0,
#         "flight_path_angle": -np.deg2rad(3)
#     }
# ]

# # Define trim cases for test data generation - interpolation
# trim_cases = [
#     # Straight and level at 85 m/s and 3000 m altitude
#     {
#         "name": "StraightLevel_85ms_3000m",
#         "airspeed": 85.0,
#         "altitude": 3000.0
#     },
#     # Straight and level at 100 m/s and 10000 m altitude
#     {
#         "name": "StraightLevel_100ms_10000m",
#         "airspeed": 100.0,
#         "altitude": 10000.0
#     },
#     # Coordinated turn at 80 m/s with +20 degree bank
#     {
#         "name": "BankedTurn_80ms_2000m_+20deg",
#         "airspeed": 80.0,
#         "altitude": 2000.0,
#         "bank_angle": np.deg2rad(20)
#     },
#     # Coordinated turn at 80 m/s with -20 degree bank
#     {
#         "name": "BankedTurn_80ms_2000m_-20deg",
#         "airspeed": 80.0,
#         "altitude": 2000.0,
#         "bank_angle": np.deg2rad(-20)
#     },
#     # Coordinated turn at 100 m/s with +10 degree bank
#     {
#         "name": "BankedTurn_100ms_10000m_+10deg",
#         "airspeed": 100.0,
#         "altitude": 10000.0,
#         "bank_angle": np.deg2rad(10)
#     },
#     # Coordinated turn at 100 m/s with -10 degree bank
#     {
#         "name": "BankedTurn_100ms_10000m_-10deg",
#         "airspeed": 100.0,
#         "altitude": 10000.0,
#         "bank_angle": np.deg2rad(-10)
#     },
#     # Constant descent with -3 deg flight path angle
#     {
#         "name": "Descent_80ms_2000m_-2deg_fltpath",
#         "airspeed": 80.0,
#         "altitude": 2000.0,
#         "flight_path_angle": -np.deg2rad(2)
#     }
# ]

# Define trim cases for test data generation - extrapolation
trim_cases = [
    # Straight and level at 60 m/s and 1000 m altitude
    {
        "name": "StraightLevel_60ms_1000m",
        "airspeed": 60.0,
        "altitude": 1000.0
    },
    # Straight and level at 120 m/s and 10000 m altitude
    {
        "name": "StraightLevel_130ms_10000m",
        "airspeed": 130.0,
        "altitude": 10000.0
    },
    # Coordinated turn at 80 m/s with +40 degree bank
    {
        "name": "BankedTurn_80ms_2000m_+40deg",
        "airspeed": 80.0,
        "altitude": 2000.0,
        "bank_angle": np.deg2rad(40)
    },
    # Coordinated turn at 80 m/s with -40 degree bank
    {
        "name": "BankedTurn_80ms_2000m_-40deg",
        "airspeed": 80.0,
        "altitude": 2000.0,
        "bank_angle": np.deg2rad(-40)
    },
    # Coordinated turn at 120 m/s with +45 degree bank
    {
        "name": "BankedTurn_120ms_10000m_+45deg",
        "airspeed": 120.0,
        "altitude": 10000.0,
        "bank_angle": np.deg2rad(45)
    },
    # Coordinated turn at 120 m/s with -45 degree bank
    {
        "name": "BankedTurn_120ms_10000m_-45deg",
        "airspeed": 120.0,
        "altitude": 10000.0,
        "bank_angle": np.deg2rad(-45)
    },
    # Constant descent with -4 deg flight path angle
    {
        "name": "Descent_80ms_2000m_-4deg_fltpath",
        "airspeed": 80.0,
        "altitude": 2000.0,
        "flight_path_angle": -np.deg2rad(4)
    }
]

# ==== Obtain trim points ====

trim_results = {}

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

    # Store the trim point results in the dictionary
    trim_results[case['name']] = result

# ==== Save trim results to file ====

# Save to file
# np.savez("trim_points.npz", **trim_results)
# np.savez("trim_points_test_interpolated.npz", **trim_results)
np.savez("trim_points_test_extrapolated.npz", **trim_results)

# ==== Evaluate trim points ====

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
        plot_title = f"State Variable Trajectory for {case['name']}"
        plot_state_variables(t_out, x_out, plot_title)

        # Cross-plot showing top-down view of trajectory
        plot_title = f"Aircraft Trajectory for {case['name']}"
        plot_trajectory(t_out, x_out, plot_title)

    else:
        print(f"No trim results available for {case['name']}")
