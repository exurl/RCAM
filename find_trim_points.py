import numpy as np
from trim import *

# Initial guess for the state and control vectors (rough estimate)
x0_guess = np.zeros(12)
u0_guess = np.zeros(5)
u0_guess[3] = np.deg2rad(5.0)  # throttle position of engine 1 setting
u0_guess[4] = np.deg2rad(5.0)  # throttle position of engine 2 setting

trim_cases = [
    # Straight and level at 80 m/s and 1000 m altitude
    {
        "name": "StraightLevel_80ms_1000m",
        "airspeed": 80.0,
        "altitude": 1000.0
    },
    # Straight and level at 180 m/s and 10000 m altitude
    {
        "name": "StraightLevel_180ms_10000m",
        "airspeed": 180.0,
        "altitude": 10000.0
    },
    # Coordinated turn at 60 m/s with 30 degree bank
    {
        "name": "BankedTurn_60ms_30deg",
        "airspeed": 60.0,
        "altitude": 1000.0,
        "bank_angle": np.deg2rad(30)
    },
    # Coordinated turn at 80 m/s with 45 degree bank
    {
        "name": "BankedTurn_80ms_45deg",
        "airspeed": 80.0,
        "altitude": 2000.0,
        "bank_angle": np.deg2rad(45)
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
    else:
        print(f"Trim failed for {case['name']}")
        print(result['result'].message)
