import numpy as np
from scipy.optimize import minimize
from typing import Optional, Dict, Any
from rcam import *

def trim_cost(xu: np.ndarray, constraints: Dict[str, Any]) -> float:
    """
    Cost function for trimming the RCAM aircraft model.
    
    Args:
        xu: Combined state and control vector [x0, u0] (length 17)
        constraints: Dictionary of optional constraints

    Returns:
        Scalar cost value
    """
    x0 = xu[:12]  # Aircraft state vector
    u0 = xu[12:]  # Control input vector

    # Evaluate plant dynamics at given state and input
    x_dot = rcam_plant(x0, u0)

    # Define weights for the cost function
    x_dot_weights = np.array([1, 1, 1, np.rad2deg(1), np.rad2deg(1), np.rad2deg(1), np.rad2deg(1), np.rad2deg(1), np.rad2deg(1)])
    
    # Cost is primarily the norm of the dynamics (steady state => x_dot ≈ 0)
    if "flight_path_angle" in constraints:
        cost = x_dot[:9] @ np.diag(x_dot_weights) @ x_dot[:9]
    else:
        x_dot_weights = np.concatenate((x_dot_weights, np.array([1])))
        cost = np.concatenate((x_dot[:9], [x_dot[11]])) @ np.diag(x_dot_weights) @ np.concatenate((x_dot[:9], [x_dot[11]]))

    # Additional cost terms for optional constraints
    u, v, w = x0[0:3]  # Body frame velocities
    phi, theta, psi = x0[6:9]  # Euler angles
    V_A = np.linalg.norm([u, v, w])  # Airspeed
    alpha = np.arctan2(w, u)  # Angle of attack
    beta = np.arcsin(v / V_A) if V_A != 0 else 0.0  # Sideslip angle

    # Penalize non-zero sideslip for coordinated flight
    cost += 10.0 * beta**2

    # Penalize split (differential) throttle inputs
    cost += 100.0 * (u0[3] - u0[4])**2

    # Penalize control inputs exceeding saturation limits
    if u0[0] < -deg2rad(25) or u0[0] > deg2rad(25):
        cost += 1000.0 * (u0[0] - np.clip(u0[0], -deg2rad(25), deg2rad(25)))**2
    if u0[1] < -deg2rad(25) or u0[1] > deg2rad(10):
        cost += 1000.0 * (u0[1] - np.clip(u0[1], -deg2rad(25), deg2rad(10)))**2
    if u0[2] < -deg2rad(30) or u0[2] > deg2rad(30):
        cost += 1000.0 * (u0[2] - np.clip(u0[2], -deg2rad(30), deg2rad(30)))**2
    if u0[3] < deg2rad(0.5) or u0[3] > deg2rad(10):
        cost += 1000.0 * (u0[3] - np.clip(u0[3], deg2rad(0.5), deg2rad(10)))**2
    if u0[4] < deg2rad(0.5) or u0[4] > deg2rad(10):
        cost += 1000.0 * (u0[4] - np.clip(u0[4], deg2rad(0.5), deg2rad(10)))**2

    # Penalize deviation from desired airspeed
    if "airspeed" in constraints:
        V_target = constraints["airspeed"]
        cost += 50.0 * (V_A - V_target) ** 2

    # Penalize deviation from desired flight path angle
    if "flight_path_angle" in constraints:
        gamma = constraints["flight_path_angle"]
        gamma_actual = theta - alpha
        cost += 10.0 * (gamma - gamma_actual) ** 2

    # Penalize deviation from desired altitude
    if "altitude" in constraints:
        h_target = constraints["altitude"]
        h = -x0[11]  # PD is down (positive downward)
        cost += 10.0 * (h - h_target) ** 2

    # Penalize deviation from desired bank angle
    if "bank_angle" in constraints:
        phi_target = constraints["bank_angle"]
        cost += 30.0 * (phi - phi_target) ** 2

    return cost

def trim_rcam(
    x0: np.ndarray,
    u0: np.ndarray,
    constraints: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Trims the RCAM aircraft model by finding steady-state state and control inputs.

    Args:
        x0: Initial guess for the state (length 12)
        u0: Initial guess for the control inputs (length 5)
        constraints: Optional dictionary specifying trim constraints
        options: Optional solver options for scipy.optimize.minimize

    Returns:
        Dictionary with keys: 'success', 'x_trim', 'u_trim', 'cost', 'result'
    """
    if constraints is None:
        constraints = {}
    if options is None:
        options = {"disp": True}

    # Initial guess for the optimizer
    xu0 = np.concatenate((x0, u0))

    # Solve the optimization problem
    result = minimize(
        trim_cost,
        xu0,
        args=(constraints,),
        method="SLSQP",
        options=options
    )

    xu_opt = result.x
    x_trim = xu_opt[:12]
    u_trim = xu_opt[12:]

    return {
        "success": result.success,
        "x_trim": x_trim,
        "u_trim": u_trim,
        "cost": result.fun,
        "result": result
    }
