import numpy as np
from numpy import deg2rad
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation


def dcm(phi: float, theta: float, psi: float) -> ArrayLike:
    """Direction cosine matrix from Euler angles

    Args:
        phi (float): rotation 3 about X''
        theta (float): rotation 2 about Y'
        psi (float): rotation 1 about Z

    Returns:
        ArrayLike: direction cosine matrix
    """
    return Rotation.from_euler("ZYX", [psi, theta, phi]).as_matrix()


def rcam_plant(x: ArrayLike, u: ArrayLike) -> ArrayLike:
    """RCAM plant dynamics

    Args:
        x (ArrayLike): plant state vector [u, v, w, p, q, r, phi, theta, psi, PN, PE, PD]
        u (ArrayLike): plant control vector [d_A, d_T, d_R, d_th1, d_th2]

    Returns:
        ArrayLike: plant state time derivative
    """
    # Constants
    g = 9.81  # gravitational acceleration
    rho = 1.225  # air density at sea level
    n = 5.5  # lift curve slope
    alpha_L0 = deg2rad(-11.5)  # zero lift AOA
    a0 = 15.212  # wing body stall regime model
    a1 = -155.2  # stall model coefficient
    a2 = 609.2  # stall model coefficient
    a3 = -768.5  # stall model coefficient
    l_t = 24.8  # tail distance
    depsilondalpha = 0.25  # downwash model
    S = 260  # wing area
    S_t = 64  # (horizontal) tail area
    c_bar = 6.6  # mean aerodynamic chord
    m = 120000  # mass
    r_E1b = np.array([0, -7.94, -1.9])  # engine 1 position in body frame
    r_E2b = np.array([0, 7.94, -1.9])  # engine 2 position in body frame
    I_b = m * np.array(
        [[40.07, 0, -2.0923], [0, 64, 0], [-2.0923, 0, 99.92]]
    )  # rotational inertia

    # Intermediate Variables
    V_A = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)  # airspeed
    alpha = np.arctan2(x[2], x[0])  # AOA
    beta = np.arcsin(x[1] / V_A)  # sideslip angle
    q = 0.5 * rho * V_A**2  # dynamic pressure
    omega_beb = np.array(
        [x[3], x[4], x[5]]
    )  # rotation w.r.t. earth in body frame
    V_b = np.array([x[0], x[1], x[2]])  # velocity in body frame

    ## AERODYNAMIC FORCES ------------------------------------------------------

    # Wing-Body Lift Coefficient in Stability Frame
    if alpha <= deg2rad(14.5):
        C_Lwb = n * (alpha - alpha_L0)  # linear regime
    else:
        C_Lwb = a3 * alpha**3 + a2 * alpha**2 + a1 * alpha + a0  # stall regime

    # Tail Lift Coefficient in Stability Frame
    epsilon = depsilondalpha * (alpha - alpha_L0)  # downwash
    alpha_t = alpha - epsilon + u[1] + 1.3 * x[4] * l_t / V_A  # tail AOA
    C_Lt = 3.1 * S_t / S * alpha_t

    # Total Lift Coefficient in Stability Frame
    C_L = C_Lwb + C_Lt

    # Drag Coefficient in Stability Frame
    C_D = 0.13 + 0.07 * (5.5 * alpha + 0.654) ** 2

    # Side Force Coeffficient in Stability Frame
    C_Y = -1.6 * beta + 0.24 * u[2]

    # Aerodynamic Forces in Stability Frame
    F_As = np.array([-C_D * q * S, C_Y * q * S, -C_L * q * S])

    # Aerodynamic Forces in Body Frame
    C_bs = np.array(
        [
            [np.cos(alpha), 0, -np.sin(alpha)],
            [0, 1, 0],
            [np.sin(alpha), 0, np.cos(alpha)],
        ]
    )  # rotation matrix
    F_Ab = C_bs @ F_As

    ## AERODYNAMIC MOMENTS -----------------------------------------------------

    # Aerodynamic Moment Coefficients about AC in Body Frame
    eta_bar = np.array(
        [
            -1.4 * beta,
            -0.59 - 3.1 * S_t * l_t / (S * c_bar) * (alpha - epsilon),
            (1 - alpha * 180 / (15 * np.pi)) * beta,
        ]
    )  # orientation effects
    dcmdx = np.array(
        [
            [-11, 0, 5],
            [0, -4.03 * S_t * (l_t**2) / (S * (c_bar**2)), 0],
            [1.7, 0, -11.5],
        ]
    ) * (
        c_bar / V_A
    )  # rotation rate sensitivity
    dcmdu = np.array(
        [
            [-0.6, 0, 0.22],
            [0, -3.1 * S_t * l_t / (S * c_bar), 0],
            [0, 0, -0.63],
        ]
    )  # control sensitivity
    C_Macb = eta_bar + dcmdx @ x[3:6] + dcmdu @ u[:3]

    # Aerodynamic Moments about AC in Body Frame
    M_Aacb = C_Macb * q * S * c_bar

    # Aerodynamic Moments about CG in Body Frame
    r_cgb = np.array(
        [0.23 * c_bar, 0, 0.10 * c_bar]
    )  # CG position in body frame
    r_acb = np.array([0.12 * c_bar, 0, 0])  # AC position in body frame
    M_Acgb = M_Aacb + np.cross(F_Ab, (r_cgb - r_acb))

    ## ENGINE FORCES AND MOMENTS -----------------------------------------------

    # Engine Forces in Body Frame
    F_E1b = np.array([u[3] * m * g, 0, 0])  # engine 1 force
    F_E2b = np.array([u[4] * m * g, 0, 0])  # engine 2 force
    F_Eb = F_E1b + F_E2b

    # Engine Moments about CG in Body Frame
    mu_bar1 = np.array(
        [r_cgb[0] - r_E1b[0], r_E1b[1] - r_cgb[1], r_cgb[2] - r_E1b[2]]
    )  # engine 1 moment arm
    mu_bar2 = np.array(
        [r_cgb[0] - r_E2b[0], r_E2b[1] - r_cgb[1], r_cgb[2] - r_E2b[2]]
    )  # engine 2 moment arm
    M_E1cgb = np.cross(mu_bar1, F_E1b)  # engine 1 moment
    M_E2cgb = np.cross(mu_bar2, F_E2b)  # engine 2 moment
    M_Ecgb = M_E1cgb + M_E2cgb

    ## GRAVITY FORCES
    F_gb = (
        np.array(
            [
                -np.sin(x[7]),
                np.cos(x[7]) * np.sin(x[6]),
                np.cos(x[7]) * np.cos(x[6]),
            ]
        )
        * m
        * g
    )

    ## FIRST-ORDER EXPLICIT SYSTEM ---------------------------------------------

    x_dot = np.zeros(len(x))

    # Linear Acceleration
    F_b = F_gb + F_Eb + F_Ab  # total forces in body frame
    x_dot[0:3] = (1 / m) * F_b - np.cross(omega_beb, V_b)

    # Rotational Acceleration
    M_cgb = M_Ecgb + M_Acgb  # total moments in body frame
    x_dot[3:6] = np.linalg.solve(
        I_b, (M_cgb - np.cross(omega_beb, (I_b @ omega_beb)))
    )

    # Euler Rates
    phi = x[6]
    theta = x[7]
    ERmat = np.array(
        [
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)],
        ]
    )  # Euler rate from body rate transform
    x_dot[6:9] = ERmat @ x[3:6]

    ## APPEND NORTH-EAST-DOWN POSITION RATES -----------------------------------

    if len(x) == 12:
        # DCM From Body to NED Frame
        phi, theta, psi = x[6:9]
        C_vb = dcm(phi, theta, psi)

        # Transform u,v,w to V_NED
        x_dot[9:12] = C_vb @ x[0:3]

    return x_dot


def rcam_actuators(x: ArrayLike, u: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """RCAM actuator dynamics

    Args:
        x (ArrayLike): actuator state vector [lag_A, lag_T, lag_R, lag_th1, lag_th2, lead_th1_1, lead_th1_2, lead_th2_1, lead_th2_2]
        u (ArrayLike): actuator control vector [d_A, d_T, d_R, d_th1, d_th2]

    Returns:
        ArrayLike: actuator control output
        ArrayLike: actuator state time derivative

    Reference: Section 2.5
    """
    x_lag = x[:5]
    x_lead = x[5:]

    # Rate limit (2nd-order Butterworth filter approximation)
    A = np.array([[-4.9635, -12.3166], [1, 0]])
    A_block = np.block([[A, np.zeros((2, 2))], [np.zeros((2, 2)), A]])
    B_block = np.array([[0, 0], [12.3166, 0], [0, 0], [0, 12.3166]])
    x_dot_lead = A_block @ x_lead + B_block @ u[-2:]
    y_lead = 12.3166 * x_lead[[1, 3]]
    y_lead = np.concatenate((np.zeros(3), y_lead))

    # Lag dynamics
    lags = 1 / np.array([0.15, 0.15, 0.3, 1.5, 1.5])
    x_dot_lag = -lags * x_lag + y_lead
    y = lags * x_lag

    # Saturation limits
    y_min = np.array([-deg2rad(25), -deg2rad(25), -deg2rad(30), 0.5, 0.5])
    y_max = np.array([deg2rad(25), deg2rad(10), deg2rad(30), 1.0, 1.0])
    y[y > y_max] = y_max[y > y_max]
    y[y < y_min] = y_min[y < y_min]

    return y, np.concatenate((x_dot_lag, x_dot_lead))


def rcam_system(x: ArrayLike, u: ArrayLike) -> ArrayLike:
    """RCAM system dynamics

    Args:
        x (ArrayLike): system state vector
        u (ArrayLike): system control vector

    Returns:
        ArrayLike: system state time derivative
    """
    y_act, x_dot_act = rcam_actuators(x[12:], u)
    x_dot_plant = rcam_plant(x[:12], y_act)
    return np.concatenate((x_dot_plant, x_dot_act))
