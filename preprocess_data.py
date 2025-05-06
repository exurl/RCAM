import numpy as np
from numpy.typing import ArrayLike


def scale_x_data(x):
    """
    Scale state vector variables x by fixed amount, such that scale is appropriate
    for use as training data for a neural network.

    EXCEPTION: Euler yaw angle is converted to sine(psi) to avoid
    discontinuity issues.

    Inputs are:
    x: n x m Numpy array, where m is the number of augmented state variables (17 or 29)

    Outputs are:
    x_scaled: n x m Numpy array
    """

    # State vector scale factors
    scale_vector_x = np.array(
        [
            1e-2,  # u
            1 / 40,  # v
            1 / 40,  # w
            1 / 0.6,  # p
            1 / 0.8,  # q
            1 / 0.3,  # r
            1 / np.radians(45),  # phi
            1 / np.radians(20),  # theta
            1,  # sin(psi)
            1e-3,  # PN
            1e-3,  # PE
            1e-3,  # PD
            1,  # cos(psi)
            1,  # total airspeed
            1,  # angle of attack
            1,  # sideslip angle
            1e-4  # dynamic pressure
        ]
    )

    # If the input includes dx/dt (length 29 columns):
    if x.shape[1] == 29:
        # Augment scale_factor_x with dx/dt scale factors, using the same scale factors 
        # for dx/dt as for the original state variables
        scale_vector_x = np.concatenate(
            (scale_vector_x, scale_vector_x[:12]), axis=0
        )

    x_scaled = x * scale_vector_x

    # Convert Euler yaw angle to sine of yaw angle
    x_scaled[:, 8] = np.sin(x_scaled[:, 8])

    return x_scaled


def scale_u_data(u):
    """
    Scale control vector variables u by fixed amount, such that scale is appropriate
    for use as training data for a neural network.

    Inputs are:
    u: n x 5 Numpy array

    Outpus are:
    u_scaled: n x 5 Numpy array
    """

    # Control vector scale factors
    scale_vector_u = np.array(
        [
            1 / np.radians(25),  # d_A
            1 / np.radians(25),  # d_T
            1 / np.radians(30),  # d_R
            1 / np.radians(10),  # d_th1
            1 / np.radians(10),  # d_th2
        ]
    )

    u_scaled = u @ np.diag(scale_vector_u)

    return u_scaled


def unscale_u_data(u):
    """
    Unscale control vector variables u by fixed amount, such that scale is restored to
    native RCAM model scale.

    Inputs are:
    u: n x 5 Numpy array

    Outputs are:
    u_unscaled: n x 5 Numpy array
    """

    # Control vector scale factors
    scale_vector_u = np.array(
        [
            1 * np.radians(25),  # d_A
            1 * np.radians(25),  # d_T
            1 * np.radians(30),  # d_R
            1 * np.radians(10),  # d_th1
            1 * np.radians(10),  # d_th2
        ]
    )

    u_unscaled = u @ np.diag(scale_vector_u)

    return u_unscaled


def unaugment_unscale_x_data(x):
    # State vector scale factors
    scale_vector_x = np.array(
        [
            1 / 1e2,  # u
            1 / 40,  # v
            1 / 40,  # w
            1 / 0.6,  # p
            1 / 0.8,  # q
            1 / 0.3,  # r
            1 / np.radians(45),  # phi
            1 / np.radians(20),  # theta
            1,  # sin(psi)
            1 / 1e3,  # PN
            1 / 1e3,  # PE
            1 / 1e3,  # PD
            1,  # cos(psi)
            1,  # total airspeed
            1,  # angle of attack
            1,  # sideslip angle
            1e-4,  # dynamic pressure
        ]
    )
    x_unscaled = x / scale_vector_x
    x_unscaled[:, 8] = np.arcsin(x_unscaled[:, 8])

    x_unaugmented = x_unscaled[:, :12]

    return x_unaugmented


def augment_data(x, augment_dxdt=False):
    """
    Augment state vector variables x by adding relevant virtual variables including:
    - cosine of yaw angle
    - total airspeed
    - angle of attack
    - sideslip angle
    - dynamic pressure

    If augment_dxdt=True, also augment state vector by adding the time derivative 
    of the full x state.

    Inputs are:
    x: n x 12 Numpy array

    Outputs are:
    x_augmented: n x 17 Numpy array (if augment_dxdt=False)
    x_augmented: n x 29 Numpy array (if augment_dxdt=True)
    """
    rho = 1.225  # air density at sea level
    cos_psi = np.cos(x[:, 11])  # cos(psi)
    V_A = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)  # airspeed
    alpha = np.arctan2(x[:, 2], x[:, 0])  # AOA
    beta = np.arcsin(x[:, 1] / V_A)  # sideslip angle
    q = 0.5 * rho * V_A**2  # dynamic pressure

    # Concatenate the original state vector with the new variables
    x_augmented = np.hstack(
        (
            x,
            cos_psi.reshape(-1, 1),
            V_A.reshape(-1, 1),
            alpha.reshape(-1, 1),
            beta.reshape(-1, 1),
            q.reshape(-1, 1),
        )
    )

    # If augment_dxdt is True, also add the time derivative of the full x state
    if augment_dxdt==True:        
        dxdt = np.gradient(x, axis=0)  # Calculate the time derivative of x
        x_augmented = np.hstack((x_augmented, dxdt))

    return x_augmented


def preprocess(x_or_u: ArrayLike, augment_dxdt=False) -> ArrayLike:
    """
        Preprocess state (x) or control (u) vector variables by scaling and augmenting.

        Args:
        x_or_u: n x 12 state array OR n x 5 control array

        Returns:
        ArrayLike: n x m state array OR n x 5 control array, where m is the number of 
        augmented state variables (17 or 29, depending on augment_dxdt)
    """
    if x_or_u.shape[1] == 5:
        # If input is control vector, scale it
        preprocessed_vector = scale_u_data(x_or_u)
    elif x_or_u.shape[1] == 12:
        # If input is state vector, augment and scale it
        preprocessed_vector = scale_x_data(augment_data(x_or_u, augment_dxdt=augment_dxdt))
    else:
        raise ValueError(
            f"Expected size of x_or_u to be 5 or 12, received size {x_or_u.size}"
        )

    return preprocessed_vector


def postprocess(x_or_u: ArrayLike) -> ArrayLike:
    """Postprocess state (x) or contrl control (u) vector variables by scaling and augmenting.

    Args:
        x_or_u (ArrayLike): n x 17 state array OR n x 5 control array

    Returns:
        ArrayLike: n x 12 state array OR n x 5 control array
    """
    if x_or_u.shape[1] == 5:
        # If input is a control vector, unscale it
        postprocessed_vector = unscale_u_data(x_or_u)
    elif x_or_u.shape[1] == 17:
        # If input is a state vector, unscale and unaugment it
        postprocessed_vector = unaugment_unscale_x_data(x_or_u)
    else:
        raise ValueError(
            f"Expected size of x_or_u to be 5 or 17, received size {x_or_u.size}"
        )

    return postprocessed_vector


if __name__ == "__main__":
    x1 = np.array([[100, 5, 8, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 100, 200, 300]])
    print(x1)
    x2 = preprocess(x1)
    print(x2)
    x3 = postprocess(x2)
    print(x3)
