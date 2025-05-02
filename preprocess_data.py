import numpy as np

def scale_x_data(x):
    '''
    Scale state vector variables x by fixed amount, such that scale is appropriate 
    for use as training data for a neural network.

    EXCEPTION: Euler yaw angle is converted to sine(psi) to avoid 
    discontinuity issues.

    Inputs are:
    x: n x 12 Numpy array

    Outpus are:
    x_scaled: n x 12 Numpy array
    '''
    
    # State vector scale factors
    scale_vector_x = np.array([
        1 / 1e2,  # u
        1 / 40,  # v
        1 / 40,  # w
        1 / 0.6,  # p
        1 / 0.8,  # q
        1 / 0.3,  # r
        1 / np.radians(45),  # phi
        1 / np.radians(20),  # theta
        1 / np.pi,  # psi
        1 / 1e3,  # PN
        1 / 1e3,  # PE
        1 / 1e3   # PD
    ])

    x_scaled = x @ np.diag(scale_vector_x)

    # Convert Euler yaw angle to sine of yaw angle
    x_scaled[:, 8] = np.sin(x_scaled[:, 8])

    return x_scaled

def scale_u_data(u):
    '''
    Scale control vector variables u by fixed amount, such that scale is appropriate 
    for use as training data for a neural network.

    Inputs are:
    u: n x 5 Numpy array

    Outpus are:
    u_scaled: n x 5 Numpy array
    '''

    # Control vector scale factors
    scale_vector_u = np.array([
        1 / np.radians(25),  # d_A
        1 / np.radians(25),  # d_T
        1 / np.radians(30),  # d_R
        1 / np.radians(10),  # d_th1
        1 / np.radians(10)   # d_th2
    ])

    u_scaled = u @ np.diag(scale_vector_u)

    return u_scaled

def unscale_u_data(u):
    '''
    Unscale control vector variables u by fixed amount, such that scale is restored to 
    native RCAM model scale.

    Inputs are:
    u: n x 5 Numpy array

    Outputs are:
    u_unscaled: n x 5 Numpy array
    '''
    
    # Control vector scale factors
    scale_vector_u = np.array([
        1 * np.radians(25),  # d_A
        1 * np.radians(25),  # d_T
        1 * np.radians(30),  # d_R
        1 * np.radians(10),  # d_th1
        1 * np.radians(10)   # d_th2
    ])

    u_unscaled = u @ np.diag(scale_vector_u)

    return u_unscaled

def augment_data(x):
    '''
    Augment state vector variables x by adding relevant virtual variables including:
    - total airspeed
    - angle of attack
    - sideslip angle
    - dynamic pressure

    Inputs are:
    x: n x 12 Numpy array

    Outputs are:
    x_augmented: n x 16 Numpy array
    '''
    rho = 1.225  # air density at sea level
    V_A = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)  # airspeed
    alpha = np.arctan2(x[:, 2], x[:, 0])  # AOA
    beta = np.arcsin(x[:, 1] / V_A)  # sideslip angle
    q = 0.5 * rho * V_A**2  # dynamic pressure

    # Concatenate the original state vector with the new variables
    x_augmented = np.hstack((
        x,
        V_A.reshape(-1, 1),
        alpha.reshape(-1, 1),
        beta.reshape(-1, 1),
        q.reshape(-1, 1)
    ))

    return x_augmented

def preprocess(x_or_u):
    '''
    Preprocess state (x) or control (u) vector variables by scaling and augmenting.

    Inputs are:
    x: n x 12 Numpy array
        OR
    u: n x 5 Numpy array

    Outputs are:
    x_scaled: n x 16 Numpy array
        OR
    u_scaled: n x 5 Numpy array
    '''
    if x_or_u.shape[1] == 5:
        # If input is control vector, scale it
        u = x_or_u
        u_scaled = scale_u_data(u)
        preprocessed_vector = u_scaled
    else:
        # If input is state vector, scale and augment it
        x = x_or_u
        x_scaled = scale_x_data(x)
        x_augmented = augment_data(x_scaled)
        preprocessed_vector = x_augmented

    return preprocessed_vector
