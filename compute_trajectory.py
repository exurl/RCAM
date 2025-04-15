import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
from typing import Any, Callable
from pathlib import Path

from rcam import rcam_system

N_PLANT_STATES = 12
N_ACTUATOR_STATES = 9
N_CONTROLS = 5


def simulate_trajectory(
    f: Callable[[ArrayLike, ArrayLike], ArrayLike],
    x0: ArrayLike,
    t_in: ArrayLike,
    u_in: ArrayLike,
    t_out: ArrayLike = None,
) -> ArrayLike:
    """Simulate trajectory of dynamical system.

    Args:
        f (callable[[ArrayLike, ArrayLike], ArrayLike]):
            System dynamics x_dot = f(x, u).
        x0 (ArrayLike): (L) initial state array.
        t_in (ArrayLike): (M) input time array.
        u_in (ArrayLike): (M) control array corresponding to t_in.
        t_out (ArrayLike, optional): (N) output time array. Defaults to None.

    Returns:
        ArrayLike: (N)x(L) array of states corresponding to t_out.
    """

    # Define ODE
    def ode_function(t, x):
        u = np.array([np.interp(t, t_in, u_in_i) for u_in_i in u_in.T])
        return f(x, u)

    # Integrate
    solution = solve_ivp(
        fun=ode_function,
        t_span=(t_out[0], t_out[-1]),
        y0=x0,
        t_eval=t_out,
        method="RK45",
    )
    assert solution.success, f"Integration failed: {solution.message}"

    return solution.t, solution.y.T


def generate_output_data(
    input_data: dict[Any, ArrayLike],
    trim_data: dict[Any, ArrayLike],
) -> dict[tuple[Any, Any], ArrayLike]:
    """Generate output data for each input + trim combination

    Args:
        input_data (Dict[Any, ArrayLike]): keys are input IDs and values are (L)x(M) arrays with
        time in first column and control DOFs in remaining columns
        trim_data (Dict[Any, ArrayLike]): keys are trim IDs and values are (M+N) arrays

    Returns:
        Dict[tuple(any, any), ArrayLike]: keys are (input_ID, trim_ID) and values are (N) arrays
                               values are numpy arrays containing the integrated state
    """
    output_data = {}

    # Fetch trim
    for trim_ID, trim in trim_data.items():
        x0 = np.concatenate(
            (trim[:N_PLANT_STATES], np.zeros(N_ACTUATOR_STATES))
        )
        u0 = trim[-N_CONTROLS:]

        # Fetch input
        for input_ID, input_trajectory in input_data.items():
            t_in = input_trajectory[:, 0]
            u_in = input_trajectory[:, 1:] + u0

            # Simulate
            t_out, x_out = simulate_trajectory(
                f=rcam_system, x0=x0, t_in=t_in, u_in=u_in, t_out=t_in
            )
            output_data[(trim_ID, input_ID)] = x_out

    return output_data


if __name__ == "__main__":
    # Define file paths
    input_file_path = Path("input_data.npy")
    trim_file_path = Path("trim_data.npy")
    output_file_path = Path("output_data.npy")

    # Load input data
    input_data = np.load(input_file_path, allow_pickle=True).item()
    input_data_DOFs = next(iter(input_data.values())).shape[1] - 1
    assert (
        input_data_DOFs == N_CONTROLS
    ), f"Input data should have {N_CONTROLS} DOFs, not {input_data_DOFs}"
    print("Loaded input data.")

    # Load trim data
    trim_data = np.load(trim_file_path, allow_pickle=True).item()
    trim_data_DOFs = len(next(iter(trim_data.values())))
    assert (
        trim_data_DOFs == N_PLANT_STATES + N_CONTROLS
    ), f"Trim data should have {N_PLANT_STATES + N_CONTROLS} DOFs, not {trim_data_DOFs}"
    print("Loaded trim data.")

    # Generate output data
    output_data = generate_output_data(input_data, trim_data)
    print("Generated output data.")

    # Save output data
    np.save(output_file_path, {"output_data": output_data})
    print(f"Saved output data to {output_file_path}")
