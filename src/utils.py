import numpy as np
from typing import Tuple
from scipy.stats import beta


def validate_sim_params(sim_params):
    if (
        isinstance(sim_params.excite_str, Tuple)
        and len(sim_params.excite_str) != sim_params.num_doses
    ):
        raise ValueError(
            f"Number of excite string values {sim_params.excite_str} "
            "must match num doses: {sim_params.num_doses}."
        )
    elif isinstance(sim_params.excite_str, Tuple) and sim_params.verbose:
        print(f"Custom regime of {sim_params.excite_str} for excite_str")
    elif sim_params.verbose:
        print(f"Fixed regime of {sim_params.excite_str} for excite_str")

    if (
        isinstance(sim_params.inhibit_str, Tuple)
        and len(sim_params.inhibit_str) != sim_params.num_doses
    ):
        raise ValueError(
            f"Number of inhibit string values {sim_params.inhibit_str} "
            "must match num doses: {sim_params.num_doses}."
        )
    elif isinstance(sim_params.inhibit_str, Tuple) and sim_params.verbose:
        print(f"Custom regime of {sim_params.inhibit_str} for inhibit_str")
    elif sim_params.verbose:
        print(f"Fixed regime of {sim_params.inhibit_str} for inhibit_str")

    if (
        isinstance(sim_params.response_style, Tuple)
        and len(sim_params.response_style) != sim_params.num_doses
    ):
        raise ValueError(
            f"Number of sim_params.response_style values {sim_params.response_style} "
            "must match num doses: {sim_params.num_doses}."
        )
    elif isinstance(sim_params.response_style, Tuple) and sim_params.verbose:
        print(f"Custom regime of {sim_params.response_style} for response_style")
    elif sim_params.verbose:
        print(f"Fixed regime of {sim_params.response_style} for response_style")

    if (
        isinstance(sim_params.placebo, Tuple)
        and len(sim_params.placebo) != sim_params.num_doses
    ):
        raise ValueError(
            f"Number of placebo values {sim_params.placebo} "
            "must match num doses: {sim_params.num_doses}."
        )
    elif isinstance(sim_params.placebo, Tuple) and sim_params.verbose:
        print(f"Custom regime of {sim_params.placebo} for placebo")
    elif sim_params.verbose:
        print(f"Fixed regime of {sim_params.placebo} for placebo")


def mean_energy(e, zs):
    return np.mean(e[zs[:, 0], zs[:, 1]])


def gradient_at_zs(e, zs):
    shape = e.shape
    i, j = zs[:, 0], zs[:, 1]

    # Handle boundaries for finite differences
    im1 = np.maximum(i - 1, 0)
    ip1 = np.minimum(i + 1, shape[0] - 1)
    jm1 = np.maximum(j - 1, 0)
    jp1 = np.minimum(j + 1, shape[1] - 1)

    # Central difference method
    grad_x = (e[ip1, j] - e[im1, j]) / 2
    grad_y = (e[i, jp1] - e[i, jm1]) / 2

    return np.stack((grad_x, grad_y), axis=-1)


def response_curve(t, style="placebo"):
    if style == "u-curve":
        return 1 - (t - 0.5) ** 2 * 4
    elif style == "beta-long":
        return beta.pdf(t, 2, 5) / 2.5
    elif style == "beta-short":
        return beta.pdf(t, 2, 20) / 3.5
    elif style == "placebo":
        return 0
    elif style == "full":
        return 1
    else:
        raise ValueError(f"Unknown style: {style}")


def empty_metrics():
    return {
        "levels": [],
        "divergence": [],
        "div_monotonicity": [],
        "local_minima": [],
        "gradient_mags": [],
        "state_counts": [0],
        "energy": [],
        "target_energy": [],
        "unique_states": [],
        "entropy": [],
        "minima_visited": [0],
    }


def increment_state_count(z, metrics, e):
    contains = any(np.array_equal(arr, z) for arr in metrics["unique_states"])
    minima_check = local_minima(e, z)

    if not contains:
        metrics["unique_states"].append(z)

    metrics["state_counts"].append(metrics["state_counts"][-1] + (not contains))
    metrics["minima_visited"].append(
        metrics["minima_visited"][-1] + (minima_check > 0 and not contains)
    )

    return metrics


def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / exp.sum()


def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler divergence between two probability distributions.

    Parameters:
    - p, q: numpy arrays representing the two probability distributions. Both p and q must be the same shape.

    Returns:
    - KL divergence value.
    """
    # Ensure that neither p nor q has zero elements, as log(0) is undefined
    p = np.clip(p, 1e-15, 1)
    q = np.clip(q, 1e-15, 1)

    # Calculate the KL divergence
    kl_div = np.sum(p * np.log(p / q))

    return kl_div


def gradient_mag(e):
    """
    Compute the gradient of e and return its average magnitude.
    """
    grad_x, grad_y = np.gradient(e)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude.mean()


def local_minima(e, z=None):
    """
    Calculate the number of local minima in a given matrix e using a vectorized approach.
    """
    # ensure e is a float array
    e = e.astype(float)

    # Create an empty padded matrix with +inf values to avoid boundary issues
    padded_z = np.pad(e, pad_width=1, mode="constant", constant_values=np.inf)

    # Check if each element is smaller than its 8 neighbors using slicing
    is_min = (
        (padded_z[1:-1, 1:-1] < padded_z[:-2, :-2])
        & (padded_z[1:-1, 1:-1] < padded_z[:-2, 1:-1])  # top-left
        & (padded_z[1:-1, 1:-1] < padded_z[:-2, 2:])  # top-center
        & (padded_z[1:-1, 1:-1] < padded_z[1:-1, :-2])  # top-right
        & (padded_z[1:-1, 1:-1] < padded_z[1:-1, 2:])  # center-left
        & (padded_z[1:-1, 1:-1] < padded_z[2:, :-2])  # center-right
        & (padded_z[1:-1, 1:-1] < padded_z[2:, 1:-1])  # bottom-left
        & (padded_z[1:-1, 1:-1] < padded_z[2:, 2:])  # bottom-center  # bottom-right
    )

    if z is None:
        # Return the number of local minima
        return np.sum(is_min)
    else:
        # return whether there is a local minima at point theta
        return is_min[z[0], z[1]] * 1.0
