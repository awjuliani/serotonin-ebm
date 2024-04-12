import functools
import numpy as np
import pytest
from unittest.mock import MagicMock

from src import utils
import src.params as params

valid_params_test_data = [
    (1, "beta-long", 0.01, 0.0, False),
    (2, "beta-long", 0.01, 0.0, False),
    (2, "beta-long", 0.01, 0.0, [False, True]),
    (3, "beta-long", 0.01, 0.0, False),
    (
        3,
        ["beta-long", "beta-long", "beta-short"],
        [0.01, 0.0, 0.1],
        [0.0, 0.1, 0.2],
        [False, True, False],
    ),
]
invalid_params_test_data = [
    (1, "beta-long", 0.01, (0.0, 0.1), False),
    (2, "beta-long", (0.01,), 0.0, False),
    (2, "beta-long", (0.01,), 0.0, (False, False, True)),
    (2, ("beta-long",), 0.01, 0.0, (False, True)),
    (2, "beta-long", (0.01,) * 3, 0.0, (False, True)),
    (3, ("beta-long",) * 4, 0.01, 0.0, False),
    (3, "beta-long", (0.01,) * 4, 0.0, False),
    (3, "beta-long", 0.01, (0.0,) * 4, False),
    (3, "beta-long", 0.01, 0.0, (False,) * 4),
    (
        2,
        ("beta-long", "beta-long", "beta-short"),
        (0.01, 0.0, 0.1),
        (0.0, 0.1, 0.2),
        (False, False),
    ),
]


@pytest.mark.parametrize(
    "num_doses,response_style,excite_str,inhibit_str,placebo",
    valid_params_test_data,
)
def test_valid_sim_params(num_doses, response_style, excite_str, inhibit_str, placebo):
    sim_params = params.SimHyperparams(
        num_doses=num_doses,
        response_style=response_style,
        excite_str=excite_str,
        inhibit_str=inhibit_str,
        placebo=placebo,
    )
    _ = utils.validate_sim_params(sim_params)


@pytest.mark.parametrize(
    "num_doses,response_style,excite_str,inhibit_str,placebo",
    invalid_params_test_data,
)
def test_invalid_sim_params(
    num_doses, response_style, excite_str, inhibit_str, placebo
):
    sim_params = params.SimHyperparams(
        num_doses=num_doses,
        response_style=response_style,
        excite_str=excite_str,
        inhibit_str=inhibit_str,
        placebo=placebo,
    )
    fn = functools.partial(utils.validate_sim_params, sim_params)
    np.testing.assert_raises(ValueError, fn)


def test_mean_energy():
    e = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    zs = np.array([[0, 0], [1, 1], [2, 2]])
    assert utils.mean_energy(e, zs) == 5.0


def test_gradient_at_zs():
    e = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    zs = np.array([[0, 0], [1, 1], [2, 2]])
    assert utils.gradient_at_zs(e, zs).all() == 1.0


def test_softmax():
    x = np.array([1, 2, 3])
    np.testing.assert_allclose(
        utils.softmax(x), np.array([0.09003057, 0.24472847, 0.66524096])
    )


def test_kl_divergence():
    p = np.array([0.1, 0.2, 0.7])
    q = np.array([0.1, 0.2, 0.7])
    np.testing.assert_allclose(utils.kl_divergence(p, q), 0.0)


def test_gradient_mag():
    e = np.array([[1, 2], [2, 3]])
    mag = utils.gradient_mag(e)
    assert 0 <= mag, "Gradient magnitude should be non-negative."


def test_local_minima_count():
    e = np.array([[5, 3, 6], [4, 2, 7], [8, 9, 3]])
    assert utils.local_minima(e) == 1, "There's only one local minimum (2)."
