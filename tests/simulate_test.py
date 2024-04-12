import numpy as np
from src import simulate
from src.landscape import LandscapeGenerator
import src.params as params

import pytest
from unittest.mock import MagicMock


simulate_test_data = [1, 2, 3]


@pytest.mark.parametrize(
    "num_doses",
    simulate_test_data,
)
def test_simulate(num_doses):
    # Tests the function calls for different num dose settings.
    init_seed = 42 * 1000
    sim_params = params.SimHyperparams(
        num_doses=num_doses,
        warmup_steps=1,
        drug_steps=1,
        cooldown_steps=1,
    )
    generator = LandscapeGenerator(
        landscape_size=sim_params.landscape_size,
        norm_type=sim_params.norm_type,
        pattern_type=sim_params.surface_pattern,
    )
    e = generator.generate(
        seed=init_seed,
        scale=sim_params.init_mag,
        blur=sim_params.init_blur,
    )
    e_star = generator.generate(
        seed=init_seed + 1,
        scale=sim_params.init_mag,
        blur=sim_params.init_blur,
    )
    metrics = {"unique_states": None}
    mock_sim_single = MagicMock(return_value=([], [], [], metrics, [], []))
    simulate.simulate_single = mock_sim_single
    simulate.simulate(
        e=e,
        init_seed=init_seed,
        sim_params=sim_params,
        generator=generator,
        e_star=e_star,
    )
    np.testing.assert_equal(mock_sim_single.call_count, num_doses)
