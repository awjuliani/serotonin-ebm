import numpy as np
import ray
import os
from scipy.ndimage import gaussian_filter
from src.landscape import LandscapeGenerator
from src import utils
from typing import Tuple


EPSILON = 1e-8
SEED_OFFSET = 9999


def gradient_step(e, zs, step_size=1.0, noise_std=0.1):
    # Extract the gradient at the z of interest.
    z_grads = utils.gradient_at_zs(e, zs)
    # Normalize the gradient for each z.
    norms = np.linalg.norm(z_grads, axis=1, keepdims=True)
    normalized_grads = z_grads / (norms + EPSILON)
    noise = np.random.randn(*zs.shape) * noise_std
    normalized_grads += noise
    # Move against the gradient (downhill).
    zs = zs - normalized_grads * step_size

    # Ensure the z stays within the bounds of the landscape.
    zs = np.clip(zs, 0, e.shape[0] - 1).astype(int)

    return zs


def gradient_steps(z_set, e, sim_params):
    """Perform k gradient steps on z_set, track and return visited zs."""
    z_set = np.array(z_set)
    visited_zs = []
    for _ in range(sim_params.grad_step_num):
        z_set = gradient_step(
            e=e,
            zs=z_set,
            step_size=sim_params.grad_step_size,
            noise_std=sim_params.grad_noise,
        )
        visited_zs.extend([(z_set[i, 0], z_set[i, 1]) for i in range(z_set.shape[0])])
    return z_set, visited_zs


def _unpack_params_step(sim_params, dose_num):
    if isinstance(sim_params.excite_str, Tuple):
        excite_str = sim_params.excite_str[dose_num]
    else:
        excite_str = sim_params.excite_str
    if isinstance(sim_params.inhibit_str, Tuple):
        inhibit_str = sim_params.inhibit_str[dose_num]
    else:
        inhibit_str = sim_params.inhibit_str
    constraint_str = sim_params.constraint_str
    plastic_str = sim_params.plastic_str
    noise_blur = sim_params.noise_blur

    return (excite_str, inhibit_str, constraint_str, plastic_str, noise_blur)


def _unpack_params_simulate_single(sim_params, dose_num):
    if isinstance(sim_params.response_style, Tuple):
        response_style = sim_params.response_style[dose_num]
    else:
        response_style = sim_params.response_style
    warmup_steps = sim_params.warmup_steps
    drug_steps = sim_params.drug_steps
    cooldown_steps = sim_params.cooldown_steps
    if isinstance(sim_params.placebo, Tuple):
        placebo = sim_params.placebo[dose_num]
    else:
        placebo = sim_params.placebo

    return (response_style, warmup_steps, drug_steps, cooldown_steps, placebo)


def step(
    drug_level,
    seed,
    generator,
    e,
    e_star,
    e_mod,
    sim_params,
    dose_num,
    z_set,
):
    (
        excite_str,
        inhibit_str,
        constraint_str,
        plastic_str,
        noise_blur,
    ) = _unpack_params_step(sim_params=sim_params, dose_num=dose_num)

    # calculate drug effect strengths
    excite_str *= drug_level
    inhibit_str *= drug_level

    # calculate 2a (excitatory) drug effect
    e_noise = generator.generate(seed, 1, noise_blur, norm_type="neg-pos")
    e_noise_delta = 0.5 * excite_str * e_noise

    # calculate 1a (inhibitory) drug effect
    e_blur = gaussian_filter(e_mod, sigma=noise_blur * 0.5)
    e_blur_delta = inhibit_str * (e_blur - e_mod)

    # calculate homeostatic constraint delta
    e_homeo_delta = constraint_str * (e - e_mod)

    # apply drug and homeostatic effects to e_mod
    e_mod = e_mod + e_noise_delta + e_blur_delta + e_homeo_delta

    # do gradient descent on zs and track visited zs
    z_set, visited_zs = gradient_steps(z_set=z_set, e=e_mod, sim_params=sim_params)

    # calculate predictive error
    for (i, j) in visited_zs:
        error = e_star[i, j] - e[i, j]
        # apply hebbian plasticity effect to e
        e[i, j] = e[i, j] + (plastic_str * error)

    return e_mod, e, z_set


def sample_zs(e, sample_type, num_samples, seed):
    if sample_type == "uniform":
        z_set = np.random.RandomState(seed).randint(
            0, e.shape[0], size=(num_samples, 2)
        )
    elif sample_type == "density":
        # turn e into a probability distribution using softmax
        p = utils.softmax(-e.flatten())
        # sample z_set from e
        z_set = np.random.RandomState(seed).choice(
            np.arange(e.size), size=(num_samples,), p=p
        )
        z_set = np.unravel_index(z_set, e.shape)
        z_set = np.array(z_set).T
    else:
        raise ValueError("sample_type must be 'random' or 'density'")
    return z_set


def simulate(
    e,
    e_star,
    init_seed,
    sim_params,
    generator,
):
    # unpack parameters
    warmup_steps = sim_params.warmup_steps
    num_doses = sim_params.num_doses
    optim_zs = sim_params.optim_zs

    # initialize z_set
    z_set = sample_zs(e, "uniform", optim_zs, init_seed)
    zs = [z_set.copy()]

    # initialize metrics
    metrics = utils.empty_metrics()
    e_mod = e.copy()
    e_mods = [e_mod]
    update_metrics(e_mod=e_mods[0], metrics=metrics, zs=zs, e_star=e_star)

    for i in range(num_doses):
        e_mods, zs, z_set, metrics, e_mod, e = simulate_single(
            e=e,
            e_star=e_star,
            e_mod=e_mod,
            z_set=z_set,
            init_seed=init_seed,
            sim_params=sim_params,
            generator=generator,
            metrics=metrics,
            e_mods=e_mods,
            zs=zs,
            dose_num=i,
        )

    # discard unique states from dictionary
    del metrics["unique_states"]
    # remove warmup steps from metrics
    metrics = {key: value[warmup_steps:] for key, value in metrics.items()}
    return e_mods, zs, metrics


def simulate_single(
    e,
    e_star,
    e_mod,
    z_set,
    init_seed,
    sim_params,
    generator,
    metrics,
    e_mods,
    zs,
    dose_num,
):
    (
        response_style,
        warmup_steps,
        drug_steps,
        cooldown_steps,
        placebo,
    ) = _unpack_params_simulate_single(sim_params=sim_params, dose_num=dose_num)

    for i in range(1, warmup_steps + drug_steps + cooldown_steps):
        if i < warmup_steps or i > warmup_steps + drug_steps or placebo:
            drug_level = 0
        else:
            use_i = i - warmup_steps - 1
            drug_level = utils.response_curve(use_i / drug_steps, response_style)
        metrics["levels"].append(drug_level)
        e_mod, e, z_set = step(
            drug_level=drug_level,
            seed=init_seed + i,
            generator=generator,
            e=e,
            e_star=e_star,
            e_mod=e_mod,
            sim_params=sim_params,
            dose_num=dose_num,
            z_set=z_set,
        )
        e_mods.append(e_mod)
        zs.append(z_set.copy())
        update_metrics(e_mod=e_mod, metrics=metrics, zs=zs, e_star=e_star)
    return e_mods, zs, z_set, metrics, e_mod, e


def update_metrics(e_mod, metrics, zs, e_star):
    p_mod = utils.softmax(-e_mod)
    p_star = utils.softmax(-e_star)
    kl_div = utils.kl_divergence(p_star, p_mod)
    metrics["divergence"].append(kl_div)
    metrics["entropy"].append(-np.sum(p_mod * np.log(p_mod + EPSILON)))
    metrics["local_minima"].append(utils.local_minima(e_mod))
    metrics["gradient_mags"].append(utils.gradient_mag(e_mod))
    metrics["target_energy"].append(utils.mean_energy(e_star, zs[-1]))
    metrics["energy"].append(utils.mean_energy(e_mod, zs[-1]))
    metrics = utils.increment_state_count(zs[-1][0], metrics, e_mod)
    if len(metrics["divergence"]) > 1:
        div_diffs = np.diff(metrics["divergence"])
        pos_divs = np.maximum(0, div_diffs)
        metrics["div_monotonicity"].append(np.sum(pos_divs))
    else:
        metrics["div_monotonicity"].append(0.0)
    return metrics


@ray.remote
def run_sim_remote(seed, sim_params, generator):
    return run_simulation(seed, sim_params, generator)


def run_simulation(seed, sim_params, generator):
    init_seed = seed * SEED_OFFSET
    x, y = generator.x, generator.y
    e = generator.generate(
        seed=init_seed,
        scale=sim_params.init_mag,
        blur=sim_params.init_blur,
    )
    e_star = generator.generate(
        seed=init_seed + SEED_OFFSET,
        scale=sim_params.init_mag,
        blur=sim_params.init_blur,
    )
    e_mods, zs, metrics = simulate(
        e=e,
        e_star=e_star,
        init_seed=init_seed,
        sim_params=sim_params,
        generator=generator,
    )
    return x, y, e_mods, zs, metrics, e_star


def experiment(sim_params):
    generator = LandscapeGenerator(
        landscape_size=sim_params.landscape_size,
        norm_type=sim_params.norm_type,
        pattern_type=sim_params.surface_pattern,
    )
    utils.validate_sim_params(sim_params)

    if sim_params.use_ray:
        # Set RAY_verbose_spill_logs=0 to disable the spill logs
        os.environ["RAY_verbose_spill_logs"] = "0"

        # Using Ray to parallelize the simulations for different seeds
        futures = [
            run_sim_remote.remote(i, sim_params, generator)
            for i in range(sim_params.num_seeds)
        ]
        results = ray.get(futures)
    else:
        results = []
        for i in range(sim_params.num_seeds):
            results.append(run_simulation(i, sim_params, generator))

    all_metrics = {}  # To keep track of all values for each metric
    # Process the results
    for x, y, e_mods, zs, metrics, e_star in results:
        for key, value in metrics.items():
            all_metrics.setdefault(key, []).append(np.array(value))

    # Stack arrays along a new dimension for statistics calculation
    for key in all_metrics:
        all_metrics[key] = np.stack(all_metrics[key])

    avg_metrics = {}
    std_error_metrics = {}
    for key in all_metrics:
        avg_metrics[key] = np.mean(all_metrics[key], axis=0)
        variance = np.var(all_metrics[key], axis=0)
        std_error_metrics[key] = np.sqrt(variance / sim_params.num_seeds)

    return (
        all_metrics,
        avg_metrics,
        std_error_metrics,
        e_mods,
        x,
        y,
        zs,
        e_star,
    )
