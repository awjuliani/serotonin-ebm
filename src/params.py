from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class SimHyperparams:
    # NOTE: placebo, excite_str, inhibit_str, response_style can either be specified
    # by a single float, in which case the value will be applied to all doses.
    # Or, a value per dose can be specified in a tuple.
    # In this case a value per dose needs to be given.
    use_ray: bool = True  # Whether to use ray for parallelization
    init_seed: int = 0  # Initial seed for reproducibility
    num_seeds: int = 100  # Number of seeds to run per experiment
    warmup_steps: int = 50  # Number of steps to run before applying drug
    drug_steps: int = 200  # Number of steps to run after applying drug
    cooldown_steps: int = 150  # Number of steps to run after drug wears off
    placebo: Union[bool, Tuple[bool]] = False  # Whether to run a placebo experiment
    landscape_size: int = 50  # Size of optimization landscape
    init_mag: int = 3  # Initial magnitude of landscape height
    init_blur: int = 5  # Initial gaussian blur applied to landscape
    noise_mag: int = 3  # Magnitude of unnormalized noise applied to landscape
    noise_blur: int = 5  # Gaussian blur applied to noise
    noise_std: float = 0.5  # Standard deviation of noise applied to landscape
    excite_str: Union[float, Tuple[float]] = 0.01  # Strength of noise applied
    inhibit_str: Union[float, Tuple[float]] = 0.0  # Mean strength of smoothing applied
    grad_step_size: float = 1.0  # Gradient step size for optimization
    grad_step_num: int = 10  # Number of gradient steps per step
    grad_noise: float = 0.05  # Noise applied to gradient steps
    optim_zs: int = 100  # Number of zs used for energy calculation
    norm_type: str = "min-max"  # Type of normalization applied to landscape
    constraint_str: str = 0.05  # Strength of homeostatic constraint term
    plastic_str: float = 0.5  # Strength of hebbian plasticity term
    response_style: Union[str, Tuple[str]] = "beta-long"  # Response function applied
    surface_pattern: str = "noise"  # Type of surface pattern
    num_doses: int = 1  # Number of doses to apply sequentially
    verbose: bool = False  # Whether to print verbosely during experiments.


@dataclass
class PlotHyperparams:
    generate: bool = False  # Whether to generate plots
    plot_type: str = "2d"  # "2d" or "3d"
    output_type: str = "gif"  # "gif" or "pdf"
    plot_style: str = "a"  # "a" "b" or "c"
    show_points: bool = True  # Whether to show points on plot
    plot_freq: int = 2  # Frequency of plotting
    minimal_timeseries: bool = True  # Whether to plot minimal timeseries
