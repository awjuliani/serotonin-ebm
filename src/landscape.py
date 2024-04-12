import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp2d


class LandscapeGenerator:
    def __init__(
        self, landscape_size, norm_type, pattern_type="sine", base_resolution=100
    ):
        self.landscape_size = landscape_size
        self.default_norm_type = norm_type
        self.pattern_type = pattern_type
        x = np.linspace(-1, 1, landscape_size)
        y = np.linspace(-1, 1, landscape_size)
        self.x, self.y = np.meshgrid(x, y)
        self.base_resolution = base_resolution
        self.base_scale = landscape_size / base_resolution

    def normalize(self, e, norm_type=None):
        if norm_type is None:
            norm_type = self.default_norm_type
        if norm_type == "min-max":
            # normalize e to have a min of 0 and a max of 1
            e = (e - np.min(e)) / (np.max(e) - np.min(e))
        elif norm_type == "bound":
            e = np.clip(e, 0, 1)
        elif norm_type == "z-score":
            # normalize e to have a mean of 0 and a std of 1
            e = (e - np.mean(e)) / np.std(e)
        elif norm_type == "neg-pos":
            # normalize e to have a min of -1 and a max of 1
            e = self.normalize(e, "min-max")
            e = 2 * e - 1
        elif norm_type == "none":
            pass
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
        return e

    def get_noise(self, rng, add_noise):
        """Return either random noise or a ones array based on the add_noise flag, resampled to landscape_size."""
        # Determine base matrix based on add_noise flag
        br = (self.base_resolution, self.base_resolution)
        base = rng.randn(*br) if add_noise else np.ones(br)

        # Shortcut when no resizing is needed
        if self.base_resolution == self.landscape_size:
            return base

        # Setup interpolation and resample in one step
        f = interp2d(
            np.linspace(0, 1, br[0]), np.linspace(0, 1, br[1]), base, kind="linear"
        )
        resampled_matrix = f(
            np.linspace(0, 1, self.landscape_size),
            np.linspace(0, 1, self.landscape_size),
        )

        return resampled_matrix

    def pattern(self, rng, add_noise=True):
        # Single dictionary for patterns
        patterns = {
            "noise": lambda noise: noise,
            "sine": lambda noise: noise * np.sin(np.sqrt(self.x**2 + self.y**2)),
            "cosine": lambda noise: noise * np.cos(np.sqrt(self.x**2 + self.y**2)),
            "tangent": lambda noise: noise * np.tan(np.sqrt(self.x**2 + self.y**2)),
            "exponent": lambda noise: noise * np.exp(-1 * (self.x**2 + self.y**2)),
            "log": lambda noise: noise * np.log(np.sqrt(self.x**2 + self.y**2)),
            "twisted_sine": lambda noise: noise * np.sin(self.x) * np.cos(self.y),
            "swirl": lambda noise: noise * np.sin(self.x**2 - self.y**2),
            "parabolic": lambda noise: 5 * noise + (self.x**2 + self.y**2),
        }

        if self.pattern_type not in patterns:
            raise ValueError(f"Unknown pattern_type: {self.pattern_type}")

        noise = self.get_noise(rng, add_noise)
        return patterns[self.pattern_type](noise)

    def generate(
        self, seed, scale, blur, normalize=True, add_noise=True, norm_type=None
    ):
        rng = np.random.RandomState(seed)
        e = self.pattern(rng, add_noise=add_noise)
        e = gaussian_filter(e, sigma=blur * self.base_scale) * scale
        if normalize:
            e = self.normalize(e, norm_type)
        return e
