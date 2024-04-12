import numpy as np
import pytest
from src.landscape import LandscapeGenerator


def test_initialization():
    lg = LandscapeGenerator(100, "min-max")
    assert lg.landscape_size == 100
    assert lg.default_norm_type == "min-max"
    assert lg.pattern_type == "sine"
    assert lg.x.shape == (100, 100)
    assert lg.y.shape == (100, 100)


@pytest.mark.parametrize(
    "norm_type,expected",
    [
        ("min-max", (0, 1)),
        ("bound", (0, 1)),
        ("z-score", (0, 1)),
    ],
)
def test_normalize(norm_type, expected):
    lg = LandscapeGenerator(100, "none")
    e = np.random.randn(5, 5)
    norm_z = lg.normalize(e, norm_type)
    if norm_type == "z-score":
        assert np.isclose(np.mean(norm_z), 0, atol=1e-5)
        assert np.isclose(np.std(norm_z), 1, atol=1e-5)
    else:
        assert np.min(norm_z) == expected[0]
        assert np.max(norm_z) == expected[1]


def test_normalize_error():
    lg = LandscapeGenerator(100, "none")
    e = np.random.randn(5, 5)
    with pytest.raises(ValueError, match="Unknown norm_type"):
        lg.normalize(e, "invalid_norm_type")


@pytest.mark.parametrize(
    "pattern_type,add_noise",
    [
        ("noise", True),
        ("sine", True),
        ("cosine", True),
        ("tangent", True),
        ("exponent", True),
        ("log", True),
        ("parabolic", True),
        ("twisted_sine", True),
        ("swirl", True),
        ("noise", False),
        ("sine", False),
        ("cosine", False),
        ("tangent", False),
        ("exponent", False),
        ("log", False),
        ("parabolic", False),
        ("twisted_sine", False),
        ("swirl", False),
    ],
)
def test_pattern(pattern_type, add_noise):
    lg = LandscapeGenerator(100, "none", pattern_type)
    rng = np.random.RandomState(0)
    e = lg.pattern(rng, add_noise=add_noise)
    assert e.shape == (100, 100)


def test_pattern_error():
    lg = LandscapeGenerator(100, "none", "invalid_pattern")
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError, match="Unknown pattern_type"):
        lg.pattern(rng)


def test_generate():
    lg = LandscapeGenerator(100, "min-max")
    e = lg.generate(seed=42, scale=2, blur=1, normalize=True)
    assert e.shape == (100, 100)
    assert np.min(e) == 0
    assert np.max(e) == 1


def test_resize():
    lg = LandscapeGenerator(500, "min-max", "sine", base_resolution=100)
    e = lg.generate(seed=42, scale=1, blur=10, normalize=True)
    assert e.shape == (500, 500)
    lg2 = LandscapeGenerator(200, "min-max", "sine", base_resolution=100)
    e2 = lg2.generate(seed=42, scale=1, blur=10, normalize=True)
    assert e2.shape == (200, 200)
