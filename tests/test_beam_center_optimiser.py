from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from gratingcalibration.beam_center_optimiser import (
    BeamCenterOptimiser,
    split_azimuth_window,
    wrap_azimuth,
)
from gratingcalibration.data_loader import DataLoader


@pytest.mark.parametrize(
    "angle, expected",
    (
        (0, 0),
        (90, 90),
        (-90, -90),
        (45, 45),
        # values outside (-180, 180] should wrap around
        (190, -170),
        (-190, 170),
        (350, -10),
        (-350, 10),
        # the +/-180 seam itself wraps to -180
        (180, -180),
        (-180, -180),
    ),
)
def test_wrap_azimuth(angle: float, expected: float) -> None:
    assert np.isclose(wrap_azimuth(angle), expected)


@pytest.mark.parametrize(
    "center, half_width, expected",
    (
        # nowhere near the +/-180 seam: a single, unsplit window
        (90, 4, [(86, 94)]),
        (0, 4, [(-4, 4)]),
        (-170, 4, [(-174, -166)]),
        # straddling the seam: split into two pieces either side of it
        (180, 4, [(176, 180.0), (-180.0, -176)]),
        (182, 4, [(178, 180.0), (-180.0, -174)]),
    ),
)
def test_split_azimuth_window(
    center: float, half_width: float, expected: list[tuple[float, float]]
) -> None:
    windows = split_azimuth_window(center, half_width)
    assert len(windows) == len(expected)
    for (lo, hi), (exp_lo, exp_hi) in zip(windows, expected, strict=True):
        assert np.isclose(lo, exp_lo)
        assert np.isclose(hi, exp_hi)


def test_azimuth_offset_shifts_integration_configs() -> None:
    """
    the four integration sectors should each rotate by the same azimuth_offset
    """
    image = np.ones((10, 10))
    mask = np.zeros((10, 10))

    fitter = BeamCenterOptimiser(
        image=image,
        mask=mask,
        beam_energy=12000,
        beamstop_center={"x": 5, "y": 5},
        azimuth_offset=7.5,
    )

    assert np.isclose(fitter.INTEGRATION_CONFIGS["y0"]["center"], 97.5)
    assert np.isclose(fitter.INTEGRATION_CONFIGS["y1"]["center"], -82.5)
    assert np.isclose(fitter.INTEGRATION_CONFIGS["x0"]["center"], 7.5)
    assert np.isclose(fitter.INTEGRATION_CONFIGS["x1"]["center"], 187.5)


@pytest.mark.parametrize("azimuth_offset", (0.0, 15.0, -20.0))
def test_beam_center_optimiser_recovers_known_center(
    make_fake_file: Callable[..., Path], azimuth_offset: float
) -> None:
    """
    a radially symmetric halo has nothing directional in it, so the optimiser
    should recover the true centre regardless of azimuth_offset - the fitted
    sectors just end up sampling the same symmetric halo at a different angle.
    """
    cx, cy = 700, 200

    # small beamstop radius so it doesn't overlap the sectors' radial ranges
    fn = make_fake_file(cx=cx, cy=cy, r=15.0)
    fake_data = DataLoader(fn)

    a, b, c, d = cx - 40, cx + 40, cy + 60, cy - 60
    cropped = fake_data.data[d:c, a:b]
    cropped_mask = fake_data.mask[d:c, a:b]
    cropped_center = {"x": cx - a, "y": cy - d}

    fitter = BeamCenterOptimiser(
        image=cropped,
        mask=cropped_mask,
        beam_energy=fake_data.energy,
        beamstop_center=cropped_center,
        optimise_direction="x",
        offset={"x": a, "y": d},
        azimuth_offset=azimuth_offset,
    )
    fitter.fit_beam_centre()
    fitter.optimise_direction = "y"
    fitter.fit_beam_centre()

    assert np.isclose(fitter.beam_center_global["x"], cx, atol=1.0)
    assert np.isclose(fitter.beam_center_global["y"], cy, atol=1.0)
