from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from gratingcalibration.data_loader import DataLoader
from gratingcalibration.detector_calibration import (
    DetectorCalibration,
    determine_pattern_angle,
    find_calibration_azimuth,
    find_multiple_sequence,
    robust_noise_floor,
)

from .helper_functions import make_fit_store


@pytest.mark.parametrize(
    "cx, cy, spots, radius",
    (
        # somewhere sensible
        (700, 200, (100, 200, 300, 400, 500, 600, 700, 820), 50),
        # somewhere different
        (100, 100, (100, 200, 300, 400, 500, 600, 700, 820), 75),
        # beamstop is overlapping the spots,
        (100, 100, (100, 200, 300, 400, 500, 600, 700, 820), 150),
    ),
)
def test_radial_range(
    make_fake_file: Callable[..., Path],
    cx: int,
    cy: int,
    spots: tuple[int, ...],
    radius: int,
) -> None:
    """
    test that we pick up the radial range correctly
    """

    fn = make_fake_file(
        cx=cx,
        cy=cy,
        r=radius,
        spot_rows=spots,
        background_intensity=5.0,
        spot_intensity=10.0,
    )
    fake_data = DataLoader(fn)

    dc = DetectorCalibration(
        image=fake_data.data,
        beam_center={"x": cx, "y": cy},
        wavelength=1e-10,  # this doesn't actually matter
    )

    lims = dc._determine_radial_range()  # pyright: ignore[reportPrivateUsage]
    lower, upper = [i / dc.PILATUS2M_PIXEL_SIZE for i in lims]

    print(lower, radius)
    # expect the lower limit to be ~ the same position as the beamstop edge.
    assert np.isclose(lower, radius, atol=5)
    # expect the upper limit to be within aroud 50 points of the last peak.
    print(upper, spots[-1])
    assert np.isclose(upper, spots[-1], atol=50)


@pytest.mark.parametrize(
    "input, output",
    (
        # input and output are the same
        (
            np.arange(10),
            np.arange(10),
        ),
        # input contains one outlier that should be excluded
        (np.array([0, 1, 2, 3, 4, 5.5]), np.array([0, 1, 2, 3, 4])),
        # input is non sequential, output should identify and
        # relabel with correct index
        (np.array([0, 1, 2, 4, 5, 6]), np.array([0, 1, 2, 4, 5, 6])),
    ),
)
def test_find_multiple_sequence(input: NDArray[Any], output: NDArray[Any]) -> None:
    """
    test that sequences get identified correctly
    """
    fit_store = find_multiple_sequence(make_fit_store(input))
    result = np.array([i["center"] for i in fit_store.values()])
    assert np.all(np.isclose(result, output))


@pytest.mark.parametrize("npt, radial_range", (((500, None), (250, (0.005, 0.01)))))
def test_azi_integration(
    make_fake_file: Callable[..., Path],
    npt: int,
    radial_range: tuple[float, float] | None,
) -> None:
    """
    test that we pick up the radial range correctly
    """

    fn = make_fake_file(
        cx=700,
        cy=200,
    )
    fake_data = DataLoader(fn)

    dc = DetectorCalibration(
        image=fake_data.data,
        beam_center={"x": 700, "y": 200},
        wavelength=1e-10,  # this doesn't actually matter
    )

    signal_arr = dc.make_signal(npt=npt, auto_radial=radial_range)

    # make sure we get out the right number of points
    assert signal_arr.shape == (2, npt)
    # make sure we get the radial range we're expecting
    if radial_range is not None:
        assert np.all(
            np.isclose(
                np.array([signal_arr[0][0], signal_arr[0][-1]]),
                np.array(radial_range),
                atol=1e-5,
            )
        )


def test_detector_distance() -> None:
    dc = DetectorCalibration(
        image=None,
        beam_center=None,
        wavelength=1,  # this doesn't actually matter
        grating_spacing=1,  # neither does this
    )

    dc.peaks = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]).T

    dc.calculate_detector_distance()
    assert dc.detector_distance is not None
    assert np.isclose(dc.detector_distance, 1)


def test_find_multiple_sequence_prefers_larger_fundamental() -> None:
    """
    peaks at these positions could be explained either by a fundamental
    spacing of ~0.0059 (indices 1, 2, 4, 5) or, equally well numerically, by
    half that spacing with every index doubled (2, 4, 8, 10). The larger,
    simpler fundamental should be preferred.
    """
    peaks = np.array([0.0059, 0.0116, 0.0233, 0.029])

    result = find_multiple_sequence(make_fit_store(peaks))

    assert sorted(result.keys()) == [1, 2, 4, 5]


@pytest.mark.parametrize(
    "profile, expect_zero",
    (
        # perfectly flat: no point-to-point noise at all
        (np.full(50, 5.0), True),
        # smooth trend, no noise: consecutive differences are ~constant
        (np.linspace(0, 10, 100), True),
        # trend plus genuine point-to-point noise
        (
            np.linspace(0, 10, 100)
            + np.random.default_rng(42).normal(0, 0.3, size=100),
            False,
        ),
    ),
)
def test_robust_noise_floor(profile: NDArray[np.float64], expect_zero: bool) -> None:
    floor = robust_noise_floor(profile)

    assert floor >= 0
    if expect_zero:
        assert np.isclose(floor, 0, atol=1e-8)
    else:
        assert floor > 0.1


def test_peak_fitter_finds_expected_peaks(
    make_fake_file: Callable[..., Path],
) -> None:
    """
    the SNR-based prominence filter shouldn't reject genuine, strong peaks
    """
    cx, cy = 700, 200
    spot_rows = (100, 200, 300, 400, 500, 600, 700, 820)

    fn = make_fake_file(
        cx=cx,
        cy=cy,
        spot_rows=spot_rows,
        background_intensity=5.0,
        spot_intensity=10.0,
    )
    fake_data = DataLoader(fn)

    dc = DetectorCalibration(
        image=fake_data.data, beam_center={"x": cx, "y": cy}, wavelength=1e-10
    )
    peaks = dc.peak_fitter()

    # the first several orders should all have been found and correctly
    # indexed (the very last one can be clipped by the auto radial range)
    assert set(range(1, 6)).issubset(set(peaks[:, 0]))


def _make_rotated_spots_image(
    shape: tuple[int, int],
    cx: float,
    cy: float,
    radius: float,
    theta: float,
    spot_sigma: float = 2.0,
    amplitude: float = 20.0,
) -> NDArray[np.float64]:
    """
    a beam centre surrounded by four spots (mimicking a grating's two
    perpendicular fringe families) at radius `radius`, rotated by `theta`
    degrees away from the detector's vertical/horizontal axes.
    """
    yy, xx = np.indices(shape)
    image = np.full(shape, 3.0)
    for base_angle in (90, -90, 0, 180):
        angle_rad = np.deg2rad(base_angle + theta)
        sx = cx + radius * np.cos(angle_rad)
        sy = cy + radius * np.sin(angle_rad)
        dist2 = (xx - sx) ** 2 + (yy - sy) ** 2
        image += amplitude * np.exp(-dist2 / (2 * spot_sigma**2))
    return image


@pytest.mark.parametrize("theta", (0.0, 10.0, -15.0, 30.0))
def test_determine_pattern_angle_recovers_known_rotation(theta: float) -> None:
    cx, cy = 150.0, 150.0
    image = _make_rotated_spots_image((300, 300), cx, cy, radius=25, theta=theta)

    estimated = determine_pattern_angle(image, {"x": cx, "y": cy}, pixel_size=172e-6)

    assert np.isclose(estimated, theta, atol=1.5)


def _make_two_family_image(
    shape: tuple[int, int],
    cx: float,
    cy: float,
    theta: float,
    strong_axis: str,
    n_orders: int = 6,
    spacing: float = 15,
    spot_sigma: float = 2.0,
    amplitude: float = 15.0,
) -> NDArray[np.float64]:
    """
    a grating-like pattern with a genuine, multi-order fringe sequence along
    one axis and only a single, non-extended order along the perpendicular
    one - i.e. only one axis is actually useful for calibration.
    """
    yy, xx = np.indices(shape)
    image = np.full(shape, 3.0)
    strong_angles = (90, -90) if strong_axis == "vertical" else (0, 180)
    weak_angles = (0, 180) if strong_axis == "vertical" else (90, -90)

    for base_angle in strong_angles:
        angle_rad = np.deg2rad(base_angle + theta)
        for order in range(1, n_orders + 1):
            r = order * spacing
            sx = cx + r * np.cos(angle_rad)
            sy = cy + r * np.sin(angle_rad)
            dist2 = (xx - sx) ** 2 + (yy - sy) ** 2
            image += amplitude * np.exp(-dist2 / (2 * spot_sigma**2))

    for base_angle in weak_angles:
        angle_rad = np.deg2rad(base_angle + theta)
        r = spacing
        sx = cx + r * np.cos(angle_rad)
        sy = cy + r * np.sin(angle_rad)
        dist2 = (xx - sx) ** 2 + (yy - sy) ** 2
        image += amplitude * np.exp(-dist2 / (2 * spot_sigma**2))

    return image


@pytest.mark.parametrize(
    "strong_axis, theta",
    (
        ("vertical", 0.0),
        ("horizontal", 0.0),
        ("vertical", 5.0),
    ),
)
def test_find_calibration_azimuth_picks_stronger_direction(
    strong_axis: str, theta: float
) -> None:
    cx, cy = 150.0, 150.0
    image = _make_two_family_image(
        (300, 300), cx, cy, theta=theta, strong_axis=strong_axis
    )

    chosen = find_calibration_azimuth(
        image, {"x": cx, "y": cy}, pixel_size=172e-6, theta=theta
    )

    expected = (90 + theta) if strong_axis == "vertical" else theta
    assert np.isclose(chosen, expected, atol=1e-6)
