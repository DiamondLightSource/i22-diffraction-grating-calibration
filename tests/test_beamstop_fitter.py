import numpy as np
import pytest
from numpy.typing import NDArray

from gratingcalibration.beamstop_fitter import FitBeamstop


def _make_beamstop_image(
    shape: tuple[int, int],
    cx: float,
    cy: float,
    radius: float,
    halo_amplitude: float = 10.0,
    background: float = 2.0,
) -> NDArray[np.float64]:
    """
    a dark disc of the given radius surrounded by a bright halo just outside
    its edge, mimicking a beamstop shadow sitting in the direct beam.
    """
    yy, xx = np.indices(shape)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    image = background + halo_amplitude * np.exp(-((dist - radius) ** 2) / (2 * 3.0**2))
    image[dist <= radius] = 0.0
    return image


@pytest.mark.parametrize(
    "cx, cy, radius",
    (
        (123.0, 87.0, 12.0),
        (60.0, 140.0, 8.0),
    ),
)
def test_find_beamstop_recovers_known_center_and_radius(
    cx: float, cy: float, radius: float
) -> None:
    image = _make_beamstop_image((200, 200), cx, cy, radius)

    fbs = FitBeamstop(image, plot=False)

    assert np.isclose(fbs.beamstop_center["x"], cx, atol=0.5)
    assert np.isclose(fbs.beamstop_center["y"], cy, atol=0.5)
    assert np.isclose(fbs.beamstop_radius, radius, atol=1.0)
