"""
Generate a synthetic diffraction-grating pattern as a .nxs file that can be
fed straight into ``i22-diffraction-grating-calibration --file ...``.

The detector geometry (shape, pixel size, beam energy, beam centre, module
gaps) is fixed to match the geometry used in .example_data/first and
.example_data/second. Only the sample-to-detector distance and the
in-plane rotation ("pattern angle") of the grating pattern are varied, via
the --distance and --pattern-angle arguments.

Usage:
    python scripts/generate_synthetic_data.py --distance 6.5 --pattern-angle 6.0

Validated against the real CLI (i22-diffraction-grating-calibration --file
...) across many (distance, pattern-angle, --seed) combinations, this
reliably recovers the input distance and pattern angle for distances from a
few metres up to ~12 m and pattern angles up to roughly +/-5 degrees
(matching .example_data, where the real patterns are rotated by a couple of
degrees at most). Beyond ~5-6 degrees, DetectorCalibration._determine_radial_
range's narrowest scan column (+/-5 px either side of the beam centre) can be
narrower than how far a rotated fringe has drifted sideways by the time it
reaches the first order, so it misses the fringe and the loop gives up
before trying a wider column that would have caught it - confirmed
independent of fringe brightness, so no amount of tuning here fixes it; the
pipeline then either errors out or, occasionally, silently locks onto the
wrong order and reports a wrong distance. That's a property of the analysis
pipeline, not of this generator: past ~5-6 degrees, always check the CLI's
printed "Detector located at ..." against the --distance you asked for, and
try a different --seed if it looks wrong or the CLI raises an error - the
noise realisation alone can decide whether a given angle calibrates cleanly.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nexusformat.nexus import (
    NXdetector,  # pyright: ignore[reportAttributeAccessIssue]
    NXentry,
    NXfield,
    NXinstrument,  # pyright: ignore[reportAttributeAccessIssue]
    NXmonochromator,  # pyright: ignore[reportAttributeAccessIssue]
    NXroot,
)
from numpy.typing import NDArray

from gratingcalibration import PILATUS2M_PIXEL_SIZE

# --- fixed detector geometry, matched to .example_data/first & second -----
DETECTOR_SHAPE = (1679, 1475)  # (rows, cols)
BEAM_CENTER_PX = {"x": 797.8, "y": 89.0}
BEAM_ENERGY_KEV = 14.0
GRATING_SPACING_M = 100e-9  # matches the CLI tool's --grating-spacing default
BEAMSTOP_RADIUS_PX = 15.0
# approximate Pilatus 2M module-gap dead rows, offset from the beam centre
# (loosely modelled on tests/helper_functions.py, moved further out so there
# is room for several grating orders before the detector goes quiet)
MODULE_GAP_BANDS = (
    (400, 450),
    (650, 700),
    (950, 1000),
    (1250, 1300),
    (1500, 1550),
)


def wavelength_from_energy_kev(energy_kev: float) -> float:
    """X-ray wavelength (m) for a given beam energy (keV)."""
    c = 299792458  # m/s
    planck = 6.62607015e-34  # Js
    ev = 1.602176634e-19  # J
    energy_ev = energy_kev * 1e3
    return (c * planck) / (energy_ev * ev)


# Order amplitude ~ ORDER_BOOST / n**ORDER_DECAY, as a multiplicative boost
# on the local background+halo (so the log-space contrast at each order is
# independent of how bright the local background happens to be there).
ORDER_BOOST_CALIBRATION_AXIS = 25.0
ORDER_DECAY = 0.6
ORDER_SIGMA_PX = 1.0
# DetectorCalibration.find_multiple_sequence caps a candidate order index at
# max_multiple=20 (see detector_calibration.py) - if the true fundamental
# would require indexing an order above that, it gets rejected outright and
# a spurious alias wins instead.
MAX_ORDER_INDEX_CALIBRATION_AXIS = 18
# Each calibration-axis order peak also trails a narrow, exponentially
# decaying streak running out along the perpendicular (horizontal, when
# pattern_angle=0) detector direction - matching the real data (see
# .example_data/first, beam_center_location.png): small, very bright spots
# sit on the vertical calibration axis, and each one has a decaying line
# running out horizontally from it. The streak is narrow across the
# calibration axis (STREAK_SIGMA_V) and decays exponentially along its own
# length (STREAK_DECAY_U_PX); it's smooth and radially broad enough that it
# never gets picked up by the calibration pipeline's own peak fitter,
# matching how the real streaks never get treated as their own order.
STREAK_SIGMA_V = 1.8
STREAK_DECAY_U_PX = 22.0
STREAK_BOOST = 10.0
# The streaks aren't smooth along their own length either - they carry a
# fine, closely-spaced fringe ripple (see .example_data/first: sampling
# pixel values along the direct-beam and order-1 streaks shows a ~4 px
# period with an amplitude far above Poisson noise - roughly 25x the ~2.5%
# shot-noise level expected at those count rates, so it's a real feature,
# not read-noise). This is what the very first correction to this generator
# was about: the x-direction spacing is much finer than the y (order)
# spacing - not a second, coincidentally similar family of order peaks.
#
# Amplitude is deliberately capped far below what was directly measured in
# the real data (~0.6 fractional ripple): find_calibration_azimuth decides
# which of chi=90+theta or chi=theta is the real calibration axis by probing
# both and taking whichever has more DetectorCalibration.peak_fitter-style
# "confirmed" (SNR > min_peak_snr=15, self-consistently spaced) peaks. At a
# large enough pattern rotation, the true order axis's radial range (and so
# its peak count) shrinks, while this closely-spaced fringe can still fit
# many periods into its own radial range; at FRINGE_AMPLITUDE=0.6 its peaks
# reached SNR ~34 and out-voted the real orders entirely (find_calibration_
# azimuth locked onto the fringe direction, and the "distance" it then
# fitted was just the fringe spacing misread as a grating order spacing -
# confirmed by reproducing the pipeline's own steps directly on
# scripts/d5.8_a10_s0's output). How close the fringe spacing (4.1 px) sits
# to pyFAI's own radial bin width also makes this a moving target: small,
# sub-pixel shifts in beam centre (from BeamCenterOptimiser's own fit) can
# swing the fringe peaks' SNR non-monotonically - 0.15 measured *safe* on
# one case's coarse beam centre and then ~29 SNR (unsafe) on the same case's
# refined centre, while 0.1 and 0.2 on the same refined centre measured
# safe. There's no amplitude that's provably safe for every beam centre/
# rotation/distance combination the pipeline might land on, so this is kept
# low enough (checked over several such cases, comfortably single-digit SNR
# in each) that it takes an unlucky combination to cross 15, rather than
# being reliably on the edge of it.
FRINGE_SPACING_PX = 4.1
FRINGE_AMPLITUDE = 0.08
# The direct beam itself streaks out horizontally right through the beam
# centre too (the "order 0" streak - see .example_data/first, the bright
# band passing straight through the beamstop). BeamCenterOptimiser's x0/x1
# sectors (beam_center_optimiser.py) integrate along exactly this direction,
# at ~13-38 px, right next to the beamstop, to symmetry-match the beam
# centre in x - without a directional signal there, that fit has nothing to
# latch onto.
STREAK_ZERO_AMPLITUDE = 1.0
# Multiplicative boost applied within the angular fan at all radii, giving
# determine_pattern_angle a clean, scale-independent signal to lock onto.
RIDGE_BOOST = 3.0
ANGULAR_SIGMA_DEG = 2.5


def build_pattern(
    distance: float, pattern_angle: float, seed: int = 0
) -> NDArray[np.float64]:
    """
    Build a synthetic Pilatus 2M frame of a diffraction grating pattern.

    The grating produces one family of diffraction order peaks, running
    radially away from the beam centre along the pattern_angle+90/+270
    directions (pyFAI's chi convention). Order n sits at radius r_n = n *
    wavelength * distance / grating_spacing (the same small-angle relation
    that DetectorCalibration.calculate_detector_distance inverts), on top of
    a smooth diffuse background/halo and a dark circular beamstop. Each
    order peak also trails a narrow, exponentially decaying streak running
    out along the perpendicular direction - matching the real data
    (.example_data/first): small, very bright spots on the calibration axis,
    each with a decaying line running out perpendicular to it.

    The angular/radial boosts are multiplicative (fractional contrast on top
    of the local background) rather than fixed additive amplitudes - tuned
    empirically against the real calibration pipeline (see
    DetectorCalibration._determine_radial_range and determine_pattern_angle)
    so that it reliably recovers the input distance and pattern angle for
    geometries similar to .example_data (sample-to-detector distances of a
    few metres upward). Order 1 is deliberately allowed to fall inside the
    near-beam windows that determine_pattern_angle and BeamCenterOptimiser's
    x0/x1/y0/y1 sectors scan (~13-58 px) - matching the real data, where
    order 1 is the single dominant feature in exactly that range (e.g.
    .example_data/first has order 1 at r~49 px) - rather than being
    artificially excluded from it. Very short distances (below ~4 m) pack
    the grating orders too close together for the peak fitter to resolve
    cleanly - this is a limitation of the analysis pipeline's fixed fitting
    window, not something the synthetic data generator can paper over.
    """
    ny, nx = DETECTOR_SHAPE
    cx, cy = BEAM_CENTER_PX["x"], BEAM_CENTER_PX["y"]
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx = xx - cx
    dy = yy - cy
    r = np.hypot(dx, dy)
    # pyFAI convention: chi = 0 along +x, chi = 90 along +y
    chi = np.degrees(np.arctan2(dy, dx))

    wavelength = wavelength_from_energy_kev(BEAM_ENERGY_KEV)
    max_r = float(np.hypot(max(cx, nx - cx), max(cy, ny - cy)))

    def angular_lobe(angle_offsets: tuple[float, float]) -> NDArray[np.float64]:
        lobe = np.zeros_like(r)
        for offset in angle_offsets:
            direction = pattern_angle + offset
            delta = (chi - direction + 180) % 360 - 180
            lobe += np.exp(-0.5 * (delta / ANGULAR_SIGMA_DEG) ** 2)
        return lobe

    lobe_calibration = angular_lobe((90.0, 270.0))
    calibration_spacing_px = (
        wavelength * distance / GRATING_SPACING_M / PILATUS2M_PIXEL_SIZE
    )
    n_orders = min(
        int(max_r / max(calibration_spacing_px, 1e-6)) + 1,
        MAX_ORDER_INDEX_CALIBRATION_AXIS,
    )

    # Rotated coordinates aligned with the pattern: u runs along the
    # perpendicular (streak) direction, v runs along the calibration axis
    # itself, so a streak attached to order n is just a narrow band around
    # v = +/- r_n that decays smoothly in |u|.
    theta_rad = np.deg2rad(pattern_angle)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    u = dx * cos_t + dy * sin_t
    v = -dx * sin_t + dy * cos_t

    order_sum = np.zeros_like(r)
    streak_sum = np.zeros_like(r)
    for n in range(1, n_orders + 1):
        r_n = n * calibration_spacing_px
        if r_n > max_r + 5:
            break
        amp = 1.0 / n**ORDER_DECAY
        order_sum += amp * np.exp(-0.5 * ((r - r_n) / ORDER_SIGMA_PX) ** 2)
        for sign in (1.0, -1.0):
            dv = v - sign * r_n
            streak_sum += (
                amp
                * np.exp(-0.5 * (dv / STREAK_SIGMA_V) ** 2)
                * np.exp(-np.abs(u) / STREAK_DECAY_U_PX)
            )
    streak_zero = (
        STREAK_ZERO_AMPLITUDE
        * np.exp(-0.5 * (v / STREAK_SIGMA_V) ** 2)
        * np.exp(-np.abs(u) / STREAK_DECAY_U_PX)
    )
    streak_zero = np.where(r >= BEAMSTOP_RADIUS_PX, streak_zero, 0.0)
    streak_sum = streak_sum + streak_zero

    # Fine fringe ripple riding on top of the streaks (see FRINGE_SPACING_PX
    # above) - a function of u alone, so it applies equally to the order-0
    # streak and every order-n streak without needing to be built into the
    # per-order loop above.
    streak_sum = streak_sum * (
        1 + FRINGE_AMPLITUDE * np.cos(2 * np.pi * u / FRINGE_SPACING_PX)
    )

    boost_calibration = ORDER_BOOST_CALIBRATION_AXIS * lobe_calibration * order_sum
    boost_streak = STREAK_BOOST * streak_sum

    fan = lobe_calibration

    background = 8.0 + 80.0 * np.exp(-r / 250.0)
    halo = 1500.0 * np.exp(-r / 70.0)
    baseline = background + halo

    image = baseline * (1 + RIDGE_BOOST * fan) * (1 + boost_calibration + boost_streak)

    for lo, hi in MODULE_GAP_BANDS:
        y0 = max(0, int(cy) + lo)
        y1 = min(ny, int(cy) + hi)
        if y0 < y1:
            image[y0:y1, :] = 0

    beamstop_mask = r <= BEAMSTOP_RADIUS_PX
    image[beamstop_mask] = 0

    image = np.clip(image, 0, None)
    # Draw Poisson noise for the calibration axis and for everything else
    # from independent RNG streams, rather than one rng.poisson() call over
    # the whole array. numpy's Generator.poisson has cross-element coupling
    # on large arrays (confirmed empirically: changing one element's value
    # changes unrelated elements' draws elsewhere in the same array), so a
    # single shared call would let unrelated tuning of the other axis change
    # the calibration axis's own noise realisation, and with it whether a
    # given --seed happens to calibrate cleanly.
    noisy = np.empty_like(image)
    calibration_mask = lobe_calibration > 0.01
    rng_calibration = np.random.default_rng(seed)
    noisy[calibration_mask] = rng_calibration.poisson(image[calibration_mask])
    rng_rest = np.random.default_rng(seed + 1_000_003)
    noisy[~calibration_mask] = rng_rest.poisson(image[~calibration_mask])

    return noisy


def write_nexus_file(image: NDArray[np.float64], outpath: Path) -> None:
    nx_detector = NXdetector()
    nx_detector.data = np.array([[image]])

    nx_instrument = NXinstrument()
    monochromator = NXmonochromator()
    monochromator.energy = NXfield(value=BEAM_ENERGY_KEV, dtype="float64", units="keV")
    nx_instrument.monochromator = monochromator

    nx_entry = NXentry()
    nx_entry.nxname = NXfield("entry1", dtype="U")
    nx_entry.instrument = nx_instrument
    nx_entry.detector = nx_detector

    nexus_output = NXroot(nx_entry)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    nexus_output.save(outpath, "w")


def write_preview_png(
    image: NDArray[np.float64], distance: float, pattern_angle: float, outpath: Path
) -> None:
    z_abs = np.abs(image)
    log_image = np.zeros_like(image)
    np.log(z_abs, out=log_image, where=(z_abs >= 1))

    # Clip the display range to percentiles rather than the raw min/max: a
    # handful of very bright pixels at the low-order peaks (order 1 in
    # particular, right next to the beamstop) would otherwise stretch the
    # colour scale so far that the much fainter, but still real, decaying
    # streaks further out get washed out to a near-uniform colour.
    nonzero = log_image[log_image > 0]
    vmin, vmax = np.percentile(nonzero, [1, 99.5])

    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(log_image, cmap="viridis", origin="upper", vmin=vmin, vmax=vmax)
    ax.plot(BEAM_CENTER_PX["x"], BEAM_CENTER_PX["y"], "r+", ms=12, mew=2)
    ax.set_title(f"distance={distance:.3f} m, pattern angle={pattern_angle:.2f} deg")
    fig.colorbar(im, ax=ax, label="log(intensity)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic diffraction-grating .nxs frame, using the "
            "same detector geometry as .example_data, for a given "
            "sample-to-detector distance and grating pattern rotation."
        )
    )
    parser.add_argument(
        "--distance",
        type=float,
        required=True,
        help="Sample-to-detector distance in metres.",
    )
    parser.add_argument(
        "--pattern-angle",
        type=float,
        required=True,
        dest="pattern_angle",
        help="Rotation of the grating pattern away from the detector axes, in degrees.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(".example_data/synthetic"),
        dest="output_path",
        help="Directory to write the .nxs and .png files to.",
    )
    parser.add_argument(
        "--outname",
        type=str,
        default=None,
        dest="outname",
        help=(
            "Base name (no extension) for the output files. Defaults to "
            "synthetic-<distance>m-<angle>deg."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the (Poisson) detector noise.",
    )
    args = parser.parse_args()

    outname = args.outname or f"synthetic-{args.distance:g}m-{args.pattern_angle:g}deg"
    nxs_path = args.output_path / f"{outname}.nxs"
    png_path = args.output_path / f"{outname}.png"

    image = build_pattern(args.distance, args.pattern_angle, seed=args.seed)
    write_nexus_file(image, nxs_path)
    write_preview_png(image, args.distance, args.pattern_angle, png_path)

    print(f"Wrote {nxs_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
