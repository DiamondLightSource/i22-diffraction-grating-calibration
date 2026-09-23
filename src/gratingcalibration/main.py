import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from gratingcalibration import PILATUS2M_PIXEL_SIZE, __version__
from gratingcalibration.beam_center_optimiser import BeamCenterOptimiser
from gratingcalibration.beamstop_fitter import FitBeamstop
from gratingcalibration.calibrant_file_writer import CalibrantFileWriter
from gratingcalibration.data_loader import DataLoader
from gratingcalibration.detector_calibration import (
    DetectorCalibration,
    determine_pattern_angle,
    find_calibration_azimuth,
)


def save_all_figures(output_dir: Path) -> None:
    """
    save all figures that have been generated along the way in the output directory
    """
    for num in plt.get_fignums():
        fig = plt.figure(num)
        label = fig.get_label() or f"fig{num}"
        fig.savefig(output_dir / f"{label}.png", dpi=200, bbox_inches="tight")


def main(args: Sequence[str] | None = None) -> None:

    parser = argparse.ArgumentParser(
        description="Calibration for a diffraction grating"
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
    )
    parser.add_argument(
        "--file", type=Path, default=None, dest="input", help="Path to input nexus file"
    )
    parser.add_argument(
        "--grating-spacing",
        type=float,
        default=100,
        dest="grating_spacing",
        help=("Spacing of diffraction grating in nm. Default 100."),
    )
    parser.add_argument(
        "--x-offset",
        type=int,
        default=40,
        dest="x_offset",
        help=(
            "Space (in pixels) to use around estimated beamstop"
            " and beam position. Default=40"
        ),
    )
    parser.add_argument(
        "--y-offset",
        type=int,
        default=60,
        dest="y_offset",
        help=(
            "Space (in pixels) to use around estimated beamstop"
            " and beam position. Default=60"
        ),
    )
    parser.add_argument(
        "--peak-prominance",
        type=float,
        default=0.1,
        dest="peak_prominance",
        help=(
            "Peak prominance to use in scipy.signal.find_peaks when "
            "finding grating fringes. Default=0.1"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default="processing",
        dest="output_path",
        help=(
            "Relative path to where calibration file will be written. "
            "Defaults to 'processing'. Accepts any pathtype, "
            "but will be relative by default. "
            "Extended paths (e.g. `processing/my/experiment`) can be given"
            ", directories will be created if not already found"
        ),
    )
    parser.add_argument(
        "--outname",
        type=str,
        default="SAXS_calibration",
        dest="outname",
        help="Name of output calibration file. Defaults to 'SAXS_calibration'",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        dest="no_plots",
        help="Don't save processing plots alongside the calibration file",
    )

    parsed_args = parser.parse_args(args)

    # -----------------------------------------------------------------------
    # STEP 0: load the data
    # -----------------------------------------------------------------------

    file_in = parsed_args.input

    if file_in is None:
        print("No input file!\n")
        sys.exit(1)

    data_in = DataLoader(filepath=file_in)
    z_corr = data_in.data
    beam_energy = data_in.energy
    det_mask = data_in.mask

    print("Starting calibration. Locating beamstop & beam center")

    # -----------------------------------------------------------------------
    # STEP 1: find the centre of the beamstop
    # this should be an approximate starting point for the centre of the beam
    # -----------------------------------------------------------------------
    beamstop = FitBeamstop(
        z_corr, plot=True, x_cut=parsed_args.x_offset, y_cut=parsed_args.y_offset
    )
    beamstop_center = beamstop.beamstop_center
    mask = det_mask + beamstop.beamstop_mask

    # -----------------------------------------------------------------------
    # STEP 1.5: work out how far the grating pattern is rotated away from the
    # detector's vertical/horizontal axes. This only needs a rough beam
    # position, so the coarse beamstop centre is accurate enough for it - no
    # need to wait for the refined beam centre from step 2, which means the
    # beam centre only has to be fitted once (with the correct rotation from
    # the start) instead of once assuming no rotation and then again after
    # measuring it.
    # -----------------------------------------------------------------------

    azimuth_offset = determine_pattern_angle(
        z_corr, beamstop_center, PILATUS2M_PIXEL_SIZE, mask=mask
    )
    print(f"Pattern rotation approx. {azimuth_offset:.3f} degrees")

    # -----------------------------------------------------------------------
    # STEP 2: find the centre of the beam
    # using beamstop centre as starting point, adjust the centre of
    # azimuthal integration in y then x to minimise difference in profiles
    # looking in opposite directions
    # -----------------------------------------------------------------------

    a, b, c, d = (
        int(beamstop_center["x"] - parsed_args.x_offset),
        int(beamstop_center["x"] + parsed_args.x_offset),
        int(beamstop_center["y"] + parsed_args.y_offset),
        # assuming we're near the top of the detector, want to be careful not to go off
        max(0, int(beamstop_center["y"] - parsed_args.y_offset)),
    )

    cropped = z_corr[d:c, a:b]
    cropped_center = {
        "x": beamstop_center["x"] - a,
        "y": beamstop_center["y"] - d,
    }

    cropped_mask = mask[d:c, a:b]

    fitter = BeamCenterOptimiser(
        image=cropped,
        mask=cropped_mask,
        beam_energy=beam_energy,
        beamstop_center=cropped_center,
        optimise_direction="x",
        offset={"x": a, "y": d},
        azimuth_offset=azimuth_offset,
    )

    fitter.fit_beam_centre()
    fitter.optimise_direction = "y"
    fitter.fit_beam_centre()

    fitter.plots(extent=(a, b, c, d))

    # -----------------------------------------------------------------------
    # STEP 3: calibrate detector position
    # Using the optimised beam position as the centre of integration,
    # look at the direction with the most spacings, and calibrate against
    # peaks as usual.
    # -----------------------------------------------------------------------
    print("Calibrating detector distance")
    calibration_azimuth = find_calibration_azimuth(
        z_corr,
        fitter.beam_center_global,
        PILATUS2M_PIXEL_SIZE,
        wavelength=fitter.wavelength,
        mask=mask,
        theta=azimuth_offset,
    )

    detector_calib = DetectorCalibration(
        image=z_corr,
        beam_center=fitter.beam_center_global,
        wavelength=fitter.wavelength,
        peak_prominance=parsed_args.peak_prominance,
        mask=mask,
        azimuth_center=calibration_azimuth,
        # convert to nm here. Probably a better way to do it but make do for now.
        grating_spacing=parsed_args.grating_spacing * 1e-9,
    )
    detector_calib.calculate_detector_distance()
    detector_calib.plots()

    print(
        f"Detector located at {detector_calib.detector_distance:.5f} "
        f"+/- {detector_calib.detector_distance_error:.5f} "
        f"{detector_calib.detector_distance_units}."
    )

    data_out = {
        "image": data_in.raw_data,
        "wavelength": {"value": fitter.wavelength, "units": fitter.wavelength_units},
        "pixel_size": {"value": PILATUS2M_PIXEL_SIZE, "units": "m"},
        "beam_center": {
            "x": fitter.beam_center_global["x"] * PILATUS2M_PIXEL_SIZE,
            "y": fitter.beam_center_global["y"] * PILATUS2M_PIXEL_SIZE,
            "units": "m",
        },
        "detector_distance": {
            "value": detector_calib.detector_distance,
            "units": detector_calib.detector_distance_units,
        },
        "detector_distance_error": {
            "value": detector_calib.detector_distance_error,
            "units": detector_calib.detector_distance_error_units,
        },
    }

    outpath = Path(parsed_args.output_path)
    outpath.mkdir(exist_ok=True, parents=True)

    outname = Path(parsed_args.outname).with_suffix(".nxs")

    CalibrantFileWriter(datadict=data_out, writepath=outpath / outname).writer()

    if not parsed_args.no_plots:
        save_all_figures(outpath)
