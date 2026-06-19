import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from gratingcalibration.beam_center_optimiser import BeamCenterOptimiser
from gratingcalibration.beamstop_fitter import FitBeamstop
from gratingcalibration.calibrant_file_writer import CalibrantFileWriter
from gratingcalibration.data_loader import DataLoader
from gratingcalibration.detector_calibration import DetectorCalibration


def save_all_figures(output_dir):
    """
    save all figures that have been generated along the way in the output directory
    """
    for num in plt.get_fignums():
        fig = plt.figure(num)
        label = fig.get_label() or f"fig{num}"
        fig.savefig(output_dir / f"{label}.png", dpi=200, bbox_inches="tight")


def main():

    parser = argparse.ArgumentParser(
        description="Calibration for a diffraction grating"
    )
    parser.add_argument(
        "--file", type=Path, default=None, dest="input", help="Path to input nexus file"
    )
    parser.add_argument(
        "--peak-prominance",
        type=float,
        default=0.1,
        dest="peak_prominance",
        help=(
            "Peak prominance to use in scipy.signal.find_peaks when"
            "finding grating fringes. Default=0.1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="calibration",
        dest="output",
        help="Path to where calibration file will be written",
    )
    parser.add_argument(
        "--save-plots",
        action="store_false",
        help="Save the plots generated as part of the calibration",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # STEP 0: load the data
    # -----------------------------------------------------------------------

    file_in = args.input

    if file_in is None:
        print("No input file!")
        sys.exit(1)

    data_in = DataLoader(filepath=file_in)
    z_corr = data_in.data
    beam_energy = data_in.energy
    det_mask = data_in.mask

    # -----------------------------------------------------------------------
    # STEP 1: find the centre of the beamstop
    # this should be an approximate starting point for the centre of the beam
    # -----------------------------------------------------------------------
    beamstop = FitBeamstop(z_corr, plot=True)
    beamstop_center = beamstop.beamstop_center
    mask = det_mask + beamstop.beamstop_mask

    # -----------------------------------------------------------------------
    # STEP 2: find the centre of the beam
    # using beamstop centre as starting point, adjust the centre of
    # azimuthal integration in y then x to minimise difference in profiles
    # looking in opposite directions
    # -----------------------------------------------------------------------

    a, b, c, d = (
        int(beamstop_center["x"] - 40),
        int(beamstop_center["x"] + 40),
        int(beamstop_center["y"] + 60),
        int(beamstop_center["y"] - 60),
    )

    cropped = z_corr[d:c, a:b]
    cropped_center = {
        "x": beamstop_center.get("x") - a,
        "y": beamstop_center.get("y") - d,
    }

    cropped_mask = mask[d:c, a:b]

    fitter = BeamCenterOptimiser(
        image=cropped,
        mask=cropped_mask,
        beam_energy=beam_energy,
        beamstop_center=cropped_center,
        optimise_direction="x",
        offset={"x": a, "y": d},
    )

    fitter.fit_beam_centre()
    fitter.optimise_direction = "y"
    fitter.fit_beam_centre()

    fitter.plots(extent=[a, b, c, d])

    # -----------------------------------------------------------------------
    # STEP 3: calibrate detector position
    # Using the optimised beam position as the centre of integration,
    # look at the direction with the most spacings, and calibrate against
    # peaks as usual.
    # -----------------------------------------------------------------------

    detector_calib = DetectorCalibration(
        image=z_corr,
        beam_center=fitter.beam_center_global,
        wavelength=fitter.wavelength,
        peak_prominance=args.peak_prominance,
    )
    detector_calib.calculate_detector_distance()
    detector_calib.plots()

    print(
        f"Detector located at {detector_calib.detector_distance:.5f} "
        "{detector_calib.detector_distance_units}. "
        "Writing calibration file."
    )

    data_out = {
        "image": data_in.raw_data,
        "wavelength": {"value": fitter.wavelength, "units": fitter.wavelength_units},
        "pixel_size": {"value": 172e-6, "units": "m"},
        "beam_center": {
            "x": fitter.beam_center_global["x"] * 172e-6,
            "y": fitter.beam_center_global["y"] * 172e-6,
            "units": "m",
        },
        "detector_distance": {
            "value": detector_calib.detector_distance,
            "units": detector_calib.detector_distance_units,
        },
    }

    out = Path(args.output)
    out.mkdir(exist_ok=True)

    CalibrantFileWriter(
        datadict=data_out, writepath=out / "SAXS_calibration.nxs"
    ).writer()

    save_all_figures(out)
