[![CI](https://github.com/DiamondLightSource/i22-diffraction-grating-calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/DiamondLightSource/i22-diffraction-grating-calibration/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/DiamondLightSource/i22-diffraction-grating-calibration/branch/main/graph/badge.svg)](https://codecov.io/gh/DiamondLightSource/i22-diffraction-grating-calibration)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# gratingcalibration

Tool for calibrating a SAXS beamline using diffraction grating. Currently under construction.


What            | Where
:---:           | :---:
Source          | <https://github.com/DiamondLightSource/i22-diffraction-grating-calibration>
Docker          | `docker run ghcr.io/diamondlightsource/i22-diffraction-grating-calibration:latest`
Releases        | <https://github.com/DiamondLightSource/i22-diffraction-grating-calibration/releases>


## Usage

```
$ calibrate --file input.nxs --output output_folder
```

The calibration NeXuS file (SAXS_calibration.nxs) will be stored in the `output_folder`, along with plots made toe show how the calibration has been performed.


## What's happening?

This calibration program works by:

1. Identifying the beamstop & fitting and approximate center
    * Shown in the `beamstop_fit.png` figure   
2. Optimising the beam centre through x/y detector profile matching
    * See `beam_profiles_xy.png` and `beam_center_location.png` to see how well the profiles have been matched, and where the beam center has been located
3. Calibrate the detector distance based on grating fringe spacing
    * The peaks found, and how they have been indexed, is shown in `detector_calibration.png`. 


## Notes

* Assumes a Pilatus 2M detector, pixel size 172e-6 m.
* Units are all SI, not more native/sensible SAXS based units (e.g. beam center position should really be in mm not m, wavelength in angstrom not m, etc.)
