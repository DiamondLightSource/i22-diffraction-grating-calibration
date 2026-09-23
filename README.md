[![CI](https://github.com/DiamondLightSource/i22-diffraction-grating-calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/DiamondLightSource/i22-diffraction-grating-calibration/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/DiamondLightSource/i22-diffraction-grating-calibration/branch/main/graph/badge.svg)](https://codecov.io/gh/DiamondLightSource/i22-diffraction-grating-calibration)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# gratingcalibration

Tool for calibrating a SAXS beamline using diffraction grating. Currently under construction.

[comment]: <> ( What            | Where )
[comment]: <> ( :---:           | :---:)
[comment]: <> ( Source          | <https://github.com/DiamondLightSource/i22-diffraction-grating-calibration>)
[comment]: <> ( Docker          | `docker run ghcr.io/diamondlightsource/i22-diffraction-grating-calibration:latest`)
[comment]: <> ( Releases        | <https://github.com/DiamondLightSource/i22-diffraction-grating-calibration/releases>)

## Installation
### Load module
On the DLS module system, you can load this directly with :

```
module load grating-calibration-i22
```

### Install from source
On the DLS module system, you can install this tool as:

```
module load uv
uv venv --python 3.12
uv pip install git+https://github.com/DiamondLightSource/i22-diffraction-grating-calibration.git
source .venv/bin/activate
```

## Usage

Simple usage:

```
$ i22-diffraction-grating-calibration --file /path/to/input.nxs --output my_calibration_folder 
```

## What's happening?

For a practical overview of the program and its use, see the [docs page](docs/docs.md)

## Notes

* Assumes a Pilatus 2M detector, pixel size 172e-6 m.
* Units are all SI, not more native/sensible SAXS based units (e.g. beam center position should really be in mm not m, wavelength in angstrom not m, etc.)
