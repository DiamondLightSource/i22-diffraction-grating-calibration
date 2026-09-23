# Synthetic data examples

The `generate_synthetic_data.py` script in this folder attempts to generate a NeXuS file that looks a bit like a diffraction grating image, given a particular distance and angle of rotation. It isn't intended for any analytical purpose and should not be used as such. It contains a number of hard coded variables that can change, e.g. the centre of the beam.

The script can be called by the `generate.sh` script to rapidly generate synthetic data at a range of distances and angles, with random seeding.

The `analyse.sh` script can then be used to analyse the generated data to check the current output of the diffraction-grating program.
