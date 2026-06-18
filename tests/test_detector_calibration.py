import numpy as np
import pytest
from nexusformat.nexus import nxload

from gratingcalibration.data_loader import DataLoader  
from gratingcalibration.detector_calibration import DetectorCalibration, find_multiple_sequence
from .helper_functions import make_fake_file, make_fit_store
import matplotlib.pyplot as plt

@pytest.mark.parametrize('cx, cy, spots, radius',
                         (
                             # somewhere sensible
                             (700, 200,
                              (100,200,300,400,500,600,700,820),
                              50),
                              # somewhere different
                             (100, 100,
                              (100,200,300,400,500,600,700,820),
                              75),
                              # beamstop is overlapping the spots, 
                             (100, 100,
                              (100,200,300,400,500,600,700,820),
                              150),
                         ))
def test_radial_range(make_fake_file, cx, cy, spots, radius):
    """
    test that we pick up the radial range correctly
    """

    fn = make_fake_file(cx=cx,
                        cy=cy,
                        r=radius,
                        spot_rows=spots,
                        background_intensity=5.0,
                        spot_intensity=10.0,
                        )    
    fake_data = DataLoader(fn)
    
    dc = DetectorCalibration(image=fake_data.data,
                             beam_center={'x': cx, 'y': cy},
                             wavelength=1e-10 # this doesn't actually matter
                             )
    
    lims = dc._determine_radial_range()
    lower, upper = [i/dc.PILATUS2M_PIXEL_SIZE for i in lims]    

    print(lower, radius)
    # expect the lower limit to be ~ the same position as the beamstop edge.
    assert np.isclose(lower, radius, atol=5)
    # expect the upper limit to be within aroud 50 points of the last peak.
    print(upper, spots[-1])
    assert np.isclose(upper, spots[-1], atol=50)


@pytest.mark.parametrize('input, output',
                         (
                            # input and output are the same
                            (np.arange(10),
                             np.arange(10),
                            ),
                            # input contains one outlier that should be excluded
                            (np.array([0,1,2,3,4,5.5]),
                             np.array([0,1,2,3,4])
                            ),
                            # input is non sequential, output should identify and 
                            # relabel with correct index
                            (np.array([0,1,2,4,5,6]),
                             np.array([0,1,2,4,5,6])
                            ),
                        )
                         )
def test_find_multiple_sequence(input, output):
    """
    test that sequences get identified correctly
    """
    fit_store = find_multiple_sequence(make_fit_store(input))
    result = np.array([i['center'] for i in fit_store.values()])
    assert np.all(np.isclose(result, output))
    

@pytest.mark.parametrize('npt, radial_range',
                         (((500,None),
                           (250,(0.005,0.01)))))
def test_azi_integration(make_fake_file, npt, radial_range):
    """
    test that we pick up the radial range correctly
    """

    fn = make_fake_file(cx=700,
                        cy=200,
                        )    
    fake_data = DataLoader(fn)
    
    dc = DetectorCalibration(image=fake_data.data,
                             beam_center={'x': 700, 'y': 200},
                             wavelength=1e-10 # this doesn't actually matter
                             )
    
    signal_arr = dc.make_signal(npt=npt, auto_radial=radial_range)
    
    # make sure we get out the right number of points
    assert signal_arr.shape == (2, npt)
    # make sure we get the radial range we're expecting
    if radial_range is not None:
        assert np.all(np.isclose(np.array([signal_arr[0][0], signal_arr[0][-1]]), 
                                 np.array(radial_range), 
                                 atol=1e-5)
                                 )


def test_detector_distance():
    dc = DetectorCalibration(image=None,
                             beam_center=None,
                             wavelength = 1, # this doesn't actually matter
                             grating_spacing = 1 # neither does this
                             )
    
    dc.peaks = np.array([[1,2,3,4,5],
                         [1,2,3,4,5]]).T
    
    dc.calculate_detector_distance()
    assert np.isclose(dc.detector_distance, 1)


    

    

