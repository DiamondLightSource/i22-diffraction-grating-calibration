import numpy as np
import pytest
from nexusformat.nexus import nxload, NeXusError, NXdata, NXfield

from gratingcalibration.data_loader import DataLoader  
from .helper_functions import make_fake_file

def test_init_loads_everything(make_fake_file):
    """
    test we know how to load the fake file and that it has the expected data in
    """

    fake_data = nxload(make_fake_file(), 'r')

    assert isinstance(fake_data.entry1.detector.data, NXfield)
    assert isinstance(fake_data.entry1.instrument.monochromator.energy, NXfield)

def test_detector_image_log_transform(make_fake_file):

    dl = DataLoader(make_fake_file())

    # expects last frame: fake_data[-1, -1]
    fake_data = nxload(make_fake_file(), 'r')
    raw = fake_data.entry1.detector.data[-1,-1]

    expected = np.zeros_like(raw, dtype=np.float64)
    mask = np.abs(raw) >= 1
    expected[mask] = np.log(np.abs(raw[mask]))

    expected = expected.astype(np.float32)

    np.testing.assert_allclose(dl.data, expected)
    assert dl.data.dtype == np.float32

def test_mask_creation(make_fake_file):
    
    dl = DataLoader(make_fake_file())
    expected_mask = np.where(dl.data < 0.5, 10, 0)

    np.testing.assert_array_equal(dl.mask, expected_mask)


def test_energy_conversion_kev_to_ev(make_fake_file):

    dl = DataLoader(make_fake_file())

    fake_data = nxload(make_fake_file(), 'r')
    raw = fake_data.entry1.instrument.monochromator.energy

    assert np.isclose(dl.energy, raw * 1e3)

def test_invalid_datapath_raises(make_fake_file):
    with pytest.raises(NeXusError):
        DataLoader(make_fake_file(), datapath="invalid/path")

def test_invalid_energypath_raises(make_fake_file):
    with pytest.raises(NeXusError):
        DataLoader(make_fake_file(), energypath="invalid/path")

def test_log_does_not_apply_below_one(make_fake_file):

    dl = DataLoader(make_fake_file())

    # expects last frame: fake_data[-1, -1]
    fake_data = nxload(make_fake_file(), 'r')
    raw = fake_data.entry1.detector.data[-1,-1]

    # where < 1 → should be zero
    below_one = np.abs(raw) < 1
    assert np.all(dl.data[below_one] == 0)
