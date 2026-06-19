import numpy as np
import pytest
from nexusformat.nexus import (
    NXdetector,
    NXentry,
    NXfield,
    NXinstrument,
    NXmonochromator,
    NXroot,
)


@pytest.fixture(scope="session")
def make_fake_file(tmp_path_factory):
    """
    write a tmp file with some fake detector data
    """
    base_dir = tmp_path_factory.mktemp("data")
    cache = {}

    def add_spot(xx, yy, cx, cy, w, r, intensity):
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        return intensity * (1 - np.tanh((dist - r) / w))

    def _make_fake_file(
        cx=700,
        cy=200,
        r=50.0,
        spot_rows=(100, 200, 300, 400, 500, 600, 700, 820),
        background_intensity=10.0,
        spot_intensity=5.0,
        detector_shape=(1679, 1475),
        detector_mask=(
            (225, 275),
            (425, 475),
            (725, 775),
            (1025, 1075),
            (1225, 1275),
            (1425, 1475),
        ),
    ):
        # --- cache key ---
        key = (
            cx,
            cy,
            r,
            spot_rows,
            background_intensity,
            spot_intensity,
            detector_shape,
        )
        if key in cache:
            return cache[key]

        ny, nx = detector_shape
        z = np.zeros((ny, nx)) + 3

        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

        # main scattering pattern
        z += add_spot(xx, yy, cx, cy, 100, r, background_intensity)

        # calibration spots
        for _, j in enumerate(spot_rows, 1):
            z += add_spot(xx, yy, cx, j + cy, 3, 5, spot_intensity)

        # beamstop
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r**2
        z[mask] = 0

        # some blank regions of no signal
        for i in detector_mask:
            z[i[0] + cy : i[1] + cy, :] = 0

        # NeXus structure
        nx_data = np.array([[z]])

        nx_detector = NXdetector()
        nx_detector.data = nx_data

        nx_instrument = NXinstrument()
        beam = NXmonochromator()
        beam.energy = NXfield(value=12, dtype="float64", units="keV")
        nx_instrument.monochromator = beam

        nx_entry = NXentry()
        nx_entry.nxname = NXfield("entry1", dtype="U")
        nx_entry.instrument = nx_instrument
        nx_entry.detector = nx_detector

        nexus_output = NXroot(nx_entry)

        # unique filename
        fn = base_dir / f"fake_{abs(hash(key))}.nxs"
        nexus_output.save(fn, "w")

        cache[key] = fn
        return fn

    return _make_fake_file


def make_fit_store(sequence):
    """
    From sequence, generate a dictionary of data

    input
    sequence: np.array
        array of ints of peak spacings

    returns
    out: dict
        out[idx] = {'data': np.array([x,y]),
                    'profile': np.array([x_plt,y_plt]),
                    'center': res.params['p_center'].value,
                    }

    """

    out = {}
    for i in sequence:
        # make some data around the sequence index
        data = np.linspace(i - 0.5, i + 0.5, 100)
        # generate a small peak centered at the index
        peak = np.exp(-((data - i) ** 2) / 0.1)

        out[i] = {
            # meant to be the raw data. downsample and slightly randomise
            "data": np.array([data[::10], np.random.normal(peak, 0.1)[::10]]),
            # the 'fitted' profile to the peak
            "profile": np.array([data, peak]),
            # the peak center
            "center": i,
        }

    return out
