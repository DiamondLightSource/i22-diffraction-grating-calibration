import numpy as np
from nexusformat.nexus import NeXusError, nxload


class DataLoader:
    """
    class for loading the data
    """

    def __init__(
        self,
        filepath,
        datapath="entry1/detector/data",
        energypath="entry1/instrument/monochromator/energy",
    ):
        """
        filepath: str
            path to file to load as nexus file
        datapath: str
            entry in the nexus file containing the frame
        energypath: str
            entry in the nexus file containing the beam energy
        """
        self.filepath = filepath
        self.file = nxload(self.filepath, "r")
        self.data = self.get_detector_image(datapath)
        self.energy = self.get_beam_energy(energypath=energypath)
        self.mask = self.make_mask()

    def get_detector_image(self, datapath):
        """
        locate the detector image in the nexus file
        """
        if datapath in self.file:
            data_entry = self.file[datapath]
        else:
            raise NeXusError(
                "Invalid nexus path to access detector image. "
                f"Expected path is: {datapath}"
            )

        z = data_entry[-1, -1]
        self.raw_data = np.array(z)

        # if z is 0, make it 0, otherwise take log(abs(z))
        z_abs = np.abs(z)
        z_corr = np.zeros_like(z, dtype=np.float64)
        np.log(z_abs, out=z_corr, where=(z_abs >= 1))

        data = np.ascontiguousarray(z_corr, dtype=np.float32)
        return data

    def get_beam_energy(self, energypath):
        """
        get the beam energy in ev from the nexus file
        """
        if energypath in self.file:
            energy_entry = self.file[energypath]
        else:
            raise NeXusError(
                "Invalid NeXuS path to access beam energy. "
                f"Expected path is {energypath}"
            )
        # handle units to make sure we work in SI
        multiplier = 1
        if energy_entry.units.lower() == "kev":
            multiplier = 1e3  # i.e. convert to ev
        return energy_entry.nxvalue * multiplier

    def make_mask(self, threshold=0.5):
        """
        some kind of detector mask for integration later.
        Where data is < threshold, mask = 10, otherwise 0.
        """
        det_mask = np.where(self.data < threshold, 10, 0)
        return det_mask
