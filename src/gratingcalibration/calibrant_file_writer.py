import numpy as np
from nexusformat.nexus import (
    NeXusError,
    NXbeam,
    NXdata,
    NXdetector,
    NXdetector_module,
    NXentry,
    NXfield,
    NXinstrument,
    NXlink,
    NXroot,
    NXsample,
    NXtransformations,
)


class CalibrantFileWriter:
    def __init__(self, datadict=None, writepath=None):
        self.datadict = datadict
        self.writepath = writepath

        # default schema for data validation. update as required.
        schema = {
            "image": np.ndarray,
            "wavelength": {"value": float, "units": str},
            "pixel_size": {"value": float, "units": str},
            "beam_center": {"x": float, "y": float, "units": str},
            "detector_distance": {"value": float, "units": str},
        }
        self.valid_data = self.data_validation(self.datadict, schema)

    def data_validation(self, data, schema):
        """
        Validate that we have all the necessary data in the input dictionary

        Returns
        -------
        bool
            True if we have all the data with attributes, False if not

        """

        # no data no problems
        if data is None:
            return False

        # If schema is a type, check directly
        if isinstance(schema, type):
            return isinstance(data, schema)

        # If schema is a dict, check structure
        if isinstance(schema, dict):
            if not isinstance(data, dict):
                return False
            if set(data.keys()) != set(schema.keys()):
                return False
            return all(self.data_validation(data[k], schema[k]) for k in schema)

        # If schema is a list, assume homogeneous list
        if isinstance(schema, list):
            if len(schema) != 1:
                raise ValueError("Schema list must have one element")
            if not isinstance(data, (list, tuple)):
                return False
            return all(self.data_validation(item, schema[0]) for item in data)

        return False

    # def sanitise_units(self):
    #     # TODO complete this and implement it in writer()

    #     length_converter = {"angstrom": {"m": 1e10, "nm": 10}, "nm": {"angstrom": 10}}

    #     desired_units = {
    #         "prop": {"angstrom": {"m": 1e10, "nm": 10}, "nm": {"angstrom"}}
    #     }

    def writer(self):
        """
        Function to write a calibration NeXuS file
        """

        if not self.valid_data:
            raise NeXusError("Don't have all the data we need, won't write a file")

        detector_image = self.datadict.get("image")
        wavelength = self.datadict.get("wavelength")
        beam_center = self.datadict.get("beam_center")
        pixel_size = self.datadict.get("pixel_size")
        detector_distance = self.datadict.get("detector_distance")
        detector_vector = np.array(
            [beam_center.get("x"), beam_center.get("y"), detector_distance.get("value")]
        )

        # this gets used a couple of times
        nx_data = NXfield(
            detector_image, dtype="int32", shape=list(detector_image.shape)
        )

        # calibration sample
        nx_calibration_sample = NXsample()
        beam = NXbeam()
        beam.incident_wavelength = NXfield(
            wavelength.get("value"), dtype="float64", units=wavelength.get("units")
        )
        nx_calibration_sample.beam = beam

        # detector
        nx_detector = NXdetector()
        nx_detector.beam_center_x = NXfield(
            beam_center.get("x"), dtype="float64", units=beam_center.get("units")
        )
        nx_detector.beam_center_y = NXfield(
            beam_center.get("y"), dtype="float64", units=beam_center.get("units")
        )
        nx_detector.data = nx_data
        nx_detector.depends_on = NXfield("./transformations/euler_c")
        nx_detector.x_pixel_size = NXfield(
            pixel_size.get("value"), dtype="float64", units=pixel_size.get("units")
        )
        nx_detector.y_pixel_size = NXfield(
            pixel_size.get("value"), dtype="float64", units=pixel_size.get("units")
        )
        nx_detector.distance = NXfield(
            detector_distance.get("value"),
            dtype="float64",
            units=detector_distance.get("units"),
        )

        # detector module
        nx_detector_module = NXdetector_module()
        nx_detector_module.data_origin = NXfield(np.array([0, 0]), dtype="int32")
        nx_detector_module.data_size = NXfield(detector_image.shape, dtype="int32")
        nx_detector_module.fast_pixel_direction = NXfield(
            pixel_size.get("value"),
            units=pixel_size.get("units"),
            dtype="float64",
            depends_on="./module_offset",
            offset=[
                0.00000000,
                0.00000000,
                0.00000000,
            ],
            offset_units="mm",
            transformation_type="translation",
            vector=[
                -1.00000000,
                0.00000000,
                0.00000000,
            ],
        )
        nx_detector_module.slow_pixel_direction = NXfield(
            pixel_size.get("value"),
            units=pixel_size.get("units"),
            dtype="float64",
            depends_on="./module_offset",
            offset=[
                0.00000000,
                0.00000000,
                0.00000000,
            ],
            offset_units="mm",
            transformation_type="translation",
            vector=[
                0.00000000,
                -1.00000000,
                0.00000000,
            ],
        )
        nx_detector_module.module_offset = NXfield(
            0,
            units="mm",
            dtype="int32",
            depends_on="../transformations/euler_c",
            offset=[
                0.00000000,
                0.00000000,
                0.00000000,
            ],
            offset_units="mm",
            transformation_type="translation",
            vector=[
                0.00000000,
                0.00000000,
                0.00000000,
            ],
        )
        nx_detector.detector_module = nx_detector_module

        # detector transformations
        nx_transformations = NXtransformations()
        nx_transformations.euler_a = NXfield(
            0,
            dtype="float64",
            units="deg",
            depends_on="./origin_offset",
            offset=[
                0.00000000,
                0.00000000,
                0.00000000,
            ],
            offset_units="mm",
            transformation_type="rotation",
            vector=[
                0.00000000,
                0.00000000,
                1.00000000,
            ],
        )
        nx_transformations.euler_b = NXfield(
            0,
            dtype="float64",
            units="deg",
            depends_on="./euler_a",
            offset=[
                0.00000000,
                0.00000000,
                0.00000000,
            ],
            offset_units="mm",
            transformation_type="rotation",
            vector=[
                0.00000000,
                1.00000000,
                0.00000000,
            ],
        )
        nx_transformations.euler_c = NXfield(
            0,
            dtype="float64",
            units="deg",
            depends_on="./euler_b",
            offset=[
                0.00000000,
                0.00000000,
                0.00000000,
            ],
            offset_units="mm",
            transformation_type="rotation",
            vector=[
                0.00000000,
                0.00000000,
                1.00000000,
            ],
        )
        nx_transformations.origin_offset = NXfield(
            np.linalg.norm(detector_vector),
            dtype="float64",
            units="m",
            depends_on=".",
            offset=[
                0.00000000,
                0.00000000,
                0.00000000,
            ],
            offset_units="m",
            transformation_type="translation",
            vector=detector_vector / np.linalg.norm(detector_vector),
        )
        nx_detector.transformations = nx_transformations

        # put everything together for the detector
        nx_instrument = NXinstrument()
        nx_instrument.detector = nx_detector

        # now put everything together for the whole entry
        nx_entry = NXentry()
        nx_entry.nxname = NXfield("entry1", dtype="U")
        nx_entry.instrument = nx_instrument
        nx_entry.calibration_sample = nx_calibration_sample

        # calibration data
        nx_calibration_data = NXdata()
        nx_calibration_data.data = NXlink("/entry1/instrument/detector/data")
        nx_entry.calibration_data = nx_calibration_data

        # write the file if we want
        if self.writepath is not None:
            print("writing file")
            nexus_output = NXroot(nx_entry)
            nexus_output.save(self.writepath, "w")
