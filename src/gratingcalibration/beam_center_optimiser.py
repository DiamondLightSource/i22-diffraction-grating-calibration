from collections.abc import Callable
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, minimize
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray
from pyFAI.integrator.azimuthal import AzimuthalIntegrator


class IntegrationConfig(TypedDict):
    npt: int
    center: float
    radial_range: tuple[float, float]


def wrap_azimuth(angle: float) -> float:
    """
    wrap an azimuthal angle (degrees) into pyFAI's expected (-180, 180] range
    """
    return ((angle + 180) % 360) - 180


def split_azimuth_window(center: float, half_width: float) -> list[tuple[float, float]]:
    """
    build the azimuth_range window(s) needed to cover `center` +/- `half_width`
    degrees, wrapped into (-180, 180].

    if the window straddles the +/-180 degree seam it has to be split into two
    pieces (one either side of the seam) since pyFAI's azimuth_range doesn't
    reliably handle a wrapped (lo > hi) range itself.
    """
    lo = wrap_azimuth(center - half_width)
    hi = wrap_azimuth(center + half_width)
    if lo <= hi:
        return [(lo, hi)]
    return [(lo, 180.0), (-180.0, hi)]


class BeamCenterOptimiser:
    def __init__(
        self,
        image: NDArray[Any],
        mask: NDArray[Any],
        beam_energy: float,
        beamstop_center: dict[str, float],
        optimise_direction: str = "x",
        offset: dict[str, float] | None = None,
        azimuth_offset: float = 0.0,
        integration_range: float = 20,
    ) -> None:
        # some constants for the integration
        self.PILATUS2M_PIXEL_SIZE = 172e-6
        # half-width (degrees) of each integration sector. A narrow sector
        # (e.g. a few degrees) is very sensitive to any thin, localised
        # angular artefact close to the beamstop - e.g. its support arm's
        # shadow - since that can dominate a small sector without being
        # averaged out. Widening the sector trades a little azimuthal
        # resolution for substantially better noise/artefact rejection: the
        # underlying halo shape being matched is symmetric over a much wider
        # angular range than the width of such artefacts.
        self.integration_range = integration_range
        # azimuth_offset lets the four integration sectors below be rotated to
        # follow the grating pattern when it isn't perfectly aligned with the
        # detector's vertical/horizontal axes (see determine_pattern_angle in
        # detector_calibration.py).
        self.azimuth_offset = azimuth_offset
        # TODO work out how to configure the radial ranges ahead of time
        # and work out how to expose them?
        self.INTEGRATION_CONFIGS: dict[str, IntegrationConfig] = {
            "y0": {
                "npt": 50,
                "center": 90 + azimuth_offset,
                "radial_range": (0.005, 0.010),
            },
            "y1": {
                "npt": 50,
                "center": -90 + azimuth_offset,
                "radial_range": (0.005, 0.010),
            },
            "x0": {
                "npt": 50,
                "center": azimuth_offset,
                "radial_range": (0.0023, 0.0065),
            },
            "x1": {
                "npt": 50,
                "center": 180 + azimuth_offset,
                "radial_range": (0.0023, 0.0065),
            },
        }

        # input data
        self.image = image
        self.mask = mask
        # make sure that the image and the mask are the same shape
        if not self.image.shape == self.mask.shape:
            raise AssertionError("image and mask not the same shape!")

        self.wavelength = self.calculate_wavelength(beam_energy)

        # set up the azimuthal integrator
        self.ai = AzimuthalIntegrator(
            pixel1=self.PILATUS2M_PIXEL_SIZE,
            pixel2=self.PILATUS2M_PIXEL_SIZE,
            wavelength=self.wavelength,
        )
        # results store to be used during optimisation
        self._results_store: dict[str, NDArray[np.float64]] = {}

        self._optimise_direction = optimise_direction  # can only be x or y

        self.beamstop_center = beamstop_center
        self.offset = offset

        # where we'll store the results of the optimsed beam center
        self.beam_center: dict[str, float] = {}
        self.beam_center_global: dict[str, float] = {}
        self.profiles: dict[str, dict[str, Any]] = {}
        self.target_configs: dict[str, IntegrationConfig] | None = None

    @property
    def optimise_direction(self) -> str:
        return self._optimise_direction

    @optimise_direction.setter
    def optimise_direction(self, new_value: str) -> None:
        """
        set whether we're optimising in x or y. Assert that these are the only
        valid directions.
        """
        if new_value not in ["x", "y"]:
            raise ValueError("Optimise direction must be 'x' or 'y'")
        self._optimise_direction = new_value

    def _setup(self) -> None:
        """
        fix x or y depending on whether we're optimising y or x respectively,
        and set up the target configurations for where we'll do the integration.
        """
        if self.optimise_direction == "x":
            # fix y
            self.ai.poni1 = self.beamstop_center["y"] * self.PILATUS2M_PIXEL_SIZE
            self.target_configs = {
                key: value
                for key, value in self.INTEGRATION_CONFIGS.items()
                if "x" in key
            }
        elif self.optimise_direction == "y":
            # fix x
            self.ai.poni2 = self.beamstop_center["x"] * self.PILATUS2M_PIXEL_SIZE
            self.target_configs = {
                key: value
                for key, value in self.INTEGRATION_CONFIGS.items()
                if "y" in key
            }

    def calculate_wavelength(self, beam_energy: float) -> float:
        """
        calculate the wavelength from the beam energy. assumes energy in keV
        """
        c = 299792458  # m/s
        plank = 6.62607015e-34  # Js
        ev = 1.602176634e-19  # J
        wavelength = (c * plank) / (beam_energy * ev)  # m
        self.wavelength_units = "m"
        return wavelength

    def _finalise_residual(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        generate the correct residual from the results store depending
        on whether we're optimising x or y
        """
        if self.optimise_direction == "x":
            ix0 = self._results_store["Ix0"]
            ix1 = self._results_store["Ix1"]
            self.profiles["x"] = {"q": q, "Ix0": ix0, "Ix1": ix1}
            return ix1 - ix0

        else:
            self.profiles["y"] = {
                "q": q,
                "Iy0": self._results_store["Iy0"],
                "Iy1": self._results_store["Iy1"],
            }
            return self._results_store["Iy1"] - self._results_store["Iy0"]

    def _set_new_center(self, pos: float) -> None:
        """
        set the new center of integration depending on the target optimiser
        """
        if self.optimise_direction == "x":
            self.ai.poni2 = pos * self.PILATUS2M_PIXEL_SIZE
        elif self.optimise_direction == "y":
            self.ai.poni1 = pos * self.PILATUS2M_PIXEL_SIZE

    def _integrate_sector(
        self, config: IntegrationConfig
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        integrate the sector described by an IntegrationConfig. When the
        sector straddles the +/-180 degree azimuth seam this means combining
        two pyFAI integrations (one either side of the seam); the two
        resulting intensity profiles are averaged together since they're
        both part of the same logical sector.
        """
        windows = split_azimuth_window(config["center"], self.integration_range)

        q: NDArray[np.float64] | None = None
        intensities = []
        for window in windows:
            q, intensity = self.ai.integrate1d(
                self.image,
                mask=self.mask,
                unit="r_m",
                npt=config["npt"],
                azimuth_range=window,
                radial_range=config["radial_range"],
            )
            intensities.append(intensity)
        assert q is not None
        return q, np.mean(intensities, axis=0)

    def _make_beam_residual(self) -> Callable[..., NDArray[np.float64]]:
        """
        the main optimisation function to target for minimization.
        """

        def residual(pars: Any, **kws: Any) -> NDArray[np.float64]:
            assert self.target_configs is not None

            pos = pars["beam_center_pos"]
            self._set_new_center(pos)

            q: NDArray[np.float64] | None = None
            for key, config in self.target_configs.items():
                q, intensity = self._integrate_sector(config)
                self._results_store[f"I{key}"] = intensity
            assert q is not None
            return self._finalise_residual(q=q)

        return residual

    def fit_beam_centre(self) -> None:
        """
        beam center fitting function. set up and run the minimization routine.
        """

        # setup the integration. this will fix the appropriate
        self._setup()

        # set up the residual function for optimisation
        residual = self._make_beam_residual()

        params = Parameters()
        params.add(
            "beam_center_pos",
            value=self.beamstop_center[self.optimise_direction],
            min=self.beamstop_center[self.optimise_direction] - 2,
            max=self.beamstop_center[self.optimise_direction] + 2,
        )

        result: Any = minimize(
            residual,
            params,
            method="leastsq",
        )

        self.beam_center[self.optimise_direction] = result.params[
            "beam_center_pos"
        ].value
        if self.offset is not None:
            # need the -.5 correction for the rounding on the extent
            # TODO: DOUBLE CHECK THIS
            self.beam_center_global[self.optimise_direction] = (
                result.params["beam_center_pos"].value
                + self.offset[self.optimise_direction]
                - 0.5
            )

    def plots(self, extent: tuple[float, float, float, float]) -> None:
        self.profile_plotter()
        self.center_plotter(extent)

    def profile_plotter(self) -> None:
        """
        plot the dual profiles that have been matched and the difference
        """
        fig = plt.figure(figsize=(10, 7.5))
        fig.set_label("beam_profiles_xy")
        gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])
        ax3 = fig.add_subplot(gs[3])
        axarr = [[ax0, ax2], [ax1, ax3]]

        ax0.set_ylabel("ln(I)", fontsize=15)
        ax2.set_ylabel("Difference", fontsize=15)

        for ax_pair, profile_direction in zip(axarr, self.profiles.keys(), strict=True):
            data = self.profiles.get(profile_direction)

            main, residual = ax_pair

            if data is not None:
                profile_keys = [key for key in data.keys() if key != "q"]
                profiles = []
                for key, colour in zip(
                    profile_keys, ["#4C9C88", "#9C5F4C"], strict=True
                ):
                    main.plot(data["q"], data[key], label=key, c=colour, lw=2)
                    main.set_title(profile_direction)
                    profiles.append(data[key])

                main.legend()
                main.set_xticklabels([])

                profile_diff = profiles[1] - profiles[0]
                residual.plot(data["q"], profile_diff, c="#332A31")
                residual.set_xlabel(
                    f"Detector distance (m)\n(sos = {sum(profile_diff**2):.2f})",
                    fontsize=15,
                )

    def center_plotter(self, extent: tuple[float, float, float, float]) -> None:

        fig, ax = plt.subplots()
        fig.set_label("beam_center_location")

        ax.imshow(self.image, extent=extent, aspect="equal")

        ax.axvline(self.beamstop_center["x"] + extent[0], ls=":", lw=1, c="#ffffc1")
        ax.axhline(
            self.beamstop_center["y"] + extent[3],
            ls=":",
            lw=1,
            c="#ffffc1",
            label="Centre of beamstop",
        )

        ax.axvline(
            self.beam_center["x"] + extent[0],
            c="#ff028d",
            ls=":",
            lw=1,
            label="Centre of beam",
        )
        ax.axhline(self.beam_center["y"] + extent[3], c="#ff028d", ls=":", lw=1)

        # lines through the beam centre showing the grating pattern's two
        # (perpendicular) fringe directions, i.e. the detector's
        # vertical/horizontal axes rotated by azimuth_offset.
        beam_point = (
            self.beam_center["x"] + extent[0],
            self.beam_center["y"] + extent[3],
        )
        for label, chi_deg in (
            (
                f"Pattern direction\n({self.azimuth_offset:.3f}°)",
                90 + self.azimuth_offset,
            ),
            (None, self.azimuth_offset),
        ):
            chi = np.deg2rad(chi_deg)
            direction_point = (
                beam_point[0] + np.cos(chi),
                beam_point[1] + np.sin(chi),
            )
            ax.axline(
                beam_point,
                direction_point,
                c="#B72818",
                ls=":",
                lw=1,
                label=label,
            )

        ax.legend(
            fontsize=5,
            loc="lower right",
            bbox_to_anchor=(1, 0),
            bbox_transform=ax.transAxes,
        )

    def tracker(
        self, params: Any, iter: int, resid: Any, *args: Any, **kws: Any
    ) -> None:
        itervalues = kws["itervalues"]
        itervalues["residuals"].append(resid.copy())
        itervalues["params"].append(list(params.valuesdict().values()))
