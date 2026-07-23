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
    azimuth_range: tuple[float, float]
    radial_range: tuple[float, float]


class BeamCenterOptimiser:
    def __init__(
        self,
        image: NDArray[Any],
        mask: NDArray[Any],
        beam_energy: float,
        beamstop_center: dict[str, float],
        optimise_direction: str = "x",
        offset: dict[str, float] | None = None,
    ) -> None:
        # some constants for the integration
        self.PILATUS2M_PIXEL_SIZE = 172e-6
        integration_range = 4
        # TODO work out how to configure the radial ranges ahead of time
        # and work out how to expose them?
        self.INTEGRATION_CONFIGS: dict[str, IntegrationConfig] = {
            "y0": {
                "npt": 50,
                "azimuth_range": (90 - integration_range, 90 + integration_range),
                "radial_range": (0.005, 0.010),
            },
            "y1": {
                "npt": 50,
                "azimuth_range": (-90 - integration_range, -90 + integration_range),
                "radial_range": (0.005, 0.010),
            },
            "x0": {
                "npt": 50,
                "azimuth_range": (-integration_range, integration_range),
                "radial_range": (0.0023, 0.0065),
            },
            "x1a": {
                "npt": 50,
                "azimuth_range": (180 - integration_range, 180),
                "radial_range": (0.0023, 0.0065),
            },
            "x1b": {
                "npt": 50,
                "azimuth_range": (-180, -180 + integration_range),
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
            ix1 = 0.5 * (self._results_store["Ix1a"] + self._results_store["Ix1b"])
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

    def _make_beam_residual(self) -> Callable[..., NDArray[np.float64]]:
        """
        the main optimisation function to target for minimization.
        """

        def residual(pars: Any, **kws: Any) -> NDArray[np.float64]:
            assert self.target_configs is not None

            pos = pars["beam_center_pos"]
            self._set_new_center(pos)

            q: NDArray[np.float64] | None = None
            for key, kw in self.target_configs.items():
                q, intensity = self.ai.integrate1d(
                    self.image,
                    mask=self.mask,
                    unit="r_m",
                    npt=kw["npt"],
                    azimuth_range=kw["azimuth_range"],
                    radial_range=kw["radial_range"],
                )
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

        ax.legend(
            fontsize=5,
            loc="center right",
            bbox_to_anchor=(1, 0.2),
            bbox_transform=ax.transAxes,
        )

    def tracker(
        self, params: Any, iter: int, resid: Any, *args: Any, **kws: Any
    ) -> None:
        itervalues = kws["itervalues"]
        itervalues["residuals"].append(resid.copy())
        itervalues["params"].append(list(params.valuesdict().values()))
