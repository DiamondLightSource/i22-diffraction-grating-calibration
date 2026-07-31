from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LinearModel, VoigtModel
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from gratingcalibration.beam_center_optimiser import split_azimuth_window


def find_multiple_sequence(
    peak_dict: dict[Any, dict[str, Any]],
    max_k: int = 10,
    tol: float = 0.05,
    max_multiple: int = 20,
    score_percentile: float = 75,
    parsimony_band: float = 1.5,
) -> dict[Any, dict[str, Any]]:

    peaks = [i["center"] for i in peak_dict.values()]

    # peaks = np.asarray(peak_array)[:,1]
    peaks = np.sort(peaks)

    # Candidate fundamentals
    candidates = []
    for k in range(1, max_k + 1):
        candidates.extend(peaks / k)

    candidates = np.array(candidates)

    scored: list[tuple[float, float]] = []

    for f in candidates:
        if np.isclose(f, 0):
            continue

        r = peaks / f
        k_round = np.round(r)

        # Reject if implies large multiples
        if np.max(k_round) > max_multiple:
            continue

        # Avoid divide-by-zero
        denom = np.maximum(1, k_round)
        errors = np.abs(r - k_round) / denom

        # a high percentile (rather than the median) of the relative errors
        # means the score stays low only when *most* peaks are well
        # explained, not just at least half of them.
        score = float(np.percentile(errors, score_percentile))
        scored.append((score, f))

    if not scored:
        raise RuntimeError("No valid fundamental found")

    # A fundamental f and any of its exact fractions (f/2, f/3, ...) explain
    # the same peaks about equally well, just with proportionally larger
    # indices - so picking whichever scores (even marginally) lowest is
    # unreliable and biases towards spuriously large multiples. Instead,
    # among all fundamentals that explain the peaks about as well as the
    # best one, prefer the largest: the simplest indexing consistent with
    # the data.
    best_score = min(score for score, _ in scored)
    band = [f for score, f in scored if score <= best_score * parsimony_band + 1e-9]
    best_f = max(band)

    # Final classification
    r = peaks / best_f
    k_round = np.round(r).astype(int)
    errors = np.abs(r - k_round)

    inliers = (errors < tol) & (k_round <= max_multiple)

    new_peak_idx = np.array(k_round[inliers], dtype=int)

    # here we convert the input keys (which run 0..n) to
    # the multiplier keys (which should run 1..n)
    # and account for anything that is extra/missing, so we
    # end up with correctly indexed peaks
    required_original_peak_idx = np.array(list(peak_dict.keys()))[inliers]
    peak_idx_convert = dict(zip(required_original_peak_idx, new_peak_idx, strict=True))

    peak_dict_out = {
        peak_idx_convert[i]: peak_dict[i] for i in required_original_peak_idx
    }

    return peak_dict_out


def determine_pattern_angle(
    image: NDArray[Any],
    beam_center: dict[str, float],
    pixel_size: float,
    wavelength: float = 1e-10,
    mask: NDArray[Any] | None = None,
    radial_range: tuple[float, float] = (0.0023, 0.010),
    npt: int = 1440,
    prominence: float = 0.3,
) -> float:
    """
    Determine how far the grating pattern is rotated (in degrees, modulo 90)
    away from being aligned with the detector's vertical/horizontal axes.

    Builds an intensity-vs-azimuthal-angle ("cake") profile close to the beam
    centre, where fringe orders from both of the grating's perpendicular
    families are normally visible, and finds the common phase of their 4-fold
    rotational symmetry. Averaging over all the peaks found (weighted by how
    prominent each one is) means this works even if the pattern is only
    visible as two asymmetric lobes either side of the beamstop, and is
    unaffected by which of the two families happens to be the stronger one -
    see find_calibration_azimuth for that.
    """
    ai = AzimuthalIntegrator(
        poni1=beam_center["y"] * pixel_size,
        poni2=beam_center["x"] * pixel_size,
        pixel1=pixel_size,
        pixel2=pixel_size,
        wavelength=wavelength,
    )
    chi, intensity = ai.integrate_radial(
        image, npt=npt, radial_range=radial_range, radial_unit="r_m", mask=mask
    )
    peaks, props = find_peaks(intensity, prominence=prominence)
    if peaks.size == 0:
        return 0.0

    weights = props["prominences"]
    phase = 4 * np.deg2rad(chi[peaks])
    theta = (
        np.rad2deg(
            np.arctan2(
                np.sum(weights * np.sin(phase)), np.sum(weights * np.cos(phase))
            )
        )
        / 4
    )
    return float(theta)


def find_calibration_azimuth(
    image: NDArray[Any],
    beam_center: dict[str, float],
    pixel_size: float,
    wavelength: float = 1e-10,
    mask: NDArray[Any] | None = None,
    theta: float | None = None,
    angle_region: float = 1.0,
) -> float:
    """
    Work out which azimuth angle the detector calibration should integrate
    along.

    The grating produces two fringe families at right angles to each other;
    once the pattern's rotation (theta, modulo 90 degrees - see
    determine_pattern_angle) is known, the family with the most/strongest
    higher-order fringes sits at either chi = 90 + theta or chi = theta. Which
    one that is isn't fixed: a rotation anywhere close to 90 degrees would
    swap them over, so both are actually probed (by looking for peaks well
    away from the beamstop, where only the "good" calibration direction still
    has resolvable fringes) rather than assuming it's always the nominally
    vertical one.
    """
    if theta is None:
        theta = determine_pattern_angle(
            image, beam_center, pixel_size, wavelength=wavelength, mask=mask
        )

    ai = AzimuthalIntegrator(
        poni1=beam_center["y"] * pixel_size,
        poni2=beam_center["x"] * pixel_size,
        pixel1=pixel_size,
        pixel2=pixel_size,
        wavelength=wavelength,
    )

    ny, nx = image.shape

    def max_radius(azimuth_center: float) -> float:
        # how far the detector extends in whichever cardinal direction is
        # nearest this azimuth - the useful radial range for probing a given
        # direction depends on which way it points, since the beam centre is
        # not usually equidistant from all four detector edges.
        azimuth = azimuth_center % 360
        edges = {
            0: nx - beam_center["x"],
            90: ny - beam_center["y"],
            180: beam_center["x"],
            270: beam_center["y"],
        }
        nearest = min(
            edges, key=lambda deg: min(abs(azimuth - deg), 360 - abs(azimuth - deg))
        )
        return edges[nearest] * pixel_size

    def strength(azimuth_center: float) -> float:
        radial_range = (0.01, max(0.02, 0.9 * max_radius(azimuth_center)))
        total = 0.0
        for window in split_azimuth_window(azimuth_center, angle_region):
            _, intensity = ai.integrate1d(
                image,
                npt=1000,
                azimuth_range=window,
                radial_range=radial_range,
                unit="r_m",
                mask=mask,
                method="csr",
            )
            _, props = find_peaks(intensity, prominence=0.3)
            total += props["prominences"].sum()
        return total

    candidates = (90 + theta, theta)
    return max(candidates, key=strength)


class DetectorCalibration:
    def __init__(
        self,
        image: NDArray[Any] | None,
        beam_center: dict[str, float] | None,
        wavelength: float,
        peak_prominance: float = 0.1,
        grating_spacing: float = 100e-9,
        angle_region: float = 1,
        mask: NDArray[Any] | None = None,
        azimuth_center: float = 90.0,
        min_peak_snr: float = 15.0,
    ) -> None:

        self.image = image
        self.beam_center = beam_center
        self.wavelength = wavelength
        self.peak_prominance = peak_prominance
        self.grating_spacing = grating_spacing  # 100 nm by default
        self.angle_region = angle_region
        self.mask = mask
        # the azimuth angle (pyFAI convention, degrees) along which the
        # grating fringes run. Defaults to 90 (straight down the detector);
        # pass the value from find_calibration_azimuth when the grating
        # pattern isn't aligned with the detector axes.
        self.azimuth_center = azimuth_center
        # minimum ratio of a candidate peak's prominence to the profile's own
        # point-to-point noise level for it to be trusted as a real fringe
        # rather than a noise fluctuation (see peak_fitter).
        self.min_peak_snr = min_peak_snr
        self.PILATUS2M_PIXEL_SIZE = 172e-6  # TODO move away from ~ hard coding this

        self.peaks: NDArray[np.float64] | None = None
        self.fit_data: dict[Any, dict[str, Any]] | None = None
        self.detector_distance: float | None = None

    def _scan_direction(self) -> tuple[int, int]:
        """
        the (dy, dx) unit step, in pixels, that most closely matches
        self.azimuth_center, using pyFAI's convention that chi=0 points along
        +x and chi=90 points along +y.
        """
        azimuth = self.azimuth_center % 360
        cardinal_directions = {0: (0, 1), 90: (1, 0), 180: (0, -1), 270: (-1, 0)}
        nearest = min(
            cardinal_directions,
            key=lambda deg: min(abs(azimuth - deg), 360 - abs(azimuth - deg)),
        )
        return cardinal_directions[nearest]

    def _radial_scan_signal(self, half_width: int) -> NDArray[np.float64]:
        """
        sum a strip of the image, `half_width` pixels either side of the beam
        centre, running away from the beam centre along whichever cardinal
        direction (down/up/left/right) is closest to self.azimuth_center.
        Used as a cheap 1D proxy for where the grating fringes/detector
        dead-zones are, without doing a full azimuthal integration.
        """
        assert self.image is not None
        assert self.beam_center is not None

        y, x = int(self.beam_center["y"]), int(self.beam_center["x"])
        dy, dx = self._scan_direction()

        avg: NDArray[np.float64]
        if dy == 1:
            avg = self.image[y:, x - half_width : x + half_width].sum(axis=1)
        elif dy == -1:
            avg = self.image[: y + 1, x - half_width : x + half_width][::-1].sum(
                axis=1
            )
        elif dx == 1:
            avg = self.image[y - half_width : y + half_width, x:].sum(axis=0)
        else:
            avg = self.image[y - half_width : y + half_width, : x + 1][:, ::-1].sum(
                axis=0
            )
        return avg

    def _determine_radial_range(self) -> tuple[float, float]:
        """
        determine an approximate radial range for integration.

        Take the beam centre and do bad integration down the
        detector where the peaks are

        Then use some basic signal processing to find the lower and
        upper limits for a radial integration range.

        The column summed to make this 1D signal starts out narrow (a
        handful of pixels either side of the beam centre), which is enough
        when the grating fringes run essentially straight down the detector.
        If that isn't enough to pick out a detector module gap (e.g. because
        the diffraction pattern isn't well aligned with the detector's
        vertical axis, or the signal is just noisier), the column is widened
        and the search retried, since a wider column keeps picking up fringes
        that have drifted sideways and averages out noise.
        """
        regions = np.array([], dtype=np.intp)
        exclude = np.array([], dtype=np.intp)
        avg = np.array([], dtype=np.float64)

        for half_width in (5, 10, 20, 30, 40):
            avg = self._radial_scan_signal(half_width)
            x = np.arange(len(avg))
            # fig, ax = plt.subplots(2,1,sharex=True)
            # ax[0].plot(x, avg)

            # now do the upper limit.
            # take a moving average of the gradient of the signal.
            # this helps us find regions where the signal is not changing much
            _filter = uniform_filter1d(np.gradient(avg), 10)
            # ie where we're getting any significant change in the signal gradient
            regions = np.where(np.abs(_filter) > 1)[0]
            # ax[1].plot(x, _filter)
            # ax[1].scatter(x[regions], _filter[regions], s=5, c='#262626')
            # where we have large gaps between the regions, because we're only
            # picking up detector segment dead zones
            exclude = np.where(np.diff(regions) > 100)[0]

            if exclude.size > 0:
                break

        if exclude.size > 0:
            # take the index of the first point and add a bit
            cut = x[regions][exclude[0]] + 20
        else:
            # never found a detector dead-zone gap, even in the widest column:
            # fall back to wherever the peak-containing signal itself runs out
            cut = int(regions[-1] + 20) if regions.size > 0 else len(avg) - 1
        cut = min(cut, len(avg) - 1)
        # convert it to detector distance
        upper = cut * self.PILATUS2M_PIXEL_SIZE

        # If we still have some beamstop in the signal we want to make sure we're not
        # including it as a peak.
        # look ahead 20 points.
        # if the average intensity 5 points ahead is higher in the first
        # section of the signal then we're still in the beamstop region.
        # smooth first so that noise right next to the beamstop edge doesn't
        # get mistaken for the start of the descent.
        smoothed = uniform_filter1d(avg, 5)
        descending = [
            j > smoothed[i : i + 10].mean() for i, j in enumerate(smoothed[: cut - 10])
        ]
        if any(descending):
            lower = x[: cut - 10][descending][0] * self.PILATUS2M_PIXEL_SIZE
        else:
            # never found a descending point (e.g. the signal is still
            # rising when the cut is reached): fall back to the brightest
            # point before the cut, which is the edge of the beamstop
            # shadow/halo.
            lower = (
                int(np.argmax(smoothed[: max(cut - 10, 1)])) * self.PILATUS2M_PIXEL_SIZE
            )
        # ax[0].axvline(x[:cut-10][descending][0])
        # plt.show()

        return (float(lower), float(upper))

    def make_signal(
        self, npt: int = 500, auto_radial: tuple[float, float] | None = None
    ) -> NDArray[np.float64]:
        """
        perform the azimuthal integration of the detector image.
        The radial range of the integration is determined by the _determine_radial_range
        function if a range is not explicitly given.

        npt: int
            number of points to integrate with in the radial range.
        """
        assert self.image is not None
        assert self.beam_center is not None

        ai = AzimuthalIntegrator(
            poni1=self.beam_center["y"] * self.PILATUS2M_PIXEL_SIZE,
            poni2=self.beam_center["x"] * self.PILATUS2M_PIXEL_SIZE,
            pixel1=self.PILATUS2M_PIXEL_SIZE,
            pixel2=self.PILATUS2M_PIXEL_SIZE,
            wavelength=self.wavelength,
        )

        radial_range = (
            self._determine_radial_range() if auto_radial is None else auto_radial
        )

        # make an I vs. q profile from a small sector centred on
        # self.azimuth_center (following the grating fringes, which run
        # straight down the detector - chi=90 - only when the pattern is
        # perfectly aligned with the detector axes). The sector is split in
        # two if it straddles the +/-180 degree azimuth seam.
        q: NDArray[np.float64] | None = None
        intensities = []
        for window in split_azimuth_window(self.azimuth_center, self.angle_region):
            q, intensity = ai.integrate1d(
                self.image,
                npt=npt,
                azimuth_range=window,
                radial_range=radial_range,
                unit="r_m",
                method="csr",
                mask=self.mask,
            )
            intensities.append(intensity)
        assert q is not None

        return np.array([q, np.mean(intensities, axis=0)])

    def peak_fitter(self) -> NDArray[np.float64]:

        self.profile = self.make_signal()
        q, intensity = self.profile

        # find what we think are the peaks
        peaks, props = find_peaks(intensity, prominence=self.peak_prominance)

        # A fixed prominence threshold doesn't adapt to how noisy a given
        # profile is, so a genuine fringe peak in a low-signal dataset can be
        # about as prominent as a noise fluctuation in a cleaner one (or vice
        # versa). Instead, additionally require each peak's prominence to be
        # a healthy multiple of the profile's own point-to-point noise level,
        # estimated robustly (so it isn't thrown off by the peaks themselves).
        noise_floor = 1.4826 * np.median(
            np.abs(np.diff(intensity) - np.median(np.diff(intensity)))
        )
        if noise_floor > 0:
            snr = props["prominences"] / noise_floor
            peaks = peaks[snr > self.min_peak_snr]

        # fit all the peaks using a Voigt peak + a linear background
        fit_store: dict[Any, dict[str, Any]] = {}
        for idx, p in enumerate(peaks):
            peak_mod = VoigtModel(prefix="p_")
            line_mod = LinearModel(prefix="lin_")
            mod = peak_mod + line_mod

            x = q[max(0, p - 5) : min(q.size, p + 6)]
            y = intensity[max(0, p - 5) : min(q.size, p + 6)]

            params = line_mod.make_params(intercept=y.min(), slope=0)
            params += peak_mod.guess(y, x=x)

            res = mod.fit(y, params, x=x)

            x_plt = np.linspace(x.min(), x.max(), num=100)
            y_plt = res.eval(x=x_plt)

            # some conditions for making sure we have a good peak fit.
            # condition 0: positive amplitude of the peak
            cond_0 = np.sign(res.params["p_amplitude"].value) > 0
            # condition 1: the peak centre is fitted within the bounds that we give
            cond_1 = x_plt[0] < res.params["p_center"].value < x_plt[-1]
            # condition 2: the r squared value of the fit is not terrible
            cond_2 = res.rsquared > 0.9
            # if we meet all the conditions, store the result.
            if all([cond_0, cond_1, cond_2]):
                fit_store[idx] = {
                    "data": np.array([x, y]),
                    "profile": np.array([x_plt, y_plt]),
                    "center": res.params["p_center"].value,
                }

        final_fit_store = find_multiple_sequence(fit_store)
        peaks = np.array(
            [
                np.array(list(final_fit_store.keys())),
                np.array([i["center"] for i in final_fit_store.values()]),
            ]
        ).T
        self.fit_data = final_fit_store
        return peaks

    def calculate_detector_distance(self) -> None:
        if self.peaks is None:
            self.peaks = self.peak_fitter()

        lin = LinearModel()
        lin_pars = lin.guess(self.peaks[:, 1], x=self.peaks[:, 0])
        lin_res = lin.fit(self.peaks[:, 1], lin_pars, x=self.peaks[:, 0])

        fringe_spacing = lin_res.params["slope"].value
        # should do some unit assertion around here
        self.detector_distance = fringe_spacing * self.grating_spacing / self.wavelength
        self.detector_distance_units = "m"

    def plots(self) -> None:
        assert self.peaks is not None
        assert self.fit_data is not None

        required_peaks = self.peaks[:, 0]

        ncols = 3
        # len(required_peaks)+1 because we want to plot the indexing as a bonus
        nrows = np.ceil((len(required_peaks) + 1) / (ncols - 1)).astype(int)
        fig = plt.figure(figsize=(4 * ncols, 3 * nrows))
        fig.set_label("detector_calibration")
        gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig)
        ax_top = fig.add_subplot(gs[0, :])
        ax_top.plot(self.profile[0], self.profile[1], marker=".", c="#262626")

        # --- Remaining rows: normal grid ---
        idx, ax = -1, ax_top
        for idx, key in enumerate(required_peaks):
            row = 1 + (idx // ncols)  # shift by 1 because row 0 is occupied
            col = idx % ncols

            ax = fig.add_subplot(gs[row, col])
            data = self.fit_data[key]
            ax.scatter(data["data"][0], data["data"][1], label="Data", c="#262626")
            ax.plot(
                data["profile"][0], data["profile"][1], c="hotpink", label="Fitted peak"
            )
            ax_top.axvline(data["center"], c="#262626", ls="--", lw=0.5)
            ax.axvline(
                data["center"], c="#262626", ls="--", lw=1, label="Fitted center"
            )
            ax.set_xticks(
                [data["data"][0][0], data["center"], data["data"][0][-1]],
                [
                    f"{data['data'][0][0]:.4f}",
                    f"{data['center']:.4f}",
                    f"{data['data'][0][-1]:.4f}",
                ],
            )

            ax_top.text(data["center"], 0, str(key), ha="right", fontweight="bold")
            ax.text(
                0.05,
                0.95,
                str(key),
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontweight="bold",
            )

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            bbox_transform=ax_top.transAxes,
        )

        ax_final = fig.add_subplot(gs[1 + ((idx + 1) // ncols), ((idx + 1) % ncols)])
        ax_final.scatter(self.peaks[:, 0], self.peaks[:, 1])
        ax_final.set_xticks(
            self.peaks[:, 0],
            [str(int(i)) for i in self.peaks[:, 0]],
        )

        lin = LinearModel()
        lin_pars = lin.guess(self.peaks[:, 1], x=self.peaks[:, 0])
        lin_res = lin.fit(self.peaks[:, 1], lin_pars, x=self.peaks[:, 0])
        ax_final.plot(
            np.arange(0, self.peaks[:, 0][-1] + 2),
            np.asarray(lin_res.eval(x=np.arange(0, self.peaks[:, 0][-1] + 2))),
            c="#262626",
            ls="--",
        )
        ax_final.set_xlabel("Peak order")
        ax_final.set_ylabel("q")

        # --- Hide unused axes ---
        total_cells = (nrows - 1) * ncols
        for j in range(len(required_peaks) + 1, total_cells):
            row = 1 + (j // ncols)
            col = j % ncols
            ax = fig.add_subplot(gs[row, col])
            ax.set_visible(False)

        fig.subplots_adjust(hspace=0.3, wspace=0.3)
