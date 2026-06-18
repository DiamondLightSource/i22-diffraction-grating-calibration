import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import LinearModel, VoigtModel
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from matplotlib.gridspec import GridSpec
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

def find_multiple_sequence(peak_dict, max_k=10, tol=0.05, max_multiple=20):

    peaks = [i['center'] for i in peak_dict.values()]

    # peaks = np.asarray(peak_array)[:,1]
    peaks = np.sort(peaks)

    # Candidate fundamentals
    candidates = []
    for k in range(1, max_k + 1):
        candidates.extend(peaks / k)

    candidates = np.array(candidates)

    best_f = None
    best_score = np.inf

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

        score = np.median(errors)

        if score < best_score:
            best_score = score
            best_f = f

    if best_f is None:
        raise RuntimeError("No valid fundamental found")

    # Final classification
    r = peaks / best_f
    k_round = np.round(r).astype(int)
    errors = np.abs(r - k_round)

    inliers = (errors < tol) & (k_round <= max_multiple)
    
    new_peak_idx = np.array(k_round[inliers], dtype=int)

    # here we convert the input keys (which run 0..n) to the multiplier keys (which should run 1..n)   
    # and account for anything that is extra/missing, so we end up with correctly indexed peaks
    required_original_peak_idx = np.array(list(peak_dict.keys()))[inliers]
    peak_idx_convert = {i:j for i,j in zip(required_original_peak_idx, new_peak_idx)}

    peak_dict_out = {peak_idx_convert[i]: peak_dict[i] for i in required_original_peak_idx}
    
    return peak_dict_out

class DetectorCalibration:
    
    def __init__(self, 
                 image,
                 beam_center,
                 wavelength,
                 peak_prominance=.1,
                 grating_spacing = 100e-9,
                 angle_region = 1,
                 ):
        
        self.image = image
        self.beam_center = beam_center
        self.wavelength = wavelength
        self.peak_prominance = peak_prominance
        self.grating_spacing = grating_spacing # 100 nm by default
        self.angle_region = angle_region
        self.PILATUS2M_PIXEL_SIZE = 172e-6 # TODO move away from ~ hard coding this 
        
        self.peaks = None
        self.fit_data = None
        self.detector_distance = None
    
    def _determine_radial_range(self):
        """
        determine an approximate radial range for integration.

        Take the beam centre and do bad integration down the detector where the peaks are

        Then use some basic signal processing to find the lower and upper limits for a radial integration range
        """

        avg = self.image[int(self.beam_center['y']):,
                         int(self.beam_center['x']-5):int(self.beam_center['x']+5)].sum(axis=1)
        x = np.arange(len(avg))
        # fig, ax = plt.subplots(2,1,sharex=True)
        # ax[0].plot(x, avg)

        # now do the upper limit.
        # take a moving average of the gradient of the signal.
        # this helps us find regions where the signal is not changing very much
        _filter = uniform_filter1d(np.gradient(avg), 10)
        # ie where we're getting any kind of significant change in the signal gradient
        regions = np.where(np.abs(_filter)>1)[0]
        # ax[1].plot(x, _filter)
        # ax[1].scatter(x[regions], _filter[regions], s=5, c='#262626')
        # where we have large gaps between the regions, because we're only picking up detector segment dead zones
        exclude = np.where(np.diff(regions)>100)[0]
        
        # take the index of the first point and add a bit
        cut = x[regions][exclude[0]] + 20
        # convert it to detector distance
        upper = cut * self.PILATUS2M_PIXEL_SIZE

        # If we still have some beamstop in the signal we want to make sure we're not
        # including it as a peak.
        # look ahead 20 points .
        # if the average intensity 5 points ahead is higher in the first
        # section of the signal then we're still in the beamstop region
        descending = [j > avg[i:i+10].mean() for i,j in enumerate(avg[:cut-10])]
        lower = x[:cut-10][descending][0] * self.PILATUS2M_PIXEL_SIZE
        # ax[0].axvline(x[:cut-10][descending][0])
        # plt.show()

        return (lower, upper)
        
    def make_signal(self, npt=500, auto_radial=None):
        """
        perform the azimuthal integration of the detector image.
        The radial range of the integration is determined by the _determine_radial_range
        function if a range is not explicitly given.

        npt: int
            number of points to integrate with in the radial range.
        """
        
        ai = AzimuthalIntegrator(
            poni1 = self.beam_center['y'] * self.PILATUS2M_PIXEL_SIZE,
            poni2 = self.beam_center['x'] * self.PILATUS2M_PIXEL_SIZE,
            pixel1 = self.PILATUS2M_PIXEL_SIZE,
            pixel2 = self.PILATUS2M_PIXEL_SIZE,
            wavelength = self.wavelength,
        )
        
        # make an I vs. q profile from a small sector around +y
        q, I = ai.integrate1d(
            self.image,
            npt=npt,
            azimuth_range=(90 - self.angle_region, 
                           90 + self.angle_region),
            radial_range = self._determine_radial_range() if auto_radial is None else auto_radial,
            unit='r_m',
            method='csr'
        )
        
        return np.array([q,I])
     
    def peak_fitter(self):
        
        self.profile = self.make_signal()
        q,I = self.profile

        # find what we think are the peaks        
        peaks,_ = find_peaks(I, prominence = self.peak_prominance)
        # fit all the peaks using a Voigt peak + a linear background
        fit_store = {}
        for idx,p in enumerate(peaks):
        
            peak_mod = VoigtModel(prefix='p_')
            line_mod = LinearModel(prefix='lin_')
            mod=peak_mod+line_mod
        
            x = q[max(0, p-5):min(q.size, p+6)]
            y = I[max(0, p-5):min(q.size, p+6)]
                
            params = line_mod.make_params(intercept=y.min(), slope=0)
            params += peak_mod.guess(y, x=x)
        
            res = mod.fit(y,params,x=x)
        
            x_plt = np.linspace(x.min(), x.max(), num=100)
            y_plt = res.eval(x=x_plt)

            # some conditions for making sure we have a good peak fit.
            # condition 0: positive amplitude of the peak
            cond_0 = np.sign(res.params['p_amplitude'].value)>0
            # condition 1: the peak centre is fitted within the bounds that we give
            cond_1 = x_plt[0] < res.params['p_center'].value < x_plt[-1]
            # condition 2: the r squared value of the fit is not terrible
            cond_2 = res.rsquared > 0.9
            # if we meet all the conditions, store the result.
            if all([cond_0, cond_1, cond_2]):
                fit_store[idx] = {'data': np.array([x,y]),
                                  'profile': np.array([x_plt,y_plt]),
                                  'center': res.params['p_center'].value,
                                  }
    
        final_fit_store = find_multiple_sequence(fit_store)
        peaks = np.array([np.array(list(final_fit_store.keys())),
                          np.array([i['center'] for i in final_fit_store.values()])]
                          ).T
        self.fit_data = final_fit_store
        return peaks

    def calculate_detector_distance(self):
        if self.peaks is None:
            self.peaks = self.peak_fitter()

        lin = LinearModel()
        lin_pars = lin.guess(self.peaks[:,1], x=self.peaks[:,0])
        lin_res = lin.fit(self.peaks[:,1], lin_pars, x=self.peaks[:,0])
        
        fringe_spacing = lin_res.params['slope'].value
        # should do some unit assertion around here
        self.detector_distance = fringe_spacing * self.grating_spacing / self.wavelength
        self.detector_distance_units = 'm'
    
    def plots(self):
        
        required_peaks = self.peaks[:,0]
        
        ncols = 3
        # len(required_peaks)+1 because we want to plot the indexing as a bonus
        nrows = np.ceil((len(required_peaks)+1) / (ncols - 1)).astype(int)
        fig = plt.figure(figsize=(4*ncols, 3*nrows))
        fig.set_label('detector_calibration')
        gs = GridSpec(nrows = nrows, 
                      ncols = ncols,
                      figure = fig
                      )
        ax_top = fig.add_subplot(gs[0,:])
        ax_top.plot(self.profile[0],
                    self.profile[1],
                    marker='.',
                    c='#262626'
                    )
        
        # --- Remaining rows: normal grid ---
        for idx, key in enumerate(required_peaks):
            row = 1 + (idx // ncols)   # shift by 1 because row 0 is occupied
            col = idx % ncols
    
            ax = fig.add_subplot(gs[row, col])
            data = self.fit_data[key]
            ax.scatter(data['data'][0],
                       data['data'][1],
                       label='Data',
                       c='#262626'
                       )
            ax.plot(data['profile'][0],
                    data['profile'][1],
                    c='hotpink',
                    label='Fitted peak'
                    )
            ax_top.axvline(data['center'], c='#262626', ls='--', lw=.5)
            ax.axvline(data['center'], c='#262626', ls='--', lw=1, 
                       label='Fitted center')
            ax.set_xticks([data['data'][0][0],
                           data['center'],
                           data['data'][0][-1]
                           ],
                          [f"{data['data'][0][0]:.4f}",
                           f"{data['center']:.4f}",
                           f"{data['data'][0][-1]:.4f}"])
            
            ax_top.text(data['center'],
                        0,
                        str(key),
                        ha='right',
                        fontweight='bold'
                        )
            ax.text(0.05,0.95,
                    str(key),
                    ha='left',va='top',
                    transform=ax.transAxes,
                    fontweight='bold'
                    )
        
        ax.legend(loc='upper right',
                  bbox_to_anchor=(0.99,0.99),
                  bbox_transform=ax_top.transAxes
                  )

        ax_final = fig.add_subplot(gs[1 + ((idx+1) // ncols), ((idx+1) % ncols)])
        ax_final.scatter(self.peaks[:,0],
                         self.peaks[:,1])
        ax_final.set_xticks(self.peaks[:,0],
                            [str(int(i)) for i in self.peaks[:,0]],
                            )
        
        
        lin = LinearModel()
        lin_pars = lin.guess(self.peaks[:,1], x=self.peaks[:,0])
        lin_res = lin.fit(self.peaks[:,1], lin_pars, x=self.peaks[:,0])
        ax_final.plot(np.arange(0, self.peaks[:,0][-1]+2),
                lin_res.eval(x=np.arange(0, self.peaks[:,0][-1]+2)),
                c='#262626', ls='--')
        ax_final.set_xlabel('Peak order')
        ax_final.set_ylabel('q')
        
        # --- Hide unused axes ---
        total_cells = (nrows - 1) * ncols
        for j in range(len(required_peaks)+1, total_cells):
            row = 1 + (j // ncols)
            col = j % ncols
            ax = fig.add_subplot(gs[row, col])
            ax.set_visible(False)

        fig.subplots_adjust(hspace=.3,wspace=.3)
