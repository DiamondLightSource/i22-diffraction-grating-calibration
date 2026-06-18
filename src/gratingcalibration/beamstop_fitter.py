
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.special import erf
from lmfit import Model
from matplotlib.patches import Circle

class FitBeamstop:
    
    def __init__(self, image, plot=False):
        self.image = image
        self.result = None
        self.beamstop_center = self.find_beamstop(plot=plot)

    @staticmethod
    def soft_disk_2d(x, y, x0, y0, radius, amplitude, background, sigma):
        """
        Soft-edged bright circular disc model.
        """
        r = np.sqrt((x - x0)**2 + (y - y0)**2)
        edge = 0.5 * (1.0 - erf((r - radius) / (np.sqrt(2) * sigma)))
        return background + amplitude * edge
    
    def fit_bright_circle(self, image, lo_x=0, lo_y=0, radius_guess=20):
        """
        Fit a bright approximately circular region in a cropped image, but return
        the fitted centre in the coordinate system of the original full image.
    
        Parameters
        ----------
        image : 2D numpy array
            Cropped image used for fitting.
        lo_x, lo_y : int or float, optional
            Top-left corner of the cropped region in the original image.
            For example, if:
                image = full_image[lo_y:hi_y, lo_x:hi_x]
            then pass the same lo_x and lo_y here.
        radius_guess : float, optional
            Initial guess for the circle radius in pixels.
    
        Returns
        -------
        result : lmfit.model.ModelResult
            Full lmfit result object.
        centre_global : tuple
            (x0, y0) fitted centre in original full-image coordinates.
        radius_fit : float
            Fitted radius.
        """
        image = np.asarray(image, dtype=float)
        ny, nx = image.shape
    
        # Coordinates in LOCAL crop frame
        yy_local, xx_local = np.indices(image.shape)
    
        # Coordinates in GLOBAL full-image frame
        xx = xx_local + lo_x
        yy = yy_local + lo_y
    
        x = xx.ravel()
        y = yy.ravel()
        z = image.ravel()
    
        # --- Initial guesses ---
        background0 = np.percentile(image, 10)
        high0 = np.percentile(image, 99)
        amplitude0 = max(high0 - background0, 1.0)
    
        threshold = background0 + 0.5 * amplitude0
        mask = image > threshold
    
        if np.any(mask):
            weights = image * mask
            total = weights.sum()
    
            if total > 0:
                # Initial centre estimate in LOCAL coords
                y0_local = (yy_local * weights).sum() / total
                x0_local = (xx_local * weights).sum() / total
            else:
                y0_local, x0_local = np.unravel_index(np.argmax(image), image.shape)
    
            # Convert initial guess to GLOBAL coords
            x0_0 = x0_local + lo_x
            y0_0 = y0_local + lo_y
    
            # Radius estimate from thresholded area
            area = np.count_nonzero(mask)
            radius0 = np.sqrt(area / np.pi)
        else:
            y0_local, x0_local = np.unravel_index(np.argmax(image), image.shape)
            x0_0 = x0_local + lo_x
            y0_0 = y0_local + lo_y
            radius0 = radius_guess
    
        if not np.isfinite(radius0) or radius0 <= 0:
            radius0 = radius_guess
    
        sigma0 = 1.5
    
        model = Model(self.soft_disk_2d, independent_vars=['x', 'y'])
    
        params = model.make_params(
            x0=float(x0_0),
            y0=float(y0_0),
            radius=float(radius0),
            amplitude=float(amplitude0),
            background=float(background0),
            sigma=float(sigma0),
        )
    
        # Bounds in GLOBAL coordinates
        params['x0'].set(min=lo_x, max=lo_x + nx - 1)
        params['y0'].set(min=lo_y, max=lo_y + ny - 1)
        params['radius'].set(min=1, max=min(nx, ny) / 2)
        params['amplitude'].set(min=0)
        params['sigma'].set(min=0.2, max=10)
    
        result = model.fit(z, params, x=x, y=y)
        
        self.extent = [lo_x - 0.5, lo_x + nx - 0.5, lo_y + ny - 0.5, lo_y - 0.5]

        return result
    
    def plot_circle_fit(self, image, fit_image, 
                        x0, y0, radius,
                        extent        
                        ):
    
        residual = image - fit_image
    
        fig, (ax0,ax1,ax2) = plt.subplots(1, 3, 
                                          figsize=(20,7.5)
                                          )
    
        # =========================
        # 1. Input + fitted circle
        # =========================
        im0 = ax0.imshow(
            image,
            cmap='viridis',
            origin='upper',
            extent=self.extent,
            interpolation='nearest',
        )
        ax0.add_patch(Circle((x0, y0), radius, fill=False, color='red', lw=2))
        ax0.plot(x0, y0, 'rx', ms=10, mew=2)
        ax0.axvline(x0, color='white', linestyle='--', alpha=0.6, lw=1)
        ax0.axhline(y0, color='white', linestyle='--', alpha=0.6, lw=1)
        ax0.set_title('Input (intensity reversed) + fitted circle')
    
        ax0.text(-.1,0.5,
                  'Fitting beamstop\nwith disc', 
                  fontsize=15,
                  ha='right', va='center',ma='center',
                  transform=ax0.transAxes,
                  rotation='vertical'
                  )
        ax0.set_aspect('equal')
        # ax0.set_box_aspect(1)
        div0 = make_axes_locatable(ax0)
        cax0 = div0.append_axes("right", "4%", pad=0.05)
        fig.colorbar(im0, cax=cax0)
    
        # ==============
        # 2. Fitted model
        # ==============
        im1 = ax1.imshow(
            fit_image,
            cmap='viridis',
            origin='upper',
            extent=self.extent,
            interpolation='nearest',
        )
        ax1.plot(x0, y0, 'rx', ms=10, mew=2)
        ax1.add_patch(Circle((x0, y0), radius, fill=False, color='white', lw=2))
        ax1.set_title('Fitted model')
        ax1.set_aspect('equal')
        # ax1.set_box_aspect(1)
        ax1.tick_params(labelleft=False)
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes("right", "4%", pad=0.05)
        fig.colorbar(im1, cax=cax1)
    
        # ============
        # 3. Residuals
        # ============
        vmax = np.percentile(np.abs(residual), 99)
        if vmax == 0:
            vmax = 1.0
    
        im2 = ax2.imshow(
            residual,
            cmap='coolwarm',
            origin='upper',
            extent=self.extent,
            interpolation='nearest',
            vmin=-vmax,
            vmax=vmax,
        )
        ax2.plot(x0, y0, 'kx', ms=8, mew=2)
        ax2.set_title('Residuals (data - fit)')
        ax2.set_aspect('equal')
        # ax2.set_box_aspect(1)
        ax2.tick_params(labelleft=False)
        div2 = make_axes_locatable(ax2)
        cax2 = div2.append_axes("right", "4%", pad=0.05)
        fig.colorbar(im2, cax=cax2)
        
        fig.subplots_adjust(wspace=.1)
        fig.set_label('beamstop_fit')
    
    def find_beamstop(self, plot=True):
        """
        Find the beamstop and fit its center from a detector image
        """
        
        # find some of the most intense pixels in the dataset are
        highest_points = np.where(np.digitize(self.image, 
                                              bins=np.arange(
                                                  np.ceil(self.image).max()
                                                  )
                                              ) > 
                                  np.ceil(self.image).max()-4)
        
        # cluster the points to find where the spots are spaced horizontally
        X = np.stack(highest_points).T
        bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
            
        # biggest cluster should be around the beamstop, and have cluster label 0.
        beamstop_cluster = X[labels == 0]
        beamstop_cluster_mean = beamstop_cluster.mean(axis=0).astype(int)        
    
        # set some limits in the image aroud the detector. 45 should be robust enough.
        beamstop_y_min = beamstop_cluster_mean[0] - 60
        beamstop_y_max = beamstop_cluster_mean[0] + 60
        beamstop_x_min = beamstop_cluster_mean[1] - 40
        beamstop_x_max = beamstop_cluster_mean[1] + 40
        
        # take the region aroud the beam/beamstop for fitting purposes
        region = self.image[beamstop_y_min:beamstop_y_max,
                            beamstop_x_min:beamstop_x_max
                            ]
        
        # make it an actual peak for fitting purposes
        inverted = region.max() - region
        
        fit_result = self.fit_bright_circle(inverted,
                                            lo_x = beamstop_x_min,
                                            lo_y = beamstop_y_min,
                                            )
        
        self.beamstop_radius = fit_result.params['radius'].value
        beamstop_center = {'x': fit_result.params['x0'].value,
                                'y': fit_result.params['y0'].value
                                }
        
        ny, nx = inverted.shape
    
        # Local pixel coordinates within the crop
        yy_local, xx_local = np.indices(inverted.shape)
    
        # Global/full-image coordinates corresponding to this crop
        xx_global = xx_local + beamstop_x_min
        yy_global = yy_local + beamstop_y_min
        
        yy, xx = np.indices(self.image.shape)
    
        self.beamstop_mask = self.soft_disk_2d(
                    xx,
                    yy,
                    x0 = fit_result.params['x0'].value,
                    y0 = fit_result.params['y0'].value,
                    radius = self.beamstop_radius,
                    amplitude = 10,
                    background = 0,
                    sigma = fit_result.params['sigma'].value,
                    )
    
        if plot:
        
            # Reconstruct fitted model on the cropped region, using global coords
            fit_image = self.soft_disk_2d(
                        xx_global,
                        yy_global,
                        x0 = fit_result.params['x0'].value,
                        y0 = fit_result.params['y0'].value,
                        radius = fit_result.params['radius'].value,
                        amplitude = fit_result.params['amplitude'].value,
                        background = fit_result.params['background'].value,
                        sigma = fit_result.params['sigma'].value,
                        )
    
            self.plot_circle_fit(inverted, 
                                 fit_image, 
                                 fit_result.params['x0'].value, 
                                 fit_result.params['y0'].value,
                                 fit_result.params['radius'].value,
                                 self.extent
                                 )
            
        return beamstop_center
