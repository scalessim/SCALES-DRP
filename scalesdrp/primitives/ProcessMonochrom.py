from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.core.scales_pkg_resources import get_resource_path
from scalesdrp.primitives.scales_basic import fits_headers_to_dataframe
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import binary_dilation
#from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
import scalesdrp.primitives.scales_basic as scbasic
import scalesdrp.primitives.mono_basic as mono_basic
import pandas as pd
import numpy as np
import pickle
from astropy.io import fits
import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy import sparse
import astropy.io.fits as pyfits
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter, shift
from scipy.ndimage import map_coordinates, convolve as nd_convolve
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree, cKDTree
from astropy.modeling.functional_models import Gaussian2D, Const2D
from astropy.modeling import fitting
from scalesdrp.core.scales_proctab import Proctab
import logging
log = logging.getLogger("SCALES")
from scalesdrp.core.scales_pkg_resources import get_resource_path



class ProcessMonochrom(BasePrimitive):
    """
    Estimate the psf centroid of all the calib images images and save
    in two pickle file one for x and one for y centroid values. Currently
    assumes wavelengths will be in filenames. Need to replace that with
    header keywords instead. Also the location of the calib filesneed to fix.
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger

        if not hasattr(self, "proctab") or self.proctab is None:
            self.proctab = Proctab(logger=self.logger if hasattr(self, "logger") else logging.getLogger("SCALES"))

    def parse_files(self,df,scmode,output_dir):
        """
        """
        df2 = df[df['IFSMODE'] == scmode]
        if len(df2) == 0:
            return [], []

        det_config = np.unique(df2['MCLOCK'])[0]
        lams = df2['MONOWAVE']
        #.iloc: argsort gives positions, but df2 keeps its row labels from the full
        #table, so [] (a label lookup) fails when another mode's files come first
        names = df2['filename'].iloc[np.argsort(lams.values)]

        lams = np.sort(lams)
        ims = []
        for name in names:
            image = pyfits.getdata(output_dir+'/'+name,memmap=False)
            if True in np.isnan(image): self.logger.info('nan in image')
            ims.append(image)
        ims = np.array(ims)
        return ims, lams/1000.0

    def get_filenames_and_mean(self,df,scmode,output_dir):
        """
        """
        df2 = df[df['IFSMODE'] == scmode]
        if len(df2) == 0:
            return [], [], []

        det_config = np.unique(df2['MCLOCK'])[0]
        lams = df2['MONOWAVE']
        #.iloc: argsort gives positions, but df2 keeps its row labels from the full
        #table, so [] (a label lookup) fails when another mode's files come first
        names = df2['filename'].iloc[np.argsort(lams.values)]

        lams = np.sort(lams)
        fullnames = []
        mean = np.zeros([2048,2048])
        for name in names:
            image = pyfits.getdata(output_dir+'/'+name,memmap=False)
            if True in np.isnan(image): self.logger.info('nan in image')
            mean+=image
            fullnames.append(output_dir+'/'+name)
        mean/=float(len(names))
        fullnames = np.array(fullnames,dtype=str)
        return fullnames, lams/1000.0, mean

    def apply_wave_range_names(self,imnames,lams,scmode,wave_range):
        """
        Keep only the masters whose monochromator wavelength is inside a range,
        e.g. to drop band-edge masters with no or faint spots, which otherwise
        add thousands of noise tracks and can bias the background subtraction.

        Args:
            imnames: list of master filenames, sorted by wavelength (from parse_files)
            lams: their wavelengths in um
            scmode: mode name, e.g. 'LowRes-L'
            wave_range: None (keep all); (min, max) in um, applied to every mode;
                or a dict {scmode: (min, max)} (modes not listed keep all masters).
                Limits are inclusive.

        Returns:
            imnames, lams with only the masters inside the range
        """
        if isinstance(wave_range, dict):
            wave_range = wave_range.get(scmode)
        if wave_range is None or len(imnames) == 0:
            return imnames, lams
        lams = np.asarray(lams)
        wmin, wmax = wave_range
        keep = (lams >= wmin) & (lams <= wmax)
        if (~keep).any():
            self.logger.info(
                f"{scmode}: mono_wave_range ({wmin}, {wmax}) um drops {(~keep).sum()} of "
                f"{len(lams)} masters, at {np.round(lams[~keep], 4).tolist()} um")
        if not keep.any():
            self.logger.warning(f"{scmode}: no masters inside mono_wave_range ({wmin}, {wmax}) um")
        return imnames[keep], lams[keep]

    def apply_wave_range(self,ims,lams,scmode,wave_range):
        """
        Keep only the masters whose monochromator wavelength is inside a range,
        e.g. to drop band-edge masters with no or faint spots, which otherwise
        add thousands of noise tracks and can bias the background subtraction.

        Args:
            ims: stack of masters, sorted by wavelength (from parse_files)
            lams: their wavelengths in um
            scmode: mode name, e.g. 'LowRes-L'
            wave_range: None (keep all); (min, max) in um, applied to every mode;
                or a dict {scmode: (min, max)} (modes not listed keep all masters).
                Limits are inclusive.

        Returns:
            ims, lams with only the masters inside the range
        """
        if isinstance(wave_range, dict):
            wave_range = wave_range.get(scmode)
        if wave_range is None or len(ims) == 0:
            return ims, lams
        lams = np.asarray(lams)
        wmin, wmax = wave_range
        keep = (lams >= wmin) & (lams <= wmax)
        if (~keep).any():
            self.logger.info(
                f"{scmode}: mono_wave_range ({wmin}, {wmax}) um drops {(~keep).sum()} of "
                f"{len(lams)} masters, at {np.round(lams[~keep], 4).tolist()} um")
        if not keep.any():
            self.logger.warning(f"{scmode}: no masters inside mono_wave_range ({wmin}, {wmax}) um")
        return ims[keep], lams[keep]

    def apply_exclude_waves_names(self,imnames,lams,scmode,exclude_waves,tol=0.001):
        """
        Drop the masters at specific monochromator wavelengths, e.g. inside
        atmospheric absorption bands (H2O at 2.737 um, CO2 at 4.260 um in
        LowRes-KLM), where almost no light gets through.

        Args:
            imnames: list of master filenames (from parse_files / apply_wave_range)
            lams: their wavelengths in um
            scmode: mode name, e.g. 'LowRes-KLM'
            exclude_waves: None (drop none); a list of wavelengths in um, applied
                to every mode; or a dict {scmode: [wavelengths]} (modes not
                listed drop none)
            tol: a master is dropped if it is within tol um of a listed
                wavelength (default 1 nm, as in the notebook)

        Returns:
            ims, lams without the excluded masters
        """
        if isinstance(exclude_waves, dict):
            exclude_waves = exclude_waves.get(scmode)
        if exclude_waves is None or len(imnames) == 0:
            return imnames, lams
        excl = np.atleast_1d(np.asarray(exclude_waves, dtype=float))
        if excl.size == 0:
            return imnames, lams
        lams = np.asarray(lams)
        drop = np.any(np.abs(lams[:, None] - excl[None, :]) < tol, axis=1)
        unmatched = [w for w in excl if not np.any(np.abs(lams - w) < tol)]
        if drop.any():
            self.logger.info(
                f"{scmode}: mono_exclude_waves drops {drop.sum()} of {len(lams)} "
                f"masters, at {np.round(lams[drop], 4).tolist()} um")
        if unmatched:
            self.logger.warning(
                f"{scmode}: no master within {tol} um of mono_exclude_waves "
                f"{unmatched} um; nothing dropped there")
        return imnames[~drop], lams[~drop]

    def apply_exclude_waves(self,ims,lams,scmode,exclude_waves,tol=0.001):
        """
        Drop the masters at specific monochromator wavelengths, e.g. inside
        atmospheric absorption bands (H2O at 2.737 um, CO2 at 4.260 um in
        LowRes-KLM), where almost no light gets through.

        Args:
            ims: stack of masters (from parse_files / apply_wave_range)
            lams: their wavelengths in um
            scmode: mode name, e.g. 'LowRes-KLM'
            exclude_waves: None (drop none); a list of wavelengths in um, applied
                to every mode; or a dict {scmode: [wavelengths]} (modes not
                listed drop none)
            tol: a master is dropped if it is within tol um of a listed
                wavelength (default 1 nm, as in the notebook)

        Returns:
            ims, lams without the excluded masters
        """
        if isinstance(exclude_waves, dict):
            exclude_waves = exclude_waves.get(scmode)
        if exclude_waves is None or len(ims) == 0:
            return ims, lams
        excl = np.atleast_1d(np.asarray(exclude_waves, dtype=float))
        if excl.size == 0:
            return ims, lams
        lams = np.asarray(lams)
        drop = np.any(np.abs(lams[:, None] - excl[None, :]) < tol, axis=1)
        unmatched = [w for w in excl if not np.any(np.abs(lams - w) < tol)]
        if drop.any():
            self.logger.info(
                f"{scmode}: mono_exclude_waves drops {drop.sum()} of {len(lams)} "
                f"masters, at {np.round(lams[drop], 4).tolist()} um")
        if unmatched:
            self.logger.warning(
                f"{scmode}: no master within {tol} um of mono_exclude_waves "
                f"{unmatched} um; nothing dropped there")
        return ims[~drop], lams[~drop]

    def monochrom_bksub(self,ims,method='mean'):
        if method=='mean':
            meanim = np.nanmean(ims,axis=0)
            means = np.array([np.nanmean(ims[x]) for x in range(len(ims))])
            bkgs = np.array([meanim*means[x]/np.nanmean(meanim) for x in range(len(means))])
        if method=='median':
            medim = np.nanmedian(ims,axis=0)
            meds = np.array([np.nanmedian(ims[x]) for x in range(len(ims))])
            bkgs = np.array([medim*meds[x]/np.nanmedian(medim) for x in range(len(meds))])
        ims_sub = ims - bkgs
        return ims_sub

    def monochrom_bksub_sing(self,im,meanim):
        im_mean = np.nanmean(im)
        bkg = meanim*im_mean/np.nanmean(meanim)
        im_sub = im - bkg
        return im_sub


    def masked_row_destripe(self,data, sigma_thresh=2.0, dilate_iter=3, n_passes=2):
        """
        Remove row-wise (horizontal band) noise from `data` without being biased
        by compact sources.

        Parameters
        ----------
        data : 2D ndarray
            Input image.
        sigma_thresh : float
            Pixels more than this many robust-sigma above the robust median are
            flagged as "source" pixels and excluded from the row statistic.
            Lower = more aggressive masking (better for heavily source-filled
            rows, but risks eating faint background structure).
        dilate_iter : int
            Number of binary dilation iterations applied to the source mask, to
            cover the faint wings of each spot (not just its bright core).
        n_passes : int
            Number of times to repeat the mask-and-subtract cycle. A second pass
            usually cleans up rows where the first-pass mask was incomplete.

        Returns
        -------
        corrected : 2D ndarray
            Destriped image (same shape as `data`).
        row_baseline : 1D ndarray
            The per-row offset that was subtracted (length = data.shape[0]).
        mask : 2D bool ndarray
            Final source mask used on the last pass.
        """
        working = data.copy()
        row_baseline = np.zeros(data.shape[0])
        mask = None

        for _ in range(n_passes):
            mean, med, std = sigma_clipped_stats(working, sigma=3.0, maxiters=5)
            mask = working > (med + sigma_thresh * std)
            mask = binary_dilation(mask, iterations=dilate_iter)
            masked = np.ma.array(working, mask=mask)
            pass_baseline = np.ma.median(masked, axis=1).filled(0.0)
            row_baseline += pass_baseline
            working = data - row_baseline[:, None]

        corrected = data - row_baseline[:, None]
        return corrected, row_baseline, mask



    def find_all_spots(self,ims_cal, lams_u, min_distance=15, thresh=50, plot_im=False, sigma=0.8,medres=False, mfilt='imgK'):
        spots = {}

        for ii in range(len(ims_cal)):
            im_cal = ims_cal[ii]

            if medres==True:
                if mfilt=='imgK':
                    y0=180
                    lam0=2.0248
                    chy=1
                    lmin=1.95
                    lmax=2.45
                    length=1822
                if mfilt=='imgLp':
                    y0=975
                    lam0=3.44805
                    chy=1
                    lmin=3.44805
                    lmax=4.09241
                    length=878
                if mfilt=='L':
                    #IFS L filter (MedRes-L), fitted to the medres_L_20261117 scan:
                    #middle of the 3 detector rows (~80 px apart), 1361 px/um
                    y0=179
                    lam0=2.9
                    chy=1
                    lmin=2.88
                    lmax=4.17
                    length=1756
                ycent,ymin,ymax=self.get_spot_yrange_medres(lams_u[ii],y0=y0,lam0=lam0,chy=chy,lmin=lmin,lmax=lmax,length=length)

                dy = (ymax-ymin)*0.5
                im_cal = im_cal[int(ymin):int(ymax)]
            data_smooth = gaussian_filter(im_cal, sigma=sigma)
            coords_yx = peak_local_max(
                data_smooth,
                min_distance=min_distance,
                threshold_abs=np.percentile(data_smooth, thresh),
                exclude_border=5
            )


            if medres==True:
                #"""
                todel = []
                for i in range(len(coords_yx)):
                    yc,xc = coords_yx[i]
                    ydiff = coords_yx[:,0]-yc
                    testrow = ydiff[np.where(np.abs(ydiff) < 15)]
                    if len(testrow) < 80:
                        todel.append(i)
                coords_yx = np.delete(coords_yx,todel,axis=0)
                #"""
                coordinates = np.column_stack([coords_yx[:, 1], coords_yx[:, 0]+int(ymin)])
            else:
                coordinates = np.column_stack([coords_yx[:, 1], coords_yx[:, 0]])
            intensities = data_smooth[coords_yx[:, 0], coords_yx[:, 1]]

            self.logger.info(f"Found {len(coords_yx)} spots in image {ii}")
            spots[ii] = {
                    'filename': 'blank',
                    'coordinates': coordinates,
                    'intensities': intensities,
                    'lam': lams_u[ii]*np.ones(len(coordinates)),
                    'n_spots': len(coordinates)
                }
            if plot_im == True:
                f = plt.figure(clear=True)
                f.add_subplot(121)
                #plt.imshow(data_smooth)
                plt.imshow(ims_cal[ii],origin='lower')
                plt.scatter(coordinates[:,0],coordinates[:,1],c='r',s=2)
                plt.title('spots in image '+str(ii))
                f.add_subplot(122)
                plt.hist(intensities,bins=100)
                plt.title('spot intensities')
                plt.show()

                f = plt.figure(clear=True)
                f.add_subplot(121)
                #plt.imshow(data_smooth)
                plt.hist(coordinates[:,0],bins=30)
                plt.title('xcoords in image '+str(ii))
                f.add_subplot(122)
                plt.hist(coordinates[:,1],bins=30)
                plt.title('ycoords in image '+str(ii))
                plt.show()
                #tmp = input('continue?')
                #stop

        return spots

    def find_all_spots_names(self, names, lams_u, meanim,
                             min_distance=15, thresh=50,
                             plot_im=False, sigma=0.8,medres=False,
                             mfilt='imgK'):
        spots = {}

        for ii in range(len(names)):
            #im_cal = ims_cal[ii]
            fname = names[ii]
            image = pyfits.getdata(fname,memmap=False)
            im_cal_tmp = self.monochrom_bksub_sing(image,meanim)
            if self.config.instrument.apply_destripe==True:
                im_cal = self.masked_row_destripe(im_cal_tmp)[0]
            else:
                im_cal = im_cal_tmp

            if medres==True:
                if mfilt=='imgK':
                    y0=180
                    lam0=2.0248
                    chy=1
                    lmin=1.95
                    lmax=2.45
                    length=1822
                if mfilt=='imgLp':
                    y0=975
                    lam0=3.44805
                    chy=1
                    lmin=3.44805
                    lmax=4.09241
                    length=878
                if mfilt=='L':
                    #IFS L filter (MedRes-L), fitted to the medres_L_20261117 scan:
                    #middle of the 3 detector rows (~80 px apart), 1361 px/um
                    y0=179
                    lam0=2.9
                    chy=1
                    lmin=2.88
                    lmax=4.17
                    length=1756
                ycent,ymin,ymax=self.get_spot_yrange_medres(lams_u[ii],y0=y0,lam0=lam0,chy=chy,lmin=lmin,lmax=lmax,length=length)

                dy = (ymax-ymin)*0.5
                im_cal = im_cal[int(ymin):int(ymax)]
            data_smooth = gaussian_filter(im_cal, sigma=sigma)
            coords_yx = peak_local_max(
                data_smooth,
                min_distance=min_distance,
                threshold_abs=np.percentile(data_smooth, thresh),
                exclude_border=5
            )


            if medres==True:
                #"""
                todel = []
                for i in range(len(coords_yx)):
                    yc,xc = coords_yx[i]
                    ydiff = coords_yx[:,0]-yc
                    testrow = ydiff[np.where(np.abs(ydiff) < 15)]
                    if len(testrow) < 80:
                        todel.append(i)
                coords_yx = np.delete(coords_yx,todel,axis=0)
                #"""
                coordinates = np.column_stack([coords_yx[:, 1], coords_yx[:, 0]+int(ymin)])
            else:
                coordinates = np.column_stack([coords_yx[:, 1], coords_yx[:, 0]])
            intensities = data_smooth[coords_yx[:, 0], coords_yx[:, 1]]

            self.logger.info(f"Found {len(coords_yx)} spots in image {ii}")
            spots[ii] = {
                    'filename': 'blank',
                    'coordinates': coordinates,
                    'intensities': intensities,
                    'lam': lams_u[ii]*np.ones(len(coordinates)),
                    'n_spots': len(coordinates)
                }
            if plot_im == True:
                f = plt.figure(clear=True)
                f.add_subplot(121)
                #plt.imshow(data_smooth)
                plt.imshow(ims_cal[ii],origin='lower')
                plt.scatter(coordinates[:,0],coordinates[:,1],c='r',s=2)
                plt.title('spots in image '+str(ii))
                f.add_subplot(122)
                plt.hist(intensities,bins=100)
                plt.title('spot intensities')
                plt.show()

                f = plt.figure(clear=True)
                f.add_subplot(121)
                #plt.imshow(data_smooth)
                plt.hist(coordinates[:,0],bins=30)
                plt.title('xcoords in image '+str(ii))
                f.add_subplot(122)
                plt.hist(coordinates[:,1],bins=30)
                plt.title('ycoords in image '+str(ii))
                plt.show()
                #tmp = input('continue?')
                #stop

        return spots


    def medres_trace_params(self,mfilt):
        """
        MedRes trace model (as in find_all_spots / remove_spot_dups): trace y moves
        by chy*length px over lmin..lmax um. Returns y0, lam0, chy, lmin, lmax, length.
        """
        if mfilt=='imgK':
            return dict(y0=180,lam0=2.0248,chy=1,lmin=1.95,lmax=2.45,length=1822)
        if mfilt=='imgLp':
            return dict(y0=975,lam0=3.44805,chy=1,lmin=3.44805,lmax=4.09241,length=878)
        if mfilt=='L':
            return dict(y0=179,lam0=2.9,chy=1,lmin=2.88,lmax=4.17,length=1756)
        raise ValueError(f"No MedRes trace model for mfilt={mfilt}")

    def track_sequentially(self,spots,max_match_distance=3,lams=None,dispersion=None):
        """
        Track spots sequentially forward through images.

        Each spot in image 0 gets a spot_id. As we move to image 1, we:
        1. Match spots from image 0 to image 1
        2. Assign same spot_id to matched spots
        3. Give new spot_ids to unmatched spots (newly appearing)

        This allows tracking diagonal motion and handles appearing/disappearing spots.

        If dispersion = (dx/dlam, dy/dlam) in px/um is given (with lams, the
        wavelength of each image), each previous spot is first moved to where the
        trace model puts it at the next wavelength, so max_match_distance only has
        to cover the model error, not the wavelength step. Without it, spots are
        matched at their previous positions (so the step must move them less than
        max_match_distance).

        Returns:
        --------
        spot_tracks : dict
            spot_tracks[spot_id] = {image_idx: (x, y, intensity, lam), ...}
        """
        n_images = len(spots)
        if dispersion is not None:
            if lams is None:
                raise ValueError("track_sequentially: dispersion needs lams")
            lams = np.asarray(lams, dtype=float)
            dispersion = np.asarray(dispersion, dtype=float)


        self.logger.info(f"\nSequential tracking through {n_images} images...")

        # Initialize with first image
        coords_0 = spots[0]['coordinates']
        intensities_0 = spots[0]['intensities']
        lams_0 = spots[0]['lam']
        n_spots_0 = len(coords_0)

        # Each spot gets a unique ID
        current_spot_ids = np.arange(n_spots_0)
        next_spot_id = n_spots_0  # Counter for new spots that appear later

        # Initialize tracking data structure
        # spot_tracks[spot_id] = {image_idx: (x, y, intensity), ...}
        spot_tracks = {}
        for spot_id in range(n_spots_0):
            spot_tracks[spot_id] = {
                0: (coords_0[spot_id, 0], coords_0[spot_id, 1], intensities_0[spot_id], lams_0[spot_id])
            }

        self.logger.info(f"  Image 0: {n_spots_0} spots initialized")

        # Track forward through remaining images
        prev_coords = coords_0
        prev_spot_ids = current_spot_ids

        for img_idx in range(1, n_images):
            curr_coords = spots[img_idx]['coordinates']
            curr_intensities = spots[img_idx]['intensities']
            curr_lams = spots[img_idx]['lam']
            n_curr_spots = len(curr_coords)

            if len(prev_coords) == 0:
                # No spots in previous image, all current spots are new
                new_spot_ids = np.arange(next_spot_id, next_spot_id + n_curr_spots)
                for i, spot_id in enumerate(new_spot_ids):
                    spot_tracks[spot_id] = {
                        img_idx: (curr_coords[i, 0], curr_coords[i, 1], curr_intensities[i], curr_lams[i])
                    }
                next_spot_id += n_curr_spots
                prev_coords = curr_coords
                prev_spot_ids = new_spot_ids
                self.logger.info(f"  Image {img_idx}: {n_curr_spots} new spots (no previous spots to match)")
                continue

            # Build KDTree for current image spots
            tree = KDTree(curr_coords)

            # Find nearest spot in current image for each spot in previous image
            # (moved along the trace to this image's wavelength, if a dispersion is given)
            if dispersion is not None:
                pred_coords = prev_coords + (lams[img_idx]-lams[img_idx-1])*dispersion
            else:
                pred_coords = prev_coords
            distances, indices = tree.query(pred_coords)

            # Track which current spots have been matched
            matched_curr_indices = set()
            curr_spot_ids = np.full(n_curr_spots, -1, dtype=int)

            n_matched = 0
            n_lost = 0

            # Match spots from previous image
            for prev_idx, (dist, curr_idx) in enumerate(zip(distances, indices)):
                if dist < max_match_distance and curr_idx not in matched_curr_indices:
                    # Good match - carry forward the spot_id
                    spot_id = prev_spot_ids[prev_idx]
                    curr_spot_ids[curr_idx] = spot_id
                    matched_curr_indices.add(curr_idx)

                    # Add to trajectory
                    spot_tracks[spot_id][img_idx] = (
                        curr_coords[curr_idx, 0],
                        curr_coords[curr_idx, 1],
                        curr_intensities[curr_idx],
                        curr_lams[curr_idx]
                    )
                    n_matched += 1
                else:
                    # Spot lost (disappeared or moved too far)
                    n_lost += 1

            # Handle new spots (unmatched in current image)
            n_new = 0
            for curr_idx in range(n_curr_spots):
                if curr_spot_ids[curr_idx] == -1:
                    # New spot appearing
                    new_spot_id = next_spot_id
                    curr_spot_ids[curr_idx] = new_spot_id
                    spot_tracks[new_spot_id] = {
                        img_idx: (curr_coords[curr_idx, 0], curr_coords[curr_idx, 1], curr_intensities[curr_idx], curr_lams[curr_idx])
                    }
                    next_spot_id += 1
                    n_new += 1

            self.logger.info(f"  Image {img_idx}: {n_matched} matched, {n_lost} lost, {n_new} new (total: {n_curr_spots})")

            # Update for next iteration
            prev_coords = curr_coords
            prev_spot_ids = curr_spot_ids

        total_spots = len(spot_tracks)
        self.logger.info(f"\nTotal unique spots tracked: {total_spots}")

        return spot_tracks

    def remove_spot_dups(self,spot_tracks,lams_u,maxdist=13,lmin=2.9,lmax=4.15,chx=1,chy=1,medres=False,mfilt='imgK'):
        """
        Function to ingest multi-wavelength spot tracks, which do not have
        indices that correspond to specific lenslets,
        and find spots that fall
        on common traces. Spots on the same trace are consolidated.

        Each partial track's position at the band centre is predicted from its
        first spot (trace model below); tracks whose predicted positions are
        within maxdist px are merged, greedily and in track order. Neighbours
        are found with a KD-tree, which gives the same result as comparing every
        pair of tracks but scales to tens of thousands of partial tracks (e.g.
        noise peaks from masters without monochromator light).
        """
        #full-length tracks correspond to one lenslet each and are kept as they are;
        #the rest are fragments that may share a trace
        spots_u = {}
        spots_d = []
        for i in range(len(spot_tracks)):
            if len(spot_tracks[i]) == len(lams_u):
                spots_u[len(spots_u)] = spot_tracks[i]
            else:
                spots_d.append(spot_tracks[i])
        if len(spots_d) == 0:
            return spots_u

        if medres==True:
            if mfilt=='imgK':
                chy=1
                lmin=1.95
                lmax=2.45
                length=1822
            if mfilt=='imgLp':
                chy=1
                lmin=3.44805
                lmax=4.09241
                length=878
            if mfilt=='L':
                chy=1
                lmin=2.88
                lmax=4.17
                length=1756
        lam = 0.5*(lmin+lmax)

        #predicted band-centre position of each fragment, from its first spot
        pos = np.zeros((len(spots_d), 2))
        for i, track in enumerate(spots_d):
            first = track[list(track.keys())[0]]
            x0, y0, lam0 = first[0], first[1], first[3]
            if medres==False:
                pos[i] = self.get_trace_pos(lam,x0,y0,lam0,chx,chy,lmin,lmax)
            else:
                yi,ymin,ymax = self.get_spot_yrange_medres(lam,y0=y0,lam0=lam0,chy=chy,lmin=lmin,lmax=lmax,length=length)
                pos[i] = (x0, yi)

        #greedy merge: each fragment not yet merged absorbs every other unmerged
        #fragment whose predicted position is closer than maxdist
        tree = cKDTree(pos)
        rem = set()
        for i in range(len(spots_d)):
            if i in rem:
                continue
            merged = spots_d[i]
            for j in sorted(tree.query_ball_point(pos[i], maxdist)):
                #query_ball_point includes distance == maxdist; the merge needs < maxdist
                if j != i and j not in rem and np.hypot(*(pos[i]-pos[j])) < maxdist:
                    rem.add(j)
                    merged = merged | spots_d[j]
            spots_u[len(spots_u)] = merged
            rem.add(i)
        return spots_u

    def get_trace_pos(self,lam,x0,y0,lam0,chx,chy,lmin,lmax,length=54,tilt=18):
        """
        Function to get the expected position of a certain wavelength
        within a trace.

        Args:
            lam: wavelength at which to calculate trace position
            x0: reference x position of trace
            y0: reference y position of trace
            lam0: reference wavelength for trace position (x0,y0)
            chx: direction of trace x movement with +ve lambda
            chy: direction of trace y movement with +ve lambda
            length: trace length in pixels (default = 54)
            tilt: trace tilt relative to vertical in deg (default = 18)

        Returns:
            xpos: trace x position at wavelength lam
            ypos: trace y position at wavelength lam
        """
        dlam = lam-lam0
        xoff = dlam/(lmax-lmin)*length*np.sin(np.radians(tilt))*chx
        yoff = dlam/(lmax-lmin)*length*np.cos(np.radians(tilt))*chy
        xpos = x0+xoff
        ypos = y0+yoff
        return xpos, ypos

    def remove_silos(self,avgs,spot_tracks,medres=True,show_plots=False):
        """
        Function to remove lenslet tracks that have no neighbors.

        Args:
            avgs: list of (x,y) trace positions for the average
                  wavelength in the mode

        Returns:
            avgs_new: list of (x,y) trace positions where lenslets
                      that have no neighbors have been removed
        """

        #create scipy KDTree using input set of positions
        kd1 = KDTree(avgs)
        lens = np.array([len(spot_tracks[x]) for x in range(len(spot_tracks))])
        mlen = np.max(lens)

        todel = []
        tracks_new = {}

        #loop through all lenslet spot positions
        for i in range(len(avgs)):
            #create KDTree for single lenslet position to search
            #for neighbors
            kd0 = KDTree(avgs[i:i+1])
            if medres==False:
                mindist = 33
                #query for neighbors within 33 pixels
                neighbors = kd0.query_ball_tree(kd1, mindist, p=2.0, eps=0)
                #if only one spot in the large KDTree is within 33 pixels
                #of the spot in question, the spot in question has no
                #neighbors (i.e. the spot in question itself is the only
                #one found
                diff = kd1.data[neighbors] - kd0.data
                if len(diff[0])==1:
                    todel.append(i)
            #delete neighborless spots from the list
            if medres==True:
                mindist = 80
                neighbors = kd0.query_ball_tree(kd1, mindist, p=2.0, eps=0)
                #if only one spot in the large KDTree is within 33 pixels
                #of the spot in question, the spot in question has no
                #neighbors (i.e. the spot in question itself is the only
                #one found
                diff = kd1.data[neighbors] - kd0.data
                if len(diff[0])<3:
                    todel.append(i)
                diff2 = kd0.data - kd1.data
                xdiff = np.abs(diff2[:,0])
                ydiff = np.abs(diff2[:,1])

                testrow = diff2[np.where(np.abs(ydiff) < 15)]
                if len(testrow) < 80:
                    todel.append(i)

                nearestdiffs = ydiff[np.where(xdiff < 30)]
                if np.sort(nearestdiffs)[1] > 2:
                    todel.append(i)

                if len(spot_tracks[i]) < 0.3*mlen:
                    todel.append(i)

        avgs_new = np.delete(avgs,todel,axis=0)

        cc=0
        for i in range(len(avgs)):
            if i not in todel:
                tracks_new[cc]=spot_tracks[i]
                cc+=1

        if show_plots==True:
            f = plt.figure(clear=True)
            plt.scatter(avgs[:,0],avgs[:,1])
            plt.scatter(avgs_new[:,0],avgs_new[:,1])
            plt.show()


        return avgs_new,tracks_new


    def measure_trace_shape(self,spot_tracks_u,lams,lmin,lmax):
        nlams = len(lams)
        lcent_ind = int(np.argmin(np.abs(lams - 0.5 * (lmin + lmax))))
        traces_filled = [trace for trace in spot_tracks_u.values()
                         if len(trace)==nlams]
        posns = np.array([[trace[k][:2] for k in range(nlams)]
                          for trace in traces_filled], dtype=float)
        offsets = posns - posns[:, lcent_ind:lcent_ind + 1]
        return np.median(offsets, axis=0), np.percentile(offsets, [16, 84], axis=0), posns

    def find_avg_spotpos_obs(self,spot_tracks_u,lams,lmin,lmax,
                             medres=False,exclude_lcent=False):
        trace_shape, trace_shape_pct, full_posns = self.measure_trace_shape(spot_tracks_u, lams, lmin, lmax)
        lcent_ind = int(np.argmin(np.abs(lams - 0.5 * (lmin + lmax))))
        avgs = np.zeros((len(spot_tracks_u), 2))
        for j in range(len(spot_tracks_u)):
            t = spot_tracks_u[j]
            if lcent_ind in t and not exclude_lcent:
                avgs[j] = t[lcent_ind][:2]
                continue
            ks = np.array([k for k in sorted(t) if not (exclude_lcent and k == lcent_ind)])
            if len(ks) == 0:
                avgs[j] = np.nan
                continue
            p = np.array([t[k][:2] for k in ks], dtype=float)
            avgs[j] = np.median(p - trace_shape[ks], axis=0)
        return avgs

    def find_avg_spotpos_model(self,spot_tracks_u,lmin,lmax,chx=1,chy=1,medres=False,mfilt='imgK',show_plots=False):
        ### uses model of the trace starting at first wavelength
        ### to predict each trace based on the spot tracks
        ### and then get x,y position of central wavelength in the trace
        avgs = []
        for j in range(len(spot_tracks_u)):
            keys = list(spot_tracks_u[j].keys())
            x0,y0 = spot_tracks_u[j][keys[0]][0],spot_tracks_u[j][keys[0]][1]
            lam0 = spot_tracks_u[j][keys[0]][3]
            if medres==False:
                lam = 0.5*(lmin+lmax)
                xj,yj = self.get_trace_pos(lam,x0,y0,lam0,chx,chy,lmin,lmax)

            if medres==True:
                if mfilt=='imgK':
                    chy=1
                    lmin=1.95
                    lmax=2.45
                    length=1822
                if mfilt=='imgLp':
                    chy=1
                    lmin=3.44805
                    lmax=4.09241
                    length=878
                if mfilt=='L':
                    chy=1
                    lmin=2.88
                    lmax=4.17
                    length=1756
                lam = 0.5*(lmin+lmax)
                xj = x0
                yj,ymin,ymax=self.get_spot_yrange_medres(lam,y0=y0,lam0=lam0,chy=chy,lmin=lmin,lmax=lmax,length=length)
            avgs.append([xj,yj])
        avgs = np.array(avgs)

        if show_plots==True:
            fig = plt.figure(clear=True)
            plt.scatter(avgs[:,0],avgs[:,1])
            plt.show()

            #plt.hist(avgs[:,1],bins=1000)
            #plt.show()

            #plt.hist(avgs[:,0],bins=1000)
            #plt.show()
        return avgs


    def fit_lensarr_xy(self, avgs, spot_tracks, lams,
                       deg=2, conf=0.25, fit_min_len=None):
        """Assign grid indices from a fitted lattice.
           Returns final_posns and diagnostics.
           The lattice is fitted only to tracks with at least fit_min_len
            wavelengths (default nlams // 4):
        """
        pts_all = np.asarray(avgs, float)
        lens = np.array([len(spot_tracks[k]) for k in range(len(pts_all))])
        nl = len(lams)

        fit_min_len = max(1, nl // 4) if fit_min_len is None else fit_min_len
        use = lens >= fit_min_len
        pts = pts_all[use]

        # Find nearest neighbors of every spot - within 30 pixels, up to 7
        d, j = cKDTree(pts).query(pts, k=7, distance_upper_bound=30)

        # Calculate offests between each point and its neighbors
        offs = np.concatenate([pts[j[:, m][np.isfinite(d[:, m])]] - pts[np.isfinite(d[:, m])] for m in range(1, 7)])

        # Spot positions are whole pixels, so a median offset would round the pitch (e.g. 20 instead
        # of 20.4) and the index would drift by a full row ~50 lenslets from the start point. Use a
        # clipped mean so the lattice vectors keep their sub-pixel value.
        def clipped_mean(v):
            m = np.median(v, axis=0)
            return v[np.all(np.abs(v - m) < 3, axis=1)].mean(axis=0)


        a = clipped_mean(offs[(offs[:, 0] > 10) & (np.abs(offs[:, 1]) < 10)])   # +x neighbour
        b = clipped_mean(offs[(offs[:, 1] > 10) & (np.abs(offs[:, 0]) < 10)])   # +y neighbour

        # Find the lenslet closest to the center of the array
        p0 = pts[np.argmin(np.hypot(*(pts - np.median(pts, axis=0)).T))]

        # Measure every lenslet's distance from the center in grid steps
        frac_idx = (pts - p0) @ np.linalg.inv(np.array([a, b]))      # (x_index, y_index), unrounded

        def design(p, deg):
            x, y = ((p - 1024) / 1024).T #get fractional difference between pixel positions and center of detector

            #build detector polynomial
            cols = [np.ones_like(x)] + [x ** i * y ** (k - i) for k in range(1, deg + 1) for i in range(k + 1)]
            return np.column_stack(cols)



        r = np.hypot(*(pts - p0).T) #fitted distance from central lenslet

        #assign lenslet indices iteratively using a polynomial fit that starts as linear and ends as quadratic
        for radius, dg in [(r_, 1 if r_ < 500 else deg) for r_ in range(200, 3001, 100)] + [(3000, deg)]:
            n = np.round(frac_idx) #rounded indices

            #get spots within 0.25 of a fitted lenslet position, within a fixed radius of the central lenslet
            good = np.all(np.abs(frac_idx - n) < conf, axis=1) & (r < radius)

            #get detector polynomial terms for desired degree (based on radius)
            A = design(pts, dg)

            #find coefficients by fitting a dg-degree polynomial to the lenslet grid indices
            coef = np.linalg.lstsq(A[good], n[good], rcond=None)[0]
            frac_idx = A @ coef

        pts = pts_all                                                # now assign every track
        frac_idx = design(pts, deg) @ coef #final grid of lenslet positions
        n = np.round(frac_idx).astype(int) #integer grid of lenslet positions
        frac = frac_idx - n #how far each lenslet is from an integer position

        # Resolve cells claimed by more than one track
        order = np.lexsort((np.abs(frac).max(axis=1), -lens))       # longest first, then closest to grid
        owner, losers = {}, []
        for k in order:
            cell = (n[k, 0], n[k, 1])
            if cell in owner:
                losers.append(k)
            else:
                owner[cell] = k

        n -= n.min(axis=0)
        nx_, ny_ = n.max(axis=0) + 1
        self.logger.info(f'NOTE: fitted grid (before dropping spurious tracks) is {ny_} x {nx_}')


        fp = np.full((ny_, nx_, 3), np.nan)
        for k in owner.values():
            fp[n[k, 1], n[k, 0]] = [n[k, 0], n[k, 1], k]

        #create function to return predicted lenslet positions based on pixel position on the detector
        to_index = lambda p: design(np.atleast_2d(np.asarray(p, float)), deg) @ coef   # pixel -> unrounded index
        return fp, dict(a=a, b=b, frac=frac, losers=np.array(losers, int), lens=lens, to_index=to_index,
                        n_fit=int(use.sum()), fit_min_len=fit_min_len)



    def get_lensarr_xy(self,avgs,maxdist=15,show_plots=False):
        """
        Function to take clean array of spot positions and
        register them into a x,y grid of lenslets.


        Args:
            avgs: array of pixel (x,y) positions for each spot
                  track at the average wavelength in the mode


        Returns:
            final_posns: array of pixel (x,y) positions for all
                         lenslets that have fallen on the detector,
                         with shape n_lens_y, n_lens_x, 2
        """


        #define lists of: (1) lenslets to search around
        to_search_around = [0]
        #(2) lenslets that have been searched around
        done_searching_around = []
        #(3) lenslets whose positions have been entered
        positions_entered = [0]
        #(4) lenslet indices organized by positions in the array
        positions = [[1000,1000]]
        #(5) lenslets' pixel positions on the detector,
        #arranged into a (ny,nx,2) array to match the lenslet
        #positions in the array
        posns_pix = np.zeros((2000,2000,2))
        posns_pix[:,:,:] = np.nan
        posns_pix[1000,1000] = avgs[0]


        #(6) lenslets' x,y positions in the lenslet array,
        #plus spot track index in lists of unique spot tracks,
        #arranged into (ny,nx,3) to match lenslet array shape
        #on first and second axis
        posns_idx = np.zeros((2000,2000,3))
        posns_idx[:,:,:] = np.nan
        posns_idx[1000,1000] = [1000,1000,0]


        #create KDTree from list of spot track positions at
        #average wavelength
        kd1 = KDTree(avgs)


        #continue searching until all lenslets have been searched
        counter=0
        ddone=1
        #while ((len(done_searching_around) < len(avgs)) and (ddone > 0)):
        while len(done_searching_around) < len(to_search_around):
            #while a search is needed, go through entries in list
            #of lenslets that have yet to be searched
            for search_lens in to_search_around:
                #print("done searching", len(done_searching_around))#, done_searching_around)
                #print("avgs", len(avgs))#, avgs)
                #print("tosearch", len(to_search_around))#, to_search_around)
                #confirm that this lenslet is not marked as done
                #print("search lens pos:",search_lens,avgs[search_lens:search_lens+1])
                if search_lens not in done_searching_around:
                    #create KDTree from single lenslet to be searched
                    kd0 = KDTree(avgs[search_lens:search_lens+1])
                    #grab xind,yind for the search lenslet, where xind and
                    #yind are the indices of the lenslet in the lenslet array
                    xind,yind = np.array(positions)[np.where(np.abs(np.array(positions_entered) - search_lens) < 1e-6)][0]


                    #get all neighbors within 23 pixels of the search lenslet
                    neighbors = kd0.query_ball_tree(kd1, 23, p=2.0, eps=0)
                    #difference the pixel position of the search lenslet
                    #with that of its neighbors
                    diff = kd1.data[neighbors] - kd0.data
                    #loop through list of neighbors
                    for ii in range(len(diff[0])):
                        #check that the index has not already been registered
                        if neighbors[0][ii] not in positions_entered:
                            entry = diff[0][ii]
                            #check whether x position is more than 15 pixels greater
                            #than the search lenslet
                            if entry[0] > maxdist:
                                #this means that the x index is one greater than the
                                #search lenslet
                                xind_new = xind + 1
                                #check whether the y value is less than 15 pixels away from
                                #the search lenslet
                                if abs(entry[1]) < maxdist:
                                    #this means x is greater and y is the same
                                    #which means we found the lenslet directly to the right
                                    yind_new = yind
                                    positions_entered.append(neighbors[0][ii])
                                    to_search_around.append(neighbors[0][ii])
                                    positions.append([xind_new,yind_new])
                                    posns_pix[yind_new,xind_new] = kd1.data[neighbors][0,ii]
                                    posns_idx[yind_new,xind_new] = [xind_new,yind_new,neighbors[0][ii]]
                            #check whether the x position is more than 15 pixels less
                            #than the search lenslet
                            elif entry[0] < -maxdist:
                                #this means that the x index is one less than the
                                #search lenslet
                                xind_new = xind - 1
                                #check whether the y value is less than 15 pixels away from
                                #the search lenslet
                                if abs(entry[1]) < maxdist:
                                    #this means x is greater and y is the same
                                    #which means we found the lenslet directly to the left
                                    yind_new = yind
                                    posns_pix[yind_new,xind_new] = kd1.data[neighbors][0,ii]
                                    posns_idx[yind_new,xind_new] = [xind_new,yind_new,neighbors[0][ii]]
                                    positions_entered.append(neighbors[0][ii])
                                    to_search_around.append(neighbors[0][ii])
                                    positions.append([xind_new,yind_new])
                            elif entry[1] > maxdist:
                                yind_new = yind+1
                                xind_new = xind
                                #if yind_new > posns.shape[0]:
                                #    zeros_row = np.zeros((1,posns.shape[1],posns.shape[2]))
                                #    posns = np.vstack((posns,zeros_row))
                                posns_pix[yind_new,xind_new] = kd1.data[neighbors][0,ii]
                                posns_idx[yind_new,xind_new] = [xind_new,yind_new,neighbors[0][ii]]
                                positions_entered.append(neighbors[0][ii])
                                to_search_around.append(neighbors[0][ii])
                                positions.append([xind_new,yind_new])
                                #done.append(neighbors[0][ii])


                            elif entry[1] < -maxdist:
                                #print('found lenslet below')
                                xind_new = xind
                                yind_new = yind-1
                                #if yind_new < 0:
                                #    zeros_row = np.zeros((1,posns.shape[1],posns.shape[2]))
                                #    posns = np.vstack((zeros_row,posns))
                                posns_pix[yind_new,xind_new] = kd1.data[neighbors][0,ii]
                                posns_idx[yind_new,xind_new] = [xind_new,yind_new,neighbors[0][ii]]
                                positions_entered.append(neighbors[0][ii])
                                to_search_around.append(neighbors[0][ii])
                                positions.append([xind_new,yind_new])
                                #done.append(neighbors[0][ii])
                            else:
                                continue
                            #elif (abs(entry[0]) < 15) and (abs(entry[1]) < 15):
                            #    print('')
                            #else:
                            #    print('')
                    done0=len(done_searching_around)
                    done_searching_around.append(search_lens)
                    ddone = len(done_searching_around)-done0


        minx = np.nanmin(posns_idx[:,:,0])
        maxx = np.nanmax(posns_idx[:,:,0])


        miny = np.nanmin(posns_idx[:,:,1])
        maxy = np.nanmax(posns_idx[:,:,1])


        posns_idx[:,:,0]-=minx
        posns_idx[:,:,1]-=miny


        final_posns = np.zeros([int(maxy),int(maxx)])
        final_posns = posns_idx[int(miny):int(maxy+1),int(minx):int(maxx+1)]


        if show_plots==True:
            dists = np.sqrt(posns_idx[:,:,0]**2 + posns_idx[:,:,1]**2)
            plt.imshow(dists)
            plt.colorbar()
            plt.show()


            f = plt.figure(figsize=(11,5))
            f.add_subplot(121)
            plt.title('L band: lenslet x positions\n'+'(in lenslet array, total='+str(int(np.nanmax(final_posns[:,:,0])+1))+')')
            plt.imshow(final_posns[:,:,0])
            plt.colorbar()
            f.add_subplot(122)
            plt.title('L band: lenslet y positions\n'+'(in lenslet array, total='+str(int(np.nanmax(final_posns[:,:,1])+1))+')')
            plt.imshow(final_posns[:,:,1])
            plt.colorbar()
            plt.show()
        return final_posns


    def assign_spots_to_grid(self, spots, spot_tracks_u, lams, lmin, lmax, to_index, conf=0.15, min_spots=3):
        """Map every detected spot to a grid cell. Returns final_posns, tracks, mid-band positions, diagnostics."""

        shape, shape_pct, full_pos = self.measure_trace_shape(spot_tracks_u,lams,lmin,lmax)

        nw = len(spots)
        W = np.concatenate([np.full(spots[w]['n_spots'], w) for w in range(nw)])
        P = np.concatenate([spots[w]['coordinates'] for w in range(nw)]).astype(float)
        I = np.concatenate([spots[w]['intensities'] for w in range(nw)])
        L = np.concatenate([spots[w]['lam'] for w in range(nw)])
        u = to_index(P - shape[W])
        n = np.round(u).astype(int)
        f = np.abs(u - n).max(axis=1)
        cells, ncollide = {}, 0
        for i in np.where(f < conf)[0]:
            c = cells.setdefault((n[i, 0], n[i, 1]), {})
            w = W[i]
            if w in c:
                ncollide += 1
                if f[i] >= c[w][1]:
                    continue
            c[w] = ((P[i, 0], P[i, 1], I[i], L[i]), f[i])
        counts = {c: len(d) for c, d in cells.items()}
        keep = [c for c in cells if counts[c] >= min_spots]
        nmin = np.min(np.array(keep), axis=0)
        tracks = {j: {w: v[0] for w, v in sorted(cells[c].items())} for j, c in enumerate(keep)}
        avgs = np.array([np.median([np.array(t[w][:2]) - shape[w] for w in t], axis=0) for t in tracks.values()])
        grid_size = 120                       # fixed lenslet-grid size for every LowRes cube
        keep_arr = np.array(keep)
        nxg_, nyg_ = keep_arr.max(axis=0) - nmin + 1
        if nxg_ > grid_size or nyg_ > grid_size:
            raise ValueError(
                f'Lenslet grid spans {nyg_} x {nxg_}, larger than the fixed '
                f'{grid_size} x {grid_size} grid; increase grid_size.')
        fpos = np.full((grid_size, grid_size, 3), np.nan)
        off = (grid_size - np.array([nxg_, nyg_])) // 2
        for j, (cx, cy) in enumerate(keep):
            fpos[cy - nmin[1] + off[1], cx - nmin[0] + off[0]] = [cx - nmin[0] + off[0], cy - nmin[1] + off[1], j]
        return fpos, tracks, avgs, dict(f=f, counts=counts, ncollide=ncollide, nspots=len(P),
                                        cells=np.array(list(counts)), cell_n=np.array(list(counts.values())))


    def fill_missing_spots(self,ims_cal,tracks_new,lams,lmin,lmax,min_fit_spots=5,fill_missing=True,cropsize=8):
        trace_shape, trace_shape_pct, full_posns = self.measure_trace_shape(tracks_new, lams, lmin, lmax)

        def fit_lenslet_traces(tracks, shape, min_fit=5):
            """Per lenslet and axis, fit pos(w) = c + s*shape[w] to its detected spots. Returns c, s of shape (n, 2)."""
            n = len(tracks)
            c, s = np.zeros((n, 2)), np.ones((n, 2))
            for j in range(n):
                ws = np.array(sorted(tracks[j]))
                p = np.array([tracks[j][w][:2] for w in ws], float)
                for ax in range(2):
                    sh = shape[ws, ax]
                    if len(ws) >= min_fit and np.ptp(sh) > 5:
                        s[j, ax], c[j, ax] = np.polyfit(sh, p[:, ax], 1)
                    else:
                        c[j, ax] = np.median(p[:, ax] - sh)
            return c, s

        def smoothed_value(im, x, y, sigma=1.2, half=6):
            xi, yi = int(round(x)), int(round(y))
            st_ = im[max(yi - half, 0):yi + half + 1, max(xi - half, 0):xi + half + 1]
            sm = gaussian_filter(st_, sigma)
            return sm[yi - max(yi - half, 0), xi - max(xi - half, 0)]

        trc_c, trc_s = fit_lenslet_traces(tracks_new, trace_shape, min_fit_spots)
        self.logger.info(f'trace stretch s: x median {np.median(trc_s[:, 0]):.3f} (16-84%: {np.percentile(trc_s[:, 0], 16):.3f}-'
          f'{np.percentile(trc_s[:, 0], 84):.3f}), y median {np.median(trc_s[:, 1]):.3f} '
          f'({np.percentile(trc_s[:, 1], 16):.3f}-{np.percentile(trc_s[:, 1], 84):.3f})')

        rng = np.random.default_rng(0)
        errs = {'per-lenslet c + s*shape': [], 'global shape (s=1)': []}
        for j in rng.choice(len(tracks_new), 3000):
            t = tracks_new[j]
            if len(t) < min_fit_spots + 1:
                continue
            w0 = rng.choice(list(t))
            rest = {w: v for w, v in t.items() if w != w0}
            c1, s1 = fit_lenslet_traces({0: rest}, trace_shape, min_fit_spots)
            truth = np.array(t[w0][:2])
            errs['per-lenslet c + s*shape'].append(np.hypot(*(c1[0] + s1[0] * trace_shape[w0] - truth)))
            p = np.array([v[:2] for v in rest.values()], float) - trace_shape[list(rest)]
            errs['global shape (s=1)'].append(np.hypot(*(np.median(p, axis=0) + trace_shape[w0] - truth)))
        for k, e in errs.items():
            self.logger.info(f'leave-one-out position error, {k:24s}: median {np.median(e):.2f} px, 90th {np.percentile(e, 90):.2f}, '
                f'max {np.max(e):.2f}')

        filled_set = {}      # (lenslet, w) -> 'gap' (inside its detected range) or 'edge' (outside it)
        kinds = np.array([])
        n_off = 0
        if fill_missing:
            half = cropsize / 2
            for j in range(len(tracks_new)):
                t = tracks_new[j]
                wlo, whi = min(t), max(t)
                for w in range(len(lams)):
                    if w in t:
                        continue
                    x, y = trc_c[j] + trc_s[j] * trace_shape[w]
                    # make_posarr crops [int(x - half), int(x + half)); int() truncates toward zero, so the
                    # crop is only full-size when x - half >= 0 (not just int(x - half) >= 0)
                    if x - half < 0 or int(x + half) > ims_cal.shape[2] or y - half < 0 or int(y + half) > ims_cal.shape[1]:
                        n_off += 1
                        continue
                    t[w] = (x, y, smoothed_value(ims_cal[w], x, y), lams[w])
                    filled_set[(j, w)] = 'gap' if wlo < w < whi else 'edge'
                tracks_new[j] = dict(sorted(t.items()))
            kinds = np.array(list(filled_set.values()))
            self.logger.info(f'filled {len(filled_set)} spots ({(kinds == "gap").sum()} gap, {(kinds == "edge").sum()} edge); '
                f'{n_off} predicted crops off the detector were left empty')

        return tracks_new, filled_set, kinds

    def make_posarr(self,ims_cal,final_posns,spot_tracks_u,medres=False,show_plots=False,cropsize=10,cut=0.01):
        maxy = final_posns.shape[0]
        maxx = final_posns.shape[1]
        if medres==False:
            sizex = self.nx
            sizey = self.ny
            diffx = sizex-maxx
            diffy = sizey-maxy
        if medres==True:
            sizex=self.nx
            sizey=self.ny
            diffx=0
            diffy=0

        posarr = np.zeros([len(ims_cal),sizey,sizex,7])
        posarr[:,:,:,:] = np.nan
        spotim_arr = np.zeros([len(ims_cal),sizey,sizex,cropsize,cropsize])
        spotim_arr[:] = np.nan

        for i in range(maxy):
            for j in range(maxx):
                xpos,ypos,lind = final_posns[i,j]
                if np.isnan(xpos)==False:
                    tofill = list(spot_tracks_u[lind].keys())
                    for k in tofill:
                        x,y,intens = spot_tracks_u[lind][k][:3]
                        # window start, then end = start + cropsize, so the
                        # crop is never larger than cropsize (int(y+c/2) and
                        # int(y-c/2) can differ by c+1 when y is a hair below
                        # an integer, e.g. y = 4.999999999999999)
                        x0 = int(np.floor(x-cropsize/2))
                        y0 = int(np.floor(y-cropsize/2))
                        xs = max(0, x0)
                        xe = min(x0+cropsize, ims_cal.shape[2])
                        ys = max(0, y0)
                        ye = min(y0+cropsize, ims_cal.shape[1])
                        # spot window entirely off the detector: a negative
                        # xe/ye would wrap around in the slice below, so
                        # leave this spot empty
                        if xe <= xs or ye <= ys:
                            continue
                        posarr[k,i+diffy,j+diffx] = [x,xs,xe,y,ys,ye,intens]

                        cropped = np.zeros(ims_cal[k,ys:ye,xs:xe].shape)
                        cropped[:] = ims_cal[k,ys:ye,xs:xe]
                        cropped[np.where(cropped<cut*np.max(cropped))] = 0.0
                        spotim_arr[k,i+diffy,j+diffx,:ye-ys,:xe-xs] = cropped

        if show_plots==True:
            f = plt.figure(clear=True)
            if medres==False:
                for i in range(20,40):
                    for j in range(40,60):
                        plt.scatter(posarr[:,i,j,0],posarr[:,i,j,3],c=range(len(posarr)))
                plt.show()
            if medres==True:
                for i in range(17):
                    for j in range(18):
                        plt.scatter(posarr[:,i,j,0],posarr[:,i,j,3],c=range(len(posarr)))
                plt.show()
        return posarr, spotim_arr


    def make_posarr_names(self,names,meanim,final_posns,spot_tracks_u,medres=False,show_plots=False,cropsize=10,cut=0.01):
        maxy = final_posns.shape[0]
        maxx = final_posns.shape[1]
        if medres==False:
            sizex = self.nx
            sizey = self.ny
            diffx = sizex-maxx
            diffy = sizey-maxy
        if medres==True:
            sizex=self.nx
            sizey=self.ny
            diffx=0
            diffy=0

        posarr = np.zeros([len(names),sizey,sizex,7])
        posarr[:,:,:,:] = np.nan
        spotim_arr = np.zeros([len(names),sizey,sizex,cropsize,cropsize])
        spotim_arr[:] = np.nan
        for imind in range(len(names)):
            self.logger.info('loading image '+str(imind))
            fname = names[imind]
            image = pyfits.getdata(fname,memmap=False)
            im_cal_tmp = self.monochrom_bksub_sing(image,meanim)
            if self.config.instrument.apply_destripe==True:
                im_cal = self.masked_row_destripe(im_cal_tmp)[0]
            else:
                im_cal = im_cal_tmp

            for i in range(maxy):
                for j in range(maxx):
                    xpos,ypos,lind = final_posns[i,j]
                    if np.isnan(xpos)==False and imind in spot_tracks_u[lind]:
                        k=imind
                        x,y,intens = spot_tracks_u[lind][k][:3]
                        # window start, then end = start + cropsize, so the
                        # crop is never larger than cropsize (int(y+c/2) and
                        # int(y-c/2) can differ by c+1 when y is a hair below
                        # an integer, e.g. y = 4.999999999999999)
                        x0 = int(np.floor(x-cropsize/2))
                        y0 = int(np.floor(y-cropsize/2))
                        xs = max(0, x0)
                        xe = min(x0+cropsize, im_cal.shape[1])
                        ys = max(0, y0)
                        ye = min(y0+cropsize, im_cal.shape[0])
                        # spot window entirely off the detector: a negative
                        # xe/ye would wrap around in the slice below, so
                        # leave this spot empty
                        if xe <= xs or ye <= ys:
                            continue
                        posarr[k,i+diffy,j+diffx] = [x,xs,xe,y,ys,ye,intens]



                        cropped = np.zeros(im_cal[ys:ye,xs:xe].shape)
                        cropped[:] = im_cal[ys:ye,xs:xe]
                        cropped[np.where(cropped<cut*np.max(cropped))] = 0.0
                        spotim_arr[k,i+diffy,j+diffx,:ye-ys,:xe-xs] = cropped

        if show_plots==True:
            f = plt.figure(clear=True)
            if medres==False:
                for i in range(20,40):
                    for j in range(40,60):
                        plt.scatter(posarr[:,i,j,0],posarr[:,i,j,3],c=range(len(posarr)))
                plt.show()
            if medres==True:
                for i in range(17):
                    for j in range(18):
                        plt.scatter(posarr[:,i,j,0],posarr[:,i,j,3],c=range(len(posarr)))
                plt.show()
        return posarr, spotim_arr

    def drop_lowsnr_spots(self,ims_cal,posarr,final_posns,
                          filled_set,spotim_arr,
                          snr_min_gap=5.0,snr_min_edge=5.0):

        # filled mask, using make_posarr's placement of grid cell (i, j) at (i + diffy, j + diffx)
        filled = np.zeros(posarr.shape[:3], bool)
        fill_kind = np.full(posarr.shape[:3], '', dtype=object)

        diffy, diffx = posarr.shape[1] - final_posns.shape[0], posarr.shape[2] - final_posns.shape[1]
        cell_of = {int(final_posns[i, j, 2]): (i + diffy, j + diffx)
                for i, j in zip(*np.where(np.isfinite(final_posns[..., 2])))}
        for (j, w), kind in filled_set.items():
            iy, ix = cell_of[j]
            filled[w, iy, ix] = True
            fill_kind[w, iy, ix] = kind


        def spot_snr(ims, posarr, r_in=5, r_out=7):
            """S/N of every posarr entry, measured on the unclipped images."""
            snr = np.full(posarr.shape[:3], np.nan)
            yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
            rr = np.hypot(xx, yy)
            core = (np.abs(xx) <= 1) & (np.abs(yy) <= 1)
            ann = (rr >= r_in) & (rr <= r_out)
            for w in range(len(posarr)):
                im = np.pad(np.asarray(ims[w], float), r_out, constant_values=np.nan)   # off-detector pixels are NaN
                iy, ix = np.where(np.isfinite(posarr[w, ..., 0]))
                xc = np.round(posarr[w, iy, ix, 0]).astype(int) + r_out
                yc = np.round(posarr[w, iy, ix, 3]).astype(int) + r_out
                stamps = im[yc[:, None, None] + yy, xc[:, None, None] + xx]     # (n, 15, 15)
                a = stamps[:, ann]
                bkg = np.nanmedian(a, axis=1)
                noise = 1.4826 * np.nanmedian(np.abs(a - bkg[:, None]), axis=1)
                sig = np.nansum(stamps[:, core], axis=1) - 9 * bkg
                snr[w, iy, ix] = sig / (3 * np.maximum(noise, 1e-12))
            return snr


        snr = spot_snr(ims_cal, posarr)
        det = np.isfinite(posarr[..., 0]) & ~filled
        groups = {'detected': det, 'filled: gap': filled & (fill_kind == 'gap'),
                'filled: edge': filled & (fill_kind == 'edge')}
        for k, m in groups.items():
            v = snr[m & np.isfinite(snr)]
            if len(v):
                self.logger.info(f'{k:30s} n={m.sum():6d}  S/N median {np.median(v):7.1f}  <3: {(v < 3).sum():5d}  <5: {(v < 5).sum():5d}')

        """
        fig, ax = plt.subplots(1, 3, figsize=(20, 5.5))
        bins = np.logspace(-1, 3.5, 80)
        for k, m in groups.items():
            v = snr[m & np.isfinite(snr)]
            ax[0].hist(np.clip(v, 0.1, None), bins, histtype='step', lw=2, label=k)
        for cut_, col_ in [(SNR_MIN_GAP, 'C1'), (SNR_MIN_EDGE, 'C2')]:
            if cut_ is not None:
                ax[0].axvline(cut_, c=col_, ls='--')
        ax[0].set(xscale='log', yscale='log', xlabel='S/N', ylabel='# spots', title='S/N by spot type'); ax[0].legend(fontsize=8)
        for a, (k, m) in zip(ax[1:], list(groups.items())[1:]):
            w_, iy, ix = np.where(m)
            sc = a.scatter(posarr[w_, iy, ix, 0], posarr[w_, iy, ix, 3], c=np.clip(snr[w_, iy, ix], 0, 20), s=3, cmap='viridis')
            plt.colorbar(sc, ax=a, label='S/N (clipped at 20)'); a.set_aspect('equal'); a.set_title(k)
            a.set_xlim(0, 2048); a.set_ylim(0, 2048)
        plt.tight_layout()
        """

        drop = np.zeros_like(filled)
        for kind_, cut_ in [('gap', snr_min_gap), ('edge', snr_min_edge)]:
            if cut_ is not None:
                d_ = filled & (fill_kind == kind_) & ~(snr >= cut_)
                self.logger.info(f'removing {d_.sum()} {kind_} fills with S/N < {cut_} (or unmeasurable)')
                drop |= d_
        posarr[drop] = np.nan
        spotim_arr[drop] = np.nan
        filled &= ~drop
        self.logger.info(f'{filled.sum()} filled spots remain in posarr')
        return posarr, spotim_arr


    def adaptive_centroid(
        self,
        image,
        x0,
        y0,
        box_size=15,
        min_box_size=5,
        shrink_factor=0.8,
        snr_threshold=3.0,
        bkg_annulus_frac=0.5,
        max_iter=20,
        tol=1e-3,
        subtract_background=True,
    ):
        """
        Compute a sub-pixel centroid via iterative, SNR-thresholded first moments
        with a shrinking box.

        Parameters
        ----------
        image : 2D ndarray
            Full image (or a large-enough cutout) containing the source.
        x0, y0 : float
            Initial guess position (pixel coordinates, x = column, y = row).
        box_size : int
            Initial half-width... actually full width of the square box (pixels).
            Should comfortably contain the PSF/spot on the first iteration.
        min_box_size : int
            Smallest allowed box width; iteration stops shrinking below this.
        shrink_factor : float
            Multiplicative factor applied to box_size each iteration (0 < f < 1).
        snr_threshold : float
            Pixels with (value - local_bkg) / noise < snr_threshold are excluded
            from the moment calculation.
        bkg_annulus_frac : float
            Fraction of the box half-width used to define an annular region
            (outer ring of the box) for local background/noise estimation.
            E.g. 0.5 means the outer 50% of the box (by radius) is used.
        max_iter : int
            Maximum number of shrink/recenter iterations.
        tol : float
            Convergence tolerance on centroid shift (pixels) between iterations.
        subtract_background : bool
            If True, subtract the estimated local background before computing
            the flux-weighted moment (recommended -- otherwise a flat background
            biases the centroid toward the box center).

        Returns
        -------
        xc, yc : float
            Sub-pixel centroid position.
        info : dict
            Diagnostics: 'n_iter', 'converged', 'final_box_size',
            'n_pixels_used', 'background', 'noise'.
        """
        image = np.asarray(image, dtype=float)
        ny, nx = image.shape

        xc, yc = float(x0), float(y0)
        box = float(box_size)
        converged = False
        n_pixels_used = 0
        bkg, noise = 0.0, 0.0

        for i in range(max_iter):
            half = box / 2.0

            # Integer cutout bounds, clipped to image edges
            x_lo = max(int(np.floor(xc - half)), 0)
            x_hi = min(int(np.ceil(xc + half)), nx)
            y_lo = max(int(np.floor(yc - half)), 0)
            y_hi = min(int(np.ceil(yc + half)), ny)

            if x_hi - x_lo < 2 or y_hi - y_lo < 2:
                # Box collapsed too far (e.g. near an edge); bail out safely
                break

            cutout = image[y_lo:y_hi, x_lo:x_hi]
            yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]

            # --- local background/noise from the outer annulus of the box ---
            r = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
            annulus_r_in = half * (1 - bkg_annulus_frac)
            annulus_mask = r >= annulus_r_in

            if subtract_background and np.any(annulus_mask):
                bkg = np.median(cutout[annulus_mask])
                noise = np.std(cutout[annulus_mask])
            else:
                bkg = 0.0
                noise = np.std(cutout) if np.std(cutout) > 0 else 1.0
            noise = max(noise, 1e-12)  # avoid divide-by-zero

            data = cutout - bkg if subtract_background else cutout

            # --- SNR mask: only keep pixels significantly above the noise ---
            snr = data / noise
            mask = snr >= snr_threshold

            if not np.any(mask):
                # Threshold too aggressive for this iteration; relax once by
                # falling back to all positive-flux pixels rather than failing.
                mask = data > 0
                if not np.any(mask):
                    break

            weights = np.clip(data, 0, None) * mask
            total_flux = weights.sum()
            if total_flux <= 0:
                break

            x_new = (weights * xx).sum() / total_flux
            y_new = (weights * yy).sum() / total_flux
            n_pixels_used = int(mask.sum())

            shift = np.hypot(x_new - xc, y_new - yc)
            xc, yc = x_new, y_new

            # Shrink the box toward the source, but don't go below min_box_size
            box = max(box * shrink_factor, min_box_size)

            if shift < tol and box <= min_box_size:
                converged = True
                break

        info = {
            "n_iter": i + 1,
            "converged": converged,
            "final_box_size": box,
            "n_pixels_used": n_pixels_used,
            "background": bkg,
            "noise": noise,
        }
        return xc, yc, info


    def get_centroids(self, spots):
        centroids = np.zeros([spots.shape[0],spots.shape[1],spots.shape[2],2])
        centroids[:] = np.nan
        for ll in range(len(spots)):
            #spots is (n_lam, ny, nx, crop, crop): loop over ny and nx (MedRes is 17 x 18,
            #so len(spots[0]) / len(spots[1]), both ny, skipped the last lenslet column)
            for yind in range(spots.shape[1]):
                for xind in range(spots.shape[2]):
                    if True not in np.isnan(spots[ll,yind,xind]):
                        if len(np.unique(spots[ll,yind,xind]))!=1:
                            xc, yc, info = self.adaptive_centroid(spots[ll,yind,xind], x0=8, y0=8, box_size=16, min_box_size = 3)
                            #print(f"lam: {ll:.3f}, yind: {yind:.3f}, xind: {xind:.3f}")
                            #print(f"Recovered centroid: ({xc:.3f}, {yc:.3f})")
                            #print(f"Info: {info}")
                            centroids[ll,yind,xind] = [xc,yc]
        return centroids




    def halfmax_fwhm_x(self, combined, oversamp, nrows_avg=3):
        """Direct half-max-crossing FWHM in the x direction, in original detector pixels."""
        def interp_crossing(i0, i1, y0, y1, target):
            return i0 + (target - y0) * (i1 - i0) / (y1 - y0)
        peak_flat = np.nanargmax(combined)
        py, px = np.unravel_index(peak_flat, combined.shape)

        row_lo, row_hi = py - nrows_avg // 2, py + nrows_avg // 2 + 1
        profile = np.nanmean(combined[row_lo:row_hi, :], axis=0)
        x = np.arange(len(profile))
        good = ~np.isnan(profile)
        x, profile = x[good], profile[good]

        peak_i = np.argmax(profile)
        base = np.nanmin(profile)
        half = base + (profile[peak_i] - base) / 2.0

        # walk outward from the peak until the profil/e drops below half max
        ri = peak_i
        while ri < len(profile) - 1 and profile[ri] > half:
            ri += 1
        li = peak_i
        while li > 0 and profile[li] > half:
            li -= 1
        if profile[ri] > half or profile[li] > half:
            return np.nan  # never crossed half max within the array

        xr = interp_crossing(ri - 1, ri, profile[ri - 1], profile[ri], half)
        xl = interp_crossing(li, li + 1, profile[li], profile[li + 1], half)

        fwhm_grid = xr - xl
        return fwhm_grid / oversamp  # convert grid pixels -> original detector pixels

    def build_oversampled_2(self, ll, spots, centroids, oversamp, ny, nx, npix=16):
        out_size = (npix + 1) * oversamp + 1
        xcc = out_size // 2
        ycc = out_size // 2
        arr = np.full((ny * nx, out_size, out_size), np.nan)
        for yind in range(ny):
            for xind in range(nx):
                centroid = centroids[ll, yind, xind]
                if np.any(np.isnan(centroid)):
                    continue
                xcen, ycen = centroid

                xpix = np.array(range(len(spots[ll,yind,xind,0])),dtype=int)#[diffx_s:-diffx_e]
                ypix = np.array(range(len(spots[ll,yind,xind])),dtype=int)#[diffy_s:-diffy_e]
                xpix_o = xpix*oversamp
                ypix_o = ypix*oversamp
                cent_o = centroid*oversamp
                cs = centroid * oversamp - np.array([xcc,ycc])
                indsx = np.array(np.round(xpix_o - cs[0]),dtype=int)
                indsy = np.array(np.round(ypix_o - cs[1]),dtype=int)

                for i in range(len(indsy)):
                    for j in range(len(indsx)):
                        if indsy[i] > 0:
                            if indsx[j] > 0:
                                if indsy[i] < out_size:
                                    if indsx[j] < out_size:
                                        arr[yind * nx + xind, indsy[i], indsx[j]] = spots[ll,yind,xind,ypix[i],xpix[j]]
        return np.nanmean(arr, axis=0)


    def downsample_model_and_residuals_2(self, ll, spots, centroids, combined, oversamp, ny, nx, posarr, npix=16,
                                         normalize=True, normtype='sum'):
        """
        For every valid spot in frame `ll`, sample the oversampled PSF model at that
        spot's exact subpixel-aligned grid locations (i.e. invert the placement used
        to build `combined`), giving a npix x npix model cutout directly comparable
        to the raw spot. Returns per-spot raw cutouts, model cutouts, and residuals.

        normalize=True: least-squares-scale the model to each spot's flux before
        differencing (accounts for spot-to-spot amplitude variation, e.g. throughput).
        """
        out_size = (npix + 1) * oversamp + 1
        xcc = out_size // 2
        ycc = out_size // 2
        arr = np.full((ny * nx, out_size, out_size), np.nan)

        npix_s = len(spots[ll,0,0])
        raws = np.full((ny, nx, npix_s, npix_s), np.nan)
        models = np.full((ny, nx, npix_s, npix_s), np.nan)
        residuals = np.full((ny, nx, npix_s, npix_s), np.nan)
        scales = np.full((ny, nx), np.nan)
        modim = np.full((2048,2048),0.0)

        for yind in range(ny):
            for xind in range(nx):
                centroid = centroids[ll, yind, xind]
                if np.any(np.isnan(centroid)):
                    #empty grid cell (no lenslet, or no spot at this wavelength): nothing to model
                    continue
                xcen, ycen = centroid

                xpix = np.array(range(len(spots[ll,yind,xind,0])),dtype=int)
                ypix = np.array(range(len(spots[ll,yind,xind])),dtype=int)
                xpix_o = xpix*oversamp
                ypix_o = ypix*oversamp
                cent_o = centroid*oversamp
                cs = centroid * oversamp - np.array([xcc,ycc])
                indsx = np.array(np.round(xpix_o - cs[0]),dtype=int)
                indsy = np.array(np.round(ypix_o - cs[1]),dtype=int)

                for i in range(len(indsy)):
                    for j in range(len(indsx)):
                        if indsy[i] > 0:
                            if indsx[j] > 0:
                                if indsy[i] < out_size:
                                    if indsx[j] < out_size:
                                        raws[yind,xind,ypix[i],xpix[j]] = spots[ll,yind,xind,ypix[i],xpix[j]]
                                        models[yind,xind,ypix[i],xpix[j]] = combined[indsy[i], indsx[j]]
                if normalize:
                    #scale = np.nansum(models[yind,xind]*raws[yind,xind]) / np.nansum(models[yind,xind]*models[yind,xind])
                    if normtype=='sum':scale = np.nansum(raws[yind,xind]) / np.nansum(models[yind,xind])
                    if normtype=='max':scale = np.nanmax(raws[yind,xind]) / np.nanmax(models[yind,xind])
                else:
                    scale = 1.0

                residuals[yind,xind] = raws[yind,xind] - models[yind,xind]*scale
                scales[yind,xind] = scale

                xc,xs,xe,yc,ys,ye,intens = posarr[ll,yind,xind]
                xc = int(xc)
                yc = int(yc)
                xs = int(xs)
                xe = int(xe)
                ys = int(ys)
                ye = int(ye)

                modtoadd = models[yind,xind,:ye-ys,:xe-xs]
                modtoadd[np.where(np.isnan(models[yind,xind,:ye-ys,:xe-xs])==True)] = 0.0
                modim[ys:ye,xs:xe] += modtoadd*scales[yind,xind]
        return raws, models, residuals, scales, modim


    def residual_rms_pct_2(self, raws, residuals):
        """
        RMS residual expressed as a percentage of each spot's peak flux.
        Returns:
          per_spot_rms_pct : RMS residual / peak * 100, one value per spot (NaN for skipped spots)
          overall_rms_pct  : RMS over ALL residual pixels from ALL spots, each first
                              normalized by its own spot's peak (i.e. a single pooled
                              "typical %" number, not dominated by the brightest spots)
        """
        good = ~np.isnan(residuals).any(axis=(2, 3))
        peaks = np.nanmax(raws, axis=(2, 3))  # per-spot peak flux

        per_spot_rms_pct = np.full((raws.shape[0],raws.shape[1]), np.nan)
        per_spot_rms_pct[good] = (
            np.sqrt(np.mean(residuals[good]**2, axis=(1, 2))) / peaks[good] * 100
        )

        normed_residuals = residuals[good] / peaks[good, None, None]
        overall_rms_pct = np.sqrt(np.mean(normed_residuals**2)) * 100

        return per_spot_rms_pct, overall_rms_pct


    def AnK_spot_model_2(self, spots, centroids, ny, nx, oversamp, ims_cal, posarr):
        models_all = []
        models_sym_all = []
        resids_all = []
        resids_sym_all = []

        modims_all = []
        modims_sym_all = []

        for ll in range(len(spots)):
            combined = self.build_oversampled_2(ll, spots, centroids, oversamp, ny, nx, npix=16)
            fwhm_x = self.halfmax_fwhm_x(combined, oversamp)
            self.logger.info(f"FWHM_x (direct half-max crossing, oversampled image) = {fwhm_x}")

            raws, models, residuals, scales, modim = self.downsample_model_and_residuals_2(ll, spots, centroids, combined, oversamp,
                                                                                      ny, nx, posarr,npix=16,normtype='sum')
            good = ~np.isnan(residuals).any(axis=(2,3))
            rms = np.sqrt(np.nanmean(residuals[good]**2))
            self.logger.info(f"n spots: {good.sum()}  overall RMS residual: {rms}")
            per_spot_rms_pct, overall_rms_pct = self.residual_rms_pct_2(raws, residuals)
            self.logger.info("overall RMS residual (%% of peak, pooled) = %.2f%%" % overall_rms_pct)

            gauss = Gaussian2D(amplitude=1, x_mean=len(combined)/2.0-0.5, y_mean=len(combined)/2.0-0.5,
                               x_stddev=fwhm_x/2.355*oversamp, y_stddev=fwhm_x/2.355*oversamp,
                               theta=None, cov_matrix=None)
            xx,yy=np.meshgrid(range(len(combined)),range(len(combined[0])))
            filt = gauss(xx,yy)
            combined2 = combined*filt
            fwhm_x = self.halfmax_fwhm_x(combined2, oversamp)
            self.logger.info(f"FWHM_x (direct half-max crossing, filtered oversampled image) = {fwhm_x}")
            raws, models_sym, residuals_sym, scales_sym, modim_sym = self.downsample_model_and_residuals_2(ll, spots, centroids, combined2, oversamp,
                                                                                              ny, nx, posarr,npix=16,normtype='max')
            good = ~np.isnan(residuals_sym).any(axis=(2,3))
            rms = np.sqrt(np.nanmean(residuals_sym[good]**2))
            self.logger.info(f"n spots: {good.sum()}  overall RMS residual (filtered): {rms}")
            per_spot_rms_pct_sym, overall_rms_pct_sym = self.residual_rms_pct_2(raws, residuals_sym)
            self.logger.info("overall RMS residual (%% of peak, pooled, filtered) = %.2f%%" % overall_rms_pct_sym)

            """
            yind=10
            xind=8
            xc,xs,xe,yc,ys,ye,intens = posarr[ll,yind,xind]
            xc = int(xc)
            yc = int(yc)
            xs = int(xs)
            xe = int(xe)
            ys = int(ys)
            ye = int(ye)

            f = plt.figure(figsize=(12,4))
            f.suptitle('y = '+str(yc)+' x = '+str(xc))
            f.add_subplot(131)
            plt.imshow(raws[yind,xind])
            plt.colorbar()
            f.add_subplot(132)
            plt.imshow(models[yind,xind]*scales[yind,xind])
            plt.colorbar()
            f.add_subplot(133)
            plt.imshow(residuals[yind,xind])
            plt.colorbar()
            plt.show()
            """

            models_all.append(models)
            models_sym_all.append(models_sym)
            resids_all.append(residuals)
            resids_sym_all.append(residuals_sym)
            modims_all.append(modim)
            modims_sym_all.append(modim_sym)

        models_all = np.array(models_all)
        resids_all = np.array(resids_all)
        models_sym_all = np.array(models_sym_all)
        resids_sym_all = np.array(resids_sym_all)

        modims_all = np.array(modims_all)
        modims_sym_all = np.array(modims_sym_all)
        resims_all = ims_cal - np.array(modims_all)
        resims_sym_all = ims_cal - np.array(modims_sym_all)

        return models_all, resids_all, modims_all, resims_all, models_sym_all, resids_sym_all, modims_sym_all, resims_sym_all


    def AnK_spot_model(self, spots, posarr, centroids, nx, ny, oversamp, npix=13):
        modims = []
        resims = []
        model_arr = np.full([len(spots),ny,nx,npix,npix],np.nan)
        resid_arr = np.full([len(spots),ny,nx,npix,npix],np.nan)
        ####need to create proper array of model and residual thumbnails organized like the spot array!!!!
        for ll in range(len(spots)):
            combined = self.build_oversampled(ll, centroids, spots, nx, ny, oversamp, npix=npix)
            fwhm_x = self.halfmax_fwhm_x(combined, oversamp)
            self.logger.info(f"FWHM_x (direct half-max crossing, oversampled image) = {fwhm_x}")
            raws, models, residuals, scales, models_stampsize, residuals_stampsize = self.downsample_model_and_residuals(combined, spots, centroids, oversamp, ll, npix=npix)
            modim = np.zeros([2048,2048])
            resim = np.zeros([2048,2048])
            for lensy in range(len(spots[0])):
                for lensx in range(len(spots[0,0])):
                    if True not in np.isnan(posarr[ll,lensy,lensx]):
                        xc,xs,xe,yc,ys,ye,intens = posarr[ll,lensy,lensx]
                        xc = int(xc)
                        yc = int(yc)
                        xs = int(xs)
                        xe = int(xe)
                        ys = int(ys)
                        ye = int(ye)
                        #print(ll,lensy,lensx,ys,ye,xs,xe,models_stampsize.shape)
                        modim[ys:ye,xs:xe]+=models_stampsize[lensy*self.nx + lensx][:ye-ys,:xe-xs] ##this matches how the spot psf thumbnails are constructed
                        resim[ys:ye,xs:xe]+=residuals_stampsize[lensy*self.nx + lensx][:ye-ys,:xe-xs] ##this matches how the spot psf thumbnails are constructed
                        model_arr[ll,lensy,lensx] = models[lensy*self.nx + lensx]
                        #print(models.shape,np.unique(models[lensy*self.nx + lensx]))
                        resid_arr[ll,lensy,lensx] = residuals[lensy*self.nx + lensx]
            modims.append(modim)
            resims.append(resim)
            good = ~np.isnan(residuals).any(axis=(1,2))
            rms = np.sqrt(np.nanmean(residuals[good]**2))
            self.logger.info(f"n spots: {good.sum()}  overall RMS residual: {rms}")
            per_spot_rms_pct, overall_rms_pct = self.residual_rms_pct(raws, residuals)
            self.logger.info("overall RMS residual (%% of peak, pooled) = %.2f%%" % overall_rms_pct)
        return model_arr, resid_arr, np.array(modims), np.array(resims)



    # ------------------------------------------------------------------
    # Anderson & King (2000, PASP 112, 1360) effective PSFs for LowRes.
    #
    # LowRes spots are only ~2-3 px FWHM, so a single spot cannot give its
    # FWHM without assuming a profile. Many spots of nearly the same PSF
    # (same wavelength, nearby in the field) sample it at different sub-pixel
    # phases; stacking them on an oversampled grid recovers the effective PSF
    # (PSF convolved with the pixel response) without a profile model.
    #
    # Unlike AnK_spot_model_2 (one PSF per wavelength from all lenslets, the
    # clipped crops and nearest-cell placement), these build one ePSF per
    # wavelength *and field region* from unclipped stamps cut from ims_cal,
    # each spot normalised by its own flux, iterating as in Anderson & King:
    # stack sigma-clipped residuals from the current ePSF, smooth with a
    # shape-preserving quartic kernel, recentre, then refit every spot's
    # centre and flux with the new ePSF.
    # ------------------------------------------------------------------

    @staticmethod
    def _AnK_smoothing_kernel(size=5, degree=4):
        """2D Savitzky-Golay kernel: least-squares polynomial of total degree
        <= degree over a size x size window, evaluated at the centre. It
        smooths noise without broadening structure that the polynomial can
        follow (degree 4 is the 'quartic' kernel used for A&K ePSFs)."""
        h = size // 2
        yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
        terms = [xx.ravel()**a * yy.ravel()**b
                 for a in range(degree + 1) for b in range(degree + 1 - a)]
        A = np.array(terms, dtype=float).T
        return np.linalg.pinv(A)[0].reshape(size, size)

    def AnK_extract_stamps(self, image, xs, ys, half=6, bkg_rin=4.5):
        """
        Cut (2*half+1)^2 stamps from an image around spots at (xs, ys), with
        each stamp's local background (median of pixels further than bkg_rin
        from the spot) removed.

        Returns:
            stamps (n, S, S), cx, cy (spot centre in stamp pixels), noise
            (robust std of the background pixels), ok (bool, len(xs): which
            spots were fully on the image; the other outputs only cover these)
        """
        xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
        S = 2 * half + 1
        xi = np.round(xs).astype(int); yi = np.round(ys).astype(int)
        ok = ((xi - half >= 0) & (yi - half >= 0)
              & (xi + half < image.shape[1]) & (yi + half < image.shape[0]))
        off = np.arange(-half, half + 1)
        rows = yi[ok, None, None] + off[None, :, None]
        cols = xi[ok, None, None] + off[None, None, :]
        stamps = np.asarray(image[rows, cols], dtype=float)
        cx = xs[ok] - (xi[ok] - half)
        cy = ys[ok] - (yi[ok] - half)
        yy, xx = np.mgrid[:S, :S]
        ring = np.hypot(xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]) > bkg_rin
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            bpix = np.where(ring, stamps, np.nan)
            bkg = np.nanmedian(bpix, axis=(1, 2))
            noise = 1.4826 * np.nanmedian(np.abs(bpix - bkg[:, None, None]), axis=(1, 2))
        stamps = stamps - bkg[:, None, None]
        return stamps, cx, cy, noise, ok

    def _AnK_recentre(self, epsf, oversamp, radius=1.5, max_iter=5):
        """Shift the ePSF so its flux centroid within `radius` detector px of
        the grid centre sits on the grid centre."""
        G = epsf.shape[0]; c0 = G // 2
        yy, xx = np.mgrid[:G, :G]
        r = np.hypot(xx - c0, yy - c0) / oversamp
        for _ in range(max_iter):
            w = np.where((r <= radius) & (epsf > 0), epsf, 0.0)
            if w.sum() <= 0:
                break
            mx = (w * xx).sum() / w.sum() - c0
            my = (w * yy).sum() / w.sum() - c0
            if np.hypot(mx, my) < 1e-3:
                break
            epsf = shift(epsf, (-my, -mx), order=3, mode='constant', cval=0.0)
        return epsf

    def _AnK_refit(self, stamps, cx, cy, flux, epsf, oversamp, fit_radius=3.5, n_steps=2,
                   max_step=0.5):
        """Gauss-Newton refit of each spot's (flux, x, y) with the ePSF as the
        model, vectorised over spots. Pixels within fit_radius are used."""
        G = epsf.shape[0]; c0 = G // 2
        dEy, dEx = np.gradient(epsf)
        S = stamps.shape[1]
        yy, xx = np.mgrid[:S, :S]
        finite = np.isfinite(stamps)
        data = np.where(finite, stamps, 0.0)
        for _ in range(n_steps):
            gx = (xx[None] - cx[:, None, None]) * oversamp + c0
            gy = (yy[None] - cy[:, None, None]) * oversamp + c0
            coords = [gy.ravel(), gx.ravel()]
            E = map_coordinates(epsf, coords, order=3, mode='constant').reshape(stamps.shape)
            Ex = map_coordinates(dEx, coords, order=3, mode='constant').reshape(stamps.shape) * oversamp
            Ey = map_coordinates(dEy, coords, order=3, mode='constant').reshape(stamps.shape) * oversamp
            w = (finite & (np.hypot(xx[None] - cx[:, None, None],
                                    yy[None] - cy[:, None, None]) <= fit_radius)).astype(float)
            J = np.stack([E, -flux[:, None, None] * Ex, -flux[:, None, None] * Ey], axis=-1)
            J *= w[..., None]
            res = (data - flux[:, None, None] * E) * w
            JTJ = np.einsum('nijk,nijl->nkl', J, J)
            JTr = np.einsum('nijk,nij->nk', J, res)
            JTJ += 1e-12 * np.eye(3)[None]
            try:
                step = np.linalg.solve(JTJ, JTr[..., None])[..., 0]
            except np.linalg.LinAlgError:
                step = np.stack([np.linalg.lstsq(a, b, rcond=None)[0] for a, b in zip(JTJ, JTr)])
            step[:, 1:] = np.clip(step[:, 1:], -max_step, max_step)
            flux = flux + step[:, 0]; cx = cx + step[:, 1]; cy = cy + step[:, 2]
        return cx, cy, flux

    def AnK_build_epsf(self, stamps, cx, cy, oversamp=4, n_iter=5, norm_radius=3.0,
                       fit_radius=3.5, clip_sigma=3.0, smooth=True):
        """
        Anderson & King effective PSF from background-subtracted stamps.

        Args:
            stamps: (n, S, S) background-subtracted spot stamps (S odd)
            cx, cy: (n,) initial spot centres in stamp pixels
            oversamp: ePSF grid points per detector pixel
            n_iter: build / refit iterations
            norm_radius: aperture radius (px) for the initial flux normalisation
            fit_radius: radius (px) used when refitting spot centres and fluxes
            clip_sigma: sigma clip applied to the samples in each ePSF grid cell
            smooth: smooth each iteration with the quartic SG kernel

        Returns:
            epsf: (G, G) with G = (S-1)*oversamp + 1; grid point k is an offset
                of (k - G//2)/oversamp detector px from the spot centre. Values
                are detector-pixel samples of a spot normalised to unit
                aperture flux.
            info: dict with the final cx, cy, flux, nstars and resid_pct (median
                over spots of the RMS residual within fit_radius, in % of the
                spot's model peak)
        """
        n, S, _ = stamps.shape
        half = S // 2
        G = 2 * half * oversamp + 1
        c0 = G // 2
        yy, xx = np.mgrid[:S, :S]
        cx = np.asarray(cx, float).copy(); cy = np.asarray(cy, float).copy()
        r = np.hypot(xx[None] - cx[:, None, None], yy[None] - cy[:, None, None])
        flux = np.nansum(np.where(r <= norm_radius, stamps, 0.0), axis=(1, 2))
        keep = flux > 0
        stamps, cx, cy, flux = stamps[keep], cx[keep], cy[keep], flux[keep]
        kern = self._AnK_smoothing_kernel(5, 4) if smooth else None
        epsf = np.zeros((G, G))
        for it in range(n_iter):
            gx = (xx[None] - cx[:, None, None]) * oversamp + c0
            gy = (yy[None] - cy[:, None, None]) * oversamp + c0
            model = map_coordinates(epsf, [gy.ravel(), gx.ravel()], order=3,
                                    mode='constant').reshape(stamps.shape)
            resid = stamps / flux[:, None, None] - model
            ix = np.round(gx).astype(int); iy = np.round(gy).astype(int)
            valid = np.isfinite(resid) & (ix >= 0) & (ix < G) & (iy >= 0) & (iy < G)
            cell = (iy * G + ix)[valid]; v = resid[valid]
            cnt = np.bincount(cell, minlength=G * G)
            with np.errstate(invalid='ignore', divide='ignore'):
                mean = np.bincount(cell, v, G * G) / cnt
                sd = np.sqrt(np.maximum(np.bincount(cell, v * v, G * G) / cnt - mean**2, 0))
            keep_s = np.abs(v - mean[cell]) <= clip_sigma * sd[cell] + 1e-30
            cnt2 = np.bincount(cell[keep_s], minlength=G * G)
            with np.errstate(invalid='ignore', divide='ignore'):
                delta = np.where(cnt2 > 0, np.bincount(cell[keep_s], v[keep_s], G * G) / cnt2, 0.0)
            epsf = epsf + delta.reshape(G, G)
            if smooth:
                epsf = nd_convolve(epsf, kern, mode='constant', cval=0.0)
            epsf = self._AnK_recentre(epsf, oversamp)
            if it < n_iter - 1:
                cx, cy, flux = self._AnK_refit(stamps, cx, cy, flux, epsf, oversamp, fit_radius)
                good = np.isfinite(flux) & (flux > 0) & np.isfinite(cx) & np.isfinite(cy)
                stamps, cx, cy, flux = stamps[good], cx[good], cy[good], flux[good]

        # residual quality of the final ePSF
        gx = (xx[None] - cx[:, None, None]) * oversamp + c0
        gy = (yy[None] - cy[:, None, None]) * oversamp + c0
        model = flux[:, None, None] * map_coordinates(
            epsf, [gy.ravel(), gx.ravel()], order=3, mode='constant').reshape(stamps.shape)
        inr = np.hypot(xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]) <= fit_radius
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            rms = np.sqrt(np.nanmean(np.where(inr, stamps - model, np.nan)**2, axis=(1, 2)))
            resid_pct = float(np.nanmedian(rms / (flux * epsf.max())) * 100)
        return epsf, dict(cx=cx, cy=cy, flux=flux, nstars=len(stamps), resid_pct=resid_pct)

    def _AnK_rotated(self, epsf, oversamp, ux, uy, step=0.02):
        """The ePSF sampled on a grid aligned with the unit vector (ux, uy): returns
        s (offsets, detector px), rot (len(s), len(s)) with rot[i, j] at offset s[i]
        along u and s[j] along the perpendicular, and the meshgrids S_, T_."""
        G = epsf.shape[0]; c0 = G // 2
        R = c0 / oversamp
        s = np.arange(-R, R + 1e-9, step)
        n = np.hypot(ux, uy); ux, uy = ux / n, uy / n
        vx, vy = -uy, ux
        S_, T_ = np.meshgrid(s, s, indexing='ij')           # S_ along u, T_ along v
        gx = c0 + oversamp * (S_ * ux + T_ * vx)
        gy = c0 + oversamp * (S_ * uy + T_ * vy)
        rot = map_coordinates(epsf, [gy.ravel(), gx.ravel()], order=3,
                              mode='constant').reshape(S_.shape)
        return s, rot, S_, T_

    def AnK_epsf_marginals(self, epsf, oversamp, ux=1.0, uy=0.0, step=0.02, window=3.0):
        """Marginal profiles of an ePSF along (ux, uy) and across it, as used by
        AnK_epsf_fwhm (each summed within +/- window px of the centre across the
        other direction). Returns s, marg_along, marg_across."""
        s, rot, S_, T_ = self._AnK_rotated(epsf, oversamp, ux, uy, step)
        if window is None:
            return s, rot.sum(axis=1), rot.sum(axis=0)
        return (s, np.where(np.abs(T_) <= window, rot, 0.0).sum(axis=1),
                np.where(np.abs(S_) <= window, rot, 0.0).sum(axis=0))

    @staticmethod
    def halfmax_rows(P, s):
        """FWHM of each row of P (profiles sampled at offsets s, evenly spaced), with
        the same half-maximum crossing rule as AnK_epsf_fwhm; NaN where a row is not
        finite, has no positive peak or does not drop below half maximum on both sides."""
        P = np.asarray(P, float)
        n, ns = P.shape
        step = s[1] - s[0]
        out = np.full(n, np.nan)
        fin = np.all(np.isfinite(P), axis=1)
        if not fin.any():
            return out
        rows = np.nonzero(fin)[0]
        Q = P[rows]
        pk = np.argmax(Q, axis=1)
        r_ = np.arange(len(rows))
        h = Q[r_, pk] / 2.0
        below = Q <= h[:, None]
        idx = np.arange(ns)
        lm = below & (idx[None, :] <= pk[:, None])
        rm = below & (idx[None, :] >= pk[:, None])
        has = lm.any(axis=1) & rm.any(axis=1) & (h > 0)
        l = ns - 1 - np.argmax(lm[:, ::-1], axis=1)
        r = np.argmax(rm, axis=1)
        has &= (l < pk) & (r > pk)
        l = np.where(has, l, 0); r = np.where(has, r, 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            xl = s[l] + (h - Q[r_, l]) / (Q[r_, l + 1] - Q[r_, l]) * step
            xr = s[r - 1] + (h - Q[r_, r - 1]) / (Q[r_, r] - Q[r_, r - 1]) * step
        out[rows] = np.where(has, xr - xl, np.nan)
        return out

    def AnK_epsf_fwhm(self, epsf, oversamp, ux=1.0, uy=0.0, step=0.02, window=3.0):
        """
        FWHMs (detector px) of an ePSF along the unit vector (ux, uy) and
        perpendicular to it.

        Returns (marg_along, marg_across, slice_along, slice_across):
            marg_*: FWHM of the marginal profile (ePSF summed across the other
                direction, within +/- window px of the centre). marg_along is
                the line spread function an extraction that sums across the
                trace sees; it is the quantity to compare with the
                Gaussian-fit FWHMs. Limiting the sum to +/- window px keeps it
                from being widened by a small residual background pedestal
                (summed over the whole stamp, a 0.4%-of-peak pedestal widened
                LowRes-L marginals by ~1-2%). None sums the whole ePSF.
            slice_*: FWHM of the profile on a line through the ePSF centre.
        """
        s, rot, S_, T_ = self._AnK_rotated(epsf, oversamp, ux, uy, step)
        mid = len(s) // 2

        def halfmax(p):
            if not np.all(np.isfinite(p)):
                return np.nan
            pk = int(np.argmax(p)); h = p[pk] / 2.0
            l = pk
            while l > 0 and p[l] > h:
                l -= 1
            r = pk
            while r < len(p) - 1 and p[r] > h:
                r += 1
            if p[l] > h or p[r] > h:
                return np.nan
            xl = s[l] + (h - p[l]) / (p[l + 1] - p[l]) * step
            xr = s[r - 1] + (h - p[r - 1]) / (p[r] - p[r - 1]) * step
            return xr - xl

        if window is None:
            marg_a, marg_c = rot.sum(axis=1), rot.sum(axis=0)
        else:
            marg_a = np.where(np.abs(T_) <= window, rot, 0.0).sum(axis=1)
            marg_c = np.where(np.abs(S_) <= window, rot, 0.0).sum(axis=0)
        return (halfmax(marg_a), halfmax(marg_c),
                halfmax(rot[:, mid]), halfmax(rot[mid, :]))

    def AnK_lowres_psfs(self, ims_cal, fitarr, lams, nreg=3, oversamp=4, half=6,
                        n_iter=5, snr_min=20.0, max_spots=1500, min_spots=50, seed=0):
        """
        Anderson & King ePSFs for LowRes spots, one per wavelength and field
        region, and their FWHMs along / across the trace.

        The lenslet grid is split into nreg x nreg regions with equal numbers
        of lenslets per row / column band (field-dependent PSF shape is then
        resolved at that scale). Spots come from the Gaussian fits (fitarr,
        used only for the initial centres and to select spots); a spot is used
        if its stamp is on the detector and its aperture S/N >= snr_min.

        Args:
            ims_cal: (nlam, 2048, 2048) background-subtracted masters
            fitarr: (nlam, ny, nx, >=6) Gaussian spot fits (fit_gauss_spots)
            lams: (nlam,) wavelengths in um
            nreg: field regions per axis
            oversamp, n_iter: passed to AnK_build_epsf
            half: stamp half-width (stamps are 2*half+1 px; neighbouring
                lenslet spots at one wavelength are >= ~19 px apart)
            snr_min, max_spots, min_spots: spot selection per region

        Returns a dict with
            epsf (nlam, nreg, nreg, G, G), and (nlam, nreg, nreg) arrays:
            fwhm_along, fwhm_across (marginal ePSF FWHM, px),
            slice_along, slice_across, gauss_along, gauss_across (median
            Gaussian-fit marginal FWHM of the same spots), disp (median px/um),
            R_along, R_across (lambda / FWHM_lambda from the ePSF FWHMs),
            nspots, resid_pct, ux, uy (median trace direction), xdet, ydet
            (median detector position); region_y, region_x (lenslet index
            bounds per region) and oversamp, half.
        """
        lams = np.asarray(lams, dtype=float)
        rng = np.random.default_rng(seed)
        nl, ny, nx = fitarr.shape[:3]
        dxdl, dydl, good = self.trace_derivatives(fitarr, lams)
        with np.errstate(invalid='ignore', divide='ignore'):
            disp = np.hypot(dxdl, dydl)
            ux, uy = dxdl / disp, dydl / disp
        xm, ym = fitarr[..., 1], fitarr[..., 2]
        gal, gac = self.gauss_fwhm_along_across(fitarr[..., 3], fitarr[..., 4], fitarr[..., 5], ux, uy)

        # region bounds: quantiles of the occupied lenslet rows / columns
        jj, ii = np.nonzero(good.any(axis=0))
        yb = np.unique(np.round(np.percentile(jj, np.linspace(0, 100, nreg + 1))).astype(int))
        xb = np.unique(np.round(np.percentile(ii, np.linspace(0, 100, nreg + 1))).astype(int))
        yb[-1] += 1; xb[-1] += 1
        nry, nrx = len(yb) - 1, len(xb) - 1
        yreg = np.full(ny, -1); xreg = np.full(nx, -1)
        for k in range(nry):
            yreg[yb[k]:yb[k + 1]] = k
        for k in range(nrx):
            xreg[xb[k]:xb[k + 1]] = k

        G = 2 * half * oversamp + 1
        shp = (nl, nry, nrx)
        out = {k: np.full(shp, np.nan) for k in
               ['fwhm_along', 'fwhm_across', 'slice_along', 'slice_across', 'gauss_along',
                'gauss_across', 'disp', 'R_along', 'R_across', 'resid_pct', 'ux', 'uy',
                'xdet', 'ydet']}
        out['nspots'] = np.zeros(shp, dtype=int)
        out['epsf'] = np.full(shp + (G, G), np.nan)
        npix_ap = np.pi * 3.0**2
        for l in range(nl):
            img = ims_cal[l]
            for ry in range(nry):
                for rx in range(nrx):
                    sel = (good[l] & np.isfinite(ux[l]) & (yreg[:, None] == ry)
                           & (xreg[None, :] == rx))
                    if sel.sum() < min_spots:
                        continue
                    stamps, cx, cy, noise, ok = self.AnK_extract_stamps(
                        img, xm[l][sel], ym[l][sel], half=half)
                    S = stamps.shape[1]
                    yy, xx = np.mgrid[:S, :S]
                    ap = np.hypot(xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]) <= 3.0
                    with np.errstate(invalid='ignore', divide='ignore'):
                        snr = np.nansum(np.where(ap, stamps, 0), axis=(1, 2)) / (noise * np.sqrt(npix_ap))
                    use = np.isfinite(snr) & (snr >= snr_min)
                    if use.sum() < min_spots:
                        continue
                    use = np.nonzero(use)[0]
                    if len(use) > max_spots:
                        use = np.sort(rng.choice(use, max_spots, replace=False))
                    epsf, info = self.AnK_build_epsf(stamps[use], cx[use], cy[use],
                                                     oversamp=oversamp, n_iter=n_iter)
                    idx = tuple(a[ok][use] for a in np.nonzero(sel))
                    u = np.array([np.nanmedian(ux[l][idx]), np.nanmedian(uy[l][idx])])
                    fa, fx, sa, sx_ = self.AnK_epsf_fwhm(epsf, oversamp, *u)
                    d = np.nanmedian(disp[l][idx])
                    o = (l, ry, rx)
                    out['epsf'][o] = epsf
                    out['fwhm_along'][o], out['fwhm_across'][o] = fa, fx
                    out['slice_along'][o], out['slice_across'][o] = sa, sx_
                    out['gauss_along'][o] = np.nanmedian(gal[l][idx])
                    out['gauss_across'][o] = np.nanmedian(gac[l][idx])
                    out['disp'][o] = d
                    out['R_along'][o] = lams[l] * d / fa
                    out['R_across'][o] = lams[l] * d / fx
                    out['nspots'][o] = info['nstars']
                    out['resid_pct'][o] = info['resid_pct']
                    out['ux'][o], out['uy'][o] = u / np.hypot(*u)
                    out['xdet'][o] = np.nanmedian(xm[l][idx])
                    out['ydet'][o] = np.nanmedian(ym[l][idx])
        out['region_y'] = np.array([yb[:-1], yb[1:]]).T
        out['region_x'] = np.array([xb[:-1], xb[1:]]).T
        out['oversamp'] = oversamp
        out['half'] = half
        return out

    def AnK_medres_psfs(self, names, meanim, fitarr, lams, nregx=3, nbin=16, oversamp=4,
                        half=8, bkg_rin=6.0, n_iter=5, snr_min=20.0, max_spots=1500,
                        min_spots=50, row_gap=20.0, seed=0, per_slice=True, gauss_centre=True):
        """
        Anderson & King ePSFs for MedRes spots, per wavelength bin and detector
        region, with the same outputs as AnK_lowres_psfs (so write_AnK_lowres_psfs,
        estimate_spectral_resolution_AnK and AnK_rectmat take them as they are).

        Differences from LowRes:
          - regions are on the detector, not bands of the lenslet index. On the
            detector the spots are 3 rows (found from gaps > row_gap px in each
            lenslet's median y offset from the other spots at the same
            wavelength) of six 17-spot slices; each slice is one image-slicer
            mirror and one lenslet x index, and can have its own PSF.
            per_slice=True (default): one region per slice (3 x 6), each lenslet
            assigned wholly to its own slice, with no blending between slices
            (they are different mirrors). per_slice=False: each detector row split
            into nregx equal-count bands in detector x, blended linearly in x
            between band centres;
          - with 17 lenslets per slice there are only 17 spots per region in one
            master, so each ePSF pools nbin consecutive masters (the PSF changes
            little over nbin steps) and is assigned their mean wavelength;
          - gauss_centre=True: each ePSF is shifted so that the centre of a 2D
            Gaussian fitted to it is on the grid centre. The builder centres on the
            flux centroid, which for the asymmetric MedRes spots is ~0.27 px off the
            Gaussian centre along the dispersion; the ePSFs are placed at
            Gaussian-fit spot centres (AnK_rectmat, spot_residuals), so without this
            the templates sit ~0.27 px off the spots (a residual dipole);
          - masters are read one at a time from names (load_calim);
          - stamps are larger (MedRes spots are ~4.5 px FWHM along the
            dispersion): 2*half+1 px, background from pixels beyond bkg_rin px.

        Args:
            names: (nl,) master filenames, sorted by wavelength
            meanim: mean master, for the background subtraction (load_calim)
            fitarr: (nl, ny, nx, >=6) Gaussian spot fits (fit_gauss_spots_names)
            lams: (nl,) master wavelengths (um)
            nregx: detector-x bands per detector row (per_slice=False only)
            nbin: consecutive masters pooled per ePSF

        Returns the AnK_lowres_psfs dict with the wavelength axis over bins, plus
            lams: (nbins,) mean wavelength of each bin (um), the ePSF wavelengths
                (pass these as lams_ank to AnK_rectmat);
            lens_fy, lens_fx: (ny, nx) each lenslet's position on the region grid
                (detector row; slice number, or fractional x band for
                per_slice=False; NaN = no spots), which AnK_rectmat uses to
                weight the region ePSFs;
            region_y, region_x: [[k, k+1]] region numbers (the regions are not
                lenslet-index bands); xcen (nry, nrx) region centres (detector x);
            gauss_dx, gauss_dy: (nbins, nry, nrx) Gaussian-centre offset of each
                ePSF from its grid centre before the gauss_centre shift (px);
            regspace: 'slice' or 'detector'.
        """
        lams = np.asarray(lams, dtype=float)
        rng = np.random.default_rng(seed)
        nl, ny, nx = fitarr.shape[:3]
        dxdl, dydl, good = self.trace_derivatives(fitarr, lams)
        with np.errstate(invalid='ignore', divide='ignore'):
            disp = np.hypot(dxdl, dydl)
            ux, uy = dxdl / disp, dydl / disp
        xm, ym = fitarr[..., 1], fitarr[..., 2]
        gal, gac = self.gauss_fwhm_along_across(fitarr[..., 3], fitarr[..., 4], fitarr[..., 5], ux, uy)

        #detector rows and x bands of each lenslet
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ymed = np.nanmedian(np.where(good, ym, np.nan).reshape(nl, -1), axis=1)
            yoff = np.nanmedian(np.where(good, ym - ymed[:, None, None], np.nan), axis=0)
            xlen = np.nanmedian(np.where(good, xm, np.nan), axis=0)
        has = np.isfinite(yoff) & np.isfinite(xlen)
        v = np.sort(yoff[has])
        gaps = np.nonzero(np.diff(v) > row_gap)[0]
        cuts = 0.5*(v[gaps] + v[gaps+1])
        row = np.full((ny, nx), -1)
        row[has] = np.searchsorted(cuts, yoff[has])
        nry = len(cuts) + 1
        band = np.full((ny, nx), -1)
        fx_l = np.full((ny, nx), np.nan)
        if per_slice:
            #slices: each lenslet x index is one 17-spot slice (one slicer mirror) on
            #one detector row; number the slices of each row left to right
            col_row = np.full(nx, -1); col_x = np.full(nx, np.nan)
            for i in range(nx):
                h = has[:, i]
                if h.any():
                    col_row[i] = np.bincount(row[h, i]).argmax()
                    col_x[i] = np.median(xlen[h, i])
            nregx = max([np.sum(col_row == r) for r in range(nry)] + [1])
            xcen = np.full((nry, nregx), np.nan)
            for r in range(nry):
                cols = np.nonzero(col_row == r)[0]
                cols = cols[np.argsort(col_x[cols])]
                for k, i in enumerate(cols):
                    m = has[:, i]
                    band[m, i] = k; fx_l[m, i] = k
                    xcen[r, k] = col_x[i]
                    row[m, i] = r
            if any(np.sum(col_row == r) != nregx for r in range(nry)):
                self.logger.warning(f"MedRes A&K: slices per detector row "
                                    f"{[int(np.sum(col_row == r)) for r in range(nry)]} differ")
        else:
            xcen = np.full((nry, nregx), np.nan)
            for r in range(nry):
                m = row == r
                e = np.percentile(xlen[m], np.linspace(0, 100, nregx + 1))
                b = np.clip(np.searchsorted(e, xlen[m], side='right') - 1, 0, nregx - 1)
                band[m] = b
                xcen[r] = [np.median(xlen[m][b == k]) if (b == k).any() else np.nan for k in range(nregx)]
                okc = np.isfinite(xcen[r])
                fx_l[m] = np.interp(xlen[m], xcen[r][okc], np.arange(nregx)[okc])
        fy_l = np.where(has & (band >= 0), row, np.nan).astype(float)

        #equal-size bins of about nbin masters (sizes differ by at most one), so there
        #is no short remainder bin with too few spots at the end of the scan
        bins = np.array_split(np.arange(nl), max(1, int(round(nl / nbin))))
        nb = len(bins)
        G = 2 * half * oversamp + 1
        shp = (nb, nry, nregx)
        out = {k: np.full(shp, np.nan) for k in
               ['fwhm_along', 'fwhm_across', 'slice_along', 'slice_across', 'gauss_along',
                'gauss_across', 'disp', 'R_along', 'R_across', 'resid_pct', 'ux', 'uy',
                'xdet', 'ydet', 'gauss_dx', 'gauss_dy']}
        out['nspots'] = np.zeros(shp, dtype=int)
        out['epsf'] = np.full(shp + (G, G), np.nan)
        out['lams'] = np.array([lams[b].mean() for b in bins])
        npix_ap = np.pi * 3.0**2
        S = 2 * half + 1
        yy, xx = np.mgrid[:S, :S]
        for ib, ls in enumerate(bins):
            self.logger.info(f"MedRes A&K ePSFs: wavelength bin {ib+1}/{nb} "
                             f"({lams[ls[0]]:.4f}-{lams[ls[-1]]:.4f} um, {len(ls)} masters)")
            acc = {(ry, rx): [] for ry in range(nry) for rx in range(nregx)}
            for l in ls:
                img = None
                for ry in range(nry):
                    for rx in range(nregx):
                        sel = good[l] & np.isfinite(ux[l]) & (row == ry) & (band == rx)
                        if not sel.any():
                            continue
                        if img is None:
                            img = self.load_calim(names[l], meanim)
                        stamps, cx, cy, noise, ok = self.AnK_extract_stamps(
                            img, xm[l][sel], ym[l][sel], half=half, bkg_rin=bkg_rin)
                        ap = np.hypot(xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]) <= 3.0
                        with np.errstate(invalid='ignore', divide='ignore'):
                            snr = np.nansum(np.where(ap, stamps, 0), axis=(1, 2)) / (noise * np.sqrt(npix_ap))
                        use = np.isfinite(snr) & (snr >= snr_min)
                        if not use.any():
                            continue
                        jj, ii = np.nonzero(sel)
                        jj, ii = jj[ok][use], ii[ok][use]
                        acc[ry, rx].append((stamps[use], cx[use], cy[use], np.full(len(jj), l), jj, ii))
            for (ry, rx), parts in acc.items():
                if not parts:
                    continue
                st, cx, cy, L, J, I = [np.concatenate(p) for p in zip(*parts)]
                if len(st) < min_spots:
                    continue
                use = np.arange(len(st))
                if len(use) > max_spots:
                    use = np.sort(rng.choice(use, max_spots, replace=False))
                epsf, info = self.AnK_build_epsf(st[use], cx[use], cy[use],
                                                 oversamp=oversamp, n_iter=n_iter)
                o = (ib, ry, rx)
                if gauss_centre:
                    epsf, gdx, gdy = self.AnK_gauss_recentre(epsf, oversamp)
                    out['gauss_dx'][o], out['gauss_dy'][o] = gdx, gdy
                idx = (L[use], J[use], I[use])
                u = np.array([np.nanmedian(ux[idx]), np.nanmedian(uy[idx])])
                fa, fx, sa, sx_ = self.AnK_epsf_fwhm(epsf, oversamp, *u)
                d = np.nanmedian(disp[idx])
                out['epsf'][o] = epsf
                out['fwhm_along'][o], out['fwhm_across'][o] = fa, fx
                out['slice_along'][o], out['slice_across'][o] = sa, sx_
                out['gauss_along'][o] = np.nanmedian(gal[idx])
                out['gauss_across'][o] = np.nanmedian(gac[idx])
                out['disp'][o] = d
                out['R_along'][o] = out['lams'][ib] * d / fa
                out['R_across'][o] = out['lams'][ib] * d / fx
                out['nspots'][o] = info['nstars']
                out['resid_pct'][o] = info['resid_pct']
                out['ux'][o], out['uy'][o] = u / np.hypot(*u)
                out['xdet'][o] = np.nanmedian(xm[idx])
                out['ydet'][o] = np.nanmedian(ym[idx])
        out['region_y'] = np.array([np.arange(nry), np.arange(nry) + 1]).T
        out['region_x'] = np.array([np.arange(nregx), np.arange(nregx) + 1]).T
        out['lens_fy'], out['lens_fx'] = fy_l, fx_l
        out['xcen'] = xcen
        out['regspace'] = 'slice' if per_slice else 'detector'
        out['oversamp'] = oversamp
        out['half'] = half
        self.logger.info(f"MedRes A&K ePSFs: {nry} detector rows x {nregx} "
                         f"{'slices' if per_slice else 'x bands'}, {nb} wavelength "
                         f"bins of {nbin} masters; median {np.median(out['nspots'][out['nspots'] > 0]) if (out['nspots'] > 0).any() else 0:.0f} "
                         f"spots per ePSF, {np.isfinite(out['fwhm_along']).sum()} of {out['fwhm_along'].size} built")
        if gauss_centre:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                self.logger.info(f"MedRes A&K ePSFs shifted to their Gaussian centres: median offset "
                                 f"dx {np.nanmedian(out['gauss_dx']):+.3f} px, dy {np.nanmedian(out['gauss_dy']):+.3f} px")
        return out

    def AnK_gauss_recentre(self, epsf, oversamp):
        """Shift an ePSF (cubic spline) so that the centre of a 2D Gaussian fitted to
        it is on the grid centre: the convention of the Gaussian-fit spot positions
        the ePSFs are placed at. Returns (shifted ePSF, dx, dy), dx, dy = the
        Gaussian centre's offset from the grid centre before the shift (detector px);
        the ePSF is returned unchanged (dx = dy = NaN) if the fit fails."""
        G = epsf.shape[0]; c0 = G // 2
        yy, xx = (np.mgrid[:G, :G] - c0) / oversamp
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = fitting.LevMarLSQFitter()(Gaussian2D(np.nanmax(epsf), 0, 0, 1, 1), xx, yy,
                                              np.nan_to_num(epsf))
            dx, dy = float(m.x_mean.value), float(m.y_mean.value)
        except Exception:
            return epsf, np.nan, np.nan
        if not (np.isfinite(dx) and np.isfinite(dy)) or max(abs(dx), abs(dy)) > 2.0:
            return epsf, np.nan, np.nan
        return shift(epsf, (-dy * oversamp, -dx * oversamp), order=3, mode='constant'), dx, dy

    def write_AnK_lowres_psfs(self, res, lams, scmode, output_dir, plot_dir):
        """Save AnK_lowres_psfs outputs to {scmode}_ank_epsf.fits (ePSF cube +
        per wavelength / region table) and {scmode}_ank_epsf.png, and log the
        median FWHMs and R per wavelength."""
        lams = np.asarray(lams, dtype=float)
        nl, nry, nrx = res['fwhm_along'].shape
        hdr = pyfits.Header()
        hdr['IFSMODE'] = scmode
        hdr['OVERSAMP'] = (int(res['oversamp']), 'ePSF grid points per detector px')
        hdr['STAMPHW'] = (int(res['half']), 'stamp half-width (px)')
        hdr['NREGY'] = nry; hdr['NREGX'] = nrx
        hdr['COMMENT'] = 'Anderson & King effective PSFs, axes (lam, region y, region x, gy, gx).'
        hdr['COMMENT'] = 'Grid centre = spot centre; values are px samples of a unit-flux spot.'
        hdus = [pyfits.PrimaryHDU(res['epsf'].astype(np.float32), header=hdr)]
        L, RY, RX = np.meshgrid(np.arange(nl), np.arange(nry), np.arange(nrx), indexing='ij')
        cols = [pyfits.Column(name='LAMBDA', format='D', unit='um', array=lams[L.ravel()]),
                pyfits.Column(name='REG_Y', format='J', array=RY.ravel()),
                pyfits.Column(name='REG_X', format='J', array=RX.ravel()),
                pyfits.Column(name='LENS_Y0', format='J', array=res['region_y'][RY.ravel(), 0]),
                pyfits.Column(name='LENS_Y1', format='J', array=res['region_y'][RY.ravel(), 1]),
                pyfits.Column(name='LENS_X0', format='J', array=res['region_x'][RX.ravel(), 0]),
                pyfits.Column(name='LENS_X1', format='J', array=res['region_x'][RX.ravel(), 1]),
                pyfits.Column(name='NSPOTS', format='J', array=res['nspots'].ravel())]
        for name, key, unit in [('XDET', 'xdet', 'pix'), ('YDET', 'ydet', 'pix'),
                                ('FWHM_ALONG', 'fwhm_along', 'pix'),
                                ('FWHM_ACROSS', 'fwhm_across', 'pix'),
                                ('SLICE_ALONG', 'slice_along', 'pix'),
                                ('SLICE_ACROSS', 'slice_across', 'pix'),
                                ('GAUSS_ALONG', 'gauss_along', 'pix'),
                                ('GAUSS_ACROSS', 'gauss_across', 'pix'),
                                ('DISP', 'disp', 'pix/um'), ('R_ALONG', 'R_along', ''),
                                ('R_ACROSS', 'R_across', ''), ('RESID_PCT', 'resid_pct', '%'),
                                ('UX', 'ux', ''), ('UY', 'uy', '')]:
            cols.append(pyfits.Column(name=name, format='D', unit=unit, array=res[key].ravel()))
        for name, key in [('GAUSS_DX', 'gauss_dx'), ('GAUSS_DY', 'gauss_dy')]:
            #MedRes: Gaussian-centre offset of each ePSF before it was shifted onto it
            if key in res:
                cols.append(pyfits.Column(name=name, format='D', unit='pix',
                                          array=np.asarray(res[key], float).ravel()))
        hdus.append(pyfits.BinTableHDU.from_columns(cols, name='REGIONS'))
        if res.get('lens_fy') is not None:
            #regions on the detector (MedRes): each lenslet's position on the region
            #grid, which AnK_rectmat needs to weight the region ePSFs
            rs = res.get('regspace', 'detector')
            hdr['REGSPACE'] = (rs, 'slice: one per 17-spot slice; detector: x bands'
                               if rs == 'slice' else 'regions: detector rows x detector-x bands')
            hdus[0].header['REGSPACE'] = hdr['REGSPACE']
            lh = pyfits.ImageHDU(np.array([res['lens_fy'], res['lens_fx']], dtype=float), name='LENSREG')
            lh.header['COMMENT'] = 'plane 0: region row (detector row), plane 1: fractional x band'
            lh.header['COMMENT'] = 'of each lenslet (lenslet y, lenslet x); NaN = lenslet has no spots'
            hdus.append(lh)
        pyfits.HDUList(hdus).writeto(output_dir + '/' + scmode + '_ank_epsf.fits', overwrite=True)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            med = {k: np.nanmedian(res[k].reshape(nl, -1), axis=1) for k in
                   ['fwhm_along', 'fwhm_across', 'gauss_along', 'gauss_across', 'R_along',
                    'R_across', 'resid_pct']}
        self.logger.info(f"{scmode} AnK ePSFs ({nry}x{nrx} field regions, median over regions)")
        self.logger.info("  lambda(um) | ePSF FWHM along across | Gauss FWHM along across | "
                         "R_along R_across | resid%")
        for l in range(nl):
            self.logger.info(f"  {lams[l]:9.4f} | {med['fwhm_along'][l]:9.2f} {med['fwhm_across'][l]:6.2f} "
                             f"| {med['gauss_along'][l]:10.2f} {med['gauss_across'][l]:6.2f} "
                             f"| {med['R_along'][l]:7.1f} {med['R_across'][l]:8.1f} "
                             f"| {med['resid_pct'][l]:5.1f}")

        try:
            ov, hw = int(res['oversamp']), int(res['half'])
            ext = (-hw - 0.5 / ov, hw + 0.5 / ov, -hw - 0.5 / ov, hw + 0.5 / ov)
            cy_, cx_ = nry // 2, nrx // 2
            have = np.nonzero(np.isfinite(res['fwhm_along'][:, cy_, cx_]))[0]
            if len(have) == 0:
                have = np.arange(nl)
            ls = np.unique(have[np.linspace(0, len(have) - 1, 3).astype(int)])
            fig = plt.figure(figsize=(4 * max(nrx, len(ls)) + 8, 4 * (nry + 1)))
            gs = fig.add_gridspec(nry + 1, max(nrx, len(ls)) + 2)
            for k, l in enumerate(ls):           # top row: central region at 3 wavelengths
                ax = fig.add_subplot(gs[0, k])
                e = res['epsf'][l, cy_, cx_]
                if np.isfinite(e).any():
                    ax.imshow(np.arcsinh(e / (0.02 * np.nanmax(e))), origin='lower', cmap='magma', extent=ext)
                    ax.contour(e, [0.5 * np.nanmax(e)], colors='c', extent=ext, origin='lower')
                    ax.annotate('', xy=(3 * res['ux'][l, cy_, cx_], 3 * res['uy'][l, cy_, cx_]),
                                xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='lime'))
                ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
                ax.set_title(f"centre region, {lams[l]:.3f} um\nFWHM along/across "
                             f"{res['fwhm_along'][l, cy_, cx_]:.2f}/{res['fwhm_across'][l, cy_, cx_]:.2f} px",
                             fontsize=9)
            lm = nl // 2                          # grid: every region at the middle wavelength
            for ry in range(nry):
                for rx in range(nrx):
                    ax = fig.add_subplot(gs[nry - ry, rx])
                    e = res['epsf'][lm, ry, rx]
                    if np.isfinite(e).any():
                        ax.imshow(np.arcsinh(e / (0.02 * np.nanmax(e))), origin='lower', cmap='magma', extent=ext)
                        ax.contour(e, [0.5 * np.nanmax(e)], colors='c', extent=ext, origin='lower')
                    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
                    ax.set_title(f"region ({ry},{rx}) {lams[lm]:.3f} um, N={res['nspots'][lm, ry, rx]}\n"
                                 f"along/across {res['fwhm_along'][lm, ry, rx]:.2f}/"
                                 f"{res['fwhm_across'][lm, ry, rx]:.2f} px", fontsize=9)
            ax = fig.add_subplot(gs[1:(nry + 1) // 2 + 1, -2:])
            for key, col, lab in [('fwhm_along', 'C0', 'ePSF along'), ('fwhm_across', 'C1', 'ePSF across')]:
                ax.plot(lams, med[key], '-', color=col, label=lab)
            ax.plot(lams, med['gauss_along'], '--', color='C0', label='Gaussian fit along')
            ax.plot(lams, med['gauss_across'], '--', color='C1', label='Gaussian fit across')
            ax.set(xlabel='wavelength (um)', ylabel='FWHM (px), median over regions'); ax.legend(fontsize=8)
            ax = fig.add_subplot(gs[(nry + 1) // 2 + 1:, -2:])
            ax.plot(lams, med['R_along'], '-', color='C0', label='R along (ePSF)')
            ax.plot(lams, med['R_across'], '-', color='C1', label='R across (ePSF)')
            ax.set(xlabel='wavelength (um)', ylabel='R, median over regions'); ax.legend(fontsize=8)
            fig.suptitle(f'{scmode}: Anderson & King ePSFs (asinh stretch, cyan = half max, '
                         f'green = trace direction +lambda; bottom grid laid out as on the detector)')
            fig.tight_layout()
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(os.path.join(plot_dir, scmode + '_ank_epsf.png'), dpi=100)
            plt.close(fig)
        except Exception as e:
            self.logger.info(f"AnK ePSF plot failed: {e}")

    def read_AnK_lowres_psfs(self, path):
        """Read a {scmode}_ank_epsf.fits written by write_AnK_lowres_psfs back
        into the parts of the AnK_lowres_psfs dict that AnK_rectmat needs.
        Returns (res, lams)."""
        with pyfits.open(path) as hdul:
            epsf = np.array(hdul[0].data, dtype=float)
            hdr = hdul[0].header
            t = hdul['REGIONS'].data
            nl, nry, nrx = epsf.shape[:3]
            lam_col = np.asarray(t['LAMBDA'], float).reshape(nl, nry, nrx)
            ry0 = np.asarray(t['LENS_Y0']).reshape(nl, nry, nrx)[0, :, 0]
            ry1 = np.asarray(t['LENS_Y1']).reshape(nl, nry, nrx)[0, :, 0]
            rx0 = np.asarray(t['LENS_X0']).reshape(nl, nry, nrx)[0, 0, :]
            rx1 = np.asarray(t['LENS_X1']).reshape(nl, nry, nrx)[0, 0, :]
            # per wavelength / region measurements (FWHMs, dispersion, ...), e.g. for
            # estimate_spectral_resolution_AnK
            tabcols = {name.lower(): np.asarray(t[name]).reshape(nl, nry, nrx)
                       for name in t.columns.names
                       if name not in ('LAMBDA', 'REG_Y', 'REG_X', 'LENS_Y0', 'LENS_Y1',
                                       'LENS_X0', 'LENS_X1')}
            lensreg = np.array(hdul['LENSREG'].data, dtype=float) if 'LENSREG' in hdul else None
        res = dict(epsf=epsf, oversamp=int(hdr['OVERSAMP']), half=int(hdr['STAMPHW']),
                   region_y=np.array([ry0, ry1]).T, region_x=np.array([rx0, rx1]).T)
        if lensreg is not None:
            res['lens_fy'], res['lens_fx'] = lensreg[0], lensreg[1]
            res['regspace'] = str(hdr.get('REGSPACE', 'detector'))
        for k, v in tabcols.items():
            res[{'r_along': 'R_along', 'r_across': 'R_across'}.get(k, k)] = v
        return res, lam_col[:, 0, 0]

    @staticmethod
    def AnK_fill_missing(ep, lams_ank):
        """Fill the ePSFs (nla, nry, nrx, G, G) of regions without one at some wavelength
        with that region's ePSF at the nearest wavelength that has one. Returns the
        filled ePSFs (NaN -> 0) and reg_ok (nry, nrx): regions with an ePSF at any
        wavelength."""
        ep = np.array(ep, dtype=float)
        lams_ank = np.asarray(lams_ank, float)
        nla, nry, nrx = ep.shape[:3]
        have = np.isfinite(ep).all(axis=(3, 4)) & (np.nan_to_num(ep).max(axis=(3, 4)) > 0)
        for ry in range(nry):
            for rx in range(nrx):
                ok = np.nonzero(have[:, ry, rx])[0]
                for l in np.nonzero(~have[:, ry, rx])[0]:
                    if len(ok):
                        ep[l, ry, rx] = ep[ok[np.argmin(np.abs(lams_ank[ok]-lams_ank[l]))], ry, rx]
        return np.nan_to_num(ep), have.any(axis=0)

    def AnK_field_weights(self, ank, ny, nx, reg_ok=None):
        """
        Bilinear field weights (ny*nx, nry*nrx) of each lenslet on the A&K region
        grid, normalised to unit sum per lenslet (all zero for a lenslet with no
        region). The lenslet's fractional position on the grid is given for MedRes
        (lens_fy, lens_fx: detector rows x detector-x bands, AnK_medres_psfs);
        otherwise (LowRes, regions are bands of the lenslet grid) it is its lenslet
        index, linear between the region centres. Outside the outermost centres the
        edge regions are used as they are. reg_ok (nry, nrx) drops regions that
        have no ePSF at all.
        """
        nry, nrx = np.asarray(ank['epsf']).shape[1:3]
        if reg_ok is None:
            reg_ok = np.ones((nry, nrx), bool)
        def axis_weights(f, nreg):
            fin = np.isfinite(f)
            f = np.where(fin, f, 0.0)
            i1 = np.clip(np.floor(f).astype(int), 0, nreg-1)
            i2 = np.minimum(i1+1, nreg-1)
            t = f - i1
            return i1, i2, t, fin
        if ank.get('lens_fy') is not None:
            FY = np.asarray(ank['lens_fy'], float); FX = np.asarray(ank['lens_fx'], float)
        else:
            def index_pos(n, bounds, nreg):
                cen = 0.5*(bounds[:, 0] + bounds[:, 1] - 1)
                return np.interp(np.arange(n), cen, np.arange(nreg))
            FY = np.broadcast_to(index_pos(ny, np.asarray(ank['region_y']), nry)[:, None], (ny, nx))
            FX = np.broadcast_to(index_pos(nx, np.asarray(ank['region_x']), nrx)[None, :], (ny, nx))
        jy1, jy2, ty, fin_y = axis_weights(FY, nry)
        ix1, ix2, tx, fin_x = axis_weights(FX, nrx)
        W = np.zeros((ny, nx, nry, nrx))
        J, I = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        fin = fin_y & fin_x
        for (ja, wy) in [(jy1, 1-ty), (jy2, ty)]:
            for (ia, wx) in [(ix1, 1-tx), (ix2, tx)]:
                np.add.at(W, (J, I, ja, ia), np.where(fin, wy*wx, 0.0))
        W *= reg_ok[None, None]
        with np.errstate(invalid='ignore', divide='ignore'):
            W /= W.sum(axis=(2, 3), keepdims=True)
        return np.nan_to_num(W).reshape(ny*nx, nry*nrx)

    def AnK_rectmat(self, ank, lams_ank, lams_out, fitarr, lams_fit, nsub=10, crop_half=4,
                    min_frac=1e-3, shift=(0.0, 0.0), npix=(2048, 2048)):
        """
        Optimal-extraction rectmat from the regional Anderson & King ePSFs,
        following the CHARIS pipeline's polychrome templates (Brandt et al. 2017;
        charis-dep buildmonochrome.make_polychrome):

          - field: each lenslet's ePSF is the bilinear blend of the four nearest
            field-region ePSFs (regions are bands of the lenslet grid, so the blend
            is done in lenslet-index space; outside the outermost region centres
            the edge regions are used as they are, without extrapolation);
          - wavelength: linear blend of the ePSFs at the two monochromator
            wavelengths around lam (the nearer one weighted more; outside the
            scanned range the end ePSF is used);
          - position: the lenslet's spot centre at lam from the cubic-in-lambda fit
            to its Gaussian spot centres (interp_gauss_spots), and the blended
            ePSF is sampled there with cubic-spline interpolation;
          - channel: each output wavelength is a channel from halfway to the
            previous to halfway to the next one; its template is the average of
            the ePSF placed at nsub wavelengths evenly spread across it (CHARIS
            uses 10; nsub=1 gives monochromatic templates at the channel centres,
            like the Gaussian OPT_intp rectmat).

        Each template is cut to a (2*crop_half+1)^2 box around the channel-centre
        position, widened by the spot motion across the channel, negative values
        and values below min_frac of its peak set to 0, and normalised to unit
        sum, i.e. the same profile rows as gen_QL_rectmat_interpd (rows
        lensx + lensy*nx + l*nx*ny, columns = flattened detector pixels), for
        scbasic.optimal_extract_horne.

        Args:
            ank: dict from AnK_lowres_psfs (or read_AnK_lowres_psfs) with epsf
                (nla, nry, nrx, G, G), oversamp, region_y, region_x
            lams_ank: (nla,) wavelengths of the ePSFs (um)
            lams_out: (nlam,) channel-centre wavelengths of the rectmat (um)
            fitarr: (nl, ny, nx, >=6) Gaussian spot fits at lams_fit
            lams_fit: (nl,) wavelengths of fitarr (um)
            shift: (dx, dy) px added to every spot position (disperser motion)

        Returns (rmat, info) with info: positions (nlam, ny, nx, 2) of the channel
        centres, nonzeros per template (nlam, ny, nx), box size.
        """
        from scipy.ndimage import spline_filter
        lams_ank = np.asarray(lams_ank, float)
        lams_out = np.asarray(lams_out, float)
        ep = np.array(ank['epsf'], dtype=float)
        nla, nry, nrx, G, _ = ep.shape
        ov = int(ank['oversamp']); c0 = G // 2
        ny, nx = fitarr.shape[1:3]
        nlam = len(lams_out)

        #regions without an ePSF at some wavelength: use the nearest wavelength that has one
        ep, reg_ok = self.AnK_fill_missing(ep, lams_ank)
        #cubic-spline coefficients (linear in the ePSF, so blends of them are the
        #coefficients of the blended ePSF, as CHARIS does with spline_filter)
        coef = np.array([[[spline_filter(ep[l, ry, rx], order=3) for rx in range(nrx)]
                          for ry in range(nry)] for l in range(nla)])

        #bilinear field weights of each lenslet on the region grid (AnK_field_weights)
        W = self.AnK_field_weights(ank, ny, nx, reg_ok)

        #channel edges and the sub-wavelengths in each channel
        if nlam > 1:
            mid = 0.5*(lams_out[1:] + lams_out[:-1])
            edges = np.concatenate([[lams_out[0]-(mid[0]-lams_out[0])], mid,
                                    [lams_out[-1]+(lams_out[-1]-mid[-1])]])
        else:
            edges = np.array([lams_out[0], lams_out[0]])
        if nsub > 1:
            frac = (np.arange(nsub)+0.5)/nsub
            lsub = edges[:-1, None] + frac[None]*(edges[1:]-edges[:-1])[:, None]
        else:
            lsub = lams_out[:, None]
        #spot centres at every sub-wavelength and at the channel centres
        allw = np.concatenate([lams_out, lsub.ravel()])
        pos = self.interp_gauss_spots(np.asarray(lams_fit, float), allw, fitarr)[..., 1:3]
        pos_c = pos[:nlam]                                   #(nlam, ny, nx, 2)
        pos_s = pos[nlam:].reshape(nlam, lsub.shape[1], ny, nx, 2)

        #optional detector shift (as rectmat_xshift / rectmat_yshift)
        pos_c = pos_c + np.asarray(shift, float)
        pos_s = pos_s + np.asarray(shift, float)

        #box half-size: crop_half plus half the spot motion across a channel (99.9th
        #percentile: a few edge lenslets with poor cubic position fits move much more)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            motion = np.nanpercentile(np.hypot(*(pos_s[:, -1]-pos_s[:, 0]).transpose(3, 0, 1, 2)), 99.9)
        h = crop_half + (int(np.ceil(0.5*motion)) if np.isfinite(motion) else 0)
        S = 2*h + 1
        off = np.arange(-h, h+1)

        rows, cols, vals = [], [], []
        nnz = np.zeros((nlam, ny, nx), dtype=int)
        lens_idx = np.arange(ny*nx)
        for c in range(nlam):
            xc, yc = pos_c[c, ..., 0].ravel(), pos_c[c, ..., 1].ravel()
            ok = np.isfinite(xc) & np.isfinite(yc) & (W.sum(1) > 0)
            ok &= (xc > -h) & (xc < npix[1]-1+h) & (yc > -h) & (yc < npix[0]-1+h)
            if not ok.any():
                continue
            li = lens_idx[ok]
            X = np.round(xc[ok]).astype(int)[:, None, None] + off[None, None, :]
            Y = np.round(yc[ok]).astype(int)[:, None, None] + off[None, :, None]
            X, Y = np.broadcast_arrays(X, Y)
            T = np.zeros(X.shape)
            for s in range(lsub.shape[1]):
                lam = lsub[c, s]
                if lam <= lams_ank[0]:
                    C = coef[0]
                elif lam >= lams_ank[-1]:
                    C = coef[-1]
                else:
                    k = np.searchsorted(lams_ank, lam) - 1
                    a = (lams_ank[k+1]-lam)/(lams_ank[k+1]-lams_ank[k])
                    C = a*coef[k] + (1-a)*coef[k+1]
                xs = pos_s[c, s, ..., 0].ravel()[ok]; ys = pos_s[c, s, ..., 1].ravel()[ok]
                good = np.isfinite(xs) & np.isfinite(ys)
                xs = np.where(good, xs, xc[ok]); ys = np.where(good, ys, yc[ok])
                gx = (X - xs[:, None, None])*ov + c0
                gy = (Y - ys[:, None, None])*ov + c0
                sub = np.zeros(X.shape)
                for r in range(nry*nrx):
                    w = W[li, r]
                    m = w > 0
                    if not m.any():
                        continue
                    smp = map_coordinates(C[r // nrx, r % nrx], [gy[m].ravel(), gx[m].ravel()],
                                          order=3, prefilter=False, mode='constant', cval=0.0)
                    sub[m] += w[m, None, None]*smp.reshape(-1, S, S)
                T += np.clip(sub, 0, None)
            ondet = (X >= 0) & (X < npix[1]) & (Y >= 0) & (Y < npix[0])
            T = np.where(ondet, T, 0.0)
            pk = T.max(axis=(1, 2), keepdims=True)
            T = np.where(T >= min_frac*pk, T, 0.0)
            tot = T.sum(axis=(1, 2), keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                T = np.where(tot > 0, T/tot, 0.0)
            keep = T > 0
            n_per = keep.sum(axis=(1, 2))
            nnz[c].ravel()[li] = n_per
            rows.append(np.repeat(li + c*ny*nx, n_per))
            cols.append(np.ravel_multi_index((Y[keep], X[keep]), npix))
            vals.append(T[keep])
            self.logger.info(f"AnK rectmat channel {c+1}/{nlam} ({lams_out[c]:.4f} um): "
                             f"{len(li)} lenslets, {n_per.sum()} nonzeros")
        rows = np.concatenate(rows) if rows else np.zeros(0, int)
        cols = np.concatenate(cols) if cols else np.zeros(0, int)
        vals = np.concatenate(vals) if vals else np.zeros(0)
        rmat = sparse.csr_matrix((vals, (rows, cols)), shape=(nlam*ny*nx, npix[0]*npix[1]))
        return rmat, dict(positions=pos_c, nnz=nnz, box=S, nsub=lsub.shape[1])

    def fit_gauss_spots_2(self, calims, posarr, show_plots=False, fix_theta=False, cropsize=10):
        """
        Modified version of the original fit_gauss_spots.

        Changes from the original:
          1. Background is now a free parameter in the fit itself
             (Gaussian2D + Const2D), rather than commented out / pre-subtracted.
          2. Box edges (xs, xe, ys, ye) are rounded to the nearest integer
             instead of truncated with int(), removing a systematic sub-pixel
             bias toward the origin.
          3. The initial guess for the centroid uses the predicted position
             (xc, yc) from posarr, expressed relative to the box, instead of
             assuming the spot sits at the exact geometric center of the box.
          4. Fit convergence is checked via fitter.fit_info and flagged if
             the fit did not succeed.
          5. Optional fix_theta=True fixes the rotation angle at 0, useful if
             spots are close to circular and theta is poorly constrained.
          6. 1-sigma parameter uncertainties are now extracted from the fit's
             covariance matrix and stored alongside each fitted value.

        Returns
        -------
        fitarr : ndarray, shape (n_l, n_y, n_x, 14)
            [amplitude, x_mean(abs), y_mean(abs), x_stddev, y_stddev, theta, background,
             amplitude_err, x_mean_err, y_mean_err, x_stddev_err, y_stddev_err,
             theta_err, background_err]
            Uncertainties are the formal 1-sigma errors from the LM covariance
            matrix (sqrt of the diagonal). They'll be NaN if the fit didn't
            converge or the covariance matrix was singular, and 0.0 for theta
            when fix_theta=True (it's fixed, not fitted). Note these are formal
            least-squares errors only -- they assume Gaussian noise and don't
            capture systematics (e.g. the profile-shape mismatch we've been
            discussing), so treat them as a lower bound on the true uncertainty.
        modims : list of ndarray
            Model images (Gaussian component only, background NOT added back in,
            so these remain directly comparable to the original modims).
        resims : ndarray
            calims - modims
        """
        fitarr = np.full((posarr.shape[0], posarr.shape[1], posarr.shape[2], 14), np.nan)
        spotim_arr = np.full([posarr.shape[0],posarr.shape[1],posarr.shape[2],cropsize,cropsize],np.nan)
        modim_arr = np.full([posarr.shape[0],posarr.shape[1],posarr.shape[2],cropsize,cropsize],np.nan)
        resim_arr = np.full([posarr.shape[0],posarr.shape[1],posarr.shape[2],cropsize,cropsize],np.nan)
        modims = []

        for ll in range(len(calims)):
            modim = np.zeros(calims[ll].shape)
            for lensx in range(posarr.shape[2]):
                for lensy in range(posarr.shape[1]):
                    xc, xs, xe, yc, ys, ye, intens = posarr[ll, lensy, lensx]
                    if True in np.isnan([xs, xe, ys, ye]):
                        continue

                    # round instead of truncate -> removes systematic sub-pixel
                    # bias in where the box actually sits
                    #xs_i = int(np.round(xs))
                    #xe_i = int(np.round(xe))
                    #ys_i = int(np.round(ys))
                    #ye_i = int(np.round(ye))

                    xs_i = int(xs)
                    xe_i = int(xe)
                    ys_i = int(ys)
                    ye_i = int(ye)

                    cropped = calims[ll, ys_i:ye_i, xs_i:xe_i].copy()
                    if cropped.size == 0:
                        continue
                    spotim_arr[ll,lensy,lensx,:ye_i-ys_i,:xe_i-xs_i] = cropped

                    y, x = np.mgrid[:(ye_i - ys_i), :(xe_i - xs_i)]

                    # initial guess: use the predicted center (xc, yc) relative
                    # to this box, instead of assuming box-center
                    x0_guess = xc - xs_i
                    y0_guess = yc - ys_i
                    bkg_guess = np.median(cropped)

                    gauss_init = Gaussian2D(
                        amplitude=intens, x_mean=x0_guess, y_mean=y0_guess,
                        x_stddev=1., y_stddev=1.
                    )
                    if fix_theta:
                        gauss_init.theta.fixed = True

                    const_init = Const2D(amplitude=bkg_guess)
                    initial_guess = gauss_init + const_init

                    fitter = fitting.LevMarLSQFitter()
                    fitted_model = fitter(initial_guess, x, y, cropped, maxiter=1000)

                    fit_ok = True
                    if fitter.fit_info.get('ierr', 1) not in (1, 2, 3, 4):
                        fit_ok = False

                    gauss_part = fitted_model[0]
                    bkg_part = fitted_model[1]

                    # --- parameter uncertainties from the fit covariance matrix ---
                    # param_cov only covers the *free* (non-fixed) parameters, in
                    # the order they appear in fitted_model.param_names, so build
                    # the name->error mapping from that subset rather than assuming
                    # a fixed column layout.
                    free_names = [name for name in fitted_model.param_names
                                  if not fitted_model.fixed[name]]
                    cov = fitter.fit_info.get('param_cov')
                    if cov is not None and cov.shape[0] == len(free_names):
                        errs = np.sqrt(np.diag(cov))
                        err_dict = dict(zip(free_names, errs))
                    else:
                        err_dict = {name: np.nan for name in free_names}

                    amp_err = err_dict.get('amplitude_0', np.nan)
                    xmean_err = err_dict.get('x_mean_0', np.nan)
                    ymean_err = err_dict.get('y_mean_0', np.nan)
                    xstd_err = err_dict.get('x_stddev_0', np.nan)
                    ystd_err = err_dict.get('y_stddev_0', np.nan)
                    # theta_0 won't be in err_dict at all when fix_theta=True
                    # (it's fixed, so there's no uncertainty to report -> 0.0)
                    theta_err = err_dict.get('theta_0', 0.0 if fix_theta else np.nan)
                    bkg_err = err_dict.get('amplitude_1', np.nan)

                    # store only the Gaussian component in modim, so modims stays
                    # directly comparable to the original (Gaussian-only) model
                    modim[ys_i:ye_i, xs_i:xe_i] += gauss_part(x, y)

                    fitarr[ll, lensy, lensx] = [
                        gauss_part.amplitude.value,
                        gauss_part.x_mean.value + xs_i,
                        gauss_part.y_mean.value + ys_i,
                        gauss_part.x_stddev.value,
                        gauss_part.y_stddev.value,
                        gauss_part.theta.value,
                        bkg_part.amplitude.value,
                        amp_err,
                        xmean_err,
                        ymean_err,
                        xstd_err,
                        ystd_err,
                        theta_err,
                        bkg_err,
                    ]

                    modim_arr[ll,lensy,lensx,:ye_i-ys_i,:xe_i-xs_i] = gauss_part(x,y)
                    resim_arr[ll,lensy,lensx,:ye_i-ys_i,:xe_i-xs_i] = cropped - fitted_model(x,y)


                    if not fit_ok:
                        self.logger.info(f"  [warn] fit did not converge cleanly at "
                              f"lensx={lensx}, lensy={lensy}, ll={ll} "
                              f"(ierr={fitter.fit_info.get('ierr')})")

                    if show_plots:
                        self.logger.info("--- Fit Results (value +/- 1-sigma) ---")
                        self.logger.info(f"Amplitude: {gauss_part.amplitude.value:.2f} +/- {amp_err:.2f}")
                        self.logger.info(f"X Center:  {gauss_part.x_mean.value:.2f} +/- {xmean_err:.2f}")
                        self.logger.info(f"Y Center:  {gauss_part.y_mean.value:.2f} +/- {ymean_err:.2f}")
                        self.logger.info(f"X Sigma:   {gauss_part.x_stddev.value:.2f} +/- {xstd_err:.2f}")
                        self.logger.info(f"Y Sigma:   {gauss_part.y_stddev.value:.2f} +/- {ystd_err:.2f}")
                        self.logger.info(f"Theta:     {gauss_part.theta.value:.2f} +/- {theta_err:.2f} rad")
                        self.logger.info(f"Background:{bkg_part.amplitude.value:.4f} +/- {bkg_err:.4f}")

                        full_model = fitted_model(x, y)
                        plt.figure(figsize=(12, 4), clear=True)
                        plt.subplot(1, 3, 1)
                        plt.title("Original Data")
                        plt.imshow(cropped, origin='lower', cmap='viridis')
                        plt.colorbar()

                        plt.subplot(1, 3, 2)
                        plt.title("Fitted Model (Gaussian + bkg)")
                        plt.imshow(full_model, origin='lower', cmap='viridis')
                        plt.colorbar()

                        plt.subplot(1, 3, 3)
                        plt.title("Residuals (Data - Model)")
                        plt.imshow(cropped - full_model, origin='lower', cmap='bwr')
                        plt.colorbar()
                        plt.tight_layout()
                        plt.show()

            modims.append(modim)

        resims = calims - np.array(modims)
        return fitarr,modims,resims,spotim_arr,modim_arr,resim_arr






    def fit_gauss_spots_names(self,names,meanim,posarr,cropsize=10,show_plots=False,cut=0.1,full_images=True):
        """
        Gaussian fit of every spot. full_images=False skips the full-detector model
        image per wavelength and the residual stack (modims, resims, returned as
        None): they are only written as debug files, and for long scans they are
        the largest arrays in the run (two float64 2048 x 2048 images per master).
        The fits and the cropped spot/model/residual arrays are unchanged.
        """
        fitarr = np.zeros([posarr.shape[0],posarr.shape[1],posarr.shape[2],6])
        fitarr[:,:,:,:] = np.nan
        modims = []
        spotim_arr = np.zeros([posarr.shape[0],posarr.shape[1],posarr.shape[2],cropsize,cropsize])
        modim_arr = np.zeros([posarr.shape[0],posarr.shape[1],posarr.shape[2],cropsize,cropsize])
        resim_arr = np.zeros([posarr.shape[0],posarr.shape[1],posarr.shape[2],cropsize,cropsize])
        resims = []
        for ll in range(len(names)):
            self.logger.info("image "+str(ll)+" of "+str(len(names)))
            fname = names[ll]
            image = pyfits.getdata(fname,memmap=False)
            im_cal_tmp = self.monochrom_bksub_sing(image,meanim)
            if self.config.instrument.apply_destripe==True:
                im_cal = self.masked_row_destripe(im_cal_tmp)[0]
            else:
                im_cal = im_cal_tmp
            modim = np.zeros(im_cal.shape) if full_images else None
            for lensx in range(posarr.shape[2]):
                for lensy in range(posarr.shape[1]):
                    xc,xs,xe,yc,ys,ye,intens = posarr[ll,lensy,lensx]
                    if True not in np.isnan([xs,xe,ys,ye]):
                        xc = int(xc)
                        yc = int(yc)
                        xs = int(xs)
                        xe = int(xe)
                        ys = int(ys)
                        ye = int(ye)
                        cropped = np.zeros(im_cal[ys:ye,xs:xe].shape)
                        cropped[:] = im_cal[ys:ye,xs:xe]
                        cropped[np.where(cropped<cut*np.max(cropped))] = 0.0
                        spotim_arr[ll,lensy,lensx,:ye-ys,:xe-xs] = cropped
                        initial_guess = Gaussian2D(amplitude=intens, x_mean=(xe-xs)*0.5, y_mean=(ye-ys)*0.5,
                                  x_stddev=1., y_stddev=1.)

                        fitter = fitting.LevMarLSQFitter()
                        y, x = np.mgrid[:(ye-ys), :(xe-xs)]
                        fitted_model = fitter(initial_guess, x, y, cropped)

                        if full_images:
                            modim[ys:ye,xs:xe]+=fitted_model(x,y)
                        modim_arr[ll,lensy,lensx,:ye-ys,:xe-xs] = fitted_model(x,y)
                        resim_arr[ll,lensy,lensx,:ye-ys,:xe-xs] = cropped - fitted_model(x,y)
                        fitarr[ll,lensy,lensx] = [fitted_model.amplitude.value,
                                                  fitted_model.x_mean.value+xs,
                                                  fitted_model.y_mean.value+ys,
                                                  fitted_model.x_stddev.value,
                                                  fitted_model.y_stddev.value,
                                                  fitted_model.theta.value]

                        if True in np.isnan(fitarr[ll,lensy,lensx]):
                            self.logger.info('nans in '+str(ll)+' '+str(lensy)+' '+str(lensx))
                        if show_plots==True:
                            self.logger.info("--- Fit Results ---")
                            self.logger.info(f"Amplitude: {fitted_model.amplitude.value:.2f}")
                            self.logger.info(f"X Center:  {fitted_model.x_mean.value:.2f}")
                            self.logger.info(f"Y Center:  {fitted_model.y_mean.value:.2f}")
                            self.logger.info(f"X Sigma:   {fitted_model.x_stddev.value:.2f}")
                            self.logger.info(f"Y Sigma:   {fitted_model.y_stddev.value:.2f}")
                            self.logger.info(f"Theta:     {fitted_model.theta.value:.2f} rad")

                            plt.figure(figsize=(12, 4),clear=True)
                            plt.subplot(1, 3, 1)
                            plt.title("Original Noisy Data")
                            plt.imshow(cropped, origin='lower', cmap='viridis')
                            plt.colorbar()

                            plt.subplot(1, 3, 2)
                            plt.title("Fitted Model Image")
                            plt.imshow(fitted_model(x, y), origin='lower', cmap='viridis')
                            plt.colorbar()

                            plt.subplot(1, 3, 3)
                            plt.title("Residuals (Data - Model)")
                            plt.imshow(cropped - fitted_model(x, y), origin='lower', cmap='bwr')
                            plt.colorbar()
                            plt.tight_layout()
                            plt.show()
                            stop
            if full_images:
                #residual per image here, since the images are not kept
                modims.append(modim)
                resims.append(im_cal-modim)

        if full_images:
            modims, resims = np.array(modims), np.array(resims)
        else:
            modims, resims = None, None
        return fitarr,modims,resims,spotim_arr,modim_arr,resim_arr


    def fit_gauss_spots(self,calims,posarr,cropsize=10,show_plots=False,cut=0.1,full_images=True):
        """
        Gaussian fit of every spot. full_images=False skips the full-detector model
        image per wavelength and the residual stack (modims, resims, returned as
        None): they are only written as debug files, and for long scans they are
        the largest arrays in the run (two float64 2048 x 2048 images per master).
        The fits and the cropped spot/model/residual arrays are unchanged.
        """
        fitarr = np.zeros([posarr.shape[0],posarr.shape[1],posarr.shape[2],6])
        fitarr[:,:,:,:] = np.nan
        modims = []
        spotim_arr = np.zeros([posarr.shape[0],posarr.shape[1],posarr.shape[2],cropsize,cropsize])
        modim_arr = np.zeros([posarr.shape[0],posarr.shape[1],posarr.shape[2],cropsize,cropsize])
        resim_arr = np.zeros([posarr.shape[0],posarr.shape[1],posarr.shape[2],cropsize,cropsize])
        for ll in range(len(calims)):
            self.logger.info("image "+str(ll)+" of "+str(len(calims)))
            modim = np.zeros(calims[ll].shape) if full_images else None
            for lensx in range(posarr.shape[2]):
                        for lensy in range(posarr.shape[1]):
                            xc,xs,xe,yc,ys,ye,intens = posarr[ll,lensy,lensx]
                            if True not in np.isnan([xs,xe,ys,ye]):
                                xc = int(xc)
                                yc = int(yc)
                                xs = int(xs)
                                xe = int(xe)
                                ys = int(ys)
                                ye = int(ye)
                                cropped = np.zeros(calims[ll,ys:ye,xs:xe].shape)
                                cropped[:] = calims[ll,ys:ye,xs:xe]
                                cropped[np.where(cropped<cut*np.max(cropped))] = 0.0
                                spotim_arr[ll,lensy,lensx,:ye-ys,:xe-xs] = cropped
                                initial_guess = Gaussian2D(amplitude=intens, x_mean=(xe-xs)*0.5, y_mean=(ye-ys)*0.5,
                                          x_stddev=1., y_stddev=1.)

                                fitter = fitting.LevMarLSQFitter()
                                y, x = np.mgrid[:(ye-ys), :(xe-xs)]
                                fitted_model = fitter(initial_guess, x, y, cropped)

                                if full_images:
                                    modim[ys:ye,xs:xe]+=fitted_model(x,y)
                                modim_arr[ll,lensy,lensx,:ye-ys,:xe-xs] = fitted_model(x,y)
                                resim_arr[ll,lensy,lensx,:ye-ys,:xe-xs] = cropped - fitted_model(x,y)
                                fitarr[ll,lensy,lensx] = [fitted_model.amplitude.value,
                                                          fitted_model.x_mean.value+xs,
                                                          fitted_model.y_mean.value+ys,
                                                          fitted_model.x_stddev.value,
                                                          fitted_model.y_stddev.value,
                                                          fitted_model.theta.value]

                                if True in np.isnan(fitarr[ll,lensy,lensx]):
                                    self.logger.info('nans in '+str(ll)+' '+str(lensy)+' '+str(lensx))
                                if show_plots==True:
                                    self.logger.info("--- Fit Results ---")
                                    self.logger.info(f"Amplitude: {fitted_model.amplitude.value:.2f}")
                                    self.logger.info(f"X Center:  {fitted_model.x_mean.value:.2f}")
                                    self.logger.info(f"Y Center:  {fitted_model.y_mean.value:.2f}")
                                    self.logger.info(f"X Sigma:   {fitted_model.x_stddev.value:.2f}")
                                    self.logger.info(f"Y Sigma:   {fitted_model.y_stddev.value:.2f}")
                                    self.logger.info(f"Theta:     {fitted_model.theta.value:.2f} rad")

                                    plt.figure(figsize=(12, 4),clear=True)
                                    plt.subplot(1, 3, 1)
                                    plt.title("Original Noisy Data")
                                    plt.imshow(cropped, origin='lower', cmap='viridis')
                                    plt.colorbar()

                                    plt.subplot(1, 3, 2)
                                    plt.title("Fitted Model Image")
                                    plt.imshow(fitted_model(x, y), origin='lower', cmap='viridis')
                                    plt.colorbar()

                                    plt.subplot(1, 3, 3)
                                    plt.title("Residuals (Data - Model)")
                                    plt.imshow(cropped - fitted_model(x, y), origin='lower', cmap='bwr')
                                    plt.colorbar()
                                    plt.tight_layout()
                                    plt.show()
                                    stop
            if full_images:
                modims.append(modim)

        if full_images:
            resims = calims-modims
        else:
            modims, resims = None, None
        return fitarr,modims,resims,spotim_arr,modim_arr,resim_arr


    def interp_gauss_spots(self,lams_in,lams_des,fitarr,show_plots=False,method='poly'):
        interp_arr = np.zeros([len(lams_des),fitarr.shape[1],fitarr.shape[2],fitarr.shape[3]])
        interp_arr[:,:,:,:] = np.nan
        for lensy in range(fitarr.shape[1]):
            for lensx in range(fitarr.shape[2]):
                gausspars = fitarr[:,lensy,lensx,:6]#A,xm,ym,xstd,ystd,theta
                #if False in np.isnan(gausspars):
                if len(np.where(np.isnan(gausspars[:,0])==False)[0]) > 0.9*len(lams_in):
                #if True not in np.isnan(gausspars):
                    for i in range(len(gausspars[0])):
                        #fint = LinearNDInterpolator(lams_in,gausspars[:,i])
                        tofit = gausspars[:,i]
                        lamsfit = lams_in[np.where(np.isnan(tofit)==False)]
                        tofit = tofit[np.where(np.isnan(tofit)==False)]
                        if method=='poly':
                            res=np.polyfit(lamsfit,tofit,3)
                            fint = np.polynomial.polynomial.Polynomial(res[::-1])
                        if method=='interp':
                            fint = interp1d(lamsfit,tofit)
                        gausspars_new = fint(lams_des)
                        if show_plots==True:
                            f = plt.figure(clear=True)
                            plt.scatter(lams_des,gausspars_new)
                            plt.scatter(lams_in,gausspars[:,i])
                            plt.plot(lams_des,gausspars_new)
                            plt.show()
                        interp_arr[:,lensy,lensx,i] = gausspars_new


        return interp_arr


    def trace_derivatives(self, fitarr, lams, poly_deg=3, min_spots=5):
        """
        Local trace direction and dispersion from the Gaussian spot fits.

        Fits x(lambda) and y(lambda) of each lenslet's fitted spot centres with
        polynomials and differentiates them. Derivatives are only evaluated
        inside the wavelength range that lenslet's good spots cover.

        Returns:
            dxdl, dydl: (nlam, ny, nx) dx/dlambda, dy/dlambda in px/um
            good: (nlam, ny, nx) bool, spots with a usable Gaussian fit
        """
        lams = np.asarray(lams, dtype=float)
        nl, ny, nx = fitarr.shape[:3]
        amp, xm, ym, sx, sy, th = [fitarr[..., i] for i in range(6)]
        good = (np.isfinite(amp) & (amp > 0) & np.isfinite(xm) & np.isfinite(ym)
                & np.isfinite(sx) & np.isfinite(sy) & np.isfinite(th))
        dxdl = np.full((nl, ny, nx), np.nan)
        dydl = np.full((nl, ny, nx), np.nan)
        for j in range(ny):
            for i in range(nx):
                g = good[:, j, i]
                if g.sum() < max(min_spots, poly_deg + 2):
                    continue
                px = np.polyfit(lams[g], xm[g, j, i], poly_deg)
                py = np.polyfit(lams[g], ym[g, j, i], poly_deg)
                inr = (lams >= lams[g].min()) & (lams <= lams[g].max())
                dxdl[inr, j, i] = np.polyval(np.polyder(px), lams[inr])
                dydl[inr, j, i] = np.polyval(np.polyder(py), lams[inr])
        return dxdl, dydl, good

    @staticmethod
    def gauss_fwhm_along_across(sx, sy, th, ux, uy):
        """
        FWHM of a 2D Gaussian (astropy Gaussian2D x_stddev, y_stddev, theta)
        projected onto the unit vector (ux, uy) and onto the perpendicular,
        i.e. the FWHMs of its marginal profiles along and across that direction.
        """
        c, s = np.cos(th), np.sin(th)
        sxx = c**2 * sx**2 + s**2 * sy**2
        syy = s**2 * sx**2 + c**2 * sy**2
        sxy = c * s * (sx**2 - sy**2)
        k = 2.0 * np.sqrt(2.0 * np.log(2.0))
        with np.errstate(invalid='ignore'):
            along = k * np.sqrt(ux**2 * sxx + 2 * ux * uy * sxy + uy**2 * syy)
            across = k * np.sqrt(uy**2 * sxx - 2 * ux * uy * sxy + ux**2 * syy)
        return along, across

    def estimate_spectral_resolution(self, fitarr, lams, mono_fwhm_nm=0.0,
                                     poly_deg=3, min_spots=5,
                                     fwhm_pix_range=(0.5, 8.0)):
        """
        Estimate the spectral resolution R = lambda / FWHM_lambda of every
        lenslet at every monochromator wavelength, from the Gaussian fits to
        the lenslet spots (fit_gauss_spots).

        For each spot, the fitted 2D Gaussian covariance is projected onto the
        local dispersion direction, giving the FWHM of the line spread
        function in detector pixels along the trace. The local dispersion
        (px/um) and its direction come from the derivative of a polynomial
        fit to that lenslet's spot positions vs wavelength. Then
            FWHM_lambda = FWHM_along_trace / (ds/dlambda),  R = lambda / FWHM_lambda.

        The monochromator line width is not removed unless mono_fwhm_nm > 0,
        in which case it is subtracted in quadrature from FWHM_lambda; without
        it, R is a lower limit on the instrument resolution.

        Args:
            fitarr: (nlam, ny, nx, >=6) [amp, x, y, x_std, y_std, theta, ...]
            lams: (nlam,) wavelengths in um
            mono_fwhm_nm: monochromator line FWHM in nm (0 = no correction)
            poly_deg: degree of the per-lenslet x(lambda), y(lambda) fits
            min_spots: minimum good spots for a lenslet to be used
            fwhm_pix_range: accepted range of along-trace FWHM (px)

        A second estimate, R_xdisp, uses the FWHM perpendicular to the trace
        instead (assuming a round monochromatic spot). It does not depend on
        the monochromator line width, so it is the one to use for MedRes,
        where the line is wide and its width unknown.

        Returns:
            dict of (nlam, ny, nx) arrays: R, fwhm_nm, fwhm_disp_pix,
            R_xdisp, fwhm_xdisp_nm, fwhm_xdisp_pix, disp_px_per_um; and
            per-wavelength summaries R_pct, R_xdisp_pct (nlam, 3) =
            16/50/84th percentiles, nspots, nspots_xdisp (nlam,).
        """
        lams = np.asarray(lams, dtype=float)
        nl = fitarr.shape[0]
        sx, sy, th = fitarr[..., 3], fitarr[..., 4], fitarr[..., 5]
        dxdl, dydl, good = self.trace_derivatives(fitarr, lams, poly_deg=poly_deg,
                                                  min_spots=min_spots)

        with np.errstate(invalid='ignore', divide='ignore'):
            disp = np.hypot(dxdl, dydl)                  # px / um
            ux, uy = dxdl / disp, dydl / disp
            fwhm_d, fwhm_x = self.gauss_fwhm_along_across(sx, sy, th, ux, uy)

            base = good & np.isfinite(disp) & (disp > 0)
            ok = (base & np.isfinite(fwhm_d)
                  & (fwhm_d > fwhm_pix_range[0]) & (fwhm_d < fwhm_pix_range[1]))
            ok_x = (base & np.isfinite(fwhm_x)
                    & (fwhm_x > fwhm_pix_range[0]) & (fwhm_x < fwhm_pix_range[1]))
            fwhm_nm = fwhm_d / disp * 1e3
            if mono_fwhm_nm and mono_fwhm_nm > 0:
                fwhm_nm = np.sqrt(fwhm_nm**2 - mono_fwhm_nm**2)
                ok &= np.isfinite(fwhm_nm)
            # cross-dispersion estimate: assumes the monochromatic spot is
            # round, so the spatial FWHM perpendicular to the trace stands in
            # for the instrumental FWHM along it. Independent of the
            # monochromator line width (used for MedRes, where that is unknown).
            fwhm_nm_x = fwhm_x / disp * 1e3
            fwhm_d[~ok] = np.nan
            fwhm_nm[~ok] = np.nan
            fwhm_x[~ok_x] = np.nan
            fwhm_nm_x[~ok_x] = np.nan
            disp[~(ok | ok_x)] = np.nan
            R = lams[:, None, None] * 1e3 / fwhm_nm
            R_x = lams[:, None, None] * 1e3 / fwhm_nm_x

        def pct(arr, mask):
            n = mask.reshape(nl, -1).sum(axis=1)
            p = np.full((nl, 3), np.nan)
            for k in range(nl):
                if n[k] > 0:
                    p[k] = np.nanpercentile(arr[k], [16, 50, 84])
            return p, n

        R_pct, nspots = pct(R, ok)
        R_x_pct, nspots_x = pct(R_x, ok_x)
        return dict(R=R, fwhm_nm=fwhm_nm, fwhm_disp_pix=fwhm_d,
                    R_xdisp=R_x, fwhm_xdisp_nm=fwhm_nm_x,
                    fwhm_xdisp_pix=fwhm_x, disp_px_per_um=disp,
                    R_pct=R_pct, nspots=nspots,
                    R_xdisp_pct=R_x_pct, nspots_xdisp=nspots_x)

    def write_spectral_resolution(self, res, lams, scmode, output_dir, plot_dir,
                                  mono_fwhm_nm=0.0):
        """Save the estimate_spectral_resolution outputs to
        {scmode}_resolution.fits and a summary plot, and log a per-wavelength table."""
        hdr = pyfits.Header()
        hdr['IFSMODE'] = scmode
        hdr['MEDR'] = (float(np.nanmedian(res['R'])), 'median R (along-trace FWHM)')
        hdr['MEDRX'] = (float(np.nanmedian(res['R_xdisp'])), 'median R (cross-trace FWHM)')
        hdr['MONOFWHM'] = (float(mono_fwhm_nm or 0.0),
                           'mono line FWHM removed in quadrature (nm)')
        hdr['COMMENT'] = 'R = lambda / FWHM_lambda from Gaussian spot fits'
        hdr['COMMENT'] = 'projected onto the local trace direction.'
        hdr['COMMENT'] = 'R_XDISP uses the FWHM perpendicular to the trace instead'
        hdr['COMMENT'] = '(round-spot assumption; independent of the mono line width).'
        if not mono_fwhm_nm:
            hdr['COMMENT'] = 'Monochromator width not removed: R is a lower limit.'
        hdus = [pyfits.PrimaryHDU(res['R'].astype(np.float32), header=hdr)]
        for name, key, unit in [('FWHM_NM', 'fwhm_nm', 'nm'),
                                ('FWHM_DISP_PIX', 'fwhm_disp_pix', 'pix'),
                                ('R_XDISP', 'R_xdisp', ''),
                                ('FWHM_XDISP_NM', 'fwhm_xdisp_nm', 'nm'),
                                ('FWHM_XDISP_PIX', 'fwhm_xdisp_pix', 'pix'),
                                ('DISPERSION', 'disp_px_per_um', 'pix/um')]:
            h = pyfits.ImageHDU(res[key].astype(np.float32), name=name)
            h.header['BUNIT'] = unit
            hdus.append(h)
        cols = [pyfits.Column(name='LAMBDA', format='D', unit='um', array=np.asarray(lams, float)),
                pyfits.Column(name='R_P16', format='D', array=res['R_pct'][:, 0]),
                pyfits.Column(name='R_MED', format='D', array=res['R_pct'][:, 1]),
                pyfits.Column(name='R_P84', format='D', array=res['R_pct'][:, 2]),
                pyfits.Column(name='NSPOTS', format='J', array=res['nspots']),
                pyfits.Column(name='RX_P16', format='D', array=res['R_xdisp_pct'][:, 0]),
                pyfits.Column(name='RX_MED', format='D', array=res['R_xdisp_pct'][:, 1]),
                pyfits.Column(name='RX_P84', format='D', array=res['R_xdisp_pct'][:, 2]),
                pyfits.Column(name='NSPOTS_X', format='J', array=res['nspots_xdisp'])]
        hdus.append(pyfits.BinTableHDU.from_columns(cols, name='SUMMARY'))
        pyfits.HDUList(hdus).writeto(output_dir + '/' + scmode + '_resolution.fits',
                                     overwrite=True)

        self.logger.info(f"{scmode} spectral resolution: median R = {hdr['MEDR']:.1f} "
                         f"(along trace), {hdr['MEDRX']:.1f} (cross trace)")
        self.logger.info("  lambda(um) | along trace: R_p16  R_med  R_p84 | cross trace: R_p16  R_med  R_p84")
        for k in range(len(lams)):
            p, q = res['R_pct'][k], res['R_xdisp_pct'][k]
            self.logger.info(f"  {lams[k]:9.4f} | {p[0]:12.1f} {p[1]:6.1f} {p[2]:6.1f} "
                             f"| {q[0]:12.1f} {q[1]:6.1f} {q[2]:6.1f}")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                rmap = np.nanmedian(res['R'], axis=0)
                rmap_x = np.nanmedian(res['R_xdisp'], axis=0)
                ratio = np.nanmedian(res['R_xdisp'] / res['R'], axis=0)
            fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))
            for pc, lab, col in [(res['R_pct'], 'along trace', 'C0'),
                                 (res['R_xdisp_pct'], 'cross trace', 'C1')]:
                ax[0].fill_between(lams, pc[:, 0], pc[:, 2], alpha=0.25, color=col)
                ax[0].plot(lams, pc[:, 1], 'o-', ms=3, color=col, label=lab + ' (median, 16-84%)')
            ax[0].set_xlabel('wavelength (um)')
            ax[0].set_ylabel('R = lambda / FWHM')
            ax[0].set_title(scmode + (' (mono width not removed)' if not mono_fwhm_nm else ''))
            ax[0].legend()
            allr = np.concatenate([rmap[np.isfinite(rmap)], rmap_x[np.isfinite(rmap_x)]])
            # robust colour range: faint, noisy lenslets at the field edges can have very high R
            # and would otherwise stretch a percentile-based range
            if allr.size:
                mid = np.median(allr); sc = 1.4826 * np.median(np.abs(allr - mid))
                vmin, vmax = mid - 4 * sc, mid + 4 * sc
            else:
                vmin, vmax = 0, 1
            for a, m, t in [(ax[1], rmap, 'median R per lenslet (along trace)'),
                            (ax[2], rmap_x, 'median R per lenslet (cross trace)')]:
                im = a.imshow(m, origin='lower', vmin=vmin, vmax=vmax)
                a.set_title(t)
                a.set_xlabel('lenslet x'); a.set_ylabel('lenslet y')
                fig.colorbar(im, ax=a)
            im = ax[3].imshow(ratio, origin='lower', cmap='RdBu_r', vmin=0.7, vmax=1.3)
            ax[3].set_title('median R_cross / R_along')
            ax[3].set_xlabel('lenslet x'); ax[3].set_ylabel('lenslet y')
            fig.colorbar(im, ax=ax[3])
            fig.tight_layout()
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(os.path.join(plot_dir, scmode + '_resolution.png'), dpi=120)
            plt.close(fig)
        except Exception as e:
            self.logger.info(f"resolution plot failed: {e}")

    def estimate_spectral_resolution_AnK(self, ank, lams, mono_fwhm_nm=0.0,
                                         fwhm_pix_range=(0.5, 8.0)):
        """
        Spectral resolution R = lambda / FWHM_lambda from the Anderson & King
        ePSFs (AnK_lowres_psfs), one value per wavelength and field region.

        The FWHM along the trace is that of the ePSF's marginal profile along
        the region's median trace direction (the line spread function an
        extraction summing across the trace sees), so no profile shape is
        assumed. It is converted to wavelength with the region's median local
        dispersion (px/um, from the same per-lenslet trace fits as the Gaussian
        estimate):
            FWHM_lambda = FWHM_along / (ds/dlambda),  R = lambda / FWHM_lambda.
        mono_fwhm_nm > 0 removes the monochromator line width in quadrature
        (otherwise R is a lower limit). R_xdisp uses the ePSF FWHM across the
        trace instead (round-spot assumption, independent of the line width).
        R_gauss, R_xdisp_gauss are the same quantities from the median
        Gaussian-fit FWHMs of the spots that went into each ePSF, for a
        like-for-like comparison.

        Args:
            ank: dict from AnK_lowres_psfs (or read_AnK_lowres_psfs)
            lams: (nlam,) wavelengths in um
            mono_fwhm_nm: monochromator line FWHM in nm (0 = no correction)
            fwhm_pix_range: accepted range of ePSF FWHMs (px)

        Returns:
            dict of (nlam, nry, nrx) arrays: R, fwhm_nm, fwhm_disp_pix,
            R_xdisp, fwhm_xdisp_nm, fwhm_xdisp_pix, disp_px_per_um, nspots,
            R_gauss, R_xdisp_gauss; per-wavelength summaries over regions
            R_pct, R_xdisp_pct (nlam, 3) = 16/50/84th percentiles, R_gauss_med,
            R_xdisp_gauss_med (nlam,), nspots, nspots_xdisp (nlam,, spots in
            the regions with a valid R) and nreg, nreg_xdisp (nlam,, regions
            with a valid R); and region_y, region_x, xdet, ydet passed through.
        """
        lams = np.asarray(lams, dtype=float)
        fa = np.array(ank['fwhm_along'], dtype=float)
        fx = np.array(ank['fwhm_across'], dtype=float)
        ga = np.array(ank['gauss_along'], dtype=float)
        gx = np.array(ank['gauss_across'], dtype=float)
        disp = np.array(ank['disp'], dtype=float)
        nsp = np.asarray(ank['nspots'])
        nl = fa.shape[0]
        lam3 = lams[:, None, None]
        lo, hi = fwhm_pix_range
        with np.errstate(invalid='ignore', divide='ignore'):
            base = np.isfinite(disp) & (disp > 0) & (nsp > 0)
            ok = base & np.isfinite(fa) & (fa > lo) & (fa < hi)
            ok_x = base & np.isfinite(fx) & (fx > lo) & (fx < hi)

            def to_nm(f, mono):
                nm = f / disp * 1e3
                if mono and mono > 0:
                    nm = np.sqrt(nm**2 - mono**2)
                return nm

            fwhm_nm = to_nm(fa, mono_fwhm_nm)
            ok &= np.isfinite(fwhm_nm)
            fwhm_nm_x = to_nm(fx, 0.0)
            gauss_nm = to_nm(ga, mono_fwhm_nm)
            gauss_nm_x = to_nm(gx, 0.0)
            fa[~ok] = np.nan; fwhm_nm[~ok] = np.nan
            fx[~ok_x] = np.nan; fwhm_nm_x[~ok_x] = np.nan
            gauss_nm[~ok] = np.nan; gauss_nm_x[~ok_x] = np.nan
            disp[~(ok | ok_x)] = np.nan
            R = lam3 * 1e3 / fwhm_nm
            R_x = lam3 * 1e3 / fwhm_nm_x
            R_g = lam3 * 1e3 / gauss_nm
            R_gx = lam3 * 1e3 / gauss_nm_x

        def pct(arr, mask):
            p = np.full((nl, 3), np.nan)
            m = np.full(nl, np.nan)
            nreg = mask.reshape(nl, -1).sum(axis=1)
            nspots = np.where(mask, nsp, 0).reshape(nl, -1).sum(axis=1)
            for k in range(nl):
                if nreg[k] > 0:
                    p[k] = np.nanpercentile(arr[k], [16, 50, 84])
            return p, nspots, nreg

        R_pct, nspots, nreg = pct(R, ok)
        R_x_pct, nspots_x, nreg_x = pct(R_x, ok_x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            R_g_med = np.nanmedian(R_g.reshape(nl, -1), axis=1)
            R_gx_med = np.nanmedian(R_gx.reshape(nl, -1), axis=1)
        out = dict(R=R, fwhm_nm=fwhm_nm, fwhm_disp_pix=fa,
                   R_xdisp=R_x, fwhm_xdisp_nm=fwhm_nm_x, fwhm_xdisp_pix=fx,
                   disp_px_per_um=disp, nspots_map=np.where(ok | ok_x, nsp, 0),
                   R_gauss=R_g, R_xdisp_gauss=R_gx,
                   R_pct=R_pct, R_xdisp_pct=R_x_pct,
                   R_gauss_med=R_g_med, R_xdisp_gauss_med=R_gx_med,
                   nspots=nspots, nspots_xdisp=nspots_x, nreg=nreg, nreg_xdisp=nreg_x)
        for k in ['region_y', 'region_x', 'xdet', 'ydet']:
            if k in ank:
                out[k] = np.asarray(ank[k])
        out['regspace'] = ank.get('regspace', 'lenslet')
        return out

    def AnK_lenslet_resolution(self, ank, lams_ank, fitarr, lams, mono_fwhm_nm=0.0,
                               fwhm_pix_range=(0.5, 8.0), step=0.02, window=3.0):
        """
        Spectral resolution of every spot from the A&K ePSF at its place on the
        detector: the ePSF the A&K rectmat uses for that lenslet (bilinear blend of
        the region ePSFs with AnK_field_weights, linear blend between the ePSF
        wavelengths around the spot's), measured like AnK_epsf_fwhm.

        Marginal profiles are linear in the ePSF, so each region ePSF's marginal
        profiles along and across the trace (along its region's median trace
        direction) are computed once and the spot's profile is the same blend of
        them; its FWHM is the half-maximum width of that blended profile. Then, as in
        estimate_spectral_resolution_AnK, FWHM_lambda = FWHM / (ds/dlambda) with the
        spot's own local dispersion (trace_derivatives), mono_fwhm_nm removed in
        quadrature from the along-trace width, and R = lambda / FWHM_lambda.

        Args:
            ank: dict from AnK_lowres_psfs / AnK_medres_psfs (or read_AnK_lowres_psfs)
            lams_ank: (nla,) ePSF wavelengths (um)
            fitarr: (nl, ny, nx, >=6) Gaussian spot fits at lams (spots, dispersion)
            lams: (nl,) master wavelengths (um)

        Returns dict of (nl, ny, nx) arrays fwhm_along, fwhm_across (px), R, R_xdisp
        (NaN where there is no good spot, no ePSF or the FWHM is outside
        fwhm_pix_range), and (ny, nx) maps R_map, R_xdisp_map (median over
        wavelength per lenslet).
        """
        lams = np.asarray(lams, float); lams_ank = np.asarray(lams_ank, float)
        nl, ny, nx = fitarr.shape[:3]
        ep, reg_ok = self.AnK_fill_missing(ank['epsf'], lams_ank)
        nla, nry, nrx = ep.shape[:3]
        ov = int(ank['oversamp'])
        W = self.AnK_field_weights(ank, ny, nx, reg_ok)          #(ny*nx, nreg)
        #region trace directions, filled like the ePSFs (nearest wavelength that has one)
        uxr = np.array(ank['ux'], float); uyr = np.array(ank['uy'], float)
        for ry in range(nry):
            for rx in range(nrx):
                ok = np.nonzero(np.isfinite(uxr[:, ry, rx]) & np.isfinite(uyr[:, ry, rx]))[0]
                for l in np.nonzero(~(np.isfinite(uxr[:, ry, rx]) & np.isfinite(uyr[:, ry, rx])))[0]:
                    if len(ok):
                        k = ok[np.argmin(np.abs(lams_ank[ok]-lams_ank[l]))]
                        uxr[l, ry, rx], uyr[l, ry, rx] = uxr[k, ry, rx], uyr[k, ry, rx]
        #marginal profiles of every region ePSF
        PA = PC = s = None
        for l in range(nla):
            for ry in range(nry):
                for rx in range(nrx):
                    if not reg_ok[ry, rx] or not np.isfinite(uxr[l, ry, rx]):
                        continue
                    s, a, c = self.AnK_epsf_marginals(ep[l, ry, rx], ov, uxr[l, ry, rx],
                                                      uyr[l, ry, rx], step=step, window=window)
                    if PA is None:
                        PA = np.zeros((nla, nry*nrx, len(s))); PC = np.zeros_like(PA)
                    PA[l, ry*nrx+rx], PC[l, ry*nrx+rx] = a, c
        out = {k: np.full((nl, ny, nx), np.nan) for k in ['fwhm_along', 'fwhm_across', 'R', 'R_xdisp']}
        if PA is None:
            out['R_map'] = out['R_xdisp_map'] = np.full((ny, nx), np.nan)
            return out
        dxdl, dydl, good = self.trace_derivatives(fitarr, lams)
        disp = np.hypot(dxdl, dydl)
        has_w = W.sum(axis=1) > 0
        for l in range(nl):
            lam = lams[l]
            if lam <= lams_ank[0]:
                pa, pc = PA[0], PC[0]
            elif lam >= lams_ank[-1]:
                pa, pc = PA[-1], PC[-1]
            else:
                k = np.searchsorted(lams_ank, lam) - 1
                a = (lams_ank[k+1]-lam)/(lams_ank[k+1]-lams_ank[k])
                pa = a*PA[k] + (1-a)*PA[k+1]; pc = a*PC[k] + (1-a)*PC[k+1]
            use = (good[l].ravel() & has_w)
            if not use.any():
                continue
            fa = np.full(ny*nx, np.nan); fc = np.full(ny*nx, np.nan)
            fa[use] = self.halfmax_rows(W[use] @ pa, s)
            fc[use] = self.halfmax_rows(W[use] @ pc, s)
            out['fwhm_along'][l] = fa.reshape(ny, nx)
            out['fwhm_across'][l] = fc.reshape(ny, nx)
        lo, hi = fwhm_pix_range
        with np.errstate(invalid='ignore', divide='ignore'):
            ok_d = np.isfinite(disp) & (disp > 0)
            fa, fc = out['fwhm_along'], out['fwhm_across']
            fa[~(ok_d & (fa > lo) & (fa < hi))] = np.nan
            fc[~(ok_d & (fc > lo) & (fc < hi))] = np.nan
            nm = fa / disp * 1e3
            if mono_fwhm_nm and mono_fwhm_nm > 0:
                nm = np.sqrt(nm**2 - mono_fwhm_nm**2)
            out['R'] = lams[:, None, None] * 1e3 / nm
            out['R_xdisp'] = lams[:, None, None] * 1e3 / (fc / disp * 1e3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            out['R_map'] = np.nanmedian(out['R'], axis=0)
            out['R_xdisp_map'] = np.nanmedian(out['R_xdisp'], axis=0)
        return out

    def spot_residuals(self, spotim_arr, modim_arr, posarr, fitarr, lams, ank=None,
                       lams_ank=None, fit_radius=3.5, nstack=3):
        """
        Residuals of the spot models, per spot, for spot_residual diagnostics.

        Gaussian: data crop - Gaussian fit model crop (fit_gauss_spots[_names]).
        A&K (if ank is given): the ePSF the A&K rectmat uses for that lenslet
        (AnK_field_weights blend of the region ePSFs, linear in wavelength between
        the ePSF wavelengths) placed at the Gaussian-fit spot centre, as the A&K
        rectmat places it, with only its flux fitted (least squares within
        fit_radius px of the centre).

        Residual RMS within fit_radius px of the centre, in % of the spot's data
        peak (the same quantity for both models; the A&K ePSF builder logs the
        same RMS relative to its model peak).

        Args:
            spotim_arr, modim_arr: (nl, ny, nx, c, c) data and Gaussian model crops;
                each spot's box (posarr xs:xe, ys:ye) is at [:ye-ys, :xe-xs]
            posarr: (nl, ny, nx, 7) spot boxes; fitarr: (nl, ny, nx, >=6) fits
            lams: (nl,) um; ank, lams_ank: A&K ePSFs and their wavelengths
            nstack: number of wavelengths (evenly spread) for the stacked residuals

        Returns dict with rms_gauss, rms_ank (nl, ny, nx, %; NaN where no spot),
        stack_lams (nstack,), and (nstack, c, c) median over spots of the
        data / peak (stack_data) and residual / peak (stack_gauss, stack_ank).
        """
        spotim_arr = np.asarray(spotim_arr); modim_arr = np.asarray(modim_arr)
        nl, ny, nx, c = spotim_arr.shape[:4]
        lams = np.asarray(lams, float)
        yy, xx = np.mgrid[:c, :c]
        out = dict(rms_gauss=np.full((nl, ny, nx), np.nan), rms_ank=np.full((nl, ny, nx), np.nan))
        stack_idx = np.unique(np.linspace(0, nl - 1, nstack).astype(int))
        out['stack_lams'] = lams[stack_idx]
        out['stack_data'] = np.full((len(stack_idx), c, c), np.nan)
        out['stack_gauss'] = np.full((len(stack_idx), c, c), np.nan)
        out['stack_ank'] = np.full((len(stack_idx), c, c), np.nan)

        have_ank = ank is not None
        if have_ank:
            from scipy.ndimage import spline_filter
            lams_ank = np.asarray(lams_ank, float)
            ep, reg_ok = self.AnK_fill_missing(ank['epsf'], lams_ank)
            nla, nry, nrx, G, _ = ep.shape
            ov = int(ank['oversamp']); c0 = G // 2
            coef = np.array([[spline_filter(ep[l, r // nrx, r % nrx], order=3)
                              for r in range(nry * nrx)] for l in range(nla)])
            W = self.AnK_field_weights(ank, ny, nx, reg_ok)

        for l in range(nl):
            p = posarr[l]
            ok = (np.isfinite(p[..., 0]) & np.all(np.isfinite(fitarr[l, ..., :3]), axis=-1)
                  & (fitarr[l, ..., 0] > 0))
            if not ok.any():
                continue
            J, I = np.nonzero(ok)
            xs, xe, ys, ye = (p[J, I, 1].astype(int), p[J, I, 2].astype(int),
                              p[J, I, 4].astype(int), p[J, I, 5].astype(int))
            inbox = (xx[None] < (xe - xs)[:, None, None]) & (yy[None] < (ye - ys)[:, None, None])
            d = np.where(inbox, spotim_arr[l, J, I].astype(float), np.nan)
            g = np.where(inbox, modim_arr[l, J, I].astype(float), np.nan)
            cx = fitarr[l, J, I, 1] - xs; cy = fitarr[l, J, I, 2] - ys
            ap = inbox & (np.hypot(xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]) <= fit_radius)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                pk = np.nanmax(np.where(inbox, d, np.nan), axis=(1, 2))
                good = np.isfinite(pk) & (pk > 0) & (ap.sum(axis=(1, 2)) > 0)
                rg = 100 * np.sqrt(np.nanmean(np.where(ap, d - g, np.nan)**2, axis=(1, 2))) / pk
            out['rms_gauss'][l, J[good], I[good]] = rg[good]
            a_res = None
            if have_ank:
                lam = lams[l]
                if lam <= lams_ank[0]:
                    C = coef[0]
                elif lam >= lams_ank[-1]:
                    C = coef[-1]
                else:
                    k = np.searchsorted(lams_ank, lam) - 1
                    a = (lams_ank[k+1] - lam) / (lams_ank[k+1] - lams_ank[k])
                    C = a * coef[k] + (1 - a) * coef[k+1]
                li = J * nx + I
                gx = (xx[None] - cx[:, None, None]) * ov + c0
                gy = (yy[None] - cy[:, None, None]) * ov + c0
                m = np.zeros(d.shape)
                for r in range(C.shape[0]):
                    w = W[li, r]; sel = w > 0
                    if not sel.any():
                        continue
                    smp = map_coordinates(C[r], [gy[sel].ravel(), gx[sel].ravel()], order=3,
                                          prefilter=False, mode='constant', cval=0.0)
                    m[sel] += w[sel, None, None] * smp.reshape(-1, c, c)
                hasw = W[li].sum(axis=1) > 0
                with np.errstate(invalid='ignore', divide='ignore'):
                    dm = np.where(ap, d * m, 0).sum(axis=(1, 2))
                    mm = np.where(ap, m * m, 0).sum(axis=(1, 2))
                    f = dm / mm
                    a_res = np.where(inbox, d - f[:, None, None] * m, np.nan)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        ra = 100 * np.sqrt(np.nanmean(np.where(ap, a_res, np.nan)**2, axis=(1, 2))) / pk
                gooda = good & hasw & np.isfinite(ra) & (mm > 0)
                out['rms_ank'][l, J[gooda], I[gooda]] = ra[gooda]
            if l in stack_idx:
                s = int(np.nonzero(stack_idx == l)[0][0])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    out['stack_data'][s] = np.nanmedian(d[good] / pk[good, None, None], axis=0)
                    out['stack_gauss'][s] = np.nanmedian((d - g)[good] / pk[good, None, None], axis=0)
                    if a_res is not None:
                        out['stack_ank'][s] = np.nanmedian(a_res[gooda] / pk[gooda, None, None], axis=0)
        out['fit_radius'] = fit_radius
        return out

    def write_spot_residuals(self, sres, lams, scmode, output_dir, plot_dir):
        """Save spot_residuals to {scmode}_spot_residuals.fits (per spot RMS, axes
        lam, lenslet y, lenslet x, and the stacked residual images) and plot
        {scmode}_spot_residuals.png (mono_basic.plot_spot_residual_diagnostics)."""
        hdr = pyfits.Header()
        hdr['IFSMODE'] = scmode
        hdr['FITRAD'] = (float(sres['fit_radius']), 'px; RMS within this radius of the spot centre')
        hdr['BUNIT'] = '%'
        hdr['COMMENT'] = 'Gaussian-fit residual RMS per spot, % of the spot data peak.'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            hdr['MEDGAUSS'] = (float(np.nanmedian(sres['rms_gauss'])), 'median Gaussian residual (%)')
            if np.isfinite(sres['rms_ank']).any():
                hdr['MEDANK'] = (float(np.nanmedian(sres['rms_ank'])), 'median A&K residual (%)')
        hdus = [pyfits.PrimaryHDU(sres['rms_gauss'].astype(np.float32), header=hdr)]
        h = pyfits.ImageHDU(sres['rms_ank'].astype(np.float32), name='RMS_ANK')
        h.header['COMMENT'] = 'A&K ePSF (as the A&K rectmat places it) residual RMS, % of data peak'
        hdus.append(h)
        hdus.append(pyfits.ImageHDU(np.asarray(lams, float), name='LAMBDA'))
        for name, key in [('STACK_DATA', 'stack_data'), ('STACK_GAUSS', 'stack_gauss'),
                          ('STACK_ANK', 'stack_ank')]:
            hdus.append(pyfits.ImageHDU(sres[key].astype(np.float32), name=name))
        hdus.append(pyfits.ImageHDU(np.asarray(sres['stack_lams'], float), name='STACK_LAMBDA'))
        pyfits.HDUList(hdus).writeto(output_dir + '/' + scmode + '_spot_residuals.fits', overwrite=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            msg = f"{scmode} spot residuals (RMS within {sres['fit_radius']} px, % of peak): Gaussian median {hdr['MEDGAUSS']:.2f}"
            if 'MEDANK' in hdr:
                msg += f", A&K median {hdr['MEDANK']:.2f}"
        self.logger.info(msg)
        try:
            mono_basic.plot_spot_residual_diagnostics(sres, lams, plot_dir,
                                                      scmode + '_spot_residuals.png', title=scmode)
        except Exception as e:
            self.logger.warning(f"spot residual plot failed: {e}")

    def write_spectral_resolution_AnK(self, res, lams, scmode, output_dir, plot_dir,
                                      mono_fwhm_nm=0.0, gauss_res=None, gauss_lams=None,
                                      lens_res=None):
        """Save the estimate_spectral_resolution_AnK outputs to
        {scmode}_AnK_resolution.fits and {scmode}_AnK_resolution.png, and log a
        per-wavelength table. The FITS layout follows {scmode}_resolution.fits
        (Gaussian fits, per lenslet) but the image axes are (lam, region y,
        region x); the SUMMARY table has the same columns plus the region
        counts and the Gaussian-fit R of the same spots. gauss_res (the
        estimate_spectral_resolution dict) adds the all-lenslet Gaussian R to
        the plot, at gauss_lams (default lams; MedRes ePSFs are per wavelength
        bin, so the Gaussian R is on the master wavelengths instead).
        lens_res (AnK_lenslet_resolution, per spot at gauss_lams) adds the A&K R of
        every spot (HDUs R_LENS, R_XDISP_LENS, FWHM_LENS_PIX, FWHM_XDISP_LENS_PIX,
        axes (lam, lenslet y, lenslet x)) and a second plot row of per-lenslet maps,
        laid out like {scmode}_resolution.png."""
        lams = np.asarray(lams, dtype=float)
        gauss_lams = lams if gauss_lams is None else np.asarray(gauss_lams, dtype=float)
        nl, nry, nrx = res['R'].shape
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            medr = float(np.nanmedian(res['R']))
            medrx = float(np.nanmedian(res['R_xdisp']))
            medrg = float(np.nanmedian(res['R_gauss']))
        hdr = pyfits.Header()
        hdr['IFSMODE'] = scmode
        hdr['MEDR'] = (medr, 'median R (A&K ePSF along-trace FWHM)')
        hdr['MEDRX'] = (medrx, 'median R (A&K ePSF cross-trace FWHM)')
        hdr['MEDRG'] = (medrg, 'median R (Gaussian fits, same spots)')
        hdr['MONOFWHM'] = (float(mono_fwhm_nm or 0.0),
                           'mono line FWHM removed in quadrature (nm)')
        hdr['NREGY'] = nry; hdr['NREGX'] = nrx
        hdr['COMMENT'] = 'R = lambda / FWHM_lambda from Anderson & King ePSFs, one per'
        hdr['COMMENT'] = 'wavelength and field region; axes (lam, region y, region x).'
        hdr['COMMENT'] = 'FWHM = marginal ePSF profile along the median trace direction.'
        hdr['COMMENT'] = 'R_XDISP uses the FWHM perpendicular to the trace instead'
        hdr['COMMENT'] = '(round-spot assumption; independent of the mono line width).'
        if not mono_fwhm_nm:
            hdr['COMMENT'] = 'Monochromator width not removed: R is a lower limit.'
        hdus = [pyfits.PrimaryHDU(res['R'].astype(np.float32), header=hdr)]
        for name, key, unit in [('FWHM_NM', 'fwhm_nm', 'nm'),
                                ('FWHM_DISP_PIX', 'fwhm_disp_pix', 'pix'),
                                ('R_XDISP', 'R_xdisp', ''),
                                ('FWHM_XDISP_NM', 'fwhm_xdisp_nm', 'nm'),
                                ('FWHM_XDISP_PIX', 'fwhm_xdisp_pix', 'pix'),
                                ('DISPERSION', 'disp_px_per_um', 'pix/um'),
                                ('R_GAUSS', 'R_gauss', ''),
                                ('R_XDISP_GAUSS', 'R_xdisp_gauss', ''),
                                ('NSPOTS', 'nspots_map', '')]:
            h = pyfits.ImageHDU(np.asarray(res[key]).astype(np.float32), name=name)
            h.header['BUNIT'] = unit
            hdus.append(h)
        cols = [pyfits.Column(name='LAMBDA', format='D', unit='um', array=lams),
                pyfits.Column(name='R_P16', format='D', array=res['R_pct'][:, 0]),
                pyfits.Column(name='R_MED', format='D', array=res['R_pct'][:, 1]),
                pyfits.Column(name='R_P84', format='D', array=res['R_pct'][:, 2]),
                pyfits.Column(name='NSPOTS', format='J', array=res['nspots']),
                pyfits.Column(name='NREG', format='J', array=res['nreg']),
                pyfits.Column(name='RX_P16', format='D', array=res['R_xdisp_pct'][:, 0]),
                pyfits.Column(name='RX_MED', format='D', array=res['R_xdisp_pct'][:, 1]),
                pyfits.Column(name='RX_P84', format='D', array=res['R_xdisp_pct'][:, 2]),
                pyfits.Column(name='NSPOTS_X', format='J', array=res['nspots_xdisp']),
                pyfits.Column(name='NREG_X', format='J', array=res['nreg_xdisp']),
                pyfits.Column(name='RG_MED', format='D', array=res['R_gauss_med']),
                pyfits.Column(name='RGX_MED', format='D', array=res['R_xdisp_gauss_med'])]
        hdus.append(pyfits.BinTableHDU.from_columns(cols, name='SUMMARY'))
        if 'region_y' in res and 'region_x' in res:
            ry, rx = np.asarray(res['region_y']), np.asarray(res['region_x'])
            RY, RX = np.meshgrid(np.arange(nry), np.arange(nrx), indexing='ij')
            rcols = [pyfits.Column(name='REG_Y', format='J', array=RY.ravel()),
                     pyfits.Column(name='REG_X', format='J', array=RX.ravel()),
                     pyfits.Column(name='LENS_Y0', format='J', array=ry[RY.ravel(), 0]),
                     pyfits.Column(name='LENS_Y1', format='J', array=ry[RY.ravel(), 1]),
                     pyfits.Column(name='LENS_X0', format='J', array=rx[RX.ravel(), 0]),
                     pyfits.Column(name='LENS_X1', format='J', array=rx[RX.ravel(), 1])]
            hdus.append(pyfits.BinTableHDU.from_columns(rcols, name='REGIONS'))
        if lens_res is not None:
            #A&K R of every spot (AnK_lenslet_resolution), axes (lam, lenslet y, lenslet x)
            for name, key, unit in [('R_LENS', 'R', ''), ('R_XDISP_LENS', 'R_xdisp', ''),
                                    ('FWHM_LENS_PIX', 'fwhm_along', 'pix'),
                                    ('FWHM_XDISP_LENS_PIX', 'fwhm_across', 'pix')]:
                h = pyfits.ImageHDU(np.asarray(lens_res[key]).astype(np.float32), name=name)
                h.header['BUNIT'] = unit
                h.header['COMMENT'] = 'per spot, from the A&K ePSF at its detector position'
                hdus.append(h)
            hdus.append(pyfits.ImageHDU(gauss_lams, name='LAMBDA_LENS'))
        pyfits.HDUList(hdus).writeto(output_dir + '/' + scmode + '_AnK_resolution.fits',
                                     overwrite=True)
        if lens_res is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                self.logger.info(f"{scmode} A&K R per spot (ePSF at each spot's position): median "
                                 f"{np.nanmedian(lens_res['R']):.1f} (along trace), "
                                 f"{np.nanmedian(lens_res['R_xdisp']):.1f} (cross trace)")

        self.logger.info(f"{scmode} spectral resolution from A&K ePSFs: median R = {medr:.1f} "
                         f"(along trace), {medrx:.1f} (cross trace); Gaussian fits of the "
                         f"same spots: {medrg:.1f}")
        self.logger.info("  lambda(um) | along trace: R_p16  R_med  R_p84 | cross trace: "
                         "R_p16  R_med  R_p84 | Gauss R_med | nreg")
        for k in range(nl):
            p, q = res['R_pct'][k], res['R_xdisp_pct'][k]
            self.logger.info(f"  {lams[k]:9.4f} | {p[0]:12.1f} {p[1]:6.1f} {p[2]:6.1f} "
                             f"| {q[0]:12.1f} {q[1]:6.1f} {q[2]:6.1f} "
                             f"| {res['R_gauss_med'][k]:11.1f} | {res['nreg'][k]:4d}")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                rmap = np.nanmedian(res['R'], axis=0)
                rmap_x = np.nanmedian(res['R_xdisp'], axis=0)
                rat = res['R'] / res['R_gauss']
                rat_x = res['R_xdisp'] / res['R_xdisp_gauss']
                rat_pct = np.nanpercentile(rat.reshape(nl, -1), [16, 50, 84], axis=1)
                ratx_pct = np.nanpercentile(rat_x.reshape(nl, -1), [16, 50, 84], axis=1)
            if lens_res is None:
                fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
            else:
                fig, axs = plt.subplots(2, 4, figsize=(22, 9.6))
                ax, ax2 = axs[0], axs[1]
            for pc, lab, col in [(res['R_pct'], 'A&K along trace', 'C0'),
                                 (res['R_xdisp_pct'], 'A&K cross trace', 'C1')]:
                ax[0].fill_between(lams, pc[:, 0], pc[:, 2], alpha=0.25, color=col)
                ax[0].plot(lams, pc[:, 1], 'o-', ms=3, color=col,
                           label=lab + ' (median, 16-84% of regions)')
            if gauss_res is not None:
                ax[0].plot(gauss_lams, gauss_res['R_pct'][:, 1], '--', color='C0',
                           label='Gaussian along (all lenslets)')
                ax[0].plot(gauss_lams, gauss_res['R_xdisp_pct'][:, 1], '--', color='C1',
                           label='Gaussian cross (all lenslets)')
            else:
                ax[0].plot(lams, res['R_gauss_med'], '--', color='C0',
                           label='Gaussian along (same spots)')
                ax[0].plot(lams, res['R_xdisp_gauss_med'], '--', color='C1',
                           label='Gaussian cross (same spots)')
            ax[0].set_xlabel('wavelength (um)')
            ax[0].set_ylabel('R = lambda / FWHM')
            ax[0].set_title(scmode + ' A&K ePSF' +
                            (' (mono width not removed)' if not mono_fwhm_nm else ''))
            ax[0].legend(fontsize=8)
            for a, m, t in [(ax[1], rmap, 'median R per region (A&K, along trace)'),
                            (ax[2], rmap_x, 'median R per region (A&K, cross trace)')]:
                # own robust colour range per map (along and across can differ a lot, e.g. Ls)
                fm = m[np.isfinite(m)]
                if fm.size:
                    mid = np.median(fm)
                    sc = max(1.4826 * np.median(np.abs(fm - mid)), 0.02 * mid)
                    vmin, vmax = mid - 4 * sc, mid + 4 * sc
                else:
                    vmin, vmax = 0, 1
                im = a.imshow(m, origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
                a.set_xticks(range(nrx)); a.set_yticks(range(nry))
                for j in range(nry):
                    for i in range(nrx):
                        if np.isfinite(m[j, i]):
                            a.text(i, j, f'{m[j, i]:.0f}', ha='center', va='center',
                                   fontsize=8, color='w')
                a.set_title(t)
                if res.get('regspace') == 'slice':
                    #MedRes, one region per 17-spot slice (AnK_medres_psfs per_slice=True)
                    a.set_xlabel('slice along the detector row (left to right)')
                    a.set_ylabel('detector row of traces')
                elif res.get('regspace') == 'detector':
                    #MedRes regions (AnK_medres_psfs): detector rows of traces x detector-x bands
                    a.set_xlabel('region x (detector x bands)')
                    a.set_ylabel('region y (detector rows of traces)')
                else:
                    a.set_xlabel('region x (lenslet x bands)')
                    a.set_ylabel('region y (lenslet y bands)')
                fig.colorbar(im, ax=a)
            for pc, lab, col in [(rat_pct, 'along trace', 'C0'), (ratx_pct, 'cross trace', 'C1')]:
                ax[3].fill_between(lams, pc[0], pc[2], alpha=0.25, color=col)
                ax[3].plot(lams, pc[1], 'o-', ms=3, color=col, label=lab)
            ax[3].axhline(1, color='k', lw=0.8)
            ax[3].set_xlabel('wavelength (um)')
            ax[3].set_ylabel('R(A&K) / R(Gaussian), same spots')
            ax[3].set_title('A&K vs Gaussian (median, 16-84% of regions)')
            ax[3].legend(fontsize=8)
            if lens_res is not None:
                #per-lenslet maps from the A&K ePSF at each spot (median over wavelength),
                #laid out and scaled like the Gaussian {scmode}_resolution.png
                lm, lmx = lens_res['R_map'], lens_res['R_xdisp_map']
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    lratio = np.nanmedian(lens_res['R_xdisp'] / lens_res['R'], axis=0)
                    gr = None
                    if gauss_res is not None and np.shape(gauss_res['R']) == np.shape(lens_res['R']):
                        gr = np.nanmedian(lens_res['R'] / gauss_res['R'], axis=0)
                def robust_range(m, floor):
                    #own range per map: along and cross R can differ a lot (MedRes ~1.7x),
                    #so a shared range washes both out
                    fm = m[np.isfinite(m)]
                    if not fm.size:
                        return 0, 1
                    mid = np.median(fm)
                    sc = max(1.4826 * np.median(np.abs(fm - mid)), floor * abs(mid))
                    return mid - 4 * sc, mid + 4 * sc
                for a, m, t in [(ax2[0], lm, 'median R per lenslet (A&K at each spot, along trace)'),
                                (ax2[1], lmx, 'median R per lenslet (A&K at each spot, cross trace)')]:
                    vmin, vmax = robust_range(m, 0.02)
                    im = a.imshow(m, origin='lower', vmin=vmin, vmax=vmax)
                    a.set_title(t, fontsize=10)
                    a.set_xlabel('lenslet x'); a.set_ylabel('lenslet y')
                    fig.colorbar(im, ax=a)
                #centred on its median (cross/along is ~1.7 in MedRes, ~1.1 in LowRes)
                vmin, vmax = robust_range(lratio, 0.01)
                im = ax2[2].imshow(lratio, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
                ax2[2].set_title('median R_cross / R_along (A&K)', fontsize=10)
                ax2[2].set_xlabel('lenslet x'); ax2[2].set_ylabel('lenslet y')
                fig.colorbar(im, ax=ax2[2])
                if gr is not None:
                    im = ax2[3].imshow(gr, origin='lower', cmap='RdBu_r', vmin=0.8, vmax=1.2)
                    ax2[3].set_title('median R(A&K) / R(Gaussian) per lenslet, along trace', fontsize=10)
                    ax2[3].set_xlabel('lenslet x'); ax2[3].set_ylabel('lenslet y')
                    fig.colorbar(im, ax=ax2[3])
                else:
                    ax2[3].axis('off')
            fig.tight_layout()
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(os.path.join(plot_dir, scmode + '_AnK_resolution.png'), dpi=120)
            plt.close(fig)
        except Exception as e:
            self.logger.info(f"A&K resolution plot failed: {e}")


    def gen_sparse_inds(self,xs,ys,xe,ye,ypix=2048,xpix=2048):
        """
        Function to take 2d x,y pixel coordinates and turn them into flattened
        coordinates for sparse matrix construction.
        """

        indsx = np.array([xval for xval in range(xs,xe) for yval in range(ys,ye)])
        indsy = np.array([yval for xval in range(xs,xe) for yval in range(ys,ye)])

        flatinds = np.ravel_multi_index((indsy,indsx),(ypix,xpix))
        return flatinds


    def crop_interpd_sparse_vals(self,gauss_pars,cut=0.05,method='optimal',cropsize=8):
        """
        Function to take gaussian spots and turn them into weights for a sparse
        extraction matrix.
        """

        if True not in np.isnan(gauss_pars):
            amplitude=gauss_pars[0]
            x_mean=gauss_pars[1]
            y_mean=gauss_pars[2]
            x_stddev=gauss_pars[3]
            y_stddev=gauss_pars[4]
            theta=gauss_pars[5]

            if self.scmode[:6]=='MedRes':
                if self.config.instrument.medres_force_ysigma==True:
                    y_stddev=gauss_pars[3]

            fitted_model = Gaussian2D(amplitude=amplitude,
                             x_mean=x_mean,
                             y_mean=y_mean,
                             x_stddev=x_stddev,
                             y_stddev=y_stddev,
                             theta=theta)


            cropsize2 = np.min([np.max([5*y_stddev,5*x_stddev]),3*cropsize])

            #ys = int(y_mean-3*y_stddev)
            #ye = int(y_mean+3*y_stddev)
            #xs = int(x_mean-3*x_stddev)
            #xe = int(x_mean+3*x_stddev)
            ys = int(np.round(y_mean-cropsize2/2))
            ye = int(np.round(y_mean+cropsize2/2))
            xs = int(np.round(x_mean-cropsize2/2))
            xe = int(np.round(x_mean+cropsize2/2))

            if ys<0: ys=0
            if ye>2047:ye=2047
            if xs<0: xs=0
            if xe>2047: xe=2047

            if xe < 0:
                return [], []
            if ye < 0:
                return [], []
            if xs > 2047:
                return [], []
            if ys > 2047:
                return [], []
            if ye-ys <= 0:
                return [], []
            if xe-xs <= 0:
                return [], []

            y, x = np.mgrid[ys:ye,xs:xe]
            modspot = fitted_model(x,y)

            #modspot[np.where(modspot < cut*np.max(modspot))]=0
            modspot/=np.sum(modspot)

            vals = np.array([modspot[yind,xind] for xind in range(0,xe-xs) for yind in range(0,ye-ys)])
            if method=='optimal':
                vals = vals

            if method=='aperture':
                vals[np.where(vals!=0)] = 1.0
            flatinds = self.gen_sparse_inds(xs,ys,xe,ye)
        return flatinds,vals

    def gen_imgs_gaussfit(self,interp_arr,cut=0.05,cropsize=8):
        """
        Function to take gaussian spots and turn them into weights for a sparse
        extraction matrix.
        """

        ims = []
        for ll in range(len(interp_arr)):
            self.logger.info('doing wavelength slice '+str(ll))
            imslice = np.zeros([2048,2048])
            for yy in range(len(interp_arr[0])):
                for xx in range(len(interp_arr[0,0])):
                    gauss_pars = interp_arr[ll,yy,xx]
                    if True not in np.isnan(gauss_pars):
                        amplitude=gauss_pars[0]
                        x_mean=gauss_pars[1]
                        y_mean=gauss_pars[2]
                        x_stddev=gauss_pars[3]
                        y_stddev=gauss_pars[4]
                        theta=gauss_pars[5]

                        fitted_model = Gaussian2D(amplitude=gauss_pars[0],
                                         x_mean=gauss_pars[1],
                                         y_mean=gauss_pars[2],
                                         x_stddev=gauss_pars[3],
                                         y_stddev=gauss_pars[4],
                                         theta=gauss_pars[5])

                        #ys = int(y_mean-3*y_stddev)
                        #ye = int(y_mean+3*y_stddev)
                        #xs = int(x_mean-3*x_stddev)
                        #xe = int(x_mean+3*x_stddev)
                        ys = int(y_mean-cropsize/2)
                        ye = int(y_mean+cropsize/2)
                        xs = int(x_mean-cropsize/2)
                        xe = int(x_mean+cropsize/2)

                        if ys<0: ys=0
                        if ye>2047:ye=2047
                        if xs<0: xs=0
                        if xe>2047: xe=2047

                        if xe < 0:
                            continue
                        if ye < 0:
                            continue
                        if xs > 2047:
                            continue
                        if ys > 2047:
                            continue
                        if ye-ys <= 0:
                            continue
                        if xe-xs <= 0:
                            continue

                        y, x = np.mgrid[ys:ye,xs:xe]
                        modspot = fitted_model(x,y)
                        modspot[np.where(modspot < cut*np.max(modspot))]=0

                        imslice[ys:ye,xs:xe]+=modspot
            ims.append(imslice)

        return np.array(ims)

    def crop_sparse_vals(self,image,xs,xe,ys,ye,cut=0.05,method='optimal'):
        """
        Function to crop lenslet PSFs down and then only select pixels above
        a certain flux threshold
        """
        #a copy: cutting and normalising a view would overwrite the input image
        #(it used to, so the stack passed on to make_lsf_lflat had unit-sum spots)
        cropped = np.array(image[ys:ye,xs:xe], dtype=float)

        cropped[np.where(cropped < cut*np.max(cropped))]=0
        cropped/=np.sum(cropped)

        vals = np.array([cropped[yind,xind] for xind in range(0,xe-xs) for yind in range(0,ye-ys)])
        if method=='optimal':
            vals = vals

        if method=='sum':
            vals[np.where(vals!=0)]=1.0
        return vals


    def gen_rectmat_inds_interpd(self,interp_arr,cut=0.05,method='optimal'):

        """
        Function to generate row and column indices for sparse matrix
        """

        matrowinds = []
        matcolinds = []
        matvals = []

        for ll in range(len(interp_arr)):
            for lensx in range(interp_arr.shape[2]):
                for lensy in range(interp_arr.shape[1]):
                    if True not in np.isnan(interp_arr[ll,lensy,lensx,:6]):
                        flatinds,vals = self.crop_interpd_sparse_vals(interp_arr[ll,lensy,lensx,:6],cut=cut,method=method)
                        for i in range(len(vals)):
                            if vals[i] > 0:
                                matvals.append(vals[i])
                                matcolinds.append(flatinds[i])
                                matrowinds.append(lensx+lensy*interp_arr.shape[2]+ll*interp_arr.shape[1]*interp_arr.shape[2])
        return matrowinds, matcolinds, matvals


    def spot_box_vals(self,spots,ll,lensy,lensx,xs,xe,ys,ye,cut=0.05,method='optimal'):
        """
        crop_sparse_vals for one lenslet at one wavelength, from either source:
          spots: the spot crops from make_posarr, (nlam, ny, nx, crop, crop); the box
                 (xs:xe, ys:ye) from posarr sits at [:ye-ys, :xe-xs] of each crop
                 (smaller than the crop where it is clipped at the detector edge), or
          spots: full background-subtracted images, (nlam, npix_y, npix_x).
        Both give the same values, since make_posarr cuts its crops from the same
        posarr box and applies the same flux cut.
        """
        if np.ndim(spots) == 5:
            return self.crop_sparse_vals(spots[ll,lensy,lensx],0,xe-xs,0,ye-ys,cut=cut,method=method)
        return self.crop_sparse_vals(spots[ll],xs,xe,ys,ye,cut=cut,method=method)

    def gen_rectmat_inds(self,calims,posarr,cut=0.05,method='optimal'):

        """
        Function to generate row and column indices for sparse matrix.
        calims: the spot crops from make_posarr (nlam, ny, nx, crop, crop), or the
        full images (nlam, npix_y, npix_x); see spot_box_vals
        """

        matrowinds = []
        matcolinds = []
        matvals = []

        for ll in range(len(calims)):
            for lensx in range(posarr.shape[2]):
                for lensy in range(posarr.shape[1]):
                    xc,xs,xe,yc,ys,ye,intens = posarr[ll,lensy,lensx]
                    if np.isnan(xc)==False:
                        xc = int(xc)
                        yc = int(yc)
                        xs = int(xs)
                        xe = int(xe)
                        ys = int(ys)
                        ye = int(ye)
                        flatinds = self.gen_sparse_inds(xs,ys,xe,ye)
                        vals = self.spot_box_vals(calims,ll,lensy,lensx,xs,xe,ys,ye,cut=cut,method=method)
                        for i in range(len(vals)):
                            if vals[i] > 0:
                                matvals.append(vals[i])
                                #matvals.append(1.0)
                                matcolinds.append(flatinds[i])
                                matrowinds.append(lensx+lensy*posarr.shape[2]+ll*posarr.shape[1]*posarr.shape[2])
        return matrowinds, matcolinds, matvals

    def rectmat_npix(self,calims,npix):
        """Detector pixels per frame: the image size if calims are full images,
        otherwise npix (spot crops or no images at all)."""
        if calims is not None and np.ndim(calims) == 3:
            return int(np.prod(np.shape(calims)[1:]))
        return int(np.prod(npix))

    def gen_QL_rectmat(self,calims,posarr,cut=0.05,method='optimal',interp=False,npix=(2048,2048)):
        """
        Function to generate rectmat from the cal unit spots: the spot crops from
        make_posarr (as _perform passes) or the full images (see spot_box_vals).
        """

        matrowinds,matcolinds,matvals = self.gen_rectmat_inds(calims,posarr,cut=cut,method=method)
        rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(np.prod(posarr.shape[:3]),self.rectmat_npix(calims,npix)))
        return rmat

    def gen_QL_rectmat_interpd(self,calims,interp_arr,cut=0.05,method='optimal',interp=False,npix=(2048,2048)):
        """
        Function to generate rectmat from the interpolated Gaussian spot fits. No
        images are needed: calims (None, or images whose size sets the detector
        size) is kept only for older callers.
        """

        matrowinds,matcolinds,matvals = self.gen_rectmat_inds_interpd(interp_arr,cut=cut,method=method)
        rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(np.prod(interp_arr.shape[:3]),self.rectmat_npix(calims,npix)))
        return rmat



    def gen_c2_rectmat_inds(self,calims,posarr,cut=0.05):

        """
        Function to generate row and column indices for sparse matrix
        for all 108 x 108 lenslets and wavelengths.
        calims: spot crops or full images, as in gen_rectmat_inds
        """

        matrowinds = []
        matcolinds = []
        matvals = []
        for ll in range(len(calims)):
            for lensx in range(posarr.shape[2]):
                for lensy in range(posarr.shape[1]):
                    xc,xs,xe,yc,ys,ye,intens = posarr[ll,lensy,lensx]
                    if np.isnan(xc)==False:
                        xc = int(xc)
                        yc = int(yc)
                        xs = int(xs)
                        xe = int(xe)
                        ys = int(ys)
                        ye = int(ye)
                        flatinds = self.gen_sparse_inds(xs,ys,xe,ye)
                        vals = self.spot_box_vals(calims,ll,lensy,lensx,xs,xe,ys,ye,cut=cut,method='optimal')
                        for i in range(len(vals)):
                            if vals[i] > 0:
                                matvals.append(vals[i])
                                matrowinds.append(flatinds[i])
                                matcolinds.append(lensx+lensy*posarr.shape[2]+ll*posarr.shape[1]*posarr.shape[2])
        return matrowinds, matcolinds, matvals

    def gen_c2_rectmat_inds_interpd(self,interp_arr,cut=0.05):

        """
        Function to generate row and column indices for sparse matrix
        """

        matrowinds = []
        matcolinds = []
        matvals = []

        for ll in range(len(interp_arr)):
            for lensx in range(interp_arr.shape[2]):
                for lensy in range(interp_arr.shape[1]):
                    if True not in np.isnan(interp_arr[ll,lensy,lensx,:6]):
                        flatinds,vals = self.crop_interpd_sparse_vals(interp_arr[ll,lensy,lensx,:6],cut=cut,method='optimal')
                        for i in range(len(vals)):
                            if vals[i] > 0:
                                matvals.append(vals[i])
                                matrowinds.append(flatinds[i])
                                matcolinds.append(lensx+lensy*interp_arr.shape[2]+ll*interp_arr.shape[1]*interp_arr.shape[2])
        return matrowinds, matcolinds, matvals



    def gen_C2_rectmat(self,calims,posarr,cut=0.05,npix=(2048,2048)):
        """
        Function to generate rectmat from the cal unit spots: the spot crops from
        make_posarr (as _perform passes) or the full images (see spot_box_vals).
        """

        matrowinds,matcolinds,matvals = self.gen_c2_rectmat_inds(calims,posarr,cut=cut)
        rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(self.rectmat_npix(calims,npix),np.prod(posarr.shape[:3])))
        return rmat

    def gen_C2_rectmat_interpd(self,calims,interp_arr,cut=0.05,npix=(2048,2048)):
        """
        Function to generate rectmat from the interpolated Gaussian spot fits. No
        images are needed: calims (None, or images whose size sets the detector
        size) is kept only for older callers.
        """

        matrowinds,matcolinds,matvals = self.gen_c2_rectmat_inds_interpd(interp_arr,cut=cut)
        rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(self.rectmat_npix(calims,npix),np.prod(interp_arr.shape[:3]),))
        return rmat

    def medres_row_slots(self,x,nslot=17,nscol=6,xpix=2048):
        """
        Give each trace in one MedRes detector row its slot (0 .. nscol*nslot-1).

        A row is nscol supercolumns of nslot lenslets. Within a supercolumn the
        traces are one pitch p apart; between supercolumns the gap is about 2p.
        A missing lenslet inside a supercolumn also leaves a 2p gap, so gaps alone
        can't place the traces. Instead:
          1. p = typical spacing of neighbouring traces.
          2. Anchors = runs of exactly nslot traces, each one pitch from the next
             (complete supercolumns). Their first x fixes the supercolumn period P
             and the start of each supercolumn (a straight line through the anchor
             starts predicts the others, which are then refined on their own traces).
          3. Each trace goes to the nearest slot (start + k*p) if within 0.3p;
             anything else (spurious traces at the detector edge or between
             supercolumns) is left out.
          4. Which nscol consecutive supercolumns are the real ones: the choice that
             places the most traces, then leaves the fewest empty on-detector slots.
             So supercolumns cut by the left or right edge are found without any x
             threshold.

        Returns slots (-1 = not assigned) and an info dict.
        """
        x = np.asarray(x, dtype=float)
        slots = np.full(len(x), -1, dtype=int)
        info = dict(pitch=np.nan, period=np.nan, n_anchor=0, n_assigned=0,
                    n_rejected=len(x), partial={})
        if len(x) < nslot:
            self.logger.warning(f"MedRes row with {len(x)} traces: too few to register")
            return slots, info
        order = np.argsort(x)
        xs = x[order]
        dx = np.diff(xs)
        p = np.median(dx)
        p = np.median(dx[np.abs(dx-p) < 0.25*p])
        tol = 0.3*p

        #anchors: runs of exactly nslot traces spaced by one pitch. A run of nslot+1 or
        #nslot+2 is a complete supercolumn plus a stray trace about a pitch from one end;
        #those are kept aside and resolved against the clean anchors below
        adj = np.abs(dx-p) < tol
        clean, ambig = [], []
        i = 0
        while i < len(xs):
            j = i
            while j < len(dx) and adj[j]:
                j += 1
            nrun = j-i+1
            if nrun == nslot:
                clean.append(xs[i])
            elif nslot < nrun <= nslot+2:
                ambig.append(xs[i:i+nrun-nslot+1])   #possible first traces
            i = j+1
        info['pitch'] = p
        P_nom = (nslot+1)*p   #supercolumn gap is about 2p

        def fit_line(a):
            #start of supercolumn r = coef[0]*r + coef[1], r counted from the first anchor
            a = np.sort(np.asarray(a))
            r = np.round((a-a[0])/P_nom).astype(int)
            if len(a) > 1 and len(np.unique(r)) > 1:
                return np.polyfit(r, a, 1), r, a
            return np.array([P_nom, a[0]]), r, a

        if len(clean) == 0 and len(ambig) > 0:
            #no clean anchor: fall back on the most evenly spaced window of each ambiguous run
            for cand in ambig:
                k0 = np.searchsorted(xs, cand[0])
                dev = np.abs(dx[k0:k0+len(cand)+nslot-2]-p)
                w = np.argmin([dev[s:s+nslot-1].max() for s in range(len(cand))])
                clean.append(cand[w])
            ambig = []
        if len(clean) == 0:
            self.logger.warning("MedRes row has no complete supercolumn: can't register it")
            return slots, info
        coef, ra, anchors = fit_line(clean)
        #drop anchors that disagree with the others (a stray trace can complete a
        #17-run in place of a missing lenslet, shifting it by a pitch); refit
        if len(anchors) >= 3:
            res = np.abs(anchors - np.polyval(coef, ra))
            if np.any(res > 0.5*p) and np.sum(res <= 0.5*p) >= 2:
                coef, ra, anchors = fit_line(anchors[res <= 0.5*p])
        #ambiguous runs: take the start closest to the line, if it is close
        for cand in ambig:
            r = np.round((cand-coef[1])/coef[0])
            res = np.abs(cand-np.polyval(coef, r))
            if res.min() < 0.5*p:
                anchors = np.append(anchors, cand[np.argmin(res)])
        coef, ra, anchors = fit_line(anchors)
        P = coef[0]
        info['period'] = P
        info['n_anchor'] = len(anchors)
        r_lo = int(np.floor((xs[0]-coef[1])/P))-1
        r_hi = int(np.ceil((xs[-1]-coef[1])/P))+1
        starts = {}
        for r in range(r_lo-nscol, r_hi+nscol+1):
            if r in ra:
                starts[r] = anchors[np.where(ra==r)[0][0]]
                continue
            s = np.polyval(coef, r)
            #refine on this supercolumn's own traces (twice: the first pass can be a few px off)
            for _ in range(2):
                w = (x > s-0.5*p) & (x < s+(nslot-0.5)*p)
                if not np.any(w):
                    break
                k = np.round((x[w]-s)/p)
                good = np.abs(x[w]-s-k*p) < tol
                if not np.any(good):
                    break
                s = np.median(x[w][good]-k[good]*p)
            starts[r] = s

        #nearest slot of every trace
        rr = np.full(len(x), np.nan)
        kk = np.full(len(x), -1, dtype=int)
        resid = np.full(len(x), np.inf)
        for r, s in starts.items():
            k = np.round((x-s)/p)
            res = np.abs(x-s-k*p)
            ok = (k >= 0) & (k < nslot) & (res < tol) & (res < resid)
            rr[ok] = r
            kk[ok] = k[ok].astype(int)
            resid[ok] = res[ok]
        #one trace per slot: keep the closest
        for key in set(zip(rr[np.isfinite(rr)], kk[np.isfinite(rr)])):
            same = np.where((rr == key[0]) & (kk == key[1]))[0]
            if len(same) > 1:
                drop = same[np.argsort(resid[same])[1:]]
                rr[drop] = np.nan

        #which nscol consecutive supercolumns are the real ones
        best = None
        for o in range(r_lo-nscol, r_hi+1):
            inwin = np.isfinite(rr) & (rr >= o) & (rr < o+nscol)
            nempty = 0
            for r in range(o, o+nscol):
                sx = starts[r] + p*np.arange(nslot)
                ondet = (sx > 0) & (sx < xpix-1)
                filled = np.zeros(nslot, bool)
                filled[kk[inwin & (rr == r)]] = True
                nempty += np.sum(ondet & ~filled)
            score = (np.sum(inwin), -nempty)
            if best is None or score > best[0]:
                best = (score, o)
        o = best[1]
        inwin = np.isfinite(rr) & (rr >= o) & (rr < o+nscol)
        slots[inwin] = ((rr[inwin]-o)*nslot + kk[inwin]).astype(int)
        info['n_assigned'] = int(np.sum(inwin))
        info['n_rejected'] = int(len(x)-np.sum(inwin))
        for s in range(nscol):
            n = int(np.sum((slots >= s*nslot) & (slots < (s+1)*nslot)))
            if n < nslot:
                info['partial'][s] = n
        return slots, info

    def get_medres_lensarr_xy(self,avgs,show_plots=True,nslot=17,nscol=6,dist=26):
        """
        Register MedRes traces to lenslets. avgs are the trace positions at the band
        centre. They are split into the 3 detector rows (clusters in y separated by
        more than dist px; the 3 most populated are kept), then each row is placed on
        its 6 x 17 slot lattice by medres_row_slots, and supercolumns are mapped to
        lenslet columns with scol_map.
        """
        avgs = np.asarray(avgs)
        inds0 = np.array(range(len(avgs)),dtype='int')
        nrowslot = nslot*nscol

        #detector rows: clusters in y
        ysort = np.argsort(avgs[:,1])
        breaks = np.where(np.diff(avgs[ysort,1]) > dist)[0]+1
        clusters = np.split(ysort, breaks)
        if len(clusters) < 3:
            raise ValueError(f"MedRes registration: found {len(clusters)} rows of traces, expected 3")
        if len(clusters) > 3:
            sizes = [len(c) for c in clusters]
            self.logger.warning(f"MedRes registration: {len(clusters)} groups of traces in y "
                                f"(sizes {sizes}); keeping the 3 largest")
            keep = sorted(np.argsort(sizes)[-3:])
            clusters = [clusters[k] for k in keep]

        posns_idx = np.zeros([3,nrowslot,3])
        posns_idx[:,:,:] = np.nan
        posns_pix = np.zeros([3,nrowslot,5])
        posns_pix[:,:,:] = np.nan

        if show_plots==True:
            f = plt.figure(clear=True)
            for c in clusters:
                plt.scatter(avgs[c,0],avgs[c,1],label=str(len(c)))
            plt.legend()
            plt.show()

        for yc, c in enumerate(clusters):
            slots, info = self.medres_row_slots(avgs[c,0], nslot=nslot, nscol=nscol)
            self.logger.info(f"MedRes row {yc}: {len(c)} traces, {info['n_assigned']} registered, "
                             f"{info['n_rejected']} left out; pitch {info['pitch']:.2f} px, "
                             f"supercolumn period {info['period']:.1f} px, {info['n_anchor']} complete "
                             f"supercolumns; partial supercolumns (index: traces) {info['partial']}")
            for t, i in zip(c, slots):
                if i < 0:
                    continue
                posns_idx[yc,i] = [i,yc,inds0[t]]
                posns_pix[yc,i] = [i,yc,avgs[t,0],avgs[t,1],inds0[t]]

        if show_plots==True:
            f = plt.figure(clear=True)
            plt.imshow(posns_idx[:,:,2])
            plt.colorbar()
            plt.show()

            f = plt.figure(clear=True)
            plt.title('x pixel')
            plt.imshow(posns_pix[:,:,2])
            plt.colorbar()
            plt.show()

            f = plt.figure(clear=True)
            plt.title('y pixel')
            plt.imshow(posns_pix[:,:,3])
            plt.colorbar()
            plt.show()


        ###supercolumn map: [row (bottom=lower), column on slicer]
        scol_map = [[0, [ 1, 15,  4, 12,  7,  9]],
                    [1, [17,  2, 14,  5, 11,  8]],
                    [2, [ 0, 16,  3, 13,  6, 10]]]

        posns_idx_2 = np.zeros([17,18,3])
        posns_pix_2 = np.zeros([17,18,7])

        for i in range(len(scol_map)):
            row = scol_map[i][0]
            cols = scol_map[i][1]
            for j in range(len(cols)):
                col = cols[j]
                posns_pix_2[:,col,:5]=posns_pix[row,j*17:(j+1)*17]
                posns_pix_2[:,col,5]=np.ones([17])*row
                posns_pix_2[:,col,6]=np.ones([17])*j
                posns_idx_2[:,col]=posns_idx[row,j*17:(j+1)*17]

        if show_plots==True:
            f = plt.figure(clear=True)
            plt.title('x pixel')
            plt.imshow(posns_pix_2[:,:,2])
            plt.colorbar()
            plt.show()

            f = plt.figure(clear=True)
            plt.title('y pixel')
            plt.imshow(posns_pix_2[:,:,3])
            plt.colorbar()
            plt.show()

            f = plt.figure(clear=True)
            plt.title('row number (on det; superrow)')
            plt.imshow(posns_pix_2[:,:,5])
            plt.colorbar()
            plt.show()

            f = plt.figure(clear=True)
            plt.title('row number (on det; within superrow)')
            plt.imshow(posns_pix_2[:,:,6])
            plt.colorbar()
            plt.show()

        return posns_idx_2

    def get_spot_yrange_medres(self,lam,y0=180,lam0=2.0248,chy=1,lmin=1.95,lmax=2.45,length=1822):
        """
        Function to get the expected position of a certain wavelength
        within a trace.

        Args:
            lam: wavelength at which to calculate trace position
            y0: reference y position of trace
            lam0: reference wavelength for trace position (x0,y0)
            chy: direction of trace y movement with +ve lambda
            length: trace length in pixels

        Returns:
            ymin: trace y lower limit
            ymax: trace y upper limit

        """
        dlam = lam-lam0
        yoff = dlam/(lmax-lmin)*length*chy
        ypos = y0+yoff
        ystart = np.max([ypos-250,0])
        yend = np.min([ypos+250,2048])
        return ypos,ystart,yend


    def set_lamlimits(self,scmode):
        if scmode.split('-')[0][:6]=='LowRes':
            self.nx = 120
            self.ny = 120
        if scmode.split('-')[0][:6]=='MedRes':
            self.nx = 18
            self.ny = 17
        if scmode == "LowRes-KLM":
            self.lmin = 2.0
            self.lmax = 5.2
        if scmode == 'LowRes-K':
            self.lmin = 1.95
            self.lmax = 2.45
        if scmode == 'MedRes-K':
            self.lmin = 1.95
            self.lmax = 2.45
            self.mfilt = 'imgK'
            self.lmin_i = 2.02
            self.lmax_i = 2.38
        if scmode == 'LowRes-L':
            self.lmin = 2.9
            self.lmax = 4.15
        if scmode == 'MedRes-L':
            self.lmin = 2.9
            self.lmax = 4.15
            #IFS L filter trace model (medres_L_20261117); interpolated rectmat grid
            #inside the scanned 2.888-4.162 um so its ends are not extrapolated
            self.mfilt = 'L'
            self.lmin_i = 2.89
            self.lmax_i = 4.16
        if scmode == 'LowRes-M':
            self.lmin = 4.5
            self.lmax = 5.2
        if scmode == 'MedRes-M':
            self.lmin = 4.5
            self.lmax = 5.2
        if scmode == 'LowRes-KL':
            self.lmin = 2.0
            self.lmax = 4.0
        if scmode == 'LowRes-Ls':
            self.lmin = 3.1
            self.lmax = 3.5

        return

    def load_calim(self,fname,meanim):
        """One background-subtracted (and optionally destriped) master, as in
        find_all_spots_names."""
        im_cal = self.monochrom_bksub_sing(pyfits.getdata(fname,memmap=False),meanim)
        if self.config.instrument.apply_destripe==True:
            im_cal = self.masked_row_destripe(im_cal)[0]
        return im_cal

    def medres_intp_nchan(self, scmode, default=600):
        """Number of MedRes interpolated-rectmat channels: config medres_intp_nchan,
        one number for every MedRes mode or a dict {mode: n} (modes not listed get
        the default, 600). E.g. MedRes-L traces are ~1730 px over 2.89-4.16 um with
        a 4.1-4.7 px FWHM along the trace, so 600 channels is 1.4-1.8 samples per
        FWHM and ~850 is Nyquist."""
        v = getattr(self.config.instrument, 'medres_intp_nchan', None)
        if isinstance(v, dict):
            v = v.get(scmode)
        if v is None:
            return default
        n = int(v)
        if n < 2:
            raise ValueError(f"medres_intp_nchan for {scmode} must be >= 2, got {v}")
        if n != default:
            self.logger.info(f"{scmode}: {n} interpolated rectmat channels (medres_intp_nchan)")
        return n

    def make_lsf_lflat(self,rmat,ims_cal,lams,ny,nx,meanim=None):
        """
        Extract every master with rmat: lsf[ii] is the cube from master ii, and the
        lenslet flat is the median over ii of its own wavelength slice.

        ims_cal: the master filenames (with meanim, the mean master for the
        background subtraction), loaded one at a time, or the images themselves.
        Each template reads master ii along the whole trace, not just inside the
        spot's own crop, so the spot crops are not enough here.
        """
        names = np.ndim(ims_cal) == 1
        lsfcube = []
        lensflatcube = []
        for ii in range(len(ims_cal)):
            im = self.load_calim(ims_cal[ii],meanim) if names else ims_cal[ii]
            scube = np.array(rmat*np.ravel(im)).reshape([len(lams),ny,nx])
            scube_lsf = scube
            #lflat = np.max(scube,axis=0)
            lflat = scube[ii]
            lsfcube.append(scube_lsf)
            lensflatcube.append(lflat)
        lensflat = np.median(lensflatcube,axis=0)
        return np.array(lsfcube),np.array(lensflat)

    def make_lsf_lflat_interpd(self,rmat,interp_arr,lams,ny,nx,cut=0.1,cropsize=8):
        lsfcube = []
        lensflatcube = []
        imgs = np.array(self.gen_imgs_gaussfit(interp_arr,cut=cut,cropsize=cropsize))
        for ii in range(len(imgs)):
            self.logger.info("Doing LSF and lensflat "+str(ii)+"/"+str(len(imgs)))
            scube = np.array(rmat*imgs[ii].flatten()).reshape([len(lams),ny,nx])
            #scube_lsf = scube/np.max(scube,axis=0)
            lflat = scube[ii]
            lsfcube.append(scube)
            lensflatcube.append(lflat)
        lensflat = np.median(lensflatcube,axis=0)
        return np.array(lsfcube),np.array(lensflat)

    def _perform(self):


        output_dir = self.config.instrument.output_directory
        plot_dir = self.config.instrument.plot_directory
        package = __name__.split('.')[0]
        calibfilepath = self.context.calib_file_path
        calib_dir = str(get_resource_path(package, calibfilepath))+'/'
        create_rectmat = bool(self.config.instrument.create_rectmat)

        if not create_rectmat:
            self.logger.info("create_rectmat = False: stopping after the CALUNIT masters; "
                             "not making rectification matrices.")
            return self.action.args

        #intermediate files (background-subtracted stack, spot crops, spot model and
        #residual images) are only written with save_debug_files = True; the
        #calibration products (rectmats, LSFs, lflats, spot fits, wavelengths,
        #positions, resolution and ePSF tables) are always written
        save_debug = bool(getattr(self.config.instrument, 'save_debug_files', False) or False)
        def write_debug(name, data):
            if save_debug:
                pyfits.writeto(output_dir+'/'+name, np.array(data), overwrite=True)
        if not save_debug:
            self.logger.info("save_debug_files = False: not writing intermediate files "
                             "(calim_stack, spot crops, spot model/residual images)")

        self.logger.info("############ started Processing Monochromator Images #############")
        df = fits_headers_to_dataframe(output_dir,pattern="*mcalunit.fits")
        if not (df.empty or ("IFSMODE" in df.columns)):
            df["IFSMODE"] = df.apply(
                lambda row: scbasic.select_ifsmode(
                    row["MODSLNAM"],
                    row["DSPRSNAM"],
                    row["FILTER"]),axis=1)

            self.logger.info(f"Updated IFSMODE values: {df['IFSMODE'].tolist()}")
        self.logger.info(f"Searching monochromator calibration files in: {output_dir}")
        self.logger.info(f"Number of matching files: {len(df)}")

        for scmode in ['LowRes-KLM','LowRes-K','LowRes-L',
                       'LowRes-M','LowRes-KL','LowRes-Ls',
                       'MedRes-K','MedRes-L','MedRes-M']:
            self.scmode=scmode
            self.logger.info("Parsing files for "+str(self.scmode))

            imnames, lams, mean = self.get_filenames_and_mean(df,scmode,output_dir)
            wave_range = getattr(self.config.instrument, 'mono_wave_range', None)
            imnames, lams = self.apply_wave_range_names(imnames, lams, scmode, wave_range)
            exclude_waves = getattr(self.config.instrument, 'mono_exclude_waves', None)
            imnames, lams = self.apply_exclude_waves_names(imnames, lams, scmode, exclude_waves)



            """
            commenting out during wavelength restructure!!
            ims, lams = self.parse_files(df,scmode,output_dir)
            #optional: only use masters inside a wavelength range (key may be absent)
            wave_range = getattr(self.config.instrument, 'mono_wave_range', None)
            ims, lams = self.apply_wave_range(ims, lams, scmode, wave_range)
            #optional: drop masters at single wavelengths (key may be absent)
            exclude_waves = getattr(self.config.instrument, 'mono_exclude_waves', None)
            ims, lams = self.apply_exclude_waves(ims, lams, scmode, exclude_waves)
            """


            if len(imnames) > 0:
                self.set_lamlimits(scmode)


                """
                commenting out during wavelength restructure!!
                ims_cal_tmp = self.monochrom_bksub(ims)
                if self.config.instrument.apply_destripe==True:
                    ims_cal = np.array([self.masked_row_destripe(im)[0] for im in ims_cal_tmp])
                else:
                    ims_cal = ims_cal_tmp
                write_debug(scmode+'_calim_stack.fits',ims_cal)
                self.logger.info("done loading images for "+str(self.scmode))
                """


                if scmode.split('-')[0] == 'LowRes':
                    csize = 8
                    self.logger.info("finding spots")
                    spots = self.find_all_spots(ims_cal,lams,plot_im=False,thresh=90.0,sigma=1.2,medres=False)
                    self.logger.info("tracking spots sequentially")
                    spot_tracks = self.track_sequentially(spots,max_match_distance=3)
                    self.logger.info("removing duplicates and silos")
                    spot_tracks_u = self.remove_spot_dups(spot_tracks,lams,
                                                lmin=self.lmin,lmax=self.lmax,medres=False)
                    avgs = self.find_avg_spotpos_obs(spot_tracks_u,lams,self.lmin,self.lmax)
                    #avgs = self.find_avg_spotpos(spot_tracks_u,self.lmin,self.lmax,medres=False,show_plots=False)
                    avgs_new,tracks_new = self.remove_silos(avgs,spot_tracks_u,medres=False)
                    self.logger.info("registering lenslets to array")
                    final_posns, fitinfo = self.fit_lensarr_xy(avgs_new, tracks_new, lams)
                    #final_posns = self.get_lensarr_xy(avgs_new,maxdist=16,show_plots=False)
                    final_posns, tracks_new, avgs_new, sinfo = self.assign_spots_to_grid(spots, spot_tracks_u, lams, self.lmin, self.lmax, fitinfo['to_index'])
                    tracks_new, filled_set, kinds = self.fill_missing_spots(ims_cal,tracks_new,lams,self.lmin,self.lmax)
                    fluxcut = self.config.instrument.rectmat_fluxcut
                    posarr, spots = self.make_posarr(ims_cal,final_posns,tracks_new,show_plots=False,medres=False,cropsize=csize,cut=fluxcut)
                    posarr, spots = self.drop_lowsnr_spots(ims_cal,posarr,final_posns,filled_set,spots)


                    pyfits.writeto(output_dir+'/'+scmode+'_posarr.fits',np.array(posarr),overwrite=True)
                    write_debug(scmode+'_cropped_spotpsf_arr.fits',spots)


                if scmode.split('-')[0] == 'MedRes':
                    csize=16

                    """
                    commenting out during wavelength restructure
                    spots = self.find_all_spots(ims_cal,lams,plot_im=False,thresh=70.0,sigma=1.2,medres=True,mfilt=self.mfilt)
                    """

                    spots = self.find_all_spots_names(imnames,lams,mean,plot_im=False,thresh=70.0,sigma=1.2,medres=True,mfilt=self.mfilt)
                    #follow each spot along its trace model (y moves ~3.6 px/nm in K), so steps
                    #of any size (3.5 nm steps, gaps left by mono_exclude_waves) track the same;
                    #6 px covers the model error (<3 px) and is well below the 19 px pitch
                    tp = self.medres_trace_params(self.mfilt)
                    disp = (0.0, tp['chy']*tp['length']/(tp['lmax']-tp['lmin']))
                    spot_tracks = self.track_sequentially(spots, max_match_distance=6,
                                                          lams=lams, dispersion=disp)
                    self.logger.info("removing duplicates and silos")
                    #mfilt picks the trace model (defaults are K band)
                    spot_tracks_u = self.remove_spot_dups(spot_tracks,lams,lmin=self.lmin,lmax=self.lmax,medres=True,mfilt=self.mfilt)
                    avgs = self.find_avg_spotpos_model(spot_tracks_u,self.lmin,self.lmax,medres=True,mfilt=self.mfilt,show_plots=False)
                    avgs_new,tracks_new = self.remove_silos(avgs,spot_tracks_u,medres=True,show_plots=False)
                    self.logger.info("registering lenslets to array")
                    final_posns = self.get_medres_lensarr_xy(avgs_new,show_plots=False)
                    fluxcut = self.config.instrument.rectmat_fluxcut



                    #commenting out during wavelength restructure
                    #posarr, spots = self.make_posarr(ims_cal,final_posns,tracks_new,show_plots=False,medres=True,cropsize=csize,cut=fluxcut)
                    self.logger.info("making posarr")
                    posarr, spots = self.make_posarr_names(imnames,mean,final_posns,tracks_new,show_plots=False,medres=True,cropsize=csize,cut=fluxcut)

                    pyfits.writeto(output_dir+'/'+scmode+'_posarr.fits',np.array(posarr),overwrite=True)
                    write_debug(scmode+'_cropped_spotpsf_arr.fits',spots)


                centroids = self.get_centroids(spots)
                pyfits.writeto(output_dir+'/'+scmode+'_centroids.fits',np.array(centroids),overwrite=True)
                pyfits.writeto(output_dir+'/'+scmode+'_lams.fits',lams,overwrite=True)
                #stop

                #the plot shows only the middle master (lams are sorted), so load just that one
                diagnostic_path = mono_basic.plot_rectmat_centroid_diagnostics(
                    ims_cal=self.load_calim(imnames[len(imnames)//2],mean),
                    posarr=posarr,
                    centroids=centroids,
                    lams=lams,
                    output_dir=plot_dir,
                    filename=f"{scmode}_centroid_diagnostics.png",
                    max_traces=100,
                    show=False)
                #ank_mods, ank_resids, ank_modims, ank_resims = self.AnK_spot_model(spots, posarr, centroids, self.nx, self.ny, 3, npix=13)

                #A&K spot model: only its saved model/residual files use it, so it is
                #only computed when they are written
                if save_debug:
                    oversamp=2
                    ank_mods, ank_resids, ank_modims, ank_resims, ank_mods_sym, ank_resids_sym, ank_modims_sym, ank_resims_sym = self.AnK_spot_model_2(spots, centroids, self.ny, self.nx, oversamp, ims_cal, posarr)
                    write_debug(scmode+'_ank_modarr.fits',ank_mods)
                    write_debug(scmode+'_ank_resarr.fits',ank_resids)
                    write_debug(scmode+'_ank_modims.fits',ank_modims)
                    write_debug(scmode+'_ank_resims.fits',ank_resims)
                self.logger.info("fitting spot PSFs for interpolated rectmats")
                #full-detector model/residual images per wavelength only feed the debug files


                #fitarr,modims,resims,spotim_arr,modim_arr,resim_arr = self.fit_gauss_spots(ims_cal,posarr,show_plots=False,cropsize=csize,cut=fluxcut,full_images=save_debug)
                fitarr,modims,resims,spotim_arr,modim_arr,resim_arr = self.fit_gauss_spots_names(imnames,mean,posarr,show_plots=False,cropsize=csize,cut=fluxcut,full_images=save_debug)
                pyfits.writeto(output_dir+'/'+
                                scmode+'_gauss_spotpars.fits',np.array(fitarr),overwrite=True)
                write_debug(scmode+'_gauss_mod_detims.fits',modims)
                write_debug(scmode+'_gauss_res_detims.fits',resims)
                write_debug(scmode+'_cropped_spotgauss_arr.fits',modim_arr)
                write_debug(scmode+'_cropped_spotgaussres_arr.fits',resim_arr)

                self.logger.info("estimating spectral resolution from spot FWHMs")
                mono_fwhm_nm = float(self.config.instrument.mono_fwhm_nm or 0.0)
                res = self.estimate_spectral_resolution(np.array(fitarr), lams,
                                                        mono_fwhm_nm=mono_fwhm_nm,
                                                        fwhm_pix_range=(0.5, csize))
                self.write_spectral_resolution(res, lams, scmode, output_dir, plot_dir,
                                               mono_fwhm_nm=mono_fwhm_nm)

                # Anderson & King ePSFs per wavelength and field region (LowRes),
                # for profile-independent FWHMs; on unless the config turns it off
                ank_flag = self.config.instrument.lowres_ank_epsf
                ank_res = None
                if scmode.split('-')[0] == 'LowRes' and (ank_flag is None or ank_flag):
                    self.logger.info("building Anderson & King ePSFs for LowRes spots")
                    nreg = int(self.config.instrument.lowres_ank_nregions or 5)
                    ank_res = self.AnK_lowres_psfs(ims_cal, np.array(fitarr), lams, nreg=nreg)
                    self.write_AnK_lowres_psfs(ank_res, lams, scmode, output_dir, plot_dir)
                    self.logger.info("estimating spectral resolution from A&K ePSF FWHMs")
                    ank_rres = self.estimate_spectral_resolution_AnK(
                        ank_res, lams, mono_fwhm_nm=mono_fwhm_nm, fwhm_pix_range=(0.5, csize))
                    #R of every spot from the A&K ePSF at its detector position
                    ank_lres = self.AnK_lenslet_resolution(ank_res, lams, np.array(fitarr), lams,
                                                           mono_fwhm_nm=mono_fwhm_nm,
                                                           fwhm_pix_range=(0.5, csize))
                    self.write_spectral_resolution_AnK(ank_rres, lams, scmode, output_dir,
                                                       plot_dir, mono_fwhm_nm=mono_fwhm_nm,
                                                       gauss_res=res, lens_res=ank_lres)
                ank_lams, ank_crop_half = lams, 4

                # MedRes: the same per detector region (detector rows x x-bands), each
                # ePSF pooling medres_ank_nbin consecutive masters; masters read by name
                mank_flag = getattr(self.config.instrument, 'medres_ank_epsf', None)
                if scmode.split('-')[0] == 'MedRes' and (mank_flag is None or mank_flag):
                    self.logger.info("building Anderson & King ePSFs for MedRes spots")
                    #one ePSF per 17-spot slice (slicer mirror) unless medres_ank_per_slice = False
                    per_slice = getattr(self.config.instrument, 'medres_ank_per_slice', None)
                    per_slice = True if per_slice is None else bool(per_slice)
                    nregx = int(getattr(self.config.instrument, 'medres_ank_nxregions', None) or 3)
                    nbin = int(getattr(self.config.instrument, 'medres_ank_nbin', None) or 16)
                    ank_res = self.AnK_medres_psfs(imnames, mean, np.array(fitarr), lams,
                                                   nregx=nregx, nbin=nbin, per_slice=per_slice)
                    #ePSFs are per wavelength bin; boxes cover the longer MedRes spots
                    ank_lams, ank_crop_half = ank_res['lams'], 7
                    self.write_AnK_lowres_psfs(ank_res, ank_lams, scmode, output_dir, plot_dir)
                    self.logger.info("estimating spectral resolution from A&K ePSF FWHMs")
                    ank_rres = self.estimate_spectral_resolution_AnK(
                        ank_res, ank_lams, mono_fwhm_nm=mono_fwhm_nm, fwhm_pix_range=(0.5, csize))
                    #R of every spot from the A&K ePSF at its detector position
                    ank_lres = self.AnK_lenslet_resolution(ank_res, ank_lams, np.array(fitarr), lams,
                                                           mono_fwhm_nm=mono_fwhm_nm,
                                                           fwhm_pix_range=(0.5, csize))
                    self.write_spectral_resolution_AnK(ank_rres, ank_lams, scmode, output_dir,
                                                       plot_dir, mono_fwhm_nm=mono_fwhm_nm,
                                                       gauss_res=res, gauss_lams=lams,
                                                       lens_res=ank_lres)


                #spot model residuals (Gaussian fits, and the A&K ePSFs when built), all modes
                try:
                    self.logger.info("measuring spot model residuals")
                    sres = self.spot_residuals(spotim_arr, modim_arr, posarr, np.array(fitarr), lams,
                                               ank=ank_res, lams_ank=ank_lams)
                    self.write_spot_residuals(sres, lams, scmode, output_dir, plot_dir)
                except Exception as e:
                    self.logger.warning(f"spot residual diagnostics failed: {e}")

                if scmode.split('-')[0] == 'LowRes':
                    #lams_interp = lams
                    lams_interp = np.linspace(np.min(lams),np.max(lams),27)
                    interp_arr = self.interp_gauss_spots(lams,lams_interp,fitarr)
                #MedRes interpolated channels: medres_intp_nchan (one number, or a dict per
                #mode; default 600)
                nchan = self.medres_intp_nchan(scmode)
                if scmode.split('-')[0] == 'MedRes' and scmode not in ('MedRes-K','MedRes-L'):
                    lams_interp = np.linspace(self.lmin,self.lmax,nchan)
                    interp_arr = self.interp_gauss_spots(lams,lams_interp,fitarr)
                if scmode in ('MedRes-K','MedRes-L'):
                    lams_interp = np.linspace(self.lmin_i,self.lmax_i,nchan)
                    interp_arr = self.interp_gauss_spots(lams,lams_interp,fitarr)

                pyfits.writeto(output_dir+'/'+
                                scmode+'_gauss_spotpars_interpd.fits',np.array(interp_arr),
                                overwrite=True)
                self.logger.info("making chi2 rectmat")
                #OPT and chi2 rectmats from the spot crops (make_posarr's boxes, same values as
                #cutting them from the full images); the interpolated ones need no images
                C2_rmat = self.gen_C2_rectmat(spots,posarr,cut=fluxcut)
                self.logger.info("making interpolated chi2 rectmat")
                C2_rmat_interpd = self.gen_C2_rectmat_interpd(None,interp_arr,cut=fluxcut)
                self.logger.info("making interpolated optimal rectmat")
                OPT_rmat_interpd = self.gen_QL_rectmat_interpd(None,interp_arr,cut=fluxcut,method='optimal')
                self.logger.info("making interpolated optimal lsf and lflat")
                OPT_lsf_interpd,OPT_lflat_interpd = self.make_lsf_lflat_interpd(
                                                OPT_rmat_interpd,interp_arr,lams_interp,
                                                self.ny,self.nx,cut=fluxcut,cropsize=csize)
                self.logger.info("making optimal rectmat")
                OPT_rmat = self.gen_QL_rectmat(spots,posarr,cut=fluxcut,method='optimal')
                self.logger.info("making optimal lsf and lflat")
                #reads each master along the whole trace, so it loads them one at a time
                OPT_lsf,OPT_lflat = self.make_lsf_lflat(OPT_rmat,imnames,lams,
                                                self.ny,self.nx,meanim=mean)


                #medres_force_ysigma = True (MedRes) sets the interpolated Gaussian templates'
                #y sigma to their x sigma (crop_interpd_sparse_vals), so the files made from
                #them (OPT_intp / C2_intp rectmats, shifted or not, and the OPT_intp LSF and
                #lflat) get a _yforce suffix; OPT, C2 (spot crops) and A&K are unaffected
                yforce = (scmode.split('-')[0] == 'MedRes'
                          and getattr(self.config.instrument, 'medres_force_ysigma', False) == True)
                yf = '_yforce' if yforce else ''
                if yforce:
                    self.logger.info("medres_force_ysigma = True: interpolated Gaussian rectmats, "
                                     "LSF and lflat are written with the suffix _yforce")

                if self.config.instrument.rectmat_xshift!=0 or self.config.instrument.rectmat_yshift!=0:
                    interp_shift_arr = np.zeros(interp_arr.shape)
                    interp_shift_arr[:,:,:,:] = np.nan
                    interp_shift_arr[:] = interp_arr[:]
                    interp_shift_arr[:,:,:,1]+=self.config.instrument.rectmat_xshift
                    interp_shift_arr[:,:,:,2]+=self.config.instrument.rectmat_yshift

                    self.logger.info("making interpolated shifted optimal rectmat")
                    OPT_rmat_interpd_shift = self.gen_QL_rectmat_interpd(None,interp_shift_arr,cut=fluxcut,method='optimal')
                    self.logger.info("making interpolated shifted chi2 rectmat")
                    C2_rmat_interpd_shift = self.gen_C2_rectmat_interpd(None,interp_shift_arr,cut=fluxcut)
                    scbasic.save_rectmat(output_dir+'/'+
                                    scmode+'_C2_intp_rectmat'+yf+'_dx'+str(self.config.instrument.rectmat_xshift)+
                                    '_dy'+str(self.config.instrument.rectmat_yshift)+
                                    '.npz',C2_rmat_interpd_shift,
                                    cube_shape=interp_shift_arr.shape[:3])
                    scbasic.save_rectmat(output_dir+'/'+
                                    scmode+'_OPT_intp_rectmat'+yf+'_dx'+str(self.config.instrument.rectmat_xshift)+
                                    '_dy'+str(self.config.instrument.rectmat_yshift)+
                                    '.npz',OPT_rmat_interpd_shift,
                                    cube_shape=interp_shift_arr.shape[:3])


                self.logger.info("writing out rectmats")
                # cube_shape = (nlam, ny, nx): the same array shapes the
                # gen_*_rectmat functions use to lay out the flux axis
                scbasic.save_rectmat(output_dir+'/'+
                                scmode+'_OPT_rectmat.npz',OPT_rmat,
                                cube_shape=posarr.shape[:3])
                pyfits.writeto(output_dir+'/'+
                                scmode+'_OPT_lsf.fits',OPT_lsf,overwrite=True)
                pyfits.writeto(output_dir+'/'+
                                scmode+'_OPT_lflat.fits',OPT_lflat,overwrite=True)
                scbasic.save_rectmat(output_dir+'/'+
                                scmode+'_OPT_intp_rectmat'+yf+'.npz',OPT_rmat_interpd,
                                cube_shape=interp_arr.shape[:3])
                pyfits.writeto(output_dir+'/'+
                                scmode+'_OPT_intp_lsf'+yf+'.fits',OPT_lsf_interpd,
                                overwrite=True)
                pyfits.writeto(output_dir+'/'+
                                scmode+'_OPT_intp_lflat'+yf+'.fits',OPT_lflat_interpd,
                                overwrite=True)
                scbasic.save_rectmat(output_dir+'/'+
                                scmode+'_C2_rectmat.npz',C2_rmat,
                                cube_shape=posarr.shape[:3])
                scbasic.save_rectmat(output_dir+'/'+
                                scmode+'_C2_intp_rectmat'+yf+'.npz',C2_rmat_interpd,
                                cube_shape=interp_arr.shape[:3])
                pyfits.writeto(output_dir+'/'+scmode+'_intp_lams.fits',
                    lams_interp,overwrite=True)

                #which detector trace each lenslet (x, y) is in the rectmat used for
                #extraction (positions from its template centroids), over the middle master
                try:
                    self.logger.info("plotting rectmat lenslet map")
                    mono_basic.plot_rectmat_lenslet_map(
                        OPT_rmat_interpd, interp_arr.shape[:3], lams_interp,
                        self.load_calim(imnames[len(imnames)//2],mean), plot_dir,
                        f"{scmode}_rectmat_lenslet_map.png", title=scmode+' OPT_intp')
                except Exception as e:
                    self.logger.warning(f"rectmat lenslet map plot failed: {e}")

                #CHARIS-style optimal-extraction rectmat from the regional A&K ePSFs
                #(LowRes and MedRes; same rows/columns and wavelengths as *_OPT_intp_rectmat)
                ank_rmat_flag = getattr(self.config.instrument, 'ank_rectmat', None)
                if ank_res is not None and (ank_rmat_flag is None or ank_rmat_flag):
                    nsub = int(getattr(self.config.instrument, 'ank_rectmat_nsub', None) or 10)
                    self.logger.info(f"making A&K interpolated rectmat ({nsub} sub-wavelengths per channel)")
                    ank_rmat, ank_info = self.AnK_rectmat(ank_res, ank_lams, lams_interp, np.array(fitarr),
                                                          lams, nsub=nsub, crop_half=ank_crop_half)
                    scbasic.save_rectmat(output_dir+'/'+scmode+'_AnK_OPT_intp_rectmat.npz', ank_rmat,
                                         cube_shape=(len(lams_interp), self.ny, self.nx))
                    #A&K chi2 rectmat: the same template weights with rows and columns swapped,
                    #as gen_C2_rectmat_interpd is to gen_QL_rectmat_interpd
                    self.logger.info("making A&K interpolated chi2 rectmat")
                    scbasic.save_rectmat(output_dir+'/'+scmode+'_AnK_C2_intp_rectmat.npz',
                                         ank_rmat.T.tocsr(),
                                         cube_shape=(len(lams_interp), self.ny, self.nx))
                    xsh = self.config.instrument.rectmat_xshift
                    ysh = self.config.instrument.rectmat_yshift
                    if xsh != 0 or ysh != 0:
                        self.logger.info("making shifted A&K interpolated rectmats")
                        ank_rmat_s, _ = self.AnK_rectmat(ank_res, ank_lams, lams_interp, np.array(fitarr),
                                                         lams, nsub=nsub, shift=(xsh, ysh),
                                                         crop_half=ank_crop_half)
                        scbasic.save_rectmat(output_dir+'/'+scmode+'_AnK_OPT_intp_rectmat_dx'+str(xsh)+
                                             '_dy'+str(ysh)+'.npz', ank_rmat_s,
                                             cube_shape=(len(lams_interp), self.ny, self.nx))
                        scbasic.save_rectmat(output_dir+'/'+scmode+'_AnK_C2_intp_rectmat_dx'+str(xsh)+
                                             '_dy'+str(ysh)+'.npz', ank_rmat_s.T.tocsr(),
                                             cube_shape=(len(lams_interp), self.ny, self.nx))

            log_string = ProcessMonochrom.__module__
            self.logger.info(log_string)
        self.logger.info("+++++++++++++ Succesfully created and saved the rectification matrix")
        self.logger.info("####################### Calibration files associated with all the input files are created :)")

        return self.action.args
