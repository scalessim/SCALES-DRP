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
        names = df2['filename'][np.argsort(lams)]

        lams = np.sort(lams)
        ims = []
        for name in names:
            image = pyfits.getdata(output_dir+'/'+name,memmap=False)
            if True in np.isnan(image): self.logger.info('nan in image')
            ims.append(image)
        ims = np.array(ims)
        return ims, lams/1000.0

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


    def track_sequentially(self,spots,max_match_distance=3):
        """
        Track spots sequentially forward through images.

        Each spot in image 0 gets a spot_id. As we move to image 1, we:
        1. Match spots from image 0 to image 1
        2. Assign same spot_id to matched spots
        3. Give new spot_ids to unmatched spots (newly appearing)

        This allows tracking diagonal motion and handles appearing/disappearing spots.

        Returns:
        --------
        tracking_df : DataFrame
            DataFrame with tracking results
        """
        n_images = len(spots)


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
            distances, indices = tree.query(prev_coords)

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
        """


        #define dictionary for unique traces
        spots_u = {}
        #define dictionary for spot tracks that may
        #duplicate traces
        spots_d = {}

        uc = 0
        dc = 0
        for i in range(len(spot_tracks)):
            #if a spot track has locations for every wavelength
            #it corresponds to one lenslet and does not need to
            #be consolidated
            if len(spot_tracks[i]) == len(lams_u):
                spots_u[uc] = spot_tracks[i]
                uc += 1
            #if not, add this set of spot positions to the list
            #that may duplicate traces
            else:
                spots_d[dc] = spot_tracks[i]
                dc += 1

        #start looking for lists of spot positions that can
        #be consolidated (because they fall on one trace)

        #define list of spot track indices that are duplicates
        rem = []
        for i in range(len(spots_d)):
            keys = list(spots_d[i].keys())
            #grab spot's first x,y position and first wavelength
            x0,y0 = spots_d[i][keys[0]][0],spots_d[i][keys[0]][1]
            lam0 = spots_d[i][keys[0]][3]
            #get average wavelength and calculate expected x,y position
            #at that wavelength

            ####need to add a new get trace pos function for medium-res mode!!!
            if medres==False:
                lam = 0.5*(lmin+lmax)
                xi,yi = self.get_trace_pos(lam,x0,y0,lam0,chx,chy,lmin,lmax)

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
                xi = x0
                lam = 0.5*(lmin+lmax)
                yi,ymin,ymax=self.get_spot_yrange_medres(lam,y0=y0,lam0=lam0,chy=chy,lmin=lmin,lmax=lmax,length=length)
            #if my track isn't already flagged as a duplicate
            if i not in rem:
                merged = spots_d[i]
                #loop through all other tracks
                for j in range(len(spots_d)):
                    if i != j:
                        #if the other track isn't a duplicate
                        if j not in rem:
                            #get the other track's position at the central wavelegnth
                            keys = list(spots_d[j].keys())
                            x0,y0 = spots_d[j][keys[0]][0],spots_d[j][keys[0]][1]
                            lam0 = spots_d[j][keys[0]][3]
                            lam = 0.5*(lmin+lmax)
                            if medres==False:
                                xj,yj = self.get_trace_pos(lam,x0,y0,lam0,chx,chy,lmin,lmax)

                            if medres==True:
                                xj = x0
                                yj,ymin,ymax=self.get_spot_yrange_medres(lam,y0=y0,lam0=lam0,chy=chy,lmin=lmin,lmax=lmax,length=length)
                            #get the distance between the other spot's central wavelength
                            #position and the original one
                            dist = np.sqrt((xi-xj)**2+(yi-yj)**2)
                            #if the distance between the two central wavelength positions
                            #is smaller than the maximum allowed, then they're the same
                            if dist < maxdist:
                                #append j index to list to be removed
                                rem.append(j)
                                merged = merged | spots_d[j]
                #append merged track to dictionary containing unique lenslet tracks
                spots_u[uc] = merged
                uc += 1
                #append i in dex to list to be removed
                rem.append(i)
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
                        xs = np.max([0,int(x-cropsize/2)])
                        xe = np.min([int(x+cropsize/2),len(ims_cal[k])])
                        ys = np.max([0,int(y-cropsize/2)])
                        ye = np.min([int(y+cropsize/2),len(ims_cal[k])])
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
            for yind in range(len(spots[0])):
                for xind in range(len(spots[1])):
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
                    self.logger.info(yind,xind,' cent is nan')
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
            self.logger.info("FWHM_x (direct half-max crossing, oversampled image) =", fwhm_x)

            raws, models, residuals, scales, modim = self.downsample_model_and_residuals_2(ll, spots, centroids, combined, oversamp,
                                                                                      ny, nx, posarr,npix=16,normtype='sum')
            good = ~np.isnan(residuals).any(axis=(2,3))
            rms = np.sqrt(np.nanmean(residuals[good]**2))
            self.logger.info("n spots:", good.sum(), "  overall RMS residual:", rms)
            per_spot_rms_pct, overall_rms_pct = self.residual_rms_pct_2(raws, residuals)
            self.logger.info("overall RMS residual (%% of peak, pooled) = %.2f%%" % overall_rms_pct)

            gauss = Gaussian2D(amplitude=1, x_mean=len(combined)/2.0-0.5, y_mean=len(combined)/2.0-0.5,
                               x_stddev=fwhm_x/2.355*oversamp, y_stddev=fwhm_x/2.355*oversamp,
                               theta=None, cov_matrix=None)
            xx,yy=np.meshgrid(range(len(combined)),range(len(combined[0])))
            filt = gauss(xx,yy)
            combined2 = combined*filt
            fwhm_x = self.halfmax_fwhm_x(combined2, oversamp)
            self.logger.info("FWHM_x (direct half-max crossing, filtered oversampled image) =", fwhm_x)
            raws, models_sym, residuals_sym, scales_sym, modim_sym = self.downsample_model_and_residuals_2(ll, spots, centroids, combined2, oversamp,
                                                                                              ny, nx, posarr,npix=16,normtype='max')
            good = ~np.isnan(residuals_sym).any(axis=(2,3))
            rms = np.sqrt(np.nanmean(residuals_sym[good]**2))
            self.logger.info("n spots:", good.sum(), "  overall RMS residual (filtered):", rms)
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
            self.logger.info("FWHM_x (direct half-max crossing, oversampled image) =", fwhm_x)
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
            self.logger.info("n spots:", good.sum(), "  overall RMS residual:", rms)
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
        hdus.append(pyfits.BinTableHDU.from_columns(cols, name='REGIONS'))
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






    def fit_gauss_spots(self,calims,posarr,cropsize=10,show_plots=False,cut=0.1):
        fitarr = np.zeros([posarr.shape[0],posarr.shape[1],posarr.shape[2],6])
        fitarr[:,:,:,:] = np.nan
        modims = []
        spotim_arr = np.zeros([posarr.shape[0],posarr.shape[1],posarr.shape[2],cropsize,cropsize])
        modim_arr = np.zeros([posarr.shape[0],posarr.shape[1],posarr.shape[2],cropsize,cropsize])
        resim_arr = np.zeros([posarr.shape[0],posarr.shape[1],posarr.shape[2],cropsize,cropsize])
        for ll in range(len(calims)):
            self.logger.info("image "+str(ll)+" of "+str(len(calims)))
            modim = np.zeros(calims[ll].shape)
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
            modims.append(modim)

        resims = calims-modims
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
            vmin, vmax = np.percentile(allr, [2, 98]) if allr.size else (0, 1)
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
        cropped = image[ys:ye,xs:xe]

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


    def gen_rectmat_inds(self,calims,posarr,cut=0.05,method='optimal'):

        """
        Function to generate row and column indices for sparse matrix
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
                        vals = self.crop_sparse_vals(calims[ll],xs,xe,ys,ye,cut=cut,method=method)
                        for i in range(len(vals)):
                            if vals[i] > 0:
                                matvals.append(vals[i])
                                #matvals.append(1.0)
                                matcolinds.append(flatinds[i])
                                matrowinds.append(lensx+lensy*posarr.shape[2]+ll*posarr.shape[1]*posarr.shape[2])
        return matrowinds, matcolinds, matvals


    def gen_QL_rectmat(self,calims,posarr,cut=0.05,method='optimal',interp=False):
        """
        Function to generate rectmat from cube of cal unit images.
        """

        matrowinds,matcolinds,matvals = self.gen_rectmat_inds(calims,posarr,cut=cut,method=method)
        rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(np.prod(posarr.shape[:3]),np.prod(calims[0].shape)))
        return rmat

    def gen_QL_rectmat_interpd(self,calims,interp_arr,cut=0.05,method='optimal',interp=False):
        """
        Function to generate rectmat from cube of cal unit images.
        """

        matrowinds,matcolinds,matvals = self.gen_rectmat_inds_interpd(interp_arr,cut=cut,method=method)
        rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(np.prod(interp_arr.shape[:3]),np.prod(calims[0].shape)))
        return rmat



    def gen_c2_rectmat_inds(self,calims,posarr,cut=0.05):

        """
        Function to generate row and column indices for sparse matrix
        for all 108 x 108 lenslets and wavelengths.
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
                        vals = self.crop_sparse_vals(calims[ll],xs,xe,ys,ye,cut=cut,method='optimal')
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



    def gen_C2_rectmat(self,calims,posarr,cut=0.05):
        """
        Function to generate rectmat from cube of cal unit images.
        """

        matrowinds,matcolinds,matvals = self.gen_c2_rectmat_inds(calims,posarr,cut=cut)
        rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(np.prod(calims[0].shape),np.prod(posarr.shape[:3])))
        return rmat

    def gen_C2_rectmat_interpd(self,calims,interp_arr,cut=0.05):
        """
        Function to generate rectmat from cube of cal unit images.
        """

        matrowinds,matcolinds,matvals = self.gen_c2_rectmat_inds_interpd(interp_arr,cut=cut)
        rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(np.prod(calims[0].shape),np.prod(interp_arr.shape[:3]),))
        return rmat

    def get_medres_lensarr_xy(self,avgs,show_plots=True):
        inds0 = np.array(range(len(avgs)),dtype='int')
        dist=26
        row1 = avgs[np.where(avgs[:,1] < np.min(avgs[:,1]) + dist)]
        row1inds = inds0[np.where(avgs[:,1] < np.min(avgs[:,1]) + dist)]
        rem = np.delete(avgs,np.where(avgs[:,1] < np.min(avgs[:,1]) + dist),axis=0)
        indsrem = np.delete(inds0,np.where(avgs[:,1] < np.min(avgs[:,1]) + dist),axis=0)
        row2 = rem[np.where(rem[:,1] < np.min(rem[:,1])+dist)]
        row2inds = indsrem[np.where(rem[:,1] < np.min(rem[:,1])+dist)]
        row3 = np.delete(rem,np.where(rem[:,1] < np.min(rem[:,1])+dist),axis=0)
        row3inds = np.delete(indsrem,np.where(rem[:,1] < np.min(rem[:,1])+dist),axis=0)

        if show_plots==True:
            f = plt.figure(clear=True)
            plt.scatter(row1[:,0],row1[:,1],label=str(len(row1)))
            plt.scatter(row2[:,0],row2[:,1],label=str(len(row2)))
            plt.scatter(row3[:,0],row3[:,1],label=str(len(row3)))
            plt.legend()
            plt.show()
        #stop

        posns_idx = np.zeros([3,102,3])
        posns_idx[:,:,:] = np.nan
        posns_pix = np.zeros([3,102,5])
        posns_pix[:,:,:] = np.nan


        row1sortx = row1[np.argsort(row1[:,0])]
        row1indsortx = row1inds[np.argsort(row1[:,0])]
        row2sortx = row2[np.argsort(row2[:,0])]
        row2indsortx = row2inds[np.argsort(row2[:,0])]
        row3sortx = row3[np.argsort(row3[:,0])]
        row3indsortx = row3inds[np.argsort(row3[:,0])]


        for yc,rowarr,rowind in [[0,row1sortx,row1indsortx],
                                 [1,row2sortx,row2indsortx],
                                 [2,row3sortx,row3indsortx]]:
            if len(rowarr) == 102:
                for i in range(len(row1sortx)):
                    posns_idx[yc,i] = [i,yc,rowind[i]]
                    posns_pix[yc,i] = [i,yc,rowarr[i,0],rowarr[i,1],rowind[i]]
            if len(rowarr) < 102:

                minx = np.min(rowarr)
                maxx = np.max(rowarr)
                if minx < 50 and maxx < 1998:
                    #print('spots are off on the left!')
                    for i in range(102-len(rowarr),102):
                        posns_idx[yc,i] = [i,yc,rowind[i-(102-len(rowarr))]]
                        posns_pix[yc,i] = [i,yc,
                                            rowarr[i-(102-len(rowarr)),0],
                                            rowarr[i-(102-len(rowarr)),1],
                                            rowind[i-(102-len(rowarr))]]
                elif minx > 50 and maxx > 1998:
                    #print('spots are off on the right!')
                    for i in range(len(rowarr)):
                        posns_idx[yc,i] = [i,yc,rowind[i]]
                        posns_pix[yc,i] = [i,yc,rowarr[i,0],rowarr[i,1],rowind[i]]
                else:
                    self.logger.info('Uh oh I cant tell which direction we lost a spot!!')

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

    def make_lsf_lflat(self,rmat,ims_cal,lams,ny,nx):
        lsfcube = []
        lensflatcube = []
        for ii in range(len(ims_cal)):
            scube = np.array(rmat*ims_cal[ii].flatten()).reshape([len(lams),ny,nx])
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
            self.logger.info("Skipping rectification matrix generation.")
            return self.action.args

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
            ims, lams = self.parse_files(df,scmode,output_dir)

            if len(ims) > 0:
                self.set_lamlimits(scmode)
                ims_cal_tmp = self.monochrom_bksub(ims)
                if self.config.instrument.apply_destripe==True:
                    ims_cal = np.array([self.masked_row_destripe(im)[0] for im in ims_cal_tmp])
                else:
                    ims_cal = ims_cal_tmp
                pyfits.writeto(output_dir+'/'+scmode+'_calim_stack.fits',ims_cal,overwrite=True)
                self.logger.info("done loading images for "+str(self.scmode))
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
                    pyfits.writeto(output_dir+'/'+scmode+'_cropped_spotpsf_arr.fits',np.array(spots),overwrite=True)


                if scmode.split('-')[0] == 'MedRes':
                    csize=16
                    spots = self.find_all_spots(ims_cal,lams,plot_im=False,thresh=70.0,sigma=1.2,medres=True,mfilt=self.mfilt)
                    spot_tracks = self.track_sequentially(spots, max_match_distance=13)
                    self.logger.info("removing duplicates and silos")
                    spot_tracks_u = self.remove_spot_dups(spot_tracks,lams,lmin=self.lmin,lmax=self.lmax,medres=True)
                    avgs = self.find_avg_spotpos_model(spot_tracks_u,self.lmin,self.lmax,medres=True,show_plots=False)
                    avgs_new,tracks_new = self.remove_silos(avgs,spot_tracks_u,medres=True,show_plots=False)
                    self.logger.info("registering lenslets to array")
                    final_posns = self.get_medres_lensarr_xy(avgs_new,show_plots=False)
                    fluxcut = self.config.instrument.rectmat_fluxcut
                    posarr, spots = self.make_posarr(ims_cal,final_posns,tracks_new,show_plots=False,medres=True,cropsize=csize,cut=fluxcut)
                    pyfits.writeto(output_dir+'/'+scmode+'_posarr.fits',np.array(posarr),overwrite=True)
                    pyfits.writeto(output_dir+'/'+scmode+'_cropped_spotpsf_arr.fits',np.array(spots),overwrite=True)


                centroids = self.get_centroids(spots)
                pyfits.writeto(output_dir+'/'+scmode+'_centroids.fits',np.array(centroids),overwrite=True)
                pyfits.writeto(output_dir+'/'+scmode+'_lams.fits',lams,overwrite=True)
                #stop
                diagnostic_path = mono_basic.plot_rectmat_centroid_diagnostics(
                    ims_cal=ims_cal,
                    posarr=posarr,
                    centroids=centroids,
                    lams=lams,
                    output_dir=plot_dir,
                    filename=f"{scmode}_centroid_diagnostics.png",
                    max_traces=100,
                    show=False)
                #ank_mods, ank_resids, ank_modims, ank_resims = self.AnK_spot_model(spots, posarr, centroids, self.nx, self.ny, 3, npix=13)

                oversamp=2
                ank_mods, ank_resids, ank_modims, ank_resims, ank_mods_sym, ank_resids_sym, ank_modims_sym, ank_resims_sym = self.AnK_spot_model_2(spots, centroids, self.ny, self.nx, oversamp, ims_cal, posarr)

                pyfits.writeto(output_dir+'/'+
                                scmode+'_ank_modarr.fits',np.array(ank_mods),
                                overwrite=True)
                pyfits.writeto(output_dir+'/'+
                                scmode+'_ank_resarr.fits',np.array(ank_resids),
                                overwrite=True)
                pyfits.writeto(output_dir+'/'+
                                scmode+'_ank_modims.fits',np.array(ank_modims),
                                overwrite=True)
                pyfits.writeto(output_dir+'/'+
                                scmode+'_ank_resims.fits',np.array(ank_resims),
                                overwrite=True)
                #stop
                self.logger.info("fitting spot PSFs for interpolated rectmats")
                fitarr,modims,resims,spotim_arr,modim_arr,resim_arr = self.fit_gauss_spots(ims_cal,posarr,show_plots=False,cropsize=csize,cut=fluxcut)
                pyfits.writeto(output_dir+'/'+
                                scmode+'_gauss_spotpars.fits',np.array(fitarr),overwrite=True)
                pyfits.writeto(output_dir+'/'+
                                scmode+'_gauss_mod_detims.fits',np.array(modims),overwrite=True)
                pyfits.writeto(output_dir+'/'+
                                scmode+'_gauss_res_detims.fits',np.array(resims),overwrite=True)
                pyfits.writeto(output_dir+'/'+
                                scmode+'_cropped_spotgauss_arr.fits',np.array(modim_arr),
                                overwrite=True)
                pyfits.writeto(output_dir+'/'+
                                scmode+'_cropped_spotgaussres_arr.fits',np.array(resim_arr),
                                overwrite=True)

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
                if scmode.split('-')[0] == 'LowRes' and (ank_flag is None or ank_flag):
                    self.logger.info("building Anderson & King ePSFs for LowRes spots")
                    nreg = int(self.config.instrument.lowres_ank_nregions or 3)
                    ank_res = self.AnK_lowres_psfs(ims_cal, np.array(fitarr), lams, nreg=nreg)
                    self.write_AnK_lowres_psfs(ank_res, lams, scmode, output_dir, plot_dir)


                if scmode.split('-')[0] == 'LowRes':
                    #lams_interp = lams
                    lams_interp = np.linspace(np.min(lams),np.max(lams),27)
                    interp_arr = self.interp_gauss_spots(lams,lams_interp,fitarr)
                if scmode.split('-')[0] == 'MedRes' and scmode!='MedRes-K':
                    lams_interp = np.linspace(self.lmin,self.lmax,600)
                    interp_arr = self.interp_gauss_spots(lams,lams_interp,fitarr)
                if scmode == 'MedRes-K':
                    lams_interp = np.linspace(self.lmin_i,self.lmax_i,600)
                    interp_arr = self.interp_gauss_spots(lams,lams_interp,fitarr)

                pyfits.writeto(output_dir+'/'+
                                scmode+'_gauss_spotpars_interpd.fits',np.array(interp_arr),
                                overwrite=True)
                self.logger.info("making chi2 rectmat")
                C2_rmat = self.gen_C2_rectmat(ims_cal,posarr,cut=fluxcut)
                self.logger.info("making interpolated chi2 rectmat")
                C2_rmat_interpd = self.gen_C2_rectmat_interpd(ims_cal,interp_arr,cut=fluxcut)
                self.logger.info("making interpolated optimal rectmat")
                OPT_rmat_interpd = self.gen_QL_rectmat_interpd(ims_cal,interp_arr,cut=fluxcut,method='optimal')
                self.logger.info("making interpolated optimal lsf and lflat")
                OPT_lsf_interpd,OPT_lflat_interpd = self.make_lsf_lflat_interpd(
                                                OPT_rmat_interpd,interp_arr,lams_interp,
                                                self.ny,self.nx,cut=fluxcut,cropsize=csize)
                self.logger.info("making optimal rectmat")
                OPT_rmat = self.gen_QL_rectmat(ims_cal,posarr,cut=fluxcut,method='optimal')
                self.logger.info("making optimal lsf and lflat")
                OPT_lsf,OPT_lflat = self.make_lsf_lflat(OPT_rmat,ims_cal,lams,
                                                self.ny,self.nx)


                if self.config.instrument.rectmat_xshift!=0 or self.config.instrument.rectmat_yshift!=0:
                    interp_shift_arr = np.zeros(interp_arr.shape)
                    interp_shift_arr[:,:,:,:] = np.nan
                    interp_shift_arr[:] = interp_arr[:]
                    interp_shift_arr[:,:,:,1]+=self.config.instrument.rectmat_xshift
                    interp_shift_arr[:,:,:,2]+=self.config.instrument.rectmat_yshift

                    self.logger.info("making interpolated shifted optimal rectmat")
                    OPT_rmat_interpd_shift = self.gen_QL_rectmat_interpd(ims_cal,interp_shift_arr,cut=fluxcut,method='optimal')
                    self.logger.info("making interpolated shifted chi2 rectmat")
                    C2_rmat_interpd_shift = self.gen_C2_rectmat_interpd(ims_cal,interp_shift_arr,cut=fluxcut)
                    scbasic.save_rectmat(output_dir+'/'+
                                    scmode+'_C2_intp_rectmat_dx'+str(self.config.instrument.rectmat_xshift)+
                                    '_dy'+str(self.config.instrument.rectmat_yshift)+
                                    '.npz',C2_rmat_interpd_shift,
                                    cube_shape=interp_shift_arr.shape[:3])
                    scbasic.save_rectmat(output_dir+'/'+
                                    scmode+'_OPT_intp_rectmat_dx'+str(self.config.instrument.rectmat_xshift)+
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
                                scmode+'_OPT_intp_rectmat.npz',OPT_rmat_interpd,
                                cube_shape=interp_arr.shape[:3])
                pyfits.writeto(output_dir+'/'+
                                scmode+'_OPT_intp_lsf.fits',OPT_lsf_interpd,
                                overwrite=True)
                pyfits.writeto(output_dir+'/'+
                                scmode+'_OPT_intp_lflat.fits',OPT_lflat_interpd,
                                overwrite=True)
                scbasic.save_rectmat(output_dir+'/'+
                                scmode+'_C2_rectmat.npz',C2_rmat,
                                cube_shape=posarr.shape[:3])
                scbasic.save_rectmat(output_dir+'/'+
                                scmode+'_C2_intp_rectmat.npz',C2_rmat_interpd,
                                cube_shape=interp_arr.shape[:3])
                pyfits.writeto(output_dir+'/'+scmode+'_intp_lams.fits',
                    lams_interp,overwrite=True)

            log_string = ProcessMonochrom.__module__
            self.logger.info(log_string)
        self.logger.info("+++++++++++++ Succesfully created and saved the rectification matrix")
        self.logger.info("####################### Calibration files associated with all the input files are created :)")

        return self.action.args
