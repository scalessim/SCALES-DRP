from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.core.scales_pkg_resources import get_resource_path
from scalesdrp.primitives.scales_basic import fits_headers_to_dataframe
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import binary_dilation
#from scalesdrp.primitives.scales_file_primitives import scales_fits_writer

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
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from astropy.modeling.functional_models import Gaussian2D, Const2D
from astropy.modeling import fitting



class ProcessMonochrom(BasePrimitive):
    """
    Estimate the psf centroid of all the calib images images and save
    in two pickle file one for x and one for y centroid values. Currently
    assumes wavelengths will be in filenames. Need to replace that with
    header keywords instead. Also the location of the calib filesneed to fix.
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.redux_dir = os.path.join(self.action.args.dirname, "redux")
        self.logger = context.pipeline_logger
        self.context = context

    def parse_files(self,df,scmode):
        """
        """
        df2 = df[df['IFSMODE'] == scmode]
        if len(df2) == 0:
            return [], []

        det_config = np.unique(df2['MCLOCK'])[0]

        calibfilepath = self.context.calib_file_path
        package = __name__.split('.')[0]
        calib_path = str(get_resource_path(package, calibfilepath))+'/'
        if det_config =='9.0 MHz': #fast0.6
            flat = pyfits.getdata(calib_path+self.context.flat_ifs_fast0p6)
        elif det_config =='5.0 MHz': #fast1.0
            flat = pyfits.getdata(calib_path+self.context.flat_ifs_fast1)
        elif det_config =='5.0 MHz': #fast1.0
            flat = pyfits.getdata(calib_path+self.context.flat_ifs_fast1)

        lams = df2['MONOWAVE']
        names = df2['filename'][np.argsort(lams)]

        lams = np.sort(lams)
        ims = []
        for name in names:
            image = pyfits.getdata(self.redux_dir+'/'+name,memmap=False)
            if True in np.isnan(image): print('nan in image')
            ims.append(image/flat)
            #ims.append(pyfits.getdata(self.redux_dir+'/'+name))
            #ims.append(pyfits.getdata(self.redux_dir+'/'+name)/flat)
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

            print(f"Found {len(coords_yx)} spots in image {ii}")
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


        print(f"\nSequential tracking through {n_images} images...")

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

        print(f"  Image 0: {n_spots_0} spots initialized")

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
                print(f"  Image {img_idx}: {n_curr_spots} new spots (no previous spots to match)")
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

            print(f"  Image {img_idx}: {n_matched} matched, {n_lost} lost, {n_new} new (total: {n_curr_spots})")

            # Update for next iteration
            prev_coords = curr_coords
            prev_spot_ids = curr_spot_ids

        total_spots = len(spot_tracks)
        print(f"\nTotal unique spots tracked: {total_spots}")

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

    def find_avg_spotpos(self,spot_tracks_u,lmin,lmax,chx=1,chy=1,medres=False,mfilt='imgK',show_plots=False):
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


    def make_posarr(self,ims_cal,final_posns,spot_tracks_u,medres=False,show_plots=False,cropsize=10,cut=0.01):
        maxy = final_posns.shape[0]
        maxx = final_posns.shape[1]
        if medres==False:
            sizex = 112
            sizey = 112
            diffx = sizex-maxx
            diffy = sizey-maxy
        if medres==True:
            sizex=18
            sizey=17
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
                    print(yind,xind,' cent is nan')
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
            print("FWHM_x (direct half-max crossing, oversampled image) =", fwhm_x)

            raws, models, residuals, scales, modim = self.downsample_model_and_residuals_2(ll, spots, centroids, combined, oversamp,
                                                                                      ny, nx, posarr,npix=16,normtype='sum')
            good = ~np.isnan(residuals).any(axis=(2,3))
            rms = np.sqrt(np.nanmean(residuals[good]**2))
            print("n spots:", good.sum(), "  overall RMS residual:", rms)
            per_spot_rms_pct, overall_rms_pct = self.residual_rms_pct_2(raws, residuals)
            print("overall RMS residual (%% of peak, pooled) = %.2f%%" % overall_rms_pct)

            gauss = Gaussian2D(amplitude=1, x_mean=len(combined)/2.0-0.5, y_mean=len(combined)/2.0-0.5,
                               x_stddev=fwhm_x/2.355*oversamp, y_stddev=fwhm_x/2.355*oversamp,
                               theta=None, cov_matrix=None)
            xx,yy=np.meshgrid(range(len(combined)),range(len(combined[0])))
            filt = gauss(xx,yy)
            combined2 = combined*filt
            fwhm_x = self.halfmax_fwhm_x(combined2, oversamp)
            print("FWHM_x (direct half-max crossing, filtered oversampled image) =", fwhm_x)
            raws, models_sym, residuals_sym, scales_sym, modim_sym = self.downsample_model_and_residuals_2(ll, spots, centroids, combined2, oversamp,
                                                                                              ny, nx, posarr,npix=16,normtype='max')
            good = ~np.isnan(residuals_sym).any(axis=(2,3))
            rms = np.sqrt(np.nanmean(residuals_sym[good]**2))
            print("n spots:", good.sum(), "  overall RMS residual (filtered):", rms)
            per_spot_rms_pct_sym, overall_rms_pct_sym = self.residual_rms_pct_2(raws, residuals_sym)
            print("overall RMS residual (%% of peak, pooled, filtered) = %.2f%%" % overall_rms_pct_sym)


            yind=10
            xind=8
            xc,xs,xe,yc,ys,ye,intens = posarr[ll,yind,xind]
            xc = int(xc)
            yc = int(yc)
            xs = int(xs)
            xe = int(xe)
            ys = int(ys)
            ye = int(ye)

            """
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
            #plt.imshow(combined)
            #plt.colorbar()
            #plt.show()
            fwhm_x = self.halfmax_fwhm_x(combined, oversamp)
            print("FWHM_x (direct half-max crossing, oversampled image) =", fwhm_x)
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
                        print(ll,lensy,lensx,ys,ye,xs,xe,models_stampsize.shape)
                        modim[ys:ye,xs:xe]+=models_stampsize[lensy*self.nx + lensx][:ye-ys,:xe-xs] ##this matches how the spot psf thumbnails are constructed
                        resim[ys:ye,xs:xe]+=residuals_stampsize[lensy*self.nx + lensx][:ye-ys,:xe-xs] ##this matches how the spot psf thumbnails are constructed
                        model_arr[ll,lensy,lensx] = models[lensy*self.nx + lensx]
                        print(models.shape,np.unique(models[lensy*self.nx + lensx]))
                        resid_arr[ll,lensy,lensx] = residuals[lensy*self.nx + lensx]
            modims.append(modim)
            resims.append(resim)
            good = ~np.isnan(residuals).any(axis=(1,2))
            rms = np.sqrt(np.nanmean(residuals[good]**2))
            print("n spots:", good.sum(), "  overall RMS residual:", rms)
            per_spot_rms_pct, overall_rms_pct = self.residual_rms_pct(raws, residuals)
            print("overall RMS residual (%% of peak, pooled) = %.2f%%" % overall_rms_pct)
        return model_arr, resid_arr, np.array(modims), np.array(resims)



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
            print(ll)
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
                        print(f"  [warn] fit did not converge cleanly at "
                              f"lensx={lensx}, lensy={lensy}, ll={ll} "
                              f"(ierr={fitter.fit_info.get('ierr')})")

                    if show_plots:
                        print("--- Fit Results (value +/- 1-sigma) ---")
                        print(f"Amplitude: {gauss_part.amplitude.value:.2f} +/- {amp_err:.2f}")
                        print(f"X Center:  {gauss_part.x_mean.value:.2f} +/- {xmean_err:.2f}")
                        print(f"Y Center:  {gauss_part.y_mean.value:.2f} +/- {ymean_err:.2f}")
                        print(f"X Sigma:   {gauss_part.x_stddev.value:.2f} +/- {xstd_err:.2f}")
                        print(f"Y Sigma:   {gauss_part.y_stddev.value:.2f} +/- {ystd_err:.2f}")
                        print(f"Theta:     {gauss_part.theta.value:.2f} +/- {theta_err:.2f} rad")
                        print(f"Background:{bkg_part.amplitude.value:.4f} +/- {bkg_err:.4f}")

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

                                #print(ll,lensy,lensx)
                                #print(fitted_model.x_mean.value,fitted_model.y_mean.value)
                                if True in np.isnan(fitarr[ll,lensy,lensx]):
                                    print('nans in '+str(ll)+' '+str(lensy)+' '+str(lensx))
                                if show_plots==True:
                                    print("--- Fit Results ---")
                                    print(f"Amplitude: {fitted_model.amplitude.value:.2f}")
                                    print(f"X Center:  {fitted_model.x_mean.value:.2f}")
                                    print(f"Y Center:  {fitted_model.y_mean.value:.2f}")
                                    print(f"X Sigma:   {fitted_model.x_stddev.value:.2f}")
                                    print(f"Y Sigma:   {fitted_model.y_stddev.value:.2f}")
                                    print(f"Theta:     {fitted_model.theta.value:.2f} rad")

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
                        print(lensy,lensx,tofit)
                        lamsfit = lams_in[np.where(np.isnan(tofit)==False)]
                        tofit = tofit[np.where(np.isnan(tofit)==False)]
                        if method=='poly':
                            res=np.polyfit(lamsfit,tofit,3)
                            fint = np.polynomial.polynomial.Polynomial(res[::-1])
                        if method=='interp':
                            fint = interp1d(lamsfit,tofit)
                        gausspars_new = fint(lams_des)
                        print(gausspars_new)
                        print('==========================================')
                        if show_plots==True:
                            f = plt.figure(clear=True)
                            plt.scatter(lams_des,gausspars_new)
                            plt.scatter(lams_in,gausspars[:,i])
                            plt.plot(lams_des,gausspars_new)
                            plt.show()
                        interp_arr[:,lensy,lensx,i] = gausspars_new


        return interp_arr


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
            print('doing wavelength slice '+str(ll))
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

        #print('doing c2 rectmat lowres')
        matrowinds,matcolinds,matvals = self.gen_c2_rectmat_inds(calims,posarr,cut=cut)
        #print(len(matrowinds),len(matcolinds),len(matvals))
        rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(np.prod(calims[0].shape),np.prod(posarr.shape[:3])))
        return rmat

    def gen_C2_rectmat_interpd(self,calims,interp_arr,cut=0.05):
        """
        Function to generate rectmat from cube of cal unit images.
        """

        matrowinds,matcolinds,matvals = self.gen_c2_rectmat_inds_interpd(interp_arr,cut=cut)
        #print(len(matrowinds),len(matcolinds),len(matvals))
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
                    print('spots are off on the left!')
                    for i in range(102-len(rowarr),102):
                        posns_idx[yc,i] = [i,yc,rowind[i-(102-len(rowarr))]]
                        posns_pix[yc,i] = [i,yc,
                                            rowarr[i-(102-len(rowarr)),0],
                                            rowarr[i-(102-len(rowarr)),1],
                                            rowind[i-(102-len(rowarr))]]
                elif minx > 50 and maxx > 1998:
                    print('spots are off on the right!')
                    for i in range(len(rowarr)):
                        posns_idx[yc,i] = [i,yc,rowind[i]]
                        posns_pix[yc,i] = [i,yc,rowarr[i,0],rowarr[i,1],rowind[i]]
                else:
                    print('I cant tell which direction we lost a spot!!')

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
        if scmode == "LowRes-KLM":
            self.lmin = 2.0
            self.lmax = 5.2
            self.nx = 112
            self.ny = 112
        if scmode == 'LowRes-K':
            self.lmin = 1.95
            self.lmax = 2.45
            self.nx = 112
            self.ny = 112
        if scmode == 'MedRes-K':
            self.lmin = 1.95
            self.lmax = 2.45
            self.mfilt = 'imgK'
            self.lmin_i = 2.02
            self.lmax_i = 2.38
            self.nx = 18
            self.ny = 17
        if scmode == 'LowRes-L':
            self.lmin = 2.9
            self.lmax = 4.15
            self.nx = 112
            self.ny = 112
        if scmode == 'MedRes-L':
            self.lmin = 2.9
            self.lmax = 4.15
            self.nx = 18
            self.ny = 17
        if scmode == 'LowRes-M':
            self.lmin = 4.5
            self.lmax = 5.2
            self.nx = 112
            self.ny = 112
        if scmode == 'MedRes-M':
            self.lmin = 4.5
            self.lmax = 5.2
            self.nx = 18
            self.ny = 17
        if scmode == 'LowRes-KL':
            self.lmin = 2.0
            self.lmax = 4.0
            self.nx = 112
            self.ny = 112
        if scmode == 'LowRes-Ls':
            self.lmin = 3.1
            self.lmax = 3.5
            self.nx = 112
            self.ny = 112

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
            print('scube '+str(ii)+' of '+str(len(imgs)))
            print(rmat.shape)
            print(imgs[ii].shape)
            scube = np.array(rmat*imgs[ii].flatten()).reshape([len(lams),ny,nx])
            #scube_lsf = scube/np.max(scube,axis=0)
            lflat = scube[ii]
            lsfcube.append(scube)
            lensflatcube.append(lflat)
        lensflat = np.median(lensflatcube,axis=0)
        return np.array(lsfcube),np.array(lensflat)



    def _perform(self):

        self.logger.info("Process Monochromator Images")
        df = fits_headers_to_dataframe(self.redux_dir,pattern="*mcalunit.fits")


        for scmode in ['LowRes-KLM','LowRes-K','LowRes-L',
                       'LowRes-M','LowRes-KL','LowRes-Ls',
                       'MedRes-K','MedRes-L','MedRes-M']:
            self.scmode=scmode
            self.logger.info("Parsing files for "+str(self.scmode))
            ims, lams = self.parse_files(df,scmode)
            if len(ims) > 0:
                self.set_lamlimits(scmode)
                ims_cal_tmp = self.monochrom_bksub(ims)
                if self.config.instrument.apply_destripe==True:
                    ims_cal = np.array([self.masked_row_destripe(im)[0] for im in ims_cal_tmp])
                else:
                    ims_cal = ims_cal_tmp
                self.logger.info("done loading images for "+str(self.scmode))
                print(ims_cal.shape)
                if scmode.split('-')[0] == 'LowRes':
                    csize = 8
                    self.logger.info("finding spots")
                    spots = self.find_all_spots(ims_cal,lams,plot_im=False,thresh=90.0,sigma=1.2,medres=False)
                    self.logger.info("tracking spots sequentially")
                    spot_tracks = self.track_sequentially(spots,max_match_distance=3)
                    self.logger.info("removing duplicates and silos")
                    spot_tracks_u = self.remove_spot_dups(spot_tracks,lams,
                                                lmin=self.lmin,lmax=self.lmax,medres=False)
                    avgs = self.find_avg_spotpos(spot_tracks_u,self.lmin,self.lmax,medres=False,show_plots=False)
                    avgs_new,tracks_new = self.remove_silos(avgs,spot_tracks_u,medres=False)
                    self.logger.info("registering lenslets to array")
                    final_posns = self.get_lensarr_xy(avgs_new,maxdist=16,show_plots=False)
                    fluxcut = self.config.instrument.rectmat_fluxcut
                    posarr, spots = self.make_posarr(ims_cal,final_posns,tracks_new,show_plots=False,medres=False,cropsize=csize,cut=fluxcut)
                    pyfits.writeto(self.redux_dir+'/'+
                                    scmode+'_posarr.fits',np.array(posarr),overwrite=True)
                    pyfits.writeto(self.redux_dir+'/'+
                                    scmode+'_cropped_spotpsf_arr.fits',np.array(spots),
                                    overwrite=True)


                if scmode.split('-')[0] == 'MedRes':
                    csize=16
                    spots = self.find_all_spots(ims_cal,lams,plot_im=False,thresh=70.0,sigma=1.2,medres=True,mfilt=self.mfilt)
                    spot_tracks = self.track_sequentially(spots, max_match_distance=13)
                    self.logger.info("removing duplicates and silos")
                    spot_tracks_u = self.remove_spot_dups(spot_tracks,lams,lmin=self.lmin,lmax=self.lmax,medres=True)
                    avgs = self.find_avg_spotpos(spot_tracks_u,self.lmin,self.lmax,medres=True,show_plots=False)
                    avgs_new,tracks_new = self.remove_silos(avgs,spot_tracks_u,medres=True,show_plots=False)
                    self.logger.info("registering lenslets to array")
                    final_posns = self.get_medres_lensarr_xy(avgs_new,show_plots=False)
                    fluxcut = self.config.instrument.rectmat_fluxcut
                    posarr, spots = self.make_posarr(ims_cal,final_posns,tracks_new,show_plots=False,medres=True,cropsize=csize,cut=fluxcut)
                    pyfits.writeto(self.redux_dir+'/'+
                                    scmode+'_posarr.fits',np.array(posarr),overwrite=True)
                    pyfits.writeto(self.redux_dir+'/'+
                                    scmode+'_cropped_spotpsf_arr.fits',np.array(spots),
                                    overwrite=True)





                pyfits.writeto(self.redux_dir+'/'+
                                scmode+'_calim_stack.fits',ims_cal,overwrite=True)
                centroids = self.get_centroids(spots)
                #ank_mods, ank_resids, ank_modims, ank_resims = self.AnK_spot_model(spots, posarr, centroids, self.nx, self.ny, 3, npix=13)

                oversamp=2
                ank_mods, ank_resids, ank_modims, ank_resims, ank_mods_sym, ank_resids_sym, ank_modims_sym, ank_resims_sym = self.AnK_spot_model_2(spots, centroids, self.ny, self.nx, oversamp, ims_cal, posarr)

                pyfits.writeto(self.redux_dir+'/'+
                                scmode+'_ank_modarr.fits',np.array(ank_mods),
                                overwrite=True)
                pyfits.writeto(self.redux_dir+'/'+
                                scmode+'_ank_resarr.fits',np.array(ank_resids),
                                overwrite=True)
                pyfits.writeto(self.redux_dir+'/'+
                                scmode+'_ank_modims.fits',np.array(ank_modims),
                                overwrite=True)
                pyfits.writeto(self.redux_dir+'/'+
                                scmode+'_ank_resims.fits',np.array(ank_resims),
                                overwrite=True)


                stop
                self.logger.info("fitting spot PSFs for interpolated rectmats")
                fitarr,modims,resims,spotim_arr,modim_arr,resim_arr = self.fit_gauss_spots(ims_cal,posarr,show_plots=False,cropsize=csize,cut=fluxcut)
                pyfits.writeto(self.redux_dir+'/'+
                                scmode+'_gauss_spotpars.fits',np.array(fitarr),overwrite=True)
                pyfits.writeto(self.redux_dir+'/'+
                                scmode+'_gauss_mod_detims.fits',np.array(modims),overwrite=True)
                pyfits.writeto(self.redux_dir+'/'+
                                scmode+'_gauss_res_detims.fits',np.array(resims),overwrite=True)
                pyfits.writeto(self.redux_dir+'/'+
                                scmode+'_cropped_spotgauss_arr.fits',np.array(modim_arr),
                                overwrite=True)
                pyfits.writeto(self.redux_dir+'/'+
                                scmode+'_cropped_spotgaussres_arr.fits',np.array(resim_arr),
                                overwrite=True)


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

                print(len(np.where(np.isnan(interp_arr)==False)[0]))
                pyfits.writeto(self.redux_dir+'/'+
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


                if self.context.rectmat_xshift!=0 or self.context.rectmat_yshift!=0:
                    interp_shift_arr = np.zeros(interp_arr.shape)
                    interp_shift_arr[:,:,:,:] = np.nan
                    interp_shift_arr[:] = interp_arr[:]
                    interp_shift_arr[:,:,:,1]+=self.context.rectmat_xshift
                    interp_shift_arr[:,:,:,2]+=self.context.rectmat_yshift

                    self.logger.info("making interpolated shifted optimal rectmat")
                    OPT_rmat_interpd_shift = self.gen_QL_rectmat_interpd(ims_cal,interp_shift_arr,cut=fluxcut,method='optimal')
                    self.logger.info("making interpolated shifted chi2 rectmat")
                    C2_rmat_interpd_shift = self.gen_C2_rectmat_interpd(ims_cal,interp_shift_arr,cut=fluxcut)
                    sparse.save_npz(self.redux_dir+'/'+
                                    scmode+'_C2_intp_rectmat_dx'+str(self.context.rectmat_xshift)+
                                    '_dy'+str(self.context.rectmat_yshift)+
                                    '.npz',C2_rmat_interpd_shift)
                    sparse.save_npz(self.redux_dir+'/'+
                                    scmode+'_OPT_intp_rectmat_dx'+str(self.context.rectmat_xshift)+
                                    '_dy'+str(self.context.rectmat_yshift)+
                                    '.npz',OPT_rmat_interpd_shift)


                self.logger.info("writing out rectmats")
                sparse.save_npz(self.redux_dir+'/'+
                                scmode+'_OPT_rectmat.npz',OPT_rmat)
                pyfits.writeto(self.redux_dir+'/'+
                                scmode+'_OPT_lsf.fits',OPT_lsf,overwrite=True)
                pyfits.writeto(self.redux_dir+'/'+
                                scmode+'_OPT_lflat.fits',OPT_lflat,overwrite=True)
                sparse.save_npz(self.redux_dir+'/'+
                                scmode+'_OPT_intp_rectmat.npz',OPT_rmat_interpd)
                pyfits.writeto(self.redux_dir+'/'+
                                scmode+'_OPT_intp_lsf.fits',OPT_lsf_interpd,
                                overwrite=True)
                pyfits.writeto(self.redux_dir+'/'+
                                scmode+'_OPT_intp_lflat.fits',OPT_lflat_interpd,
                                overwrite=True)
                sparse.save_npz(self.redux_dir+'/'+
                                scmode+'_C2_rectmat.npz',C2_rmat)
                sparse.save_npz(self.redux_dir+'/'+
                                scmode+'_C2_intp_rectmat.npz',C2_rmat_interpd)
                pyfits.writeto(self.redux_dir+'/'+scmode+'_lams.fits',
                    lams,overwrite=True)
                pyfits.writeto(self.redux_dir+'/'+scmode+'_intp_lams.fits',
                    lams_interp,overwrite=True)


            log_string = ProcessMonochrom.__module__
            self.logger.info(log_string)
        return self.action.args
