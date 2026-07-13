from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.core.scales_pkg_resources import get_resource_path
from scalesdrp.primitives.scales_basic import fits_headers_to_dataframe
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
from astropy.modeling.functional_models import Gaussian2D
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


    def make_posarr(self,ims_cal,final_posns,spot_tracks_u,medres=False,show_plots=False,cropsize=10):
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
        return posarr


    def fit_gauss_spots(self,calims,posarr,show_plots=False):
        fitarr = np.zeros([posarr.shape[0],posarr.shape[1],posarr.shape[2],6])
        fitarr[:,:,:,:] = np.nan
        modims = []
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

                                initial_guess = Gaussian2D(amplitude=intens, x_mean=(xe-xs)*0.5, y_mean=(ye-ys)*0.5,
                                          x_stddev=1., y_stddev=1.)

                                fitter = fitting.LevMarLSQFitter()
                                y, x = np.mgrid[:(ye-ys), :(xe-xs)]
                                fitted_model = fitter(initial_guess, x, y, cropped)

                                modim[ys:ye,xs:xe]+=fitted_model(x,y)
                                fitarr[ll,lensy,lensx] = [fitted_model.amplitude.value,
                                                          fitted_model.x_mean.value+xs,
                                                          fitted_model.y_mean.value+ys,
                                                          fitted_model.x_stddev.value,
                                                          fitted_model.y_stddev.value,
                                                          fitted_model.theta.value]
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
        return fitarr,modims,resims


    def interp_gauss_spots(self,lams_in,lams_des,fitarr,show_plots=False,method='poly'):
        interp_arr = np.zeros([len(lams_des),fitarr.shape[1],fitarr.shape[2],fitarr.shape[3]])
        interp_arr[:,:,:,:] = np.nan
        for lensy in range(fitarr.shape[1]):
            for lensx in range(fitarr.shape[2]):
                gausspars = fitarr[:,lensy,lensx]#A,xm,ym,xstd,ystd,theta
                if False in np.isnan(gausspars):
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

            modspot[np.where(modspot < cut*np.max(modspot))]=0
            modspot/=np.sum(modspot)

            vals = np.array([modspot[yind,xind] for xind in range(0,xe-xs) for yind in range(0,ye-ys)])
            if method=='optimal':
                vals = vals

            if method=='aperture':
                vals[np.where(vals!=0)] = 1.0
            flatinds = self.gen_sparse_inds(xs,ys,xe,ye)
        return flatinds,vals

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
                    if True not in np.isnan(interp_arr[ll,lensy,lensx]):
                        flatinds,vals = self.crop_interpd_sparse_vals(interp_arr[ll,lensy,lensx],cut=cut,method=method)
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
                    if True not in np.isnan(interp_arr[ll,lensy,lensx]):
                        flatinds,vals = self.crop_interpd_sparse_vals(interp_arr[ll,lensy,lensx],cut=cut,method='optimal')
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
        if scmode == 'LowRes-K':
            self.lmin = 1.95
            self.lmax = 2.45
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
            scube = np.array(rmat*ims_cal[ii].reshape([2048*2048,1])).reshape([len(lams),ny,nx])
            scube_lsf = scube/np.max(scube,axis=0)
            lflat = np.max(scube,axis=0)
            lsfcube.append(scube_lsf)
            lensflatcube.append(lflat/np.median(lflat))
        lensflat = np.median(lflat,axis=0)
        return np.array(lsfcube),np.array(lensflat)




    def _perform(self):

        self.logger.info("Process Monochromator Images")
        df = fits_headers_to_dataframe(self.redux_dir,pattern="*mcalunit.fits")


        for scmode in ['LowRes-KLM','LowRes-K','LowRes-L',
                       'LowRes-M','LowRes-KL','LowRes-Ls',
                       'MedRes-K','MedRes-L','MedRes-M']:
            ims, lams = self.parse_files(df,scmode)
            if len(ims) > 0:
                self.set_lamlimits(scmode)
                ims_cal = self.monochrom_bksub(ims)
                print(ims_cal.shape)
                if scmode.split('-')[0] == 'LowRes':
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
                    posarr = self.make_posarr(ims_cal,final_posns,tracks_new,show_plots=False,medres=False,cropsize=8)


                if scmode.split('-')[0] == 'MedRes':
                    spots = self.find_all_spots(ims_cal,lams,plot_im=False,thresh=70.0,sigma=1.2,medres=True,mfilt=self.mfilt)
                    spot_tracks = self.track_sequentially(spots, max_match_distance=13)
                    self.logger.info("removing duplicates and silos")
                    spot_tracks_u = self.remove_spot_dups(spot_tracks,lams,lmin=self.lmin,lmax=self.lmax,medres=True)
                    avgs = self.find_avg_spotpos(spot_tracks_u,self.lmin,self.lmax,medres=True,show_plots=False)
                    avgs_new,tracks_new = self.remove_silos(avgs,spot_tracks_u,medres=True,show_plots=False)
                    self.logger.info("registering lenslets to array")
                    final_posns = self.get_medres_lensarr_xy(avgs_new,show_plots=False)
                    posarr = self.make_posarr(ims_cal,final_posns,tracks_new,show_plots=False,medres=True,cropsize=8)


                self.logger.info("fitting spot PSFs for interpolated rectmats")
                fitarr,modims,resims = self.fit_gauss_spots(ims_cal,posarr,show_plots=False)

                if scmode.split('-')[0] == 'LowRes':
                    lams_interp = lams
                    interp_arr = self.interp_gauss_spots(lams,lams_interp,fitarr)
                if scmode.split('-')[0] == 'MedRes' and scmode!='MedRes-K':
                    lams_interp = np.linspace(self.lmin,self.lmax,1900)
                    interp_arr = self.interp_gauss_spots(lams,lams_interp,fitarr)
                if scmode == 'MedRes-K':
                    lams_interp = np.linspace(self.lmin_i,self.lmax_i,1900)
                    interp_arr = self.interp_gauss_spots(lams,lams_interp,fitarr)

                self.logger.info("making chi2 rectmat")
                C2_rmat = self.gen_C2_rectmat(ims_cal,posarr,cut=0.01)
                self.logger.info("making interpolated chi2 rectmat")
                C2_rmat_interpd = self.gen_C2_rectmat_interpd(ims_cal,interp_arr,cut=0.01)
                self.logger.info("making interpolated optimal rectmat")
                OPT_rmat_interpd = self.gen_QL_rectmat_interpd(ims_cal,interp_arr,cut=0.01,method='optimal')
                self.logger.info("making interpolated optimal lsf and lflat")
                OPT_lsf_interpd,OPT_lflat_interpd = self.make_lsf_lflat(
                                                OPT_rmat_interpd,ims_cal,lams_interp,
                                                self.ny,self.nx)
                self.logger.info("making optimal rectmat")
                OPT_rmat = self.gen_QL_rectmat(ims_cal,posarr,cut=0.01,method='optimal')
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
                    OPT_rmat_interpd_shift = self.gen_QL_rectmat_interpd(ims_cal,interp_shift_arr,cut=0.01,method='optimal')
                    self.logger.info("making interpolated shifted chi2 rectmat")
                    C2_rmat_interpd_shift = self.gen_C2_rectmat_interpd(ims_cal,interp_shift_arr,cut=0.01)
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
                pyfits.writeto(self.redux_dir+'/'+scmode+'_intp_lams',
                    lams_interp,overwrite=True)


            log_string = ProcessMonochrom.__module__
            self.logger.info(log_string)
        return self.action.args
