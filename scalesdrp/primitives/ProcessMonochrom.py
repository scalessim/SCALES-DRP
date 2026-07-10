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
        det_config = np.unique(df2['MCLOCK'])[0]

        calibfilepath = self.context.calib_file_path
        package = __name__.split('.')[0]
        calib_path = str(get_resource_path(package, calibfilepath))+'/'
        if det_config =='9.0 MHz': #fast0.6
            flat = pyfits.getdata(calib_path+self.context.flat_ifs_9mhz)

        lams = df2['MONOWAVE']
        names = df2['filename'][np.argsort(lams)]

        lams = np.sort(lams)
        ims = []
        for name in names:
            ims.append(pyfits.getdata(self.redux_dir+'/'+name)/flat)
        ims = np.array(ims)
        return ims, lams

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
                ycent,ymin,ymax=get_spot_yrange_medres(lams_u[ii],y0=y0,lam0=lam0,chy=chy,lmin=lmin,lmax=lmax,length=length)

                print(ycent,ymin,ymax,lams_u[ii])
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
                    #print(len(testrow))
                    if len(testrow) < 80:
                        #plt.scatter(coords_yx[:,1],ydiff)
                        #plt.scatter(xc,0)
                        #plt.show()
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
            print('----------------------------------')
            print('doing '+str(i)+' of '+str(len(spots_d)))
            keys = list(spots_d[i].keys())
            #grab spot's first x,y position and first wavelength
            x0,y0 = spots_d[i][keys[0]][0],spots_d[i][keys[0]][1]
            lam0 = spots_d[i][keys[0]][3]
            print(x0,y0,lam0)
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
                yi,ymin,ymax=get_spot_yrange_medres(lam,y0=y0,lam0=lam0,chy=chy,lmin=lmin,lmax=lmax,length=length)
                print(lam,yi)
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
                                yj,ymin,ymax=get_spot_yrange_medres(lam,y0=y0,lam0=lam0,chy=chy,lmin=lmin,lmax=lmax,length=length)
                            #get the distance between the other spot's central wavelength
                            #position and the original one
                            dist = np.sqrt((xi-xj)**2+(yi-yj)**2)
                            #if the distance between the two central wavelength positions
                            #is smaller than the maximum allowed, then they're the same
                            if dist < maxdist:
                                #append j index to list to be removed
                                rem.append(j)
                                merged = merged | spots_d[j]
                #if you accidentally merged tracks and ended up with more positions
                #than the number of wavelengths, then throw an error
                if len(merged) > len(lams_u):
                    print('uh oh bad merging!!!')
                    stop
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
                #print(diff)
                #stop
                diff2 = kd0.data - kd1.data
                xdiff = np.abs(diff2[:,0])
                ydiff = np.abs(diff2[:,1])

                #####add this test (looking for being in line with other spots)
                #### to the initial spot finding step!!
                testrow = diff2[np.where(np.abs(ydiff) < 15)]
                if len(testrow) < 80:
                    todel.append(i)

                nearestdiffs = ydiff[np.where(xdiff < 30)]
                if np.sort(nearestdiffs)[1] > 2:
                    todel.append(i)


                #####LEFT OFF HERE
                if len(spot_tracks[i]) < 0.3*mlen:
                    todel.append(i)
                #print(kd0.data)
                #plt.scatter(diff2[:,0],diff2[:,1])
                #plt.show()
                #stop

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
                yj,ymin,ymax=get_spot_yrange_medres(lam,y0=y0,lam0=lam0,chy=chy,lmin=lmin,lmax=lmax,length=length)
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
            print(ddone)
            #while a search is needed, go through entries in list
            #of lenslets that have yet to be searched
            for search_lens in to_search_around:
                print("done searching", len(done_searching_around))#, done_searching_around)
                print("avgs", len(avgs))#, avgs)
                print("tosearch", len(to_search_around))#, to_search_around)
                #confirm that this lenslet is not marked as done
                print("search lens pos:",search_lens,avgs[search_lens:search_lens+1])
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
                            elif (abs(entry[0]) < 15) and (abs(entry[1]) < 15):
                                print('found search lenslet - do nothing!')
                            else:
                                print('uh oh didnt find a neighbor!')
                                #stop
                    done0=len(done_searching_around)
                    done_searching_around.append(search_lens)
                    ddone = len(done_searching_around)-done0
                    print(ddone)


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


###########################################################################################




    def get_init_centroid_region(self,lensx,lensy,xstart,ystart,
                                 spaxsize=19, subf=8):
        """
        Function to get initial cropped image around expected
        lenslet PSF region.
        """

        x = xstart + spaxsize*lensx
        y = ystart + spaxsize*lensy
        xs = np.max([0,x-subf])
        ys = np.max([0,y-subf])
        xe = np.min([x+subf,2048])
        ye = np.min([y+subf,2048])
        return xs,ys,xe,ye


    def get_medres_init_pos(self):
        shifts = np.zeros([18,17,2]) #x spaxel, y spaxel, x and y shifts

        dx = 1.5e-3/18e-6 ##x shift between supercols (pix)
        dy = 18.0 #y shift between spaxels in each supercol (pix)
        dy0 = 17*18 + 3 #y shift between starting spaxel in each supercol (pix)
        #dy0 = 6.0 #y shift between starting spaxel in each supercol (pix)

        scol1 = [1,17,4,13,7,10] ##spaxel cols in each supercol, indexed from 1
        scol2 = [18,3,15,6,12,9]
        scol3 = [2,16,5,14,8,11]


        scols = [scol1,scol2,scol3]
        for i in range(len(scols)):
            for j in range(len(scols[i])):
                sc = scols[i][j]
                shifts[sc-1,:,0] = i*dx ###big shift between each supercolumn (there are 3)
                shifts[sc-1,:,1] = j*dy0 + i*dy/3.0  ###shift each column so that there's an extra pixel between them (there are 6 in each supercolumn)
                for ii in range(len(shifts[sc-1])):
                    shifts[sc-1,ii,1] += dy*ii
        shifts[:,:,0]+=10
        shifts[:,:,1]+=10

        return shifts


    def get_init_centroid_region_medres(self,lensx,lensy,shifts,subf=8):
        """
        Function to get initial cropped image around expected
        lenslet PSF region.
        """

        x,y = shifts[lensx,lensy]

        xs = np.max([0,int(x-subf)])
        ys = np.max([0,int(y-subf)])
        xe = np.min([int(x+subf),2048])
        ye = np.min([int(y+subf),2048])
        return xs,ys,xe,ye

    def crop_image(self,image,xs,ys,xe,ye):
        """
        Function to crop a 2D image in x and y to desired coords.
        """
        return image[ys:ye,xs:xe]

    def pixel_centroid(self,image):
        """
        Function to centroid based on brightest pixel in image.
        """
        peak = np.where(image==np.max(image))
        x,y = peak[1][0],peak[0][0]
        return x,y

    def replace_edge_centroids(self,xc,yc):
        """
        Function to replace centroids that are on edge of frame
        with nan values. Helps with figuring out whether lenslet
        PSFs have fallen off (or started off) the detector.
        """

        if 2048-xc < 2:
            return np.nan,np.nan
        if 2048-yc < 2:
            return np.nan,np.nan
        else:
            return xc,yc

    def get_init_centers_lowres(self,names,lams):
        """
        Function to get the starting point lenslet PSF position
        for the first wavelength in the set of cal-unit images.
        This just centroids based on brightest pixels, and then
        replaces edge PSF centroids with nans.
        """
        xstart = 0
        ystart = 0
        cents = np.zeros([108,108,2])
        calim = fits.getdata(names[0])
        for lensx in range(108):
            for lensy in range(108):
                xs,ys,xe,ye = self.get_init_centroid_region(lensx,lensy,xstart=xstart,ystart=ystart)
                calcrop = self.crop_image(calim,xs,ys,xe,ye)
                cx,cy = self.pixel_centroid(calcrop)
                cx,cy = self.replace_edge_centroids(cx,cy)
                cents[lensy,lensx] = [cy+ys,cx+xs]
        return cents

    def get_init_centers_medres(self,names,lams):
        """
        Function to get the starting point lenslet PSF position
        for the first wavelength in the set of cal-unit images.
        This just centroids based on brightest pixels, and then
        replaces edge PSF centroids with nans.
        """
        xstart = 0
        ystart = 0
        cents = np.zeros([17,18,2])
        calim = fits.getdata(names[0])
        for lensx in range(18):
            for lensy in range(17):
                shifts = self.get_medres_init_pos()
                xs,ys,xe,ye = self.get_init_centroid_region_medres(lensx,lensy,shifts)
                calcrop = self.crop_image(calim,xs,ys,xe,ye)
                #plt.imshow(calcrop)
                #plt.title(str(shifts[lensx,lensy])+' '+str(xs)+' '+str(xe)+' '+str(ys)+' '+str(ye))
                #plt.savefig('test.png')
                #stop
                cx,cy = self.pixel_centroid(calcrop)
                cx,cy = self.replace_edge_centroids(cx,cy)
                cents[lensy,lensx] = [cy+ys,cx+xs]
        return cents

    def gen_linear_trace_medres(self,lls,lmin,lmax,length=1900,x0=0,y0=0):
        """
        Returns rough coordinates of linear trace for given set of
        wavelengths and with a given starting point
        """
        dlam = lmax-lmin
        xoffs = np.zeros(lls.shape)
        yoffs = (lls-lmin)/dlam*length
        return xoffs+x0,yoffs+y0

    def gen_linear_trace_lowres(self,lls,lmin,lmax,tilt=18,length=54,x0=0,y0=0):
        """
        Returns rough coordinates of linear trace for given set of
        wavelengths and with a given starting point
        """
        dlam = lmax-lmin
        xoffs = (lls-lmin)/dlam*length*np.sin(np.radians(tilt))
        yoffs = (lls-lmin)/dlam*length*np.cos(np.radians(tilt))
        return xoffs+x0,yoffs+y0

    def ingest_calims_cube(self,names):
        """
        Ingests cube of cal unit images according to list of wavelengths.
        Need to replace this with something that actually follows file naming
        scheme at Keck!
        """
        calims = np.array([fits.getdata(name,memmap=False) for name in names])
        print(calims.shape)
        #calims = np.array([fits.getdata(name)[0] for name in names])
        return calims

    def get_trace_centroid_region(self,xx,yy,imsize=2048,subf=8):
        """
        Returns starting and ending coordinates of trace.
        """
        xs = np.max([int(np.round(xx-subf)),0])
        ys = np.max([int(np.round(yy-subf)),0])
        xe = np.min([int(np.round(xx+subf)),2048])
        ye = np.min([int(np.round(yy+subf)),2048])
        return xs,ys,xe,ye

    def get_pixcoords(self,image):
        """
        Returns pixel coordinates of cropped image. These will get used
        to fill specific indices in a sparse matrix array.
        """
        xys = np.array([[[y,x] for x in range(len(image[0]))] for y in range(len(image))])
        return xys

    def check_allnans_centroid(self,image):
        """
        check for nans in images to flag bad centroid values.
        """
        imsize = len(image)
        if len(np.where(np.isnan(image)==False)[0])==0:
            cx,cy = np.nan,np.nan
        else:
            peak = np.where(image==np.nanmax(image))
            cx,cy = peak[1][0],peak[0][0]
        if (cx == imsize or cy == imsize):
            cx,cy = np.nan,np.nan
        return cx,cy

    def get_pixel_trace_medres(self,lams,calcube,lensx,lensy,x0,y0,subf=10):
        """
        Function that returns wavelength-dependent centroid values for cube of
        cal unit images taken at a range of wavelengths, for one lenslet.
        """
        cents = []
        for ll in range(len(lams)):
            xx,yy = self.gen_linear_trace_medres(lams[ll],self.lmin,self.lmax,x0=x0,y0=y0)
            xs,ys,xe,ye = self.get_trace_centroid_region(xx,yy)
            calcrop = self.crop_image(calcube[ll],xs,ys,xe,ye).copy()
            xys = self.get_pixcoords(calcrop)
            if np.prod(xys.shape)!=0:
                dists = self.get_dists(xys,xx,yy,xs,ys)
                calcrop[np.where(dists > subf)] = np.nan
                cx,cy = self.check_allnans_centroid(calcrop)
                centx,centy = cx+xs,cy+ys
            else:
                centx,centy = np.nan,np.nan
            cents.append([centx,centy])
        return np.array(cents)

    def get_pixel_trace(self,lams,calcube,lensx,lensy,x0,y0,subf=10):
        """
        Function that returns wavelength-dependent centroid values for cube of
        cal unit images taken at a range of wavelengths, for one lenslet.
        """
        cents = []
        for ll in range(len(lams)):
            xx,yy = self.gen_linear_trace_lowres(lams[ll],self.lmin,self.lmax,x0=x0,y0=y0)
            xs,ys,xe,ye = self.get_trace_centroid_region(xx,yy)
            calcrop = self.crop_image(calcube[ll],xs,ys,xe,ye).copy()
            xys = self.get_pixcoords(calcrop)
            if np.prod(xys.shape)!=0:
                dists = self.get_dists(xys,xx,yy,xs,ys)
                calcrop[np.where(dists > subf)] = np.nan
                cx,cy = self.check_allnans_centroid(calcrop)
                centx,centy = cx+xs,cy+ys
            else:
                centx,centy = np.nan,np.nan
            cents.append([centx,centy])
        return np.array(cents)

    def get_dists(self,xys,xx,yy,xs,ys):
        """
        Function to calculate distances between xy coordinates
        """
        diffs = np.array(xys - np.array([yy-ys,xx-xs]))
        dists = np.sqrt(diffs[:,:,0]**2 + diffs[:,:,1]**2)
        return dists

    def get_calunit_centroids_medres(self,names,lams,calims,init_cents):
        """
        Function that returns all 108 x 108 pixel traces measured from cube of
        cal-unit images.
        """
        centsarr = np.empty([len(names),17,18,2])
        for lensx in range(18):
            for lensy in range(17):
                x0,y0 = init_cents[lensy,lensx]
                pixtrace = self.get_pixel_trace_medres(lams,calims,lensx,lensy,x0,y0)
                centsarr[:,lensy,lensx] = pixtrace
                #plt.imshow(np.sum(calims,axis=0))
                #plt.scatter(centsarr[:,lensy,lensx,0],centsarr[:,lensy,lensx,1])
                #plt.savefig('test.png')
                #stop
        return centsarr

    def get_calunit_centroids(self,names,lams,calims,init_cents):
        """
        Function that returns all 108 x 108 pixel traces measured from cube of
        cal-unit images.
        """
        centsarr = np.empty([len(names),108,108,2])
        for lensx in range(108):
            for lensy in range(108):
                x0,y0 = init_cents[lensy,lensx]
                pixtrace = self.get_pixel_trace(lams,calims,lensx,lensy,x0,y0)
                centsarr[:,lensy,lensx] = pixtrace
        return centsarr

    def parse_centers(self,centsarr,ll,lensy,lensx):
        """
        Function to pull lenslet PSF centers from 3D array.
        """
        centx,centy = centsarr[ll,lensy,lensx]
        return centx,centy


    def gen_sparse_inds(self,xs,ys,xe,ye):
        """
        Function to take 2d x,y pixel coordinates and turn them into flattened
        coordinates for sparse matrix construction.
        """

        indsx = np.array([xval for xval in range(xs,xe) for yval in range(ys,ye)])
        indsy = np.array([yval for xval in range(xs,xe) for yval in range(ys,ye)])

        flatinds = np.ravel_multi_index((indsy,indsx),(2048,2048))
        return flatinds

    def crop_sparse_vals(self,image,xs,xe,ys,ye,cut=0.05):
        """
        Function to crop lenslet PSFs down and then only select pixels above
        a certain flux threshold (currently set to 5% of max by default).
        """
        cropped = image[ys:ye,xs:xe]
        cropped[np.where(cropped < cut*np.max(cropped))]=0
        cropped/=np.sum(cropped)
        vals = np.array([cropped[yind,xind] for xind in range(0,xe-xs) for yind in range(0,ye-ys)])
        return vals

    def gen_rectmat_inds_medres(self,calims,centsarr,apsize=8):

        """
        Function to generate row and column indices for sparse matrix
        for all 108 x 108 lenslets and wavelengths.
        """

        matrowinds = []
        matcolinds = []
        matvals = []
        count=0

        for ll in range(len(calims)):
            for lensx in range(18):
                for lensy in range(17):
                    centx,centy = self.parse_centers(centsarr,ll,lensy,lensx)

                    if np.isnan(centx)==False:
                        centx = int(centx)
                        centy = int(centy)
                        xs,ys,xe,ye = self.get_trace_centroid_region(centx,centy,subf=apsize//2)
                        flatinds = self.gen_sparse_inds(xs,ys,xe,ye)
                        vals = self.crop_sparse_vals(calims[ll],xs,xe,ys,ye)
                        for i in range(len(vals)):
                            matvals.append(vals[i])
                            matcolinds.append(flatinds[i])
                            matrowinds.append(count)
                    count+=1
        return matrowinds, matcolinds, matvals

    def gen_rectmat_inds(self,calims,centsarr,apsize=8):

        """
        Function to generate row and column indices for sparse matrix
        for all 108 x 108 lenslets and wavelengths.
        """

        matrowinds = []
        matcolinds = []
        matvals = []
        count=0

        for ll in range(len(calims)):
            for lensx in range(108):
                for lensy in range(108):
                    centx,centy = self.parse_centers(centsarr,ll,lensy,lensx)

                    if np.isnan(centx)==False:
                        centx = int(centx)
                        centy = int(centy)
                        xs,ys,xe,ye = self.get_trace_centroid_region(centx,centy,subf=apsize//2)
                        flatinds = self.gen_sparse_inds(xs,ys,xe,ye)
                        vals = self.crop_sparse_vals(calims[ll],xs,xe,ys,ye)
                        for i in range(len(vals)):
                            matvals.append(vals[i])
                            matcolinds.append(flatinds[i])
                            matrowinds.append(count)
                    count+=1
        return matrowinds, matcolinds, matvals


    def gen_QL_rectmat_medres(self,calims,centsarr,apsize=8):
        """
        Function to generate rectmat from cube of cal unit images.
        """

        matrowinds,matcolinds,matvals = self.gen_rectmat_inds_medres(calims,centsarr,apsize=apsize)
        rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(1900*17*18,2048*2048))
        return rmat


    def gen_QL_rectmat(self,calims,centsarr,apsize=8):
        """
        Function to generate rectmat from cube of cal unit images.
        """

        matrowinds,matcolinds,matvals = self.gen_rectmat_inds(calims,centsarr,apsize=apsize)
        rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(54*108*108,2048*2048))
        return rmat



    def gen_c2_rectmat_inds_medres(self,calims,centsarr,apsize=8,cut=0.001):

        """
        Function to generate row and column indices for sparse matrix
        for all 108 x 108 lenslets and wavelengths.
        """

        matrowinds = []
        matcolinds = []
        matvals = []
        count=0

        for ll in range(len(calims)):
            for lensx in range(18):
                for lensy in range(17):
                    #print('lenslet:',lensy,lensx)
                    centx,centy = self.parse_centers(centsarr,ll,lensy,lensx)

                    if np.isnan(centx)==False:
                        centx = int(centx)
                        centy = int(centy)
                        xs,ys,xe,ye = self.get_trace_centroid_region(centx,centy,subf=apsize//2)
                        flatinds = self.gen_sparse_inds(xs,ys,xe,ye)
                        vals = self.crop_sparse_vals(calims[ll],xs,xe,ys,ye,cut=cut)
                        for i in range(len(vals)):
                            if vals[i]>cut:
                                matvals.append(vals[i])
                                matrowinds.append(flatinds[i])
                                matcolinds.append(count)
                    count+=1
        return matrowinds, matcolinds, matvals

    def gen_c2_rectmat_inds(self,calims,centsarr,apsize=8,cut=0.001):

        """
        Function to generate row and column indices for sparse matrix
        for all 108 x 108 lenslets and wavelengths.
        """

        matrowinds = []
        matcolinds = []
        matvals = []
        count=0

        for ll in range(len(calims)):
            for lensx in range(108):
                for lensy in range(108):
                    #print('lenslet:',lensy,lensx)
                    centx,centy = self.parse_centers(centsarr,ll,lensy,lensx)

                    if np.isnan(centx)==False:
                        centx = int(centx)
                        centy = int(centy)
                        xs,ys,xe,ye = self.get_trace_centroid_region(centx,centy,subf=apsize//2)
                        flatinds = self.gen_sparse_inds(xs,ys,xe,ye)
                        vals = self.crop_sparse_vals(calims[ll],xs,xe,ys,ye,cut=cut)
                        for i in range(len(vals)):
                            if vals[i]>cut:
                                matvals.append(vals[i])
                                matrowinds.append(flatinds[i])
                                matcolinds.append(count)
                    count+=1
        return matrowinds, matcolinds, matvals




    def gen_C2_rectmat_medres(self,calims,centsarr,apsize=8):
        """
        Function to generate rectmat from cube of cal unit images.
        """

        matrowinds,matcolinds,matvals = self.gen_c2_rectmat_inds_medres(calims,centsarr,apsize=apsize)
        print(len(matrowinds),len(matcolinds),len(matvals))
        rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(2048*2048,len(calims)*17*18))
        return rmat

    def gen_C2_rectmat(self,calims,centsarr,apsize=8):
        """
        Function to generate rectmat from cube of cal unit images.
        """

        matrowinds,matcolinds,matvals = self.gen_c2_rectmat_inds(calims,centsarr,apsize=apsize)
        print(len(matrowinds),len(matcolinds),len(matvals))
        rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(2048*2048,len(calims)*108*108))
        return rmat


    def set_lamlimits(self,scmode):
        if scmode == "LowRes-SED":
            self.lmin = 2.0
            self.lmax = 5.2
        if scmode == 'LowRes-K':
            self.lmin = 1.95
            self.lmax = 2.45
        if scmode == 'MedRes-K':
            self.lmin = 1.95
            self.lmax = 2.45
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
        if scmode == 'LowRes-H2O':
            self.lmin = 2.0
            self.lmax = 4.0
        if scmode == 'LowRes-PAH':
            self.lmin = 3.1
            self.lmax = 3.5

        return


    def _perform(self):

        self.logger.info("Process Monochromator Images")
        df = fits_headers_to_dataframe(self.redux_dir,pattern="*mcalunit.fits")


        for scmode in ['LowRes-L']:
            ims, lams = self.parse_files(df,scmode)
            ims_cal = self.monochrom_bksub(ims)
            print(np.nanmean(ims_cal,axis=(1,2)))
            spots = self.find_all_spots(ims_cal,lams,plot_im=False,thresh=90.0,sigma=1.2,medres=False)
            spot_tracks = self.track_sequentially(spots,max_match_distance=3)
            spot_tracks_u = self.remove_spot_dups(spot_tracks,lams,lmin=2.9,lmax=4.15,medres=False)
            avgs = self.find_avg_spotpos(spot_tracks_u,2.9,4.15,medres=False,show_plots=False)
            avgs_new,tracks_new = self.remove_silos(avgs,spot_tracks_u,medres=False)
            final_posns = self.get_lensarr_xy(avgs_new,maxdist=16,show_plots=False)

            stop




            stop
            self.set_lamlimits(scmode)
            print(names,lams_fs)

            calims = self.ingest_calims_cube(names)
            if scmode[:3]=='Low':
                icents = self.get_init_centers_lowres(names, lams_fs)
                cents = self.get_calunit_centroids(names, lams_fs, calims, icents)
                QL_rmat = self.gen_QL_rectmat(calims,cents)
                C2_rmat = self.gen_C2_rectmat(calims,cents)
            else:
                icents = self.get_init_centers_medres(names, lams_fs)
                cents = self.get_calunit_centroids_medres(names, lams_fs, calims, icents)
                QL_rmat = self.gen_QL_rectmat_medres(calims,cents)
                C2_rmat = self.gen_C2_rectmat_medres(calims,cents)

            print("started saving the files")

            sparse.save_npz(self.action.args.dirname+'/'+
                            scmode+'_QL_rectmat.npz',QL_rmat)
            sparse.save_npz(self.action.args.dirname+'/'+
                            scmode+'_C2_rectmat.npz',C2_rmat)


            log_string = CentroidEstimate.__module__
            self.logger.info(log_string)
        return self.action.args
