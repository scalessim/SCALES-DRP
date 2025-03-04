from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer

import pandas as pd
import numpy as np
import pickle
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pkg_resources
import os
from scipy.optimize import minimize
from scipy import sparse
import astropy.io.fits as pyfits


class QuickLook(BasePrimitive):
    """
	Quicklook  extraction : produce 3d cube
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger
        print("Optimal Extract object created")

    def _perform(self):

        self.logger.info("Quicklook Extraction")
        tab = self.context.proctab.search_proctab(
            frame=self.action.args.ccddata, target_type='OBJECT',
            nearest=True)
        self.logger.info("%d object frames found" % len(tab))

        is_obj = ('OBJECT' in self.action.args.ccddata.header['IMTYPE'])
	
        if len(tab) > 0:
            mdname = get_master_name(tab, target_type)
            print("*************** READING IMAGE: %s" % mdname)
            obj = scales_fits_reader(
                os.path.join(self.config.instrument.cwd, 'redux',mdname))[0]

        def parse_files(indir,filebase):
            """
            Function to parse files in cal unit data directory.
            Currently assumes wavelengths will be in filenames.
            Need to replace that with header keywords instead!
            """
            files = os.listdir(indir)
            lams = []
            for ff in files:
                if ff[:len(filebase)]==filebase:
                    lams.append(ff.split(filebase)[1].split('.fits')[0])
            lams_f = np.array(lams,dtype='float')
            lams_fs = np.sort(lams_f)
            return lams_fs

        def load_calim(indir,filebase,lam):
            """
            Function to load in a single cal unit image. Like the 
            file parsing, this assumes the wavelength info is in the
            filename. Need to replace this with header keywords instead!
            """
            calim = fits.getdata(indir+filebase+str(lam)+'.fits')[0]
            return calim


        def get_init_centroid_region(lensx,lensy,spaxsize=19,xstart=0,ystart=0,subf=8):
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


        def crop_cube(cube,xs,ys,xe,ye):
            """
            Function to crop a 3D cube in x and y to desired coords.
            """
            return cube[:,ys:ye,xs:xe]

        def crop_image(image,xs,ys,xe,ye):
            """
            Function to crop a 2D image in x and y to desired coords.
            """
            return image[ys:ye,xs:xe]

        def pixel_centroid(image):
            """
            Function to centroid based on brightest pixel in image. 
            """
            peak = np.where(image==np.max(image))
            x,y = peak[1][0],peak[0][0]
            return x,y

        def replace_edge_centroids(xc,yc,imsize=2048):
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

        def get_init_centers_lowres(indir,filebase,lams_fs):
            """
            Function to get the starting-point lenslet PSF positions
            for the first wavelength in the set of cal-unit images. 
            This just centroids based on brightest pixels, and then
            replaces edge PSF centroids with nans. 
            """
            xstart = 0
            ystart = 0
            cents = np.zeros([108,108,2])
            calim = load_calim(indir,filebase,lams_fs[0])
            for lensx in range(108):
                for lensy in range(108):
                    xs,ys,xe,ye = get_init_centroid_region(lensx,lensy,xstart=xstart,ystart=ystart)
                    calcrop = crop_image(calim,xs,ys,xe,ye)
                    cx,cy = pixel_centroid(calcrop)
                    cx,cy = replace_edge_centroids(cx,cy)
                    cents[lensy,lensx] = [cy+ys,cx+xs]
            return cents


        def gen_linear_trace_lowres(lls,lmin,lmax,tilt=18,length=54,x0=0,y0=0):
            """
            Returns rough coordinates of linear trace for given set of 
            wavelengths and with a given starting point. 
            """
            dlam = lmax-lmin
            xoffs = (lls-lmin)/dlam*length*np.sin(np.radians(tilt))
            yoffs = (lls-lmin)/dlam*length*np.cos(np.radians(tilt))
            return xoffs+x0,yoffs+y0

        def ingest_calims_cube(indir,filebase,lams_fs):
            """
            Ingests cube of cal unit images according to list of wavelengths.
            Need to replace this with something that actually follows file
            naming scheme at Keck!
            """
            calims = np.array([fits.getdata(indir+filebase+str(lams_fs[x])+'.fits')[0] for x in range(len(lams_fs))])
            return calims


        def get_trace_centroid_region(xx,yy,imsize=2048,subf=8):
            """
            Returns starting and ending xy coordinates of trace. 
            """
            xs = np.max([int(np.round(xx-subf)),0])
            ys = np.max([int(np.round(yy-subf)),0])
            xe = np.min([int(np.round(xx+subf)),2048])
            ye = np.min([int(np.round(yy+subf)),2048])
            return xs,ys,xe,ye


        def get_pixcoords(image):
            """
            Returns pixel coordinates of cropped image. These will get used
            to fill specific indices in a sparse matrix array. 
            """
            xys = np.array([[[y,x] for x in range(len(image[0]))] for y in range(len(image))])
            return xys


        def check_allnans_centroid(image):
            """
            Checks for nans in images to flag bad centroid values.
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


        def get_pixel_trace(lams_fs,calcube,lensx,lensy,lmin,lmax,x0,y0,subf=10):
            """
            Function that returns wavelength-dependent centroid values for cube of
            cal unit images taken at a range of wavelengths, for one lenslet.
            """
            cents = []
            for ll in range(len(lams_fs)):
                xx,yy = gen_linear_trace_lowres(lams_fs[ll],lmin,lmax,x0=x0,y0=y0)
                xs,ys,xe,ye = get_trace_centroid_region(xx,yy)
                calcrop = crop_image(calcube[ll],xs,ys,xe,ye).copy()
                xys = get_pixcoords(calcrop)
                if np.prod(xys.shape)!=0:
                    dists = get_dists(xys,xx,yy,xs,ys)
                    calcrop[np.where(dists > subf)] = np.nan
                    cx,cy = check_allnans_centroid(calcrop)
                    centx,centy = cx+xs,cy+ys
                else:
                    centx,centy = np.nan,np.nan
                cents.append([centx,centy])
            return np.array(cents)

        def get_dists(xys,xx,yy,xs,ys):
            """
            Function to calculate distances between xy coordinates
            """
            diffs = np.array(xys - np.array([yy-ys,xx-xs]))
            dists = np.sqrt(diffs[:,:,0]**2 + diffs[:,:,1]**2)
            return dists


        def get_calunit_centroids(indir,filebase,lams_fs,lmin,lmax,init_cents):
            """
            Function that returns all 108 x 108 pixel traces measured from cube of 
            cal-unit images.
            """
            calims = ingest_calims_cube(indir,filebase,lams_fs)
            centsarr = np.empty([len(lams_fs),108,108,2])
            for lensx in range(108):
                for lensy in range(108):
                    x0,y0 = init_cents[lensy,lensx]
                    pixtrace = get_pixel_trace(lams_fs,calims,lensx,lensy,lmin,lmax,x0,y0)
                    centsarr[:,lensy,lensx] = pixtrace
            return centsarr


        def gen_sparse_inds(xs,ys,xe,ye):
            """
            Function to take 2d x,y pixel coordinates and turn them into flattened
            coordinates for sparse matrix construction.
            """
            indsx = np.array([xval for xval in range(xs,xe) for yval in range(ys,ye)])
            indsy = np.array([yval for xval in range(xs,xe) for yval in range(ys,ye)])
            flatinds = np.ravel_multi_index((indsy,indsx),(2048,2048))
            return flatinds


        def crop_sparse_vals(image,xs,xe,ys,ye):
            """
            Function to crop lenslet PSFs down and then only select pixels above
            a certain flux threshold (currently set to 5% of max).
            """
            cropped = image[ys:ye,xs:xe]
            cropped[np.where(cropped < 0.05*np.max(cropped))]=0
            cropped/=np.sum(cropped)
            vals = np.array([cropped[yind,xind] for xind in range(0,xe-xs) for yind in range(0,ye-ys)])
            return vals

        def parse_centers(centsarr,ll,lensy,lensx):
            """
            Function to pull lenslet PSF centers from 3D array.
            """
            centx,centy = centsarr[ll,lensy,lensx]
            return centx,centy

        def gen_rectmat_inds(calims,centsarr,apsize=8):
            """
            Function to generate row and column indices for sparse matrix
            for all 108 x 108 lenslets and wavelengths. 
            """
            matrowinds = []
            matcolinds = []
            matvals = []
            count=0

            for ll in range(len(calims)):
                for lensy in range(108):
                    for lensx in range(108):
                        centx,centy = parse_centers(centsarr,ll,lensy,lensx)
                        if np.isnan(centx)==False:
                            centx = int(centx)
                            centy = int(centy)
                            xs,ys,xe,ye = get_trace_centroid_region(centx,centy,subf=apsize//2)
                            flatinds = gen_sparse_inds(xs,ys,xe,ye)
                            vals = crop_sparse_vals(calims[ll],xs,xe,ys,ye)
                            for i in range(len(vals)):
                                matvals.append(vals[i])
                                matcolinds.append(flatinds[i])
                                matrowinds.append(count)
                        count+=1
            return matrowinds, matcolinds, matvals

        def gen_rectmat(indir,filebase,lams_fs,centsarr,apsize=8):
            """
            Function to generate rectmat from cube of cal unit images.
            """

            calims = ingest_calims_cube(indir,filebase,lams_fs)
            matrowinds,matcolinds,matvals = gen_rectmat_inds(calims,centsarr,apsize=apsize)
            rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(54*108*108,2048*2048))
            return rmat


        
        caldir = pkg_resources.resource_filename('scalesdrp','calib/')
        filebase = 'calibration_2.0_5.2_'

        imtest = self.action.args.ccddata.data
        lams_fs = parse_files(caldir,filebase)
        initcents = get_init_centers_lowres(caldir,filebase,lams_fs)
        cents = get_calunit_centroids(caldir,filebase,lams_fs,2.0,5.0,initcents)
        rmat = gen_rectmat(caldir,filebase,lams_fs,cents)

        data_f = np.matrix(imtest.flatten()).T
        cube = rmat*data_f
        cube2 = np.array(cube).reshape(54,108,108)
        
    
        # update with new image
        self.action.args.ccddata.data = cube2 

        log_string = QuickLook.__module__
        self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)


        if is_obj:	    
            scales_fits_writer(self.action.args.ccddata, table=self.action.args.table,output_file=self.action.args.name,output_dir=self.config.instrument.output_directory,suffix="quicklook_cube")

        self.context.proctab.update_proctab(frame=self.action.args.ccddata, suffix="opt", newtype='OBJECT',
                filename=self.action.args.ccddata.header['OFNAME'])
        self.context.proctab.write_proctab(
                tfil=self.config.instrument.procfile)

        return self.action.args
    # END: class QuickLook()
