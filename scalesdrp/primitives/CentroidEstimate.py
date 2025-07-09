from keckdrpframework.primitives.base_primitive import BasePrimitive
#from scalesdrp.primitives.scales_file_primitives import scales_fits_writer

import pandas as pd
import numpy as np
import pickle
from astropy.io import fits
import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)
import pkg_resources
from scipy import sparse
import astropy.io.fits as pyfits
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt


class CentroidEstimate(BasePrimitive):
    """
    Estimate the psf centroid of all the calib images images and save
    in two pickle file one for x and one for y centroid values. Currently
    assumes wavelengths will be in filenames. Need to replace that with
    header keywords instead. Also the location of the calib filesneed to fix.
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger
        #print('action arguments in primitive',self.action.args.dirname)

    def parse_files(self,dt,scmode):
        """
        """

        dt = dt[dt['IMTYPE'] == 'CALUNIT']
        dt = dt[dt['SCMODE'] == scmode]

        lams = dt['CAL-LAM']
        names = dt.index[np.argsort(lams)]
        lams = np.sort(lams)
        return names, lams


    def parse_files_old(self,indir,filebase):
        """
        Read wavelngth from the filename. Ned to replace and
        read it from header
        """

        print('parsing files')
        files = os.listdir(indir)
        lams = []
        for ff in files:
            if ff[:len(filebase)]==filebase:
                lams.append(ff.split(filebase)[1].split('.fits')[0])
        lams_f = np.array(lams,dtype='float')
        lams_fs = np.sort(lams_f)
        return lams_fs

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

        print('trying to do centroids')
        self.logger.info("Centroid Estimation")
        dt = self.context.data_set.data_table

        for scmode in ['MedRes-K']:
        #for scmode in ['LowRes-SED','LowRes-K','LowRes-L','LowRes-M','LowRes-H2O']:
            self.set_lamlimits(scmode)
            names, lams_fs = self.parse_files(dt,scmode)
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



            sparse.save_npz(self.action.args.dirname+'/'+
                            scmode+'_QL_rectmat.npz',QL_rmat)
            sparse.save_npz(self.action.args.dirname+'/'+
                            scmode+'_C2_rectmat.npz',C2_rmat)



            """
            x=[]
            y=[]
            for i in range(0,len(lams_fs)):
                x1 = cents[i,:,:,0]
                y1 = cents[i,:,:,1]
                x.append(x1)
                y.append(y1)

            file1_path = os.path.join(calib_path,'lowres_psf_x.pickle')
            file2_path = os.path.join(calib_path,'lowres_psf_y.pickle')
            file3_path = os.path.join(calib_path,'QL_rectmat.npz')
            file4_path = os.path.join(calib_path,'C2_rectmat.npz')

            with open(file1_path, 'wb') as file1:
                pickle.dump(x, file1)

            with open(file2_path, 'wb') as file2:
                pickle.dump(y, file2)

            sparse.save_npz(file3_path,QL_rmat)
            sparse.save_npz(file4_path,C2_rmat)
            """


            log_string = CentroidEstimate.__module__
            self.logger.info(log_string)
        return self.action.args
