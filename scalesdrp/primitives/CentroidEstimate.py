from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer

import pandas as pd
import numpy as np
import pickle
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pkg_resources
from scipy import sparse
import astropy.io.fits as pyfits
from scipy.optimize import curve_fit
import os


class CentroidEstimate(BasePrimitive):
    """
	Estimate the psf centroid of the calib images 
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger
        print("Pickle files created")

    def _perform(self):

        self.logger.info("Centroid Estimation")

#        is_obj = ('CALIB' in self.action.args.ccddata.header['IMTYPE'])
        
        calib_path = pkg_resources.resource_filename('scalesdrp','calib/')

        def parse_files(indir,filebase):
            files = os.listdir(indir)
            lams = []
            for ff in files:
                if ff[:len(filebase)]==filebase:
                    lams.append(ff.split(filebase)[1].split('.fits')[0])
            lams_f = np.array(lams,dtype='float')
            lams_fs = np.sort(lams_f)
            return lams_fs

        def load_calim(indir,filebase,lam):
            calim = pyfits.getdata(indir+filebase+str(lam)+'.fits')[0]
            return calim

        def read_fits(indir,file):
            calim = pyfits.getdata(indir+file)[0]
            return calim

        def get_init_centroid_region(lensx,lensy,xstart,ystart):
            spaxsize=19
            subf=8
            x = xstart + spaxsize*lensx
            y = ystart + spaxsize*lensy
            xs = np.max([0,x-subf])
            ys = np.max([0,y-subf])
            xe = np.min([x+subf,2048])
            ye = np.min([y+subf,2048])
            return xs,ys,xe,ye

        def crop_cube(cube,xs,ys,xe,ye):
            return cube[:,ys:ye,xs:xe]

        def crop_image(image,xs,ys,xe,ye):
            return image[ys:ye,xs:xe]

        def pixel_centroid(image):
            peak = np.where(image==np.max(image))
            x,y = peak[1][0],peak[0][0]
            return x,y

        def replace_edge_centroids(xc,yc):
            if 2048-xc < 2:
                return np.nan,np.nan
            if 2048-yc < 2:
                return np.nan,np.nan
            else:
                return xc,yc

        def get_init_centers_lowres(indir,filebase,lams_fs,calim):
            xstart = 3
            ystart = 3
            xcen=[]
            ycen=[]
            cents = np.zeros([108,108,2])
            for lensx in range(108):
                for lensy in range(108):
                    xs,ys,xe,ye = get_init_centroid_region(lensx,lensy,xstart,ystart)
                    calcrop = crop_image(calim,xs,ys,xe,ye)
                    cx,cy = pixel_centroid(calcrop)
                    if (lensx==0 and lensy==0):
                        xstart,ystart=cx+xs,cy+ys
                    cx,cy = replace_edge_centroids(cx,cy)
                    cents[lensy,lensx] = [cy+ys,cx+xs]

                    xcen.append(cents[0])
                    ycen.append(cents[1])

            return xcen,ycen


        def gaussian2d(xy, A, x0, y0, sigma_x, sigma_y):
            x, y = xy
            return A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))


        def get_best_centers_lowres(indir,filebase,lams_fs,calim,xstart,ystart):
            cents = np.zeros([108,108,2])
            xcen=[]
            ycen=[]
            amp=[]
            xcen_err=[]
            ycen_err=[]
            for lensy in range(0,108):
                for lensx in range(0,108):
                    xs,ys,xe,ye = get_init_centroid_region(lensx,lensy,xstart,ystart)
                    xs,ys = replace_edge_centroids(xs,ys)
                    if not np.isnan(xs) or not np.isnan(ys):
                        calcrop = crop_image(calim,xs,ys,xe,ye)
                        cx,cy = pixel_centroid(calcrop)
                        calcrop = calcrop / np.max(calcrop)
                        try:
                            amp = np.max(calcrop)
                            x0 = cx
                            y0 = cy
                            rows, cols = calcrop.shape
                            y = np.arange(rows)
                            x = np.arange(cols)
                            X, Y = np.meshgrid(x, y)
                            popt, pcov = curve_fit(gaussian2d, (X.ravel(),Y.ravel()),calcrop.ravel(), p0=(amp, cx, cy, 1.0, 1.1),maxfev = 2000,bounds = ([0, 0, 0, 0.1, 0.1], [2, 15, 15, 3, 3]))
                            A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit = popt
                            perr = np.sqrt(np.diag(pcov))
                            xcen.append(x0_fit+xs)
                            ycen.append(y0_fit+ys)
                            xcen_err.append(sigma_x_fit)
                            ycen_err.append(sigma_y_fit)
                        except (RuntimeError, ValueError) as e:
                            xcen.append(cx+xs)
                            ycen.append(cy+ys)
                            xcen_err.append(0.95)
                            ycen_err.append(0.95)
                    else:
                        xcen.append(np.nan)
                        ycen.append(np.nan)
                        xcen_err.append(np.nan)
                        ycen_err.append(np.nan)
            return xcen,ycen,xcen_err,ycen_err


        filebase = 'calibration_2.0_5.2_'
        lams_fs = parse_files(calib_path,filebase)
        x=[]
        y=[]
        xerr=[]
        yerr=[]
        xstart=[0,0,0,0,0,1,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,5,6,6,6,7,7,7,8,8,9,9,9,10,10,10,11,11,11,12,13,13,13,14,14,15,15,16,16,17,17,18,18]

        ystart=[0,0,1,2,2,3,4,4,5,6,7,7,8,9,10,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,36,37,38,39,41,42,43,45,46,48,49,51,52,54,55]

        for files in range(0,len(lams_fs)):
            calim = load_calim(calib_path,filebase,lams_fs[files])
            xcen,ycen,x_sig,y_sig = get_best_centers_lowres(calib_path,filebase,lams_fs,calim,xstart[files],ystart[files])
            x.append(xcen)
            y.append(ycen)
            xerr.append(x_sig)
            yerr.append(y_sig)

        file1_path = os.path.join(calib_path,'lowres_psf_x.pickle')
        file2_path = os.path.join(calib_path,'lowres_psf_y.pickle')
        file3_path = os.path.join(calib_path,'lowres_psf_xerr.pickle')
        file4_path = os.path.join(calib_path,'lowres_psf_yerr.pickle')


        with open(file1_path, 'wb') as file1:
            pickle.dump(x, file1)

        with open(file2_path, 'wb') as file2:
            pickle.dump(y, file2)

        with open(file3_path, 'wb') as file3:
            pickle.dump(xerr, file3)

        with open(file4_path, 'wb') as file4:
            pickle.dump(yerr, file4)

        log_string = CentroidEstimate.__module__
        self.logger.info(log_string)
        return self.action.args
    # END: class CentroidEstimate()
