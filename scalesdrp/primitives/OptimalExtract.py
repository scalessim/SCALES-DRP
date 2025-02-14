from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer

import pandas as pd
import numpy as np
import pickle
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pkg_resources



class OptimalExtract(BasePrimitive):
    """
	Optimal extraction trial 1
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger
        print("Optimal Extract object created")

    def _perform(self):

        self.logger.info("Optimal Extraction")
        tab = self.context.proctab.search_proctab(
            frame=self.action.args.ccddata, target_type='OBJECT',
            nearest=True)
        self.logger.info("%d object frames found" % len(tab))

        is_obj = ('OBJECT' in self.action.args.ccddata.header['IMTYPE'])
	
        if len(tab) > 0:
            mdname = get_master_name(tab, target_type)
            print("*************** READING IMAGE: %s" % mdname)
            obj = kcwi_fits_reader(
                os.path.join(self.config.instrument.cwd, 'redux',
                             mdname))[0]
        
        calib_path = pkg_resources.resource_filename('scalesdrp','calib/')

        x_new = pd.read_pickle(calib_path+'psf_x.pickle')
        y_new = pd.read_pickle(calib_path+'psf_y.pickle')


        gain  = 1.20 #need to take from header and loop over ampplifier

        readnoise = 5.0

        spectra = []
        spectra_err = []

        data = self.action.args.ccddata.data
       
        filename = np.loadtxt(calib_path + 'fname.txt',unpack=True,dtype=str)

        for m in range(0,len(filename),1):
            calib1=fits.getdata(calib_path + filename[m])
            calib=calib1[0]
            data_y1 = y_new[m]
            data_x1 = x_new[m]
            spectra1 = []
            spectra1_err = []

            for k in range(0,len(data_x1),1):
                if ~np.isnan(data_y1[k]) or ~np.isnan(data_x1[k]):
                    data_y=int(np.round(data_y1[k]))
                    data_x=int(np.round(data_x1[k]))
                    calib_crop1 = calib[data_y-2:data_y+3,data_x-2:data_x+3]
                    calib_crop = calib_crop1/np.mean(calib_crop1)
                    calib_crop_err = np.sqrt(calib_crop*gain+readnoise**2)
                    calib_crop_err_mean = np.sqrt(np.sum(calib_crop_err**2)) / (len(calib_crop1)*readnoise)
                    calib_err = calib_crop*np.sqrt((calib_crop_err/calib_crop1)**2 + (calib_crop_err_mean/calib_crop)**2)
                    data_crop = data[data_y-2:data_y+3,data_x-2:data_x+3]
                    data_crop_err = np.sqrt(data_crop*gain+readnoise**2)
                    norm_data = data_crop * calib_crop
                    norm_data_err = norm_data*np.sqrt((data_crop_err/data_crop)**2+ (calib_err/calib_crop)**2)
                    flux = np.sum(norm_data)
                    flux_err = np.sqrt(np.sum(norm_data_err ** 2))
                    spectra1.append(flux)
                    spectra1_err.append(flux_err)
                else:
                    spectra1.append(np.nan)
                    spectra1_err.append(np.nan)

            spectra.append(spectra1)
            spectra_err.append(spectra1_err)

        d = []
        d_err = []
        for k in range(0,len(spectra),1):
            d1 = np.array(spectra[k]).reshape((108, 108))
            d1_err = np.array(spectra_err[k]).reshape((108, 108))

            d.append(d1)
            d_err.append(d1_err)

        d = np.array(d)
        d_err = np.array(d_err)

            
        # update with new image
        self.action.args.ccddata.data = d 

        log_string = OptimalExtract.__module__
        self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)


        if is_obj:	    
            scales_fits_writer(self.action.args.ccddata, table=self.action.args.table,output_file=self.action.args.name,output_dir=self.config.instrument.output_directory,suffix="opt")

        self.context.proctab.update_proctab(frame=self.action.args.ccddata, suffix="opt", newtype='OBJECT',
                filename=self.action.args.ccddata.header['OFNAME'])
        self.context.proctab.write_proctab(
                tfil=self.config.instrument.procfile)

        return self.action.args
    # END: class OptimalExtract()
