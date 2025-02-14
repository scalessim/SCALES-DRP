from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer

import numpy as np
import pkg_resources
import pickle
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
from scipy.optimize import minimize




class LeastExtract(BasePrimitive):
    """
	Chi extraction trial 1
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger
        print("Chi square Extract object created")

    def _perform(self):

        # parameters
        # image sections for each amp

#	self.context.proctab = Proctab()
#    	self.context.proctab.read_proctab(framework.config.instrument.procfile)



        self.logger.info("Chi Square Extraction Started")
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
        x_err = pd.read_pickle(calib_path+'psf_xerr.pickle')
        y_err = pd.read_pickle(calib_path+'psf_yerr.pickle')

        readnoise = 5.0 
        gain = 1.20 
        data_crop = self.action.args.ccddata.data
        
        final_flux = []

        for m in range(6555,6556):

            data_y1=[]
            data_x1=[]
            data_x_err1=[]
            data_y_err1=[]

            for mm in range(0,len(x_new)):
                if ~np.isnan(y_new[mm][m]) or ~np.isnan(x_new[mm][m]):
                    data_y1.append(y_new[mm][m])
                    data_x1.append(x_new[mm][m])

                    data_x_err1.append(x_err[mm][m])
                    data_y_err1.append(y_err[mm][m])
                else:
                    final_flux.append(np.nan)
        data_y = []
        data_x = []
        amp = []

        for k in range(0,len(data_x1),1):
            if data_x1[k] < 1.0 :
                data_x1[k] = 1.0
            if data_y1[k] < 1.0 :
                data_y1[k] = 1.0

            amp1 = data_crop[int(np.round(data_y1[k]))-1:int(np.round(data_y1[k]))+1,int(np.round(data_x1[k]))-1:int(np.round(data_x1[k]))+1]
            print(amp1)
#            amp.append(np.max(amp1))
           

#            data_y.append(int(np.round(data_y1[k])))
#            data_x.append(int(np.round(data_x1[k])))

        n_gaussians = len(data_x1)
        x_centroids = data_x1
        y_centroids = data_y1
        sigma_x = np.array(data_x_err1)
        sigma_y = np.array(data_y_err1)
#        initial_amplitudes = amp

        ly=int(abs(data_y1[0]))-2
        uy=int(abs(data_y1[len(data_y1)-1]))+2

        lx=int(abs(data_x1[0]))-2
        ux=int(abs(data_x1[len(data_x1)-1]))+2

        data = data_crop[ly:uy, lx:ux]

        x = np.linspace(lx, ux,data.shape[1])
        y = np.linspace(ly, uy,data.shape[0])
        X, Y = np.meshgrid(x, y)

#        result = minimize(
#            total_residual,
#            initial_amplitudes,
#            args=(X, Y, data, x_centroids, y_centroids, sigma_x, sigma_y),
#            method='L-BFGS-B',
#            bounds=[(0, None)] * n_gaussians  # Amplitudes should be non-negative)

#        fitted_amplitudes = result.x

#        model_flux = np.sum([gaussian_2d(X, Y, x_centroids[j], y_centroids[j], sigma_x[j], sigma_y[j], fitted_amplitudes[j]) for j in range(n_gaussians)],axis=0)

#        residual_flux = data - model_flux






       
            
        # update with new image
#        self.action.args.ccddata.data = obj 

        log_string = LeastExtract.__module__
        self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)

        if self.config.instrument.saveintims:
            scales_fits_writer(
                self.action.args.ccddata, table=self.action.args.table,
                output_file=self.action.args.name,
                output_dir=self.config.instrument.output_directory,
                suffix="trim")

        if is_obj:	    
	        scales_fits_writer(self.action.args.ccddata, table=self.action.args.table,output_file=self.action.args.name,output_dir=self.config.instrument.output_directory,suffix="ec")
        self.context.proctab.update_proctab(frame=self.action.args.ccddata, suffix="ec", newtype='OBJECT',
                filename=self.action.args.ccddata.header['OFNAME'])
        self.context.proctab.write_proctab(
                tfil=self.config.instrument.procfile)
        return self.action.args
    # END: class LeastExtract()
