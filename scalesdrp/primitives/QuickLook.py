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
        self.logger.info("+++++++++++ Quicklook Started +++++++++++")

        ols_t_global = None

        def ols_pack_parms(a, b, c):
            return np.array([a, b, c])
        def ols_unpack_parms(p):
            a_p, b_p, c_p = p
            return a_p, b_p, c_p

        def ols_model_fn(p_model):
            a_m, b_m, c_m = ols_unpack_parms(p_model)
            global ols_t_global
            if ols_t_global is None:
                raise ValueError("Global time array 'ols_t_global' for OLS model_fn is not set.")
            return a_m + b_m * ols_t_global + c_m * ols_t_global**2

        def create_slope_image(ramp):
            nim_s = ramp.shape[0]
            ols_t_global = np.arange(nim_s)
            readtimes_for_covar_sci = np.arange(nim_s).astype(float)
            B_ols_sci = np.zeros((2048, 2048), dtype=float)
            for i_r in range(2048):
                for j_c in range(2048):
                    a_g=ramp[0,i_r,j_c]
                    b_g=(ramp[1,i_r,j_c]-ramp[0,i_r,j_c]) if nim_s>1 else 1.0; c_g=0.0
                    sp=ols_pack_parms(a_g,b_g,c_g); imdat_p=ramp[:,i_r,j_c]
                    w_idx=np.where(imdat_p < SATURATION_SCIENCE_OLS)[0]
                    if len(w_idx)<3: B_ols_sci[i_r,j_c]=np.nan; continue
                    imdat_v = imdat_p[w_idx]
                    def resid_fn_sci_local(p_loc): mimdat_f=ols_model_fn(p_loc)
                    return imdat_v - mimdat_f[w_idx]
                    try: 
                        p_opt_s,ier_s=leastsq(resid_fn_sci_local,sp.copy())
                        if ier_s not in [1,2,3,4]: p_opt_s = np.array([np.nan]*3)
                    except: p_opt_s=np.array([np.nan]*3)
                    _ ,b_fit_s,_ = ols_unpack_parms(p_opt_s); B_ols_sci[i_r,j_c]=b_fit_s
        
            median_B_sci_val=np.nanmedian(B_ols_sci)
            B_ols_sci[np.isnan(B_ols_sci)]= median_B_sci_val if not np.isnan(median_B_sci_val) else 0.0
            B_ols_sci[B_ols_sci<0]=0 
            return B_ols_sci

        input_data = self.action.args.ccddata.data
        ramp_img = create_slope_image(input_data)



        log_string = QuickLook.__module__
        self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)


        #if is_obj:	    
        #    scales_fits_writer(self.action.args.ccddata, table=self.action.args.table,output_file=self.action.args.name,output_dir=self.config.instrument.output_directory,suffix="quicklook_cube")

        #self.context.proctab.update_proctab(frame=self.action.args.ccddata, suffix="opt", newtype='OBJECT',
        #        filename=self.action.args.ccddata.header['OFNAME'])
        #self.context.proctab.write_proctab(
        #        tfil=self.config.instrument.procfile)

        return self.action.args
    # END: class QuickLook()
