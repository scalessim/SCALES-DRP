from keckdrpframework.primitives.base_primitive import BasePrimitive
from kcwidrp.primitives.kcwi_file_primitives import kcwi_fits_writer

import pickle
import numpy as np
import pandas as pd
from astropy.io import fits
import cv2
import scipy.optimize as opt



class LeastExtract(BasePrimitive):
    """
	Chi square extraction trial 1
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger
        print("Optimal Extract object created")

    def _perform(self):
        self.logger.info("Chi square Extraction")
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

        path = '/Users/athira/anaconda3/envs/kcwidrp/lib/python3.7/site-packages/kcwidrp/calib/'
        x_new = pd.read_pickle(path+'x_cen.pickle')
        y_new = pd.read_pickle(path+'y_cen.pickle')
        x_err = pd.read_pickle(path+'x_cen_err.pickle')
        y_err = pd.read_pickle(path+'y_cen_err.pickle')
        
        nan_array = np.full(216, np.nan)
        for i in range(41,53,1):
            x_new[i] = np.concatenate((x_new[i],nan_array))
            y_new[i] = np.concatenate((y_new[i],nan_array))
            x_err[i] = np.concatenate((x_err[i],nan_array))
            y_err[i] = np.concatenate((y_err[i],nan_array))

        nan_array = np.full(108, np.nan)
        for i in range(26,41,1):
            x_new[i] = np.concatenate((x_new[i],nan_array))
            y_new[i] = np.concatenate((y_new[i],nan_array))
            x_err[i] = np.concatenate((x_err[i],nan_array))
            y_err[i] = np.concatenate((y_err[i],nan_array))

        x=[]
        y=[]
        xerr=[]
        yerr=[]
        for j in range(0,11664,1):
            x1=[]
            y1=[]
            xerr1=[]
            yerr1=[]
            for i in range(0,len(x_new)):
                x1.append(x_new[i][j])
                y1.append(y_new[i][j])
                xerr1.append(x_err[i][j])
                yerr1.append(y_err[i][j])
            x.append(x1)
            y.append(y1)
            xerr.append(xerr1)
            yerr.append(yerr1)

        
        def gaussian_2d(amplitude, x, y, x0, y0, sigma_x, sigma_y):
            return amplitude * np.exp(-((x - x0)**2 /(2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))

        def model(params, x, y, x_centroid,y_centroid, sigma_x,sigma_y):
            model_image = np.zeros_like(x)
            for i in range(50):
                amplitude = params[i]
                x0 = x_centroid[i]
                y0 = y_centroid[i]
                sigma_x1 = sigma_x[i]
                sigma_y1 = sigma_y[i]
                model_image += gaussian_2d(amplitude, x, y, x0, y0, sigma_x1, sigma_y1)
            return model_image

        def residuals(params, data, x, y, x_centroid,y_centroid,sigma_x,sigma_y):
            model_image = model(params, x, y, x_centroid,y_centroid, sigma_x,sigma_y)
            return (data - model_image).ravel()


        def get_parallel_line_points(x1, y1, x2, y2, offset):
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            dx = dx / length
            dy = dy / length
            perp_dx = -dy
            perp_dy = dx
            offset_x1 = x1 + perp_dx * offset
            offset_y1 = y1 + perp_dy * offset
            offset_x2 = x2 + perp_dx * offset
            offset_y2 = y2 + perp_dy * offset
            return (int(offset_x1), int(offset_y1)), (int(offset_x2), int(offset_y2))


        flux_sum_model = []
        flux_sum_data = []
        flux_sum_residual = []
        final_flux = []
        data_crop = self.action.args.ccddata.data

        for m in range(11440,len(x)):
            data_y11 = np.array(y[m])
            data_x11 = np.array(x[m])
            data_x_err1 = np.array(xerr[m])
            data_y_err1 = np.array(yerr[m])
            data_y1 = data_y11[~np.isnan(data_y11)]
            data_x1 = data_x11[~np.isnan(data_x11)]
            data_x_err = data_x_err1[~np.isnan(data_x_err1)]
            data_y_err = data_y_err1[~np.isnan(data_y_err1)]

            flux_sum_model1 = []
            flux_sum_data1 = []
            flux_sum_residual1 = []
            data_y=[]
            data_x=[]
            amp=[]
            dataa=[]
            f1=[]

            for k in range(0,len(data_x1),1):
                if data_x1[k] < 1.0 :
                    data_x1[k] = 1.0
                if data_y1[k] < 1.0 :
                    data_y1[k] = 1.0
                amp1 = data_crop[int(np.round(data_y1[k]))-1:int(np.round(data_y1[k]))+1,int(np.round(data_x1[k]))-1:int(np.round(data_x1[k]))+1]
                amp.append(np.max(amp1))
                data_y.append(int(np.round(data_y1[k])))
                data_x.append(int(np.round(data_x1[k])))

            x1,y1 = data_x[0]-1, data_y[0]-1
            x2,y2 = data_x[len(data_x)-1]+1, data_y[len(data_y)-1]+1
            offset = 2
            p1,p2 = get_parallel_line_points(x1, y1, x2, y2, offset)
            p3,p4 = get_parallel_line_points(x1, y1, x2, y2, -offset)
            points = np.array([p1, p2, p4, p3]
            mask = np.zeros_like(data_crop, dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            cropped_image = data_crop * mask
            xx, yy, w, h = cv2.boundingRect(points)
            data = cropped_image[yy:yy+h, xx:xx+w]

            k_val1 = data.shape
            kval,k2 =k_val1
            x = np.linspace(data_x[0]-1, data_x[len(data_x)-1]+1, len(data[0]))
            y = np.linspace(data_y[0]-1, data_y[len(data_y)-1]+1, kval
            x, y = np.meshgrid(x, y)

            x_centroid = data_x
            y_centroid = data_y
            sigma_x = np.array(data_x_err)
            sigma_y = np.array(data_y_err)
            true_amplitudes = amp
            initial_guess = amp

            result = opt.least_squares(residuals, initial_guess,args=(data, x, y, x_centroid,y_centroid, sigma_x,sigma_y,len(data_x)))
            fitted_amplitudes = result.x
            fitted_model = model(fitted_amplitudes, x, y, x_centroid,y_centroid,sigma_x,sigma_y,len(data_x))
            residuals_image = data - fitted_model
            if len(fitted_model) < 53:
                x = 53 -len(fitted_amplitudes)
                nan_array = np.full(x, np.nan)
                fitted_model = np.append(fitted_model, nan_array)
                final_flux.append(fitted_model)
        d = np.array(final_flux).reshape((108, 108,53))



        # update with new image
        self.action.args.ccddata.data = d 

        log_string = LeastExtract.__module__
        self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)

        if self.config.instrument.saveintims:
            kcwi_fits_writer(
                '
self.action.args.ccddata, table=self.action.args.table,
                output_file=self.action.args.name,
                output_dir=self.config.instrument.output_directory,
                suffix="trim")

        if is_obj:	    
	        kcwi_fits_writer(self.action.args.ccddata, table=self.action.args.table,output_file=self.action.args.name,output_dir=self.config.instrument.output_directory,suffix="lsex")
        self.context.proctab.update_proctab(frame=self.action.args.ccddata, suffix="ec", newtype='OBJECT',
                filename=self.action.args.ccddata.header['OFNAME'])
        self.context.proctab.write_proctab(
                tfil=self.config.instrument.procfile)
        return self.action.args
    # END: class LeastExtract()
