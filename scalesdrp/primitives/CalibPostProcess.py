from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
from scalesdrp.primitives.scales_file_primitives import fits_writer_calib
import scalesdrp.primitives.saturation_create as saturation_create
import scalesdrp.primitives.linearity_create as linearity_create
import scalesdrp.primitives.readnoise_create as rn_create
import scalesdrp.primitives.bpm_create as bpm_create
import scalesdrp.primitives.scales_basic as scbasic
from scalesdrp.core.scales_pkg_resources import get_resource_path
import scalesdrp.primitives.robust as robust
import numpy as np
import time
import os
from astropy.io import fits
from scipy import sparse
import scalesdrp.primitives.reference as reference #1/f and reference pixel correction
from astropy.nddata import StdDevUncertainty
from scalesdrp.core.scales_proctab import Proctab
import logging
log = logging.getLogger("SCALES")
import matplotlib.pyplot as plt

class CalibPostProcess(BasePrimitive):

    """
    This function correct a raw read for channel bias, alternative column noise, and 1/f noise.
    The core of this function is `import scalesdrp.primitives.reference as reference`. 
    No external calibration files are involved.
        Args:
            data_image: The (N,H,W) input ramp cube.

        Returns:
            data_image: The (N,H,W) detector level corrected ramp cube.
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger

        if not hasattr(self, "proctab") or self.proctab is None:
            self.proctab = Proctab(logger=self.logger if hasattr(self, "logger") else logging.getLogger("SCALES"))

    def _perform(self):
        self.logger.info("################# Started creating master calibration files")
        output_dir = self.config.instrument.output_directory
        plot_dir = self.config.instrument.plot_directory
        package = __name__.split('.')[0]
        calibfilepath = self.context.calib_file_path
        calib_dir = str(get_resource_path(package, calibfilepath))+'/'    
        create_linearity = bool(self.config.instrument.create_linearity_coeff)
        create_sat = bool(self.config.instrument.create_sat_map)
        create_bpm = bool(self.config.instrument.create_bpm)
        create_readnoise = bool(self.config.instrument.create_readnoise)

        linearity_groups = []
        saturation_files = []
        bpm_groups = []
        l1_groups = []
        readnoise_grp=[]

        proctab_path = os.path.join(output_dir,"scales.proc")
        self.proctab.read_proctab(proctab_path)
        if self.proctab.proctab is None:
            raise RuntimeError(f"Could not read processing table: {proctab_path}")
        proctab = self.proctab.proctab.to_pandas()
            
        l1_dt = scbasic.get_proc_entries(proctab,suffix="_L1")
        if len(l1_dt) == 0:
            msg = (
                "No L1 products were found in the processing table. "
                "Cannot create master calibration products.")
            self.logger.error(msg)
            raise RuntimeError(msg)

        l1_groups = scbasic.group_files_by_header(l1_dt)

        if len(l1_groups) == 0:
            msg = (
                "L1 products were found, but no valid "
                "calibration groups could be created.")
            self.logger.error(msg)
            raise RuntimeError(msg)

        self.logger.info(
            f"Created {len(l1_groups)} L1 calibration groups.")

        for group_index, group in enumerate(l1_groups, start=1):
            filenames = group["filenames"]
            params = group["params"]
            camera = group["params"]["camera"]
            mclock = group["params"]["mclock"]
            exptime = group["params"]["exptime"]
            filenames = group["filenames"]
            if not self.context.clobber:
                existing_master = (
                    scbasic.find_existing_master_for_l1_group(
                        l1_filenames=filenames,
                        redux_dir=output_dir))
                if existing_master is not None:
                    self.logger.info(
                        f"Master calibration already exists "
                        f"for group {group_index}/{len(l1_groups)}: "
                        f"{existing_master}. "
                        "Skipping master creation.")
                    continue
                
            self.logger.info(
                f"Building master calibration "
                f"{group_index}/{len(l1_groups)} "
                f"from {len(filenames)} L1 files.")
            self.logger.info(
                f"Group parameters: {params}")
                
            data_stack = []
            uncert_stack = []
            chisq_stack = []
            dq_stack = []
            hdrm = None

            for filename in filenames:
                filepath = os.path.join(output_dir,filename)
                    
                if not os.path.isfile(filepath):
                    msg = (
                        f"L1 file listed in processing table "
                        f"does not exist: {filepath}")
                    self.logger.error(msg)
                    raise RuntimeError(msg)

                with fits.open(filepath,memmap=False) as hdul:
                    data = np.asarray(hdul[0].data,dtype=np.float32)
                    hdrm = hdul[0].header.copy()

                    if data.ndim != 2:
                        raise ValueError(
                            f"L1 DATA must be 2-D. "
                            f"{filename} has shape {data.shape}.")

                    data_stack.append(data)

                    if "UNCERT" not in hdul:
                        raise RuntimeError(
                            f"UNCERT extension missing from "
                            f"{filename}.")
                    uncert = np.asarray(hdul["UNCERT"].data,dtype=np.float32)

                    if uncert.shape != data.shape:
                        raise ValueError(
                            f"UNCERT shape {uncert.shape} does not "
                            f"match DATA shape {data.shape} in "
                            f"{filename}.")
                    uncert_stack.append(uncert)
                        
                    if "CHISQ" not in hdul:
                        raise RuntimeError(
                            f"CHISQ extension missing from "
                            f"{filename}.")
                    chisq = np.asarray(
                        hdul["CHISQ"].data,
                        dtype=np.float32)

                    if chisq.shape != data.shape:
                        raise ValueError(
                            f"CHISQ shape {chisq.shape} does not "
                            f"match DATA shape {data.shape} in "
                            f"{filename}.")
                    chisq_stack.append(chisq)

                    if "DQ" not in hdul:
                        raise RuntimeError(
                            f"DQ extension missing from "
                            f"{filename}.")

                    dq = np.asarray(
                        hdul["DQ"].data,
                        dtype=np.uint16)

                    if dq.shape != data.shape:
                        raise ValueError(
                            f"DQ shape {dq.shape} does not "
                            f"match DATA shape {data.shape} in "
                            f"{filename}.")
                    dq_stack.append(dq)

            data_stack = np.stack(data_stack,axis=0)
            uncert_stack = np.stack(uncert_stack,axis=0)
            chisq_stack = np.stack(chisq_stack,axis=0)
            dq_stack = np.stack(dq_stack,axis=0)
            n_slopes = data_stack.shape[0]
                
            self.logger.info(f"Master input stack shape: {data_stack.shape}")
            combine_method = self.config.instrument.master_comb_met

            (master,
            master_uncert,
            master_chisq,
            master_dq) = scbasic.build_master_from_stack(
            data_stack=data_stack,
            sigma_stack=uncert_stack,
            chisq_stack=chisq_stack,
            quality_stack=dq_stack,
            method=combine_method)

            self.logger.info(
                f"Master calibration created for group "
                f"{group_index}.")


            imtype = str(params["imtype"]).strip().upper()
                
            if imtype == "DARK":
                hdrm["HISTORY"] = "Master detector DARK created"
                hdrm["HISTORY"] = (f"Combined {n_slopes} L1 DARK slope images")
                hdrm["HISTORY"] = (f"Master combination method: {combine_method}")
                hdrm["HISTORY"] = ("UNCERT propagated during master combination")
                hdrm["HISTORY"] = ("CHISQ propagated as median of contributing L1 values")
                hdrm["HISTORY"] = ("DQ propagated using bitwise OR of input L1 DQ maps")

                fits_writer_calib(
                    data=master,
                    header=hdrm,
                    redux_dir=output_dir,
                    input_filename=filename,
                    suffix="_mdark",
                    uncert=master_uncert,
                    dq=master_dq,
                    chisq=master_chisq)

                scbasic.proctab_update(
                    header=hdrm,
                    output_dir=output_dir,
                    input_filename=filename,
                    suffix="_mdark",
                    frame=None,
                    proctab=self.proctab)
                self.logger.info(
                    f"Master DARK created from {n_slopes} "
                    f"L1 slope images using {combine_method}.")

                plot_path = scbasic.plot_master_calib(
                    master=master,
                    master_dq=master_dq,
                    output_dir=plot_dir,
                    filename=f"master_{camera}_{mclock}_{imtype}.png",
                    title=f"Master Calibration: {camera} | {mclock} | {imtype}",
                    unit="DN/s",
                    exclude_bad=True,
                    show=False)
                self.logger.info(f"Master calibration diagnostic saved to {plot_path}")

            if imtype == "BIAS":
                hdrm["HISTORY"] = "Master detector BIAS created"
                hdrm["HISTORY"] = (f"Combined {n_slopes} L1 bias slope images")
                hdrm["HISTORY"] = (f"Master combination method: {combine_method}")
                hdrm["HISTORY"] = ("UNCERT propagated during master combination")
                hdrm["HISTORY"] = ("CHISQ propagated as median of contributing L1 values")
                hdrm["HISTORY"] = ("DQ propagated using bitwise OR of input L1 DQ maps")

                fits_writer_calib(
                    data=master,
                    header=hdrm,
                    redux_dir=output_dir,
                    input_filename=filename,
                    suffix="_mbias",
                    uncert=master_uncert,
                    dq=master_dq,
                    chisq=master_chisq)

                scbasic.proctab_update(
                    header=hdrm,
                    output_dir=output_dir,
                    input_filename=filename,
                    suffix="_mbias",
                    frame=None,
                    proctab=self.proctab)
                self.logger.info(
                    f"Master BIAS created from {n_slopes} "
                    f"L1 slope images using {combine_method}.")

                plot_path = scbasic.plot_master_calib(
                    master=master,
                    master_dq=master_dq,
                    output_dir=plot_dir,
                    filename=f"master_{camera}_{mclock}_{imtype}.png",
                    title=f"Master Calibration: {camera} | {mclock} | {imtype}",
                    unit="DN/s",
                    exclude_bad=True,
                    show=False)
                self.logger.info(f"Master calibration diagnostic saved to {plot_path}")                    

            if imtype == "FLATLAMP":
                hdrm["HISTORY"] = "Master detector flatlamp created"
                hdrm["HISTORY"] = (f"Combined {n_slopes} L1 flatlamp slope images")
                hdrm["HISTORY"] = (f"Master combination method: {combine_method}")
                hdrm["HISTORY"] = ("UNCERT propagated during master combination")
                hdrm["HISTORY"] = ("CHISQ propagated as median of contributing L1 values")
                hdrm["HISTORY"] = ("DQ propagated using bitwise OR of input L1 DQ maps")

                fits_writer_calib(
                    data=master,
                    header=hdrm,
                    redux_dir=output_dir,
                    input_filename=filename,
                    suffix="_mflatlamp",
                    uncert=master_uncert,
                    dq=master_dq,
                    chisq=master_chisq)

                scbasic.proctab_update(
                    header=hdrm,
                    output_dir=output_dir,
                    input_filename=filename,
                    suffix="_mflatlamp",
                    frame=None,
                    proctab=self.proctab)
                self.logger.info(
                    f"Master FLATLAMP created from {n_slopes} "
                    f"L1 slope images using {combine_method}.")

                plot_path = scbasic.plot_master_calib(
                    master=master,
                    master_dq=master_dq,
                    output_dir=plot_dir,
                    filename=f"master_{camera}_{mclock}_{imtype}.png",
                    title=f"Master Calibration: {camera} | {mclock} | {imtype}",
                    unit="DN/s",
                    exclude_bad=True,
                    show=False)
                self.logger.info(f"Master calibration diagnostic saved to {plot_path}")

            if imtype == "CALUNIT":
                monowave = str(params["monowave"]).strip().upper()
                wl_val = float(monowave)
                wl_str = f"{wl_val:.3f}".rstrip("0").rstrip(".")
                suffix = f"_{wl_str}_mcalunit"
                hdrm["HISTORY"] = "Master detector CALUNIT created"
                hdrm["HISTORY"] = (f"Combined {n_slopes} L1 calunit slope images")
                hdrm["HISTORY"] = (f"Master combination method: {combine_method}")
                hdrm["HISTORY"] = (f"Master wavelength combined: {monowave}")
                hdrm["HISTORY"] = ("UNCERT propagated during master combination")
                hdrm["HISTORY"] = ("CHISQ propagated as median of contributing L1 values")
                hdrm["HISTORY"] = ("DQ propagated using bitwise OR of input L1 DQ maps")

                fits_writer_calib(
                    data=master,
                    header=hdrm,
                    redux_dir=output_dir,
                    input_filename=filename,
                    suffix=suffix,
                    uncert=master_uncert,
                    dq=master_dq,
                    chisq=master_chisq)

                scbasic.proctab_update(
                    header=hdrm,
                    output_dir=output_dir,
                    input_filename=filename,
                    suffix=suffix,
                    frame=None,
                    proctab=self.proctab)
                self.logger.info(
                    f"Master CALUNIT created from {n_slopes} "
                    f"L1 slope images using {combine_method}.")

                plot_path = scbasic.plot_master_calib(
                    master=master,
                    master_dq=master_dq,
                    output_dir=plot_dir,
                    filename=f"master_{camera}_{mclock}_{imtype}.png",
                    title=f"Master Calibration: {camera} | {mclock} | {imtype}",
                    unit="DN/s",
                    exclude_bad=True,
                    show=False)
                self.logger.info(f"Master calibration diagnostic saved to {plot_path}")

        log_string = CalibPostProcess.__module__
        self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)

        self.logger.info("+++++++++++ All calibration processing are completed +++++++++++")

        return self.action.args
    # END: class CalibPostProcess()
