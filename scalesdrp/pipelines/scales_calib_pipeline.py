"""
Class definition for the SCALES Pipeline

The event_table defines the recipes with each type of image having a unique
entry point and trajectory in the table.

The action_planner sets up the parameters for each image type before the event
queue in the KeckDRPFramework gets started.

"""

from keckdrpframework.pipelines.base_pipeline import BasePipeline
from keckdrpframework.models.processing_context import ProcessingContext
from scalesdrp.primitives.scales_file_primitives import *


class Scales_Calib_Pipeline(BasePipeline):
    """
    Pipeline to process SCALES calibration data

    """
    name = 'SCALES-DRP'

    event_table = {
        # this method is used with the "group" option,
        # to ingest the data without triggering any processing.
        # it is defined lower in this file

        "add_only":                  ("add_to_dataframe_only", None, None),
        # For every file do this
        "next_file":                 ("ingest_file",
                                      "ingest_file_started",
                                      "file_ingested"),
        "file_ingested":             ("action_planner", None, None),
       

        # CALIB PROCESSING
        
        "process_bias":              ("ProcessBias",
                                      "calibration_processing_started",
                                      "bias_subtract_overscan"),
        "bias_subtract_overscan":    ("SubtractOverscan",
                                      "subtract_overscan_started",
                                      "bias_trim_overscan"),
        "bias_trim_overscan":        ("TrimOverscan",
                                      "trim_overscan_started",
                                      "bias_make_master"),
        "bias_make_master":          ("MakeMasterBias",
                                      "master_bias_started",
                                      "centroid_estimate"),
        "centroid_estimate":         ("CentroidEstimate",
                                      "Centroid estimation of the calib psfs started",
                                      None),

        "next_file_stop":            ("ingest_file", "file_ingested", None)
    }

    # event_table = scales_event_table

    def __init__(self, context: ProcessingContext):
        """
        Constructor
        """
        BasePipeline.__init__(self, context)
        self.cnt = 0

    def add_to_dataframe_only(self, action, context):
        self.context.pipeline_logger.info("******* ADD to DATAFRAME ONLY: %s" % action.args.name)
        return action.args

    def action_planner(self, action, context):
        try:
            self.context.pipeline_logger.info(
                "******* FILE TYPE DETERMINED AS %s" % action.args.imtype)
        except (AttributeError, TypeError, ValueError):
            self.context.pipeline_logger.warn(
                "******* FILE TYPE is NOT determined. "
                "No processing is possible.")
            return False

        groupid = action.args.groupid
        camera = action.args.ccddata.header['CAMERA'].upper()
        self.context.pipeline_logger.info("******* GROUPID is %s " %
                                          action.args.groupid)
        self.context.pipeline_logger.info(
            "******* STATEID is %s (%s) " %
            (action.args.ccddata.header["STATENAM"],
             action.args.ccddata.header["STATEID"]))
        self.context.pipeline_logger.info("******* CAMERA is %s " % camera)
        
        if "BIAS" in action.args.imtype:
            if action.args.ttime > 0:
                self.context.pipeline_logger.warn(
                    f"BIAS frame with exposure time = {action.args.ttime} "
                    f"> 0. Discarding.")
                return False
            bias_args = action.args
            bias_args.groupid = groupid
            bias_args.want_type = "BIAS"
            bias_args.new_type = "MBIAS"
            bias_args.min_files = context.config.instrument.bias_min_nframes
            bias_args.new_file_name = "master_bias_%s.fits" % groupid
            context.push_event("process_bias", bias_args)
        elif "DARK" in action.args.imtype:
            dark_args = action.args
            dark_args.groupid = groupid
            dark_args.want_type = "DARK"
            dark_args.new_type = "MDARK"
            dark_args.min_files = context.config.instrument.dark_min_nframes
            dark_args.new_file_name = "master_dark_%s.fits" % groupid
            dark_args.in_directory = "redux"
            context.push_event("process_dark", dark_args)
        elif "CONTBARS" in action.args.imtype:
            contbars_args = action.args
            contbars_args.groupid = groupid
            contbars_args.want_type = "CONTBARS"
            contbars_args.new_type = "MCBARS"
            contbars_args.min_files = \
                context.config.instrument.contbars_min_nframes
            contbars_args.new_file_name = "master_contbars_%s.fits" % groupid
            contbars_args.in_directory = "redux"
            context.push_event("process_contbars", contbars_args)
        elif "FLATLAMP" in action.args.imtype:
            flat_args = action.args
            flat_args.groupid = groupid
            flat_args.want_type = "FLATLAMP"
            flat_args.stack_type = "SFLAT"
            flat_args.new_type = "MFLAT"
            flat_args.min_files = context.config.instrument.flat_min_nframes
            flat_args.new_file_name = "master_flat_%s.fits" % groupid
            flat_args.in_directory = "redux"
            context.push_event("process_flat", flat_args)
        elif "DOMEFLAT" in action.args.imtype:
            flat_args = action.args
            flat_args.groupid = groupid
            flat_args.want_type = "DOMEFLAT"
            flat_args.stack_type = "SDOME"
            flat_args.new_type = "MDOME"
            flat_args.min_files = context.config.instrument.dome_min_nframes
            flat_args.new_file_name = "master_flat_%s.fits" % groupid
            flat_args.in_directory = "redux"
            context.push_event("process_flat", flat_args)
        elif "TWIFLAT" in action.args.imtype:
            flat_args = action.args
            flat_args.groupid = groupid
            flat_args.want_type = "TWIFLAT"
            flat_args.stack_type = "STWIF"
            flat_args.new_type = "MTWIF"
            flat_args.min_files = context.config.instrument.twiflat_min_nframes
            flat_args.new_file_name = "master_flat_%s.fits" % groupid
            flat_args.in_directory = "redux"
            context.push_event("process_flat", flat_args)
        elif "ARCLAMP" in action.args.imtype:
            arc_args = action.args
            arc_args.groupid = groupid
            arc_args.want_type = "ARCLAMP"
            arc_args.new_type = "MARC"
            arc_args.min_files = context.config.instrument.arc_min_nframes
            arc_args.new_file_name = "master_arc_%s.fits" % groupid
            arc_args.in_directory = "redux"
            context.push_event("process_arc", arc_args)
        return True


if __name__ == "__main__":
    """
    Standalone test
    """
    pass
