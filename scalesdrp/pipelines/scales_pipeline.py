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


class Scales_pipeline(BasePipeline):
    """
    Pipeline to process SCALES data

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
       

        # OBJECT PROCESSING
        "process_object":            ("ProcessObject",
                                      "object_processing_started",
                                      "ramp_fit"),
        "ramp_fit":                  ("RampFit",
                                      "science_ramp_fitting",
                                      "spectral_extract"),
        "spectral_extract":          ("SpectralExtract",
                                      "Extraction started",
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

        camera = action.args.ccddata.header['CAMERA'].upper()
        self.context.pipeline_logger.info("******* Observing MODE is %s " % camera)
        if action.args.in_proctab:
            if len(action.args.last_suffix) > 0:
                self.context.pipeline_logger.warn(
                    "Already processed (already in proctab up to %s)" %
                    action.args.last_suffix)
            else:
                self.context.pipeline_logger.warn(
                    "Already processed (already in proctab)")
        if action.args.in_proctab and not context.config.instrument.clobber:
            self.context.pipeline_logger.warn("Pushing noop to queue")
            context.push_event("noop", action.args)

        elif "OBJECT" in action.args.imtype:
            object_args = action.args
            object_args.new_type = "MOBJ"
            object_args.min_files = \
            context.config.instrument.object_min_nframes
            object_args.in_directory = "redux"
            context.push_event("process_object", object_args)
        return True


if __name__ == "__main__":
    """
    Standalone test
    """
    pass
