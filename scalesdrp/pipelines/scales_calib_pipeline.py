"""
Class definition for the SCALES Pipeline

The event_table defines the recipes with each type of image having a unique
entry point and trajectory in the table.

The action_planner sets up the parameters for each image type before the event
queue in the KeckDRPFramework gets started.

"""

from keckdrpframework.pipelines.base_pipeline import BasePipeline
from keckdrpframework.models.processing_context import ProcessingContext
from scalesdrp.primitives.CalibFilePrimitives import *
from scalesdrp.primitives import *
from scalesdrp.primitives.scales_file_primitives import *
#import scalesdrp.primitives

class Scales_Calib_Pipeline(BasePipeline):
    """
    Pipeline to process SCALES calibration data

    """
    name = 'SCALES-DRP'

    event_table = {

        "add_only":                 ("add_to_dataframe_only", None, None),

        # For every file do this
        "next_file":                 ("ingest_file",
                                      "ingest_file_started",
                                      "file_ingested"),

        "file_ingested":             ("action_planner", None, None),

        #CALIB PROCESSING
        "process_calib":             ("ProcessCalframe",
                                      "calib_processing_started",
                                      "calib_start"),

        "calib_start":               ("StartCalib",
                                      "Raw_calibration_file_processing_started",
                                      "calib_file_done1"),     

        "calib_file_done1":            ("check_calib_complete", None, None),



        "calib_process_done":        ("ProcessCalframe",
                                      None,
                                      "calib_post_process"),   

        "calib_post_process":        ("CalibPostProcess",
                                      "calib_post_processing_started",
                                      None),

    }

    def __init__(self, context: ProcessingContext):
        """
        Constructor
        """
        BasePipeline.__init__(self, context)
        self.cnt = 0

    def add_to_dataframe_only(self, action, context):
        self.context.pipeline_logger.info("******* ADD to DATAFRAME ONLY: %s" % action.args.name)
        return action.args

    def check_calib_complete(self, action, context):
        self.cnt += 1
        imtypes = ['BIAS', 'DARK', 'FLATLAMP', 'CALUNIT']
        n_calib_files = (
            context.data_set.data_table.IMTYPE.isin(imtypes).sum())
        context.pipeline_logger.info(
            f"Stage-1 calibration complete: "
            f"{self.cnt}/{n_calib_files}")
        if self.cnt == n_calib_files:
            context.pipeline_logger.info(
                "All Stage-1 calibration files completed. "
                "Starting Post Processing.")
            context.push_event(
                "calib_process_done",
                action.args)
        return action.args

    def action_planner(self, action, context):
        try:
            self.context.pipeline_logger.info(
                "******* FILE TYPE DETERMINED AS %s" % action.args.imtype
            )
        except (AttributeError, TypeError, ValueError):
            self.context.pipeline_logger.warning(
                "******* FILE TYPE is NOT determined. "
                "No processing is possible."
            )
            return False

        camera = action.args.ccddata.header['CAMERA'].upper()
        self.context.pipeline_logger.info(
            "******* Observing MODE is %s " % camera
        )

        if action.args.in_proctab:
            if len(action.args.last_suffix) > 0:
                self.context.pipeline_logger.warning(
                    "Already processed (already in proctab up to %s)"
                    % action.args.last_suffix
                )
            else:
                self.context.pipeline_logger.warning(
                    "Already processed (already in proctab)"
                )

        if action.args.in_proctab and not context.config.instrument.clobber:
            self.context.pipeline_logger.warning("Pushing noop to queue")
            context.push_event("noop", action.args)

        elif action.args.imtype in ["BIAS", "DARK", "FLATLAMP", "CALUNIT"]:
            context.push_event("process_calib", action.args)

        return True


if __name__ == "__main__":
    """
    Standalone test
    """
    pass
