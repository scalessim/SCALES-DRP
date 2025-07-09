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
#from scalesdrp.primitives import CentroidEstimate

class Scales_Calib_Pipeline(BasePipeline):
    """
    Pipeline to process SCALES calibration data

    """
    name = 'SCALES-DRP'

    event_table = {

        "add_only":                 ("add_to_dataframe_only", None, None),

        "next_file":                 ("ingest_file",
                                      "ingest_file_started",
                                      "file_ingested"),
        "file_ingested":             ("action_planner", None, None),

        #"add_only":                 ("AddToCalDataFrame", None, None),

        # BIAS PROCESSING
        
        "process_bias":              ("ProcessBias",
                                      "bias_processing_started",
                                      "bias_subtract_overscan"),
        "bias_subtract_overscan":    ("SubtractOverscan",
                                      "subtract_overscan_started",
                                      "bias_trim_overscan"),
        "bias_trim_overscan":        ("TrimOverscan",
                                      "trim_overscan_started",      # intb
                                      "bias_make_master"),
        "bias_make_master":          ("MakeMasterBias",
                                      "master_bias_started",        # mbias
                                      None),


        # CALIB PROCESSING
        
        #"next_file":                ("ingest_file",
        #                              "ingest_file_started",
        #                              "file_ingested"),

        "centroid_estimate":         ("CentroidEstimate",
                                      None,
                                      None),
    }


    def __init__(self, context: ProcessingContext):
        """
        Constructor
        """
        BasePipeline.__init__(self, context)
        print('initialized basepipeline')
        self.cnt = 0


    def add_to_dataframe_only(self, action, context):
        self.context.pipeline_logger.info("******* ADD to DATAFRAME ONLY: %s" %
                                          action.args.name)
        return action.args


        
       # if action.args.imtype=='CALUNIT':
       #     context.push_event("

        return action.args
