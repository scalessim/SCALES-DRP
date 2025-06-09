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

        # CALIB PROCESSING
        
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


