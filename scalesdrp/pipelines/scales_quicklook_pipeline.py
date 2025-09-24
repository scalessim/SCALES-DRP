"""
Class definition for the SCALES Pipeline

The event_table defines the recipes with each type of image having a unique
entry point and trajectory in the table.

The action_planner sets up the parameters for each image type before the event
queue in the KeckDRPFramework gets started.

"""

from keckdrpframework.pipelines.base_pipeline import BasePipeline
from keckdrpframework.models.processing_context import ProcessingContext
#from scalesdrp.primitives.scales_file_primitives import *


class Scales_Quicklook_Pipeline(BasePipeline):
    """
    Pipeline to process SCALES data

    """
    name = 'SCALES-DRP'

    event_table = {
        "add_only":                  ("add_to_dataframe_only", None, None),

        "next_file":                 ("QuickLook", None, None),
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

if __name__ == "__main__":
    """
    Standalone test
    """
    pass
