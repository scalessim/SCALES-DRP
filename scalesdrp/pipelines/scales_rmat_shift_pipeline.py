"""
Class definition for the SCALES rectmat-shift pipeline

Takes single monochromator (CALUNIT) frames, makes an L1 slope image for each
exactly as the standalone calibration pipeline does (StartCalibSA), measures
the spot shift relative to the master calunit image at the same wavelength in
an existing monochromator scan, and writes shifted OPT and C2 interpolated
rectmats (RmatShift).
"""

import os

from keckdrpframework.pipelines.base_pipeline import BasePipeline
from keckdrpframework.models.processing_context import ProcessingContext
from scalesdrp.primitives.scales_file_primitives import *


class Scales_Rmat_Shift_Pipeline(BasePipeline):
    """
    Pipeline to measure rectmat shifts from single monochromator frames
    """
    name = 'SCALES-DRP'

    event_table = {
        "add_only":                  ("add_to_dataframe_only", None, None),

        # For every file do this
        "next_file":                 ("ingest_file",
                                      "ingest_file_started",
                                      "file_ingested"),
        "file_ingested":             ("action_planner", None, None),

        # MONOCHROMATOR FRAME -> L1 -> SHIFTED RECTMATS
        "process_mono":              ("ProcessObject",
                                      "mono_processing_started",
                                      "mono_l1"),
        "mono_l1":                   ("StartCalibSA",
                                      "mono_l1_started",
                                      "rmat_shift"),
        "rmat_shift":                ("RmatShift",
                                      "rmat_shift_started",
                                      None),
    }

    def __init__(self, context: ProcessingContext):
        """
        Constructor
        """
        BasePipeline.__init__(self, context)
        self.cnt = 0

    def add_to_dataframe_only(self, action, context):
        self.context.pipeline_logger.info(
            "******* ADD to DATAFRAME ONLY: %s" % action.args.name)
        return action.args

    def action_planner(self, action, context):
        try:
            self.context.pipeline_logger.info(
                "******* FILE TYPE DETERMINED AS %s" % action.args.imtype)
        except (AttributeError, TypeError, ValueError):
            self.context.pipeline_logger.warning(
                "******* FILE TYPE is NOT determined. "
                "No processing is possible.")
            return False

        # The proctab is not used to skip frames here: StartCalibSA reuses an
        # existing L1 when clobber = False, and the shift is always remeasured.
        imtype = str(action.args.imtype).strip().upper()

        # Background frames get the same L1 processing as CALUNIT frames
        # (StartCalibSA only writes an L1 for CALUNIT), whatever their IMTYPE
        bkg = getattr(context.config.instrument, "rmat_background_frames", None) or []
        if isinstance(bkg, str):
            bkg = [bkg]
        if os.path.basename(action.args.name) in {os.path.basename(b) for b in bkg}:
            if imtype != "CALUNIT":
                hdr = action.args.ccddata.header
                hdr["ORIGIMTY"] = (imtype, "IMTYPE before rmat-shift background use")
                hdr["IMTYPE"] = "CALUNIT"
                hdr["HISTORY"] = ("IMTYPE set to CALUNIT to reduce this frame "
                                  "as a monochromator background")
                action.args.imtype = "CALUNIT"
            self.context.pipeline_logger.info(
                f"******* {action.args.name} is a background frame")
            context.push_event("process_mono", action.args)
        elif imtype == "CALUNIT":
            context.push_event("process_mono", action.args)
        else:
            self.context.pipeline_logger.warning(
                f"******* {action.args.name} has IMTYPE={imtype}; only "
                "CALUNIT (monochromator) frames can be used to measure "
                "rectmat shifts. Skipping.")
        return True


if __name__ == "__main__":
    """
    Standalone test
    """
    pass
