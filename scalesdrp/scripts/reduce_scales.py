"""
Reduce SCALES data using the SCALES DRP running on the KeckDRPFramework
"""
from keckdrpframework.core.framework import Framework
from keckdrpframework.config.framework_config import ConfigClass
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.utils.drpf_logger import getLogger
import subprocess
import time
import datetime
import argparse
import sys
import traceback
import os
import pkg_resources
import psutil
import shutil

from scalesdrp.pipelines.scales_pipeline import Scales_pipeline
from scalesdrp.core.scales_proctab import Proctab
import logging.config


def _parse_arguments(in_args: list) -> argparse.Namespace:
    description = "SCALES pipeline CLI"

    # this is a simple case where we provide a frame and a configuration file
    parser = argparse.ArgumentParser(prog=f"{in_args[0]}",
                                     description=description)
    parser.add_argument('-c', '--config', dest="SCALES_config_file", type=str,
                        help="SCALES configuration file", default=None)
    parser.add_argument('--write_config', dest="write_config",
                        help="Write out an editable config file in current dir"
                        " (scales.cfg)", action="store_true", default=False)
    parser.add_argument('-f', '--frames', nargs='*', type=str,
                        help='input image files (full path, list ok)',
                        default=None)
    parser.add_argument('-l', '--list', dest='file_list',
                        help='File containing a list of files to be processed',
                        default=None)
    parser.add_argument('-g', '--groups', dest='group_mode',
                        help='Use group mode: separate files by image type and '
                             'reduce in the correct order',
                        default=False, action='store_true')

    parser.add_argument('-i', '--infiles', dest="infiles",
                        help="Input files, or pattern to match", nargs="?")
    parser.add_argument('-d', '--directory', dest="dirname", type=str,
                        help="Input directory", nargs='?', default=None)
    # after ingesting the files,
    # do we want to continue monitoring the directory?
    parser.add_argument('-m', '--monitor', dest="monitor",
                        help='Continue monitoring the directory '
                             'after the initial ingestion',
                        action='store_true', default=False)

    # special arguments, ignore
    parser.add_argument("-I", "--ingest_data_only", dest="ingest_data_only",
                        action="store_true",
                        help="Ingest data and terminate")
    parser.add_argument("-w", "--wait_for_event", dest="wait_for_event",
                        action="store_true", help="Wait for events")
    parser.add_argument("-W", "--continue", dest="continuous",
                        action="store_true",
                        help="Continue processing, wait for ever")
    parser.add_argument("-s", "--start_queue_manager_only",
                        dest="queue_manager_only", action="store_true",
                        help="Starts queue manager only, no processing",)

    # scales specific parameters
    parser.add_argument("-p", "--proctab", dest='proctab', help='Proctab file',
                        default=None)
    #parser.add_argument("-k", "--skipsky", dest='skipsky', action="store_true",
    #                    default=False, help="Skip sky subtraction")

    out_args = parser.parse_args(in_args[1:])
    return out_args


def check_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print("Directory %s has been created" % directory)


def main():

    # Package
    pkg = 'scalesdrp'

    # get arguments
    args = _parse_arguments(sys.argv)

    if args.write_config:
        dest = os.path.join(os.getcwd(), 'scales.cfg')
        if os.path.exists(dest):
            print("Config file scales.cfg already exists in current dir")
        else:
            scales_config_file = 'configs/scales.cfg'
            scales_config_fullpath = pkg_resources.resource_filename(
                pkg, scales_config_file)
            shutil.copy(scales_config_fullpath, os.getcwd())
            print("Copied scales.cfg into current dir.  Edit and use with -c")
        sys.exit(0)

    # This check can be removed once reduce_scales processes are
    # siloed against each other
    plist = []
    for p in psutil.process_iter():
        try:
            if "reduce_scales" in p.name():
                plist.append(p)
        except psutil.NoSuchProcess:
            continue
    # plist = [p for p in psutil.process_iter() if "reduce_scales" in p.name()]
    if len(plist) > 1:
        print("DRP already running in another process, exiting")
        sys.exit(0)

    def process_subset(in_subset):
        for in_frame in in_subset.index:
            arguments = Arguments(name=in_frame)
            framework.append_event('next_file', arguments, recurrent=True)
    
    def process_list(in_list):
        for in_frame in in_list:
            arguments = Arguments(name=in_frame)
            framework.append_event('next_file', arguments, recurrent=True)

    if args.file_list:
        if '.fits' in args.file_list:
            print("\nERROR - trying to read in fits file as file list\n\n"
                  "Please use -f or --frames for direct input of fits files\n")
            sys.exit(0)

    # START HANDLING OF CONFIGURATION FILES ##########

    # check for the logs diretory
    check_directory("logs")
    # check for the plots directory
    check_directory("plots")

    framework_config_file = "configs/framework.cfg"
    framework_config_fullpath = \
        pkg_resources.resource_filename(pkg, framework_config_file)

    framework_logcfg_file = 'configs/logger.cfg'
    framework_logcfg_fullpath = \
        pkg_resources.resource_filename(pkg, framework_logcfg_file)

    # add scales specific config files # make changes here to allow this file
    # to be loaded from the command line
    if args.SCALES_config_file is None:
        scales_config_file = 'configs/scales.cfg'
        scales_config_fullpath = pkg_resources.resource_filename(
            pkg, scales_config_file)
        scales_config = ConfigClass(scales_config_fullpath, default_section='SCALES')
    else:
        # scales_config_fullpath = os.path.abspath(args.scales_config_file)
        scales_config = ConfigClass(args.SCALES_config_file, default_section='SCALES')

    # END HANDLING OF CONFIGURATION FILES ##########

    # Add current working directory to config info
    scales_config.cwd = os.getcwd()

    # check for the output directory
    check_directory(scales_config.output_directory)

    try:
        framework = Framework(Scales_pipeline, framework_config_fullpath)
        # add this line ONLY if you are using a local logging config file
        logging.config.fileConfig(framework_logcfg_fullpath)
        framework.config.instrument = scales_config
    except Exception as e:
        print("Failed to initialize framework, exiting ...", e)
        traceback.print_exc()
        sys.exit(1)
    framework.context.pipeline_logger = getLogger(framework_logcfg_fullpath,
                                                  name="SCALES")
    framework.logger = getLogger(framework_logcfg_fullpath, name="DRPF")

    if args.infiles is not None:
        framework.config.file_type = args.infiles

    # update proc table argument
    if args.proctab:
        framework.context.pipeline_logger.info(
            "Using proc table file %s" % args.proctab
        )
        framework.config.instrument.procfile = args.proctab

    framework.config.instrument.arc_min_nframes = \
	scales_config.arc_min_nframes
    framework.config.instrument.contbars_min_nframes = \
	scales_config.contbars_min_nframes = \
    framework.config.instrument.object_min_nframes = \
	scales_config.object_min_nframes
    framework.config.instrument.minoscanpix = scales_config.minoscanpix
    framework.config.instrument.oscanbuf = scales_config.oscanbuf

    # initialize the proctab and read it
    framework.context.proctab = Proctab()
    framework.context.proctab.read_proctab(framework.config.instrument.procfile)

    framework.logger.info("Framework initialized")

    # which initiates the processing
    if args.group_mode is True:
        # set the default ingestion event to None
        framework.config.default_ingestion_event = "add_only"

    # start queue manager only (useful for RPC)
    if args.queue_manager_only:
        # The queue manager runs forever.
        framework.logger.info("Starting queue manager only, no processing")
        framework.start(args.queue_manager_only)

    # single frame processing
    elif args.frames:
        frames = []
        for frame in args.frames:
            # Verify we have the correct channel selected
            #if args.lowres and 'mr' in frame:
            #    print('low-res channel requested, but medium-res files in list')
            #    qstr = input('Proceed? <cr>=yes or Q=quit: ')
            #    if 'Q' in qstr.upper():
            #        frames = []
            #        break
            #if args.medres and 'lr' in frame:
            #    print('med-res channel requested, but low-res files in list')
            #    qstr = input('Proceed? <cr>=yes or Q=quit: ')
            #    if 'Q' in qstr.upper():
            #        frames = []
            #        break
            frames.append(frame)
        framework.ingest_data(None, frames, False)

    # processing of a list of files contained in a file
    elif args.file_list:
        frames = []
        with open(args.file_list) as file_list:
            for frame in file_list:
                if "#" not in frame:
                    # Verify we have the correct channel selected
                    #if args.lowres and 'mr' in frame:
                    #    print('	Low-res channel requested, but med-res files in list')
                    #    qstr = input('Proceed? <cr>=yes or Q=quit: ')
                    #    if 'Q' in qstr.upper():
                    #        frames = []
                    #        break
                    #if args.medres and 'lr' in frame:
                    #    print('Med-res channel requested, but low-res files in list')
                    #    qstr = input('Proceed? <cr>=yes or Q=quit: ')
                    #    if 'Q' in qstr.upper():
                    #        frames = []
                    #        break
                    frames.append(frame.strip('\n'))

        framework.ingest_data(None, frames, False)

        with open(args.file_list + '_ingest', 'w') as ingest_f:
            ingest_f.write('Files ingested at: ' +
                           datetime.datetime.now().isoformat())

    # ingest an entire directory, trigger "next_file" (which is an option
    # specified in the config file) on each file,
    # optionally continue to monitor if -m is specified
    elif args.dirname is not None:

        framework.ingest_data(args.dirname, None, args.monitor)

    # implement the group mode
    if args.group_mode is True:
        data_set = framework.context.data_set

        # remove focus images and ccd clearing images from the dataset
        focus_frames = data_set.data_table[data_set.data_table.OBJECT ==
                                           "focus"].index
        ccdclear_frames = data_set.data_table[data_set.data_table.OBJECT ==
                                              "Clearing ccd"].index
        data_set.data_table.drop(focus_frames, inplace=True)
        data_set.data_table.drop(ccdclear_frames, inplace=True)

        # processing
        imtypes = ['BIAS', 'DARK', 'FLAT', 'LENSFLAT',
                   'CALIB', 'OBJECT']

        for imtype in imtypes:
            subset = data_set.data_table[
                framework.context.data_set.data_table.IMTYPE == imtype]
            if 'OBJECT' in imtype:  # Ensure that standards are processed first
                object_order = []
                standard_order = []
                for frame in subset.index:
                    if is_file_scales_std(frame, logger=framework.context.logger):
                        standard_order.append(frame)
                    else:
                        object_order.append(frame)
                order = standard_order + object_order  # Standards first
                process_list(order)
            else:
                process_subset(subset)

    framework.config.instrument.wait_for_event = args.wait_for_event
    framework.config.instrument.continuous = args.continuous

    framework.start(args.queue_manager_only, args.ingest_data_only,
                    args.wait_for_event, args.continuous)


if __name__ == "__main__":
    main()
