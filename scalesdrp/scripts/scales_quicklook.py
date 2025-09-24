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

from scalesdrp.pipelines.scales_quicklook_pipeline import Scales_Quicklook_Pipeline
from scalesdrp.core.scales_proctab import Proctab
import logging.config


def _parse_arguments(in_args: list) -> argparse.Namespace:
    description = "SCALES pipeline CLI"

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

    # in this case, we are loading an entire directory,
    # and ingesting all the files in that directory
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
    parser.add_argument("-k", "--skipsky", dest='skipsky', action="store_true",
                        default=False, help="Skip sky subtraction")

    parser.add_argument("-lr", "--lowres", dest='lowres', action="store_true",
                        default=False, help="low resolution mode on!")
    
    parser.add_argument("-mr", "--medres", dest='medres', action="store_true",
                        default=False, help="medium resolution mode on !")

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
        dest = os.path.join(os.getcwd(), 'scales_quicklook.cfg')
        if os.path.exists(dest):
            print("Config file scales_quicklook.cfg already exists in current dir")
        else:
            scales_config_file = 'configs/scales_quicklook.cfg'
            scales_config_fullpath = pkg_resources.resource_filename(
                pkg, scales_config_file)
            shutil.copy(scales_config_fullpath, os.getcwd())
            print("Copied scales_quicklook.cfg into current dir.  Edit and use with -c")
        sys.exit(0)

    # make sure user has selected a channel
    if not args.lowres and not args.medres:
        print("\nERROR - DRP can process only one channel at a time\n\n"
              "Please indicate a channel to process:\n"
              "Either medium-resolution with -mr or --medres or\n"
              "       low-resolution  with -lr or --lowres\n")
        sys.exit(0)

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
        scales_config_file = 'configs/scales_quicklook.cfg'
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
        framework = Framework(Scales_Quicklook_Pipeline, framework_config_fullpath)
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

    # check for skipsky argument
    if args.skipsky:
        def_sk = getattr(framework.config.instrument, 'skipsky', None)
        if def_sk is not None:
            framework.context.pipeline_logger.info("Skipping sky subtraction")
            framework.config.instrument.skipsky = args.skipsky

    #framework.config.default_ingestion_event = "add_only"

    # update proc table argument
    if args.proctab:
        framework.context.pipeline_logger.info(
            "Using proc table file %s" % args.proctab
        )
        framework.config.instrument.procfile = args.proctab
    else:
        if args.lowres:
            proctab = scales_config.LOWRES['procfile']
        elif args.medres:
            proctab = scales_config.MEDRES['procfile']
        else:
            proctab = scales_config.procfile
        framework.context.pipeline_logger.info(
            "Using proc table file %s" % proctab)
        framework.config.instrument.procfile = proctab

    # initialize the proctab and read it
    framework.context.proctab = Proctab()
    framework.context.proctab.read_proctab(framework.config.instrument.procfile)

    framework.logger.info("Framework initialized")
    print("=====================")
    # start queue manager only (useful for RPC)
    if args.queue_manager_only:
        # The queue manager runs forever.
        framework.logger.info("Starting queue manager only, no processing")
        framework.start(args.queue_manager_only)

    # ingest an entire directory, trigger "next_file" (which is an option
    # specified in the config file) on each file,
    # optionally continue to monitor if -m is specified
        print("=====================")
    elif args.dirname is not None:
        print("++++++++++++")
        framework.ingest_data(args.dirname, None, args.monitor)

    framework.config.instrument.wait_for_event = args.wait_for_event
    framework.config.instrument.continuous = args.continuous

    framework.start(args.queue_manager_only, args.ingest_data_only,
                    args.wait_for_event, args.continuous)


if __name__ == "__main__":
    main()
