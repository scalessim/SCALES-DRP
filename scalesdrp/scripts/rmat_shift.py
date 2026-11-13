"""
Measure rectmat shifts from single SCALES monochromator frames.

Each input CALUNIT frame is reduced to L1 (same processing as
start_scales_calib_sa), matched by wavelength and IFS mode to a master calunit
image in an existing monochromator scan, and the spot shift between the two is
measured. Shifted OPT and C2 interpolated rectmats are written to the current
directory (rmat_out_directory in the config).

Example:
    start_scales_rmat_shift -f ss260615_15477.fits -b ss260615_15476.fits \
        -r /path/to/monochrom_scan/redux

-b gives the background frame; the spot image is then the difference of the
two L1s. Without -b the frame is background-subtracted like the scan images
(scaled mean of the scan's mcalunits).
"""
from keckdrpframework.core.framework import Framework
from keckdrpframework.config.framework_config import ConfigClass
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.utils.drpf_logger import getLogger
import datetime
import argparse
import sys
import traceback
import os
import psutil
import shutil

from scalesdrp.pipelines.scales_rmat_shift_pipeline import Scales_Rmat_Shift_Pipeline
from scalesdrp.core.scales_proctab import Proctab
from scalesdrp.core.scales_pkg_resources import get_resource_path
import logging.config


CONFIG_NAME = 'scales_rmat_shift.cfg'


def _parse_arguments(in_args: list) -> argparse.Namespace:
    description = "SCALES rectmat shift CLI"

    parser = argparse.ArgumentParser(prog=f"{in_args[0]}",
                                     description=description)
    parser.add_argument('-c', '--config', dest="SCALES_config_file", type=str,
                        help="SCALES configuration file", default=None)
    parser.add_argument('--write_config', dest="write_config",
                        help="Write out an editable config file in current dir"
                        f" ({CONFIG_NAME})", action="store_true", default=False)
    parser.add_argument('-f', '--frames', nargs='*', type=str,
                        help='input monochromator image files (full path, list ok)',
                        default=None)
    parser.add_argument('-l', '--list', dest='file_list',
                        help='File containing a list of files to be processed',
                        default=None)
    parser.add_argument('-d', '--directory', dest="dirname", type=str,
                        help="Input directory", nargs='?', default=None)
    parser.add_argument('-i', '--infiles', dest="infiles",
                        help="Input files, or pattern to match", nargs="?")
    parser.add_argument('-m', '--monitor', dest="monitor",
                        help='Continue monitoring the directory '
                             'after the initial ingestion',
                        action='store_true', default=False)

    # rectmat-shift specific parameters
    parser.add_argument('-r', '--rmat_dir', dest='rmat_dir', type=str,
                        default=None,
                        help='Directory with the monochromator scan products '
                             '(*_mcalunit.fits, {IFSMODE}_lams.fits, '
                             '{IFSMODE}_gauss_spotpars[_interpd].fits)')
    parser.add_argument('-b', '--background', dest='background', nargs='*',
                        type=str, default=None,
                        help='Background frame(s) for the monochromator '
                             'frame(s); the spot image is L1(frame) - '
                             'L1(background). Without this, the scan-mean '
                             'background subtraction is used.')
    parser.add_argument('-o', '--outdir', dest='outdir', type=str,
                        default=None,
                        help='Directory for the shifted rectmats '
                             '(default: rmat_out_directory in the config, '
                             'i.e. the current directory)')
    parser.add_argument('--measure_only', dest='measure_only',
                        action='store_true', default=False,
                        help='Measure and log the shift without writing rectmats')
    parser.add_argument('--affine', dest='affine', action='store_true',
                        default=False,
                        help='Fit and apply an affine model (shift + scale + '
                             'rotation + shear) instead of a single x/y shift')

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
    parser.add_argument("-p", "--proctab", dest='proctab', help='Proctab file',
                        default=None)

    out_args = parser.parse_args(in_args[1:])
    return out_args


def check_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print("Directory %s has been created" % directory)


# context attributes StartCalibSA reads (copied from the config)
CONTEXT_KEYS = [
    'clobber', 'calib_file_path', 'create_rectmat',
    'bpm_ifs', 'bpmat_ifs', 'bpm_img', 'bpmat_img',
    'flat_ifs_fast0p6', 'sig_map_ifs_fast0p6', 'lin_coeff_ifs_fast0p6',
    'sat_map_ifs_fast0p6', 'gain_ifs_fast0p6',
    'flat_ifs_fast1', 'sig_map_ifs_fast1', 'lin_coeff_ifs_fast1',
    'sat_map_ifs_fast1', 'gain_ifs_fast1',
    'flat_ifs_slow', 'sig_map_ifs_slow', 'lin_coeff_ifs_slow',
    'sat_map_ifs_slow', 'gain_ifs_slow',
    'flat_img_fast0p6', 'sig_map_img_fast0p6', 'lin_coeff_img_fast0p6',
    'sat_map_img_fast0p6', 'gain_img_fast0p6',
    'flat_img_fast1', 'sig_map_img_fast1', 'lin_coeff_img_fast1',
    'sat_map_img_fast1', 'gain_img_fast1',
    'flat_img_slow', 'sig_map_img_slow', 'lin_coeff_img_slow',
    'sat_map_img_slow', 'gain_img_slow',
    'subtract_row_median', 'do_swap', 'nchans', 'altcol', 'channelwise',
    'amp_mean_func', 'do_acn', 'acn_avg_type', 'acn_mean_func', 'acn_smooth',
    'acn_savgol', 'acn_winsize', 'acn_order', 'resid_colsub', 'fixcol',
    'ref_avg_type', 'ref_mean_func', 'ref_smooth', 'ref_savgol',
    'ref_winsize', 'ref_order', 'pickup', 'sigma_thresh', 'dilate_iter',
    'highpass_size', 'per_amp', 'do_linearity', 'apply_sat_mask', 'apply_bpm',
]


def main():

    pkg = 'scalesdrp'
    args = _parse_arguments(sys.argv)

    if args.write_config:
        dest = os.path.join(os.getcwd(), CONFIG_NAME)
        if os.path.exists(dest):
            print(f"Config file {CONFIG_NAME} already exists in current dir")
        else:
            src = get_resource_path(pkg, 'configs/' + CONFIG_NAME)
            shutil.copy(src, os.getcwd())
            print(f"Copied {CONFIG_NAME} into current dir.  Edit and use with -c")
        sys.exit(0)

    plist = []
    for p in psutil.process_iter():
        try:
            if "scales_rmat_shift" in p.name():
                plist.append(p)
        except psutil.NoSuchProcess:
            continue
    if len(plist) > 1:
        print("DRP already running in another process, exiting")
        sys.exit(0)

    if args.file_list and '.fits' in args.file_list:
        print("\nERROR - trying to read in fits file as file list\n\n"
              "Please use -f or --frames for direct input of fits files\n")
        sys.exit(1)

    # START HANDLING OF CONFIGURATION FILES ##########
    check_directory("logs")
    check_directory("plots")

    framework_config_fullpath = str(
        get_resource_path(pkg, "configs/framework_rmat_shift.cfg"))
    framework_logcfg_fullpath = str(
        get_resource_path(pkg, 'configs/logger.cfg'))

    if args.SCALES_config_file is None:
        scales_config_fullpath = str(get_resource_path(pkg, 'configs/' + CONFIG_NAME))
    else:
        scales_config_fullpath = os.path.abspath(args.SCALES_config_file)
    scales_config = ConfigClass(scales_config_fullpath, default_section='SCALES')
    # END HANDLING OF CONFIGURATION FILES ##########

    scales_config.cwd = os.getcwd()

    # rectmat-shift settings from the command line
    if args.rmat_dir is not None:
        scales_config.rmat_scan_dir = os.path.abspath(args.rmat_dir)
    if getattr(scales_config, 'rmat_scan_dir', None) in (None, 'None', ''):
        print("\nERROR - no monochromator scan directory given. "
              "Use -r/--rmat_dir (or rmat_scan_dir in the config).\n")
        sys.exit(1)
    if not os.path.isdir(scales_config.rmat_scan_dir):
        print(f"\nERROR - scan directory {scales_config.rmat_scan_dir} "
              "does not exist.\n")
        sys.exit(1)
    if args.outdir is not None:
        scales_config.rmat_out_directory = args.outdir
    if args.measure_only:
        scales_config.rmat_measure_only = True
    if args.affine:
        scales_config.rmat_shift_affine = True
    if args.background:
        scales_config.rmat_background_frames = list(args.background)
    bkg_frames = getattr(scales_config, 'rmat_background_frames', None) or []
    if isinstance(bkg_frames, str):
        bkg_frames = [bkg_frames]
    for b in bkg_frames:
        if not os.path.isfile(b):
            print(f"\nERROR - background frame {b} does not exist.\n")
            sys.exit(1)

    # StartCalibSA only needs to make the CALUNIT L1
    scales_config.create_rectmat = True
    for key in ('create_linearity_coeff', 'create_sat_map',
                'create_bpm', 'create_readnoise'):
        setattr(scales_config, key, False)

    check_directory(scales_config.output_directory)
    check_directory(scales_config.plot_directory)
    proc_dir = os.path.dirname(scales_config.procfile)
    if proc_dir:
        check_directory(proc_dir)

    try:
        framework = Framework(Scales_Rmat_Shift_Pipeline, framework_config_fullpath)
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

    if args.proctab:
        framework.context.pipeline_logger.info(
            "Using proc table file %s" % args.proctab)
        framework.config.instrument.procfile = args.proctab
        override_proc_dir = os.path.dirname(args.proctab)
        if override_proc_dir:
            check_directory(override_proc_dir)

    # Start from an empty proctab: this pipeline never skips frames based on
    # it (StartCalibSA keeps its own), and ingest_file fails on frames that
    # are already listed (Proctab has no last_suffix method), retrying forever.
    framework.context.proctab = Proctab()
    framework.context.proctab.new_proctab()

    framework.logger.info("Framework initialized")
    framework.context.pipeline_logger.info(
        f"Monochromator scan directory: {scales_config.rmat_scan_dir}")

    if args.queue_manager_only:
        framework.logger.info("Starting queue manager only, no processing")
        framework.start(args.queue_manager_only)
        return

    # background frames first: each frame's chain runs to completion before
    # the next file starts, so their L1s exist when the shift is measured
    if bkg_frames:
        framework.ingest_data(None, list(bkg_frames), False)

    if args.frames:
        framework.ingest_data(None, list(args.frames), False)

    elif args.file_list:
        frames = []
        with open(args.file_list) as file_list:
            for frame in file_list:
                if "#" not in frame:
                    frames.append(frame.strip('\n'))
        framework.ingest_data(None, frames, False)
        with open(args.file_list + '_ingest', 'w') as ingest_f:
            ingest_f.write('Files ingested at: ' +
                           datetime.datetime.now().isoformat())

    elif args.dirname is not None:
        framework.ingest_data(args.dirname, None, args.monitor)

    for key in CONTEXT_KEYS:
        setattr(framework.context, key, getattr(scales_config, key))

    framework.config.instrument.wait_for_event = args.wait_for_event
    framework.config.instrument.continuous = args.continuous

    framework.start(args.queue_manager_only, args.ingest_data_only,
                    args.wait_for_event, args.continuous)


if __name__ == "__main__":
    main()
