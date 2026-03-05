from keckdrpframework.config.framework_config import ConfigClass
import logging.config
from scalesdrp.core.scales_pkg_resources import get_resource_path

pkg = 'scalesdrp'


def test_parse_scales_config():

    scales_config_file = 'configs/scales.cfg'
    scales_config_fullpath = get_resource_path(pkg, scales_config_file)
    scales_config = ConfigClass(scales_config_fullpath, default_section='SCALES')


def test_parse_log_config():

    framework_logcfg_file = 'configs/logger.cfg'
    framework_logcfg_fullpath = get_resource_path(pkg, framework_logcfg_file)
    # we are only checking that the config file can be parsed,
    # not that the log file can be created
    try:
        logging.config.fileConfig(framework_logcfg_fullpath)
    except FileNotFoundError:
        pass


def test_parse_framework_config():

    framework_config_file = 'configs/framework.cfg'
    framework_config_fullpath = get_resource_path(pkg, framework_config_file)
    framework_config = ConfigClass(framework_config_fullpath)
