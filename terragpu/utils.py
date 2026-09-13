import sys
import os
import logging
import configparser
from datetime import datetime  # tracking date

# -------------------------------------------------------------------------------
# module utils
# Uilities module to include methods
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Module Methods
# -------------------------------------------------------------------------------


def create_logfile(logdir='results'):
    """
    Create log file to output both stderr and stdout.
    :param args: argparser object
    :param logdir: log directory to store log file
    :return: logfile instance, stdour and stderr being logged to file
    """
    logfile = os.path.join(logdir, '{}_log.out'.format(
        datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    logging.info('See ', logfile)
    so = se = open(logfile, 'w')  # open our log file
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')  # stdout buffering
    os.dup2(so.fileno(), sys.stdout.fileno())  # redirect to the log file
    os.dup2(se.fileno(), sys.stderr.fileno())
    return logfile


def read_config(fname: str):
    """
    Read INI format configuration file.
    :params fname: filename of configuration file
    :return: configparser object with configuration parameters
    """
    try:  # try initializing config object
        config_file = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )  # initialize config parser
        config_file.read(fname)  # take first argument from script
    except configparser.ParsingError as err:  # abort if incorrect format
        raise RuntimeError('Could not parse {}.'.format(err))
    logging.info('Configuration file read.')
    return config_file
