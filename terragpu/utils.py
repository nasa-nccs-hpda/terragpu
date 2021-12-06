import sys
import os
import logging
import configparser
from datetime import datetime  # tracking date

try:
    from cuml.dask.common import utils as dask_utils
    from dask.distributed import Client
    # from dask.distributed import wait
    from dask_cuda import LocalCUDACluster
    import dask_cudf
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

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


def create_dcluster():
    # This will use all GPUs on the local host by default
    cluster = LocalCUDACluster(threads_per_worker=1)
    c = Client(cluster)

    # Query the client for all connected workers
    workers = c.has_what().keys()
    n_workers = len(workers)
    n_streams = 8  # Performance optimization
    return c, workers, n_workers, n_streams


def distribute_dcluster(X_cudf, y_cudf, n_partitions, c, workers):
    # x and y in cudf format
    # n_partitions = n_workers
    # c and workers come from the create_dcluster function

    # First convert to cudf (you would likely load in cuDF format to start)
    # X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X))
    # y_cudf = cudf.Series(y)

    # Partition with Dask
    # In this case, each worker will train on 1/n_partitions fraction of data
    X_dask = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)
    y_dask = dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)

    # Persist to cache the data in active memory
    X_dask, y_dask = \
        dask_utils.persist_across_workers(c, [X_dask, y_dask], workers=workers)

    return X_dask, y_dask
