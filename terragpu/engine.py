import os
import logging
import numpy as np
import pandas as pd
import xarray as xr
from types import ModuleType
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

_warn_array_module_once = False
_warn_df_module_once = False

# TODO:
# - ml module to choose between sklearn and cuml
# - decorator to make module selection repeatable


def array_module(xp=None):
    """
    Find the array module to use, for example **numpy** or **cupy**.
    """
    xp = xp or os.environ.get("ARRAY_MODULE", "numpy")

    if isinstance(xp, ModuleType):
        return xp
    if xp == "numpy":
        return np
    if xp == "cupy":
        try:
            import cupy as cp
            return cp
        except ModuleNotFoundError as e:
            global _warn_array_module_once
            if not _warn_array_module_once:
                # logging.warning(f'Using numpy ({e}).')
                _warn_array_module_once = True
            return np
    raise ValueError(f'ARRAY_MODULE={xp} not known')


def df_module(xf=None):
    """
    Find the dataframe module to use, for example **pandas** or **cudf**.
    """
    xf = xf or os.environ.get("DF_MODULE", "pandas")

    if isinstance(xf, ModuleType):
        return xf
    if xf == "pandas":
        return pd
    if xf == "cudf":
        try:
            import cudf as cf
            return cf
        except ModuleNotFoundError as e:
            global _warn_df_module_once
            if not _warn_df_module_once:
                # logging.warning(f'Using pandas ({e}).')
                _warn_df_module_once = True
            return pd
    raise ValueError(f'DF_MODULE={xf} not known')


def tif_module(xtif=None):
    """
    Find the dataframe module to use, for example **pandas** or **cudf**.
    """
    xtif = xtif or os.environ.get("TIF_MODULE", "xarray")

    if isinstance(xtif, ModuleType):
        return xtif
    if xtif == "xarray":
        return xr
    if xtif == "cucim":
        try:
            import cucim
            return cucim
        except ModuleNotFoundError as e:
            global _warn_df_module_once
            if not _warn_df_module_once:
                # logging.warning(f'Using pandas ({e}).')
                _warn_df_module_once = True
            return xr
    raise ValueError(f'TIF_MODULE={xtif} not known')

def configure_dask(local_directory: str = None):
    if local_directory is not None:
        cluster = LocalCUDACluster(local_directory=local_directory)
    else:
        cluster = LocalCUDACluster()
    client = Client(cluster)
    return
