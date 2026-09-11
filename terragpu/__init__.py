"""GPU-accelerated geospatial raster processing."""
from importlib.metadata import version, PackageNotFoundError
from .engine import configure_dask

try:
    __version__ = version("terragpu")
except PackageNotFoundError:
    __version__ = "2026.9.0.dev0"

__all__ = ["Raster", "configure_dask", "__version__"]


def __getattr__(name):
    if name == 'Raster':
        # xarray/Dask may probe installed GPU libraries while importing. Load
        # that stack only when the caller actually requests raster operations.
        from .array.raster import Raster
        globals()['Raster'] = Raster
        return Raster
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
