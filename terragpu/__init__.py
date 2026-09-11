"""GPU-accelerated geospatial raster processing."""
from importlib.metadata import version, PackageNotFoundError
from .engine import configure_dask
from .array.raster import Raster

try:
    __version__ = version("terragpu")
except PackageNotFoundError:
    __version__ = "2026.9.0.dev0"

__all__ = ["Raster", "configure_dask", "__version__"]
