"""TerraGPU's raster handle, independent of legacy wrappers."""
from terragpu import io


class Raster:
    """Own an open georeferenced DataArray and its file lifetime.

    Use as a context manager. Operations return DataArrays and do not mutate
    the source. Derived lazy arrays must be written/computed before closing.
    """

    def __init__(self, filename, bands=None, *, backend='numpy', chunks=None):
        self.filename = filename
        self.raster = io.imread(filename, bands=bands, backend=backend, chunks=chunks)

    @property
    def data(self):
        return self.raster

    @property
    def bands(self):
        return tuple(self.raster.attrs.get('band_names', ()))

    def index(self, name):
        from terragpu.indices.wv_indices import get_indices
        return get_indices(name)(self.raster)

    def add_indices(self, names):
        from terragpu.indices.wv_indices import add_indices
        return add_indices(self.raster, names)

    def save(self, filename, **kwargs):
        io.imsave(self.raster, filename, **kwargs)

    def close(self):
        self.raster.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
