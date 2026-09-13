"""Native PACE OCI L2 AOP swath processing with bounded spatial tiles."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from .engine import array_module

# Explicit example policy; users can select a different set of named flags.
DEFAULT_FLAGS = ('ATMFAIL', 'LAND', 'HIGLINT', 'HILT', 'HISATZEN', 'STRAYLIGHT',
                 'CLDICE', 'HISOLZEN', 'LOWLW', 'NAVWARN', 'NAVFAIL', 'PRODFAIL')


def _netcdf():
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise ImportError('Install terragpu[pace] or terragpu[viirs] for NetCDF processing') from exc
    return Dataset


class OceanColorSwath:
    """Shared resource lifetime and metadata-driven ocean-color flags."""

    def flag_mask(self, names):
        meanings = self.flags.flag_meanings.split()
        masks = np.asarray(self.flags.flag_masks).reshape(-1)
        if len(meanings) != len(masks):
            raise ValueError('Malformed flag metadata')
        result = 0
        for name in names:
            if meanings.count(name) != 1:
                raise ValueError(f'Flag must occur exactly once in metadata: {name}')
            result |= int(masks[meanings.index(name)]) & 0xffffffff
        return result

    def close(self):
        self.dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class PaceSwath(OceanColorSwath):
    """Own a grouped NetCDF handle; read decoded tiles without loading the cube.

    netCDF4 applies packed scale/offset and masks fill and valid-range violations.
    The Rrs spectral coordinate is wavelength_3d, not the sensor's wavelength list.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.dataset = _netcdf()(path)
        try:
            ds = self.dataset
            if getattr(ds, 'platform', '') != 'PACE' or getattr(ds, 'instrument', '') != 'OCI':
                raise ValueError('Expected PACE OCI data')
            self.rrs = ds.groups['geophysical_data'].variables['Rrs']
            self.flags = ds.groups['geophysical_data'].variables['l2_flags']
            self.latitude = ds.groups['navigation_data'].variables['latitude']
            self.longitude = ds.groups['navigation_data'].variables['longitude']
            wavelengths = ds.groups['sensor_band_parameters'].variables['wavelength_3d']
            self.wavelengths = np.asarray(wavelengths[:].astype('float64').filled(np.nan))
            spatial_dims = ('number_of_lines', 'pixels_per_line')
            if (self.rrs.ndim != 3 or self.rrs.dimensions[:2] != spatial_dims
                    or any(v.dimensions != spatial_dims or v.shape != self.rrs.shape[:2]
                           for v in (self.flags, self.latitude, self.longitude))
                    or self.wavelengths.shape != (self.rrs.shape[2],)
                    or not np.all(np.isfinite(self.wavelengths))
                    or not np.all(np.diff(self.wavelengths) > 0)):
                raise ValueError('Inconsistent swath grids or spectral coordinate')
            if getattr(wavelengths, 'units', '') != 'nm' or getattr(self.rrs, 'units', '') != 'sr^-1':
                raise ValueError('Expected wavelengths in nm and Rrs in sr^-1')
            if not np.issubdtype(self.flags.dtype, np.integer):
                raise ValueError('Quality flags must be integers')
            self.shape = self.rrs.shape[:2]
        except Exception:
            self.close()
            raise

    def read_rrs(self, window, spectral):
        return self.rrs[window + (spectral,)].astype("float32").filled(np.nan)


def _process_swath(source, destination, *, reader, backend='numpy', tile_size=128,
                   wavelength_range=(400., 700.), reject_flags=DEFAULT_FLAGS):
    """Write mean Rrs = trapezoid integral / wavelength span on native samples.

    Uses samples inside the requested interval, without interpolating endpoints;
    records actual endpoints in the output. Requires every selected band to be
    valid. This is a spectral reduction example, not NASA's AVW or chlorophyll
    algorithm. Output retains two-dimensional swath coordinates, not a map grid.
    """
    if backend not in {'numpy', 'cupy'}:
        raise ValueError('backend must be numpy or cupy')
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError('tile_size must be a positive integer')
    bounds = np.asarray(wavelength_range, dtype=float)
    reject_flags = tuple(reject_flags)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError('wavelength_range must contain increasing finite bounds')
    destination = Path(destination)
    if destination.suffix != '.nc':
        raise ValueError('Output must be a .nc NetCDF file')
    if destination.exists():
        raise FileExistsError(destination)
    xp = array_module(backend)
    timing = dict(read_seconds=0., compute_seconds=0., write_seconds=0., tiles=0, valid_pixels=0)
    started = time.perf_counter()
    temporary = None
    try:
        with reader(source) as src:
            chosen = np.flatnonzero((src.wavelengths >= bounds[0]) & (src.wavelengths <= bounds[1]))
            if len(chosen) < 2:
                raise ValueError('Requested interval must contain at least two wavelengths')
            wave = src.wavelengths[chosen]
            spectral = slice(int(chosen[0]), int(chosen[-1])+1)
            # Trapezoid weights support nonuniform wavelength spacing.
            delta = np.diff(wave)
            weights = np.zeros(len(wave), dtype='float64')
            weights[:-1] += delta / 2
            weights[1:] += delta / 2
            weights = xp.asarray(weights / (wave[-1]-wave[0]), dtype=xp.float32)
            mask = src.flag_mask(tuple(reject_flags))
            if backend == 'cupy':
                xp.cuda.get_current_stream().synchronize()
            fd, temporary = tempfile.mkstemp(prefix='.terragpu-pace-', suffix='.nc', dir=destination.parent)
            os.close(fd)
            with _netcdf()(temporary, 'w', format='NETCDF4') as dst:
                dims = ('number_of_lines', 'pixels_per_line')
                for name, size in zip(dims, src.shape):
                    dst.createDimension(name, size)
                options = dict(zlib=True, complevel=4, chunksizes=tuple(min(tile_size, s) for s in src.shape))
                output = dst.createVariable('mean_Rrs', 'f4', dims, fill_value=np.nan, **options)
                output.setncatts(dict(units='sr^-1', long_name='Wavelength-weighted mean remote sensing reflectance',
                                     coordinates='latitude longitude'))
                coordinates = []
                for name, unit in (('latitude', 'degrees_north'), ('longitude', 'degrees_east')):
                    v = dst.createVariable(name, 'f4', dims, fill_value=np.nan, **options)
                    v.setncatts(dict(standard_name=name, units=unit))
                    coordinates.append(v)
                dst.setncatts(dict(Conventions='CF-1.8', source_file=Path(source).name,
                                  source_platform=str(getattr(src.dataset, 'platform', 'unknown')),
                                  source_instrument=str(getattr(src.dataset, 'instrument', 'unknown')),
                                  source_processing_version=str(getattr(src.dataset, 'processing_version', 'unknown')),
                                  reduction='trapezoid integral divided by actual wavelength span; all samples required',
                                  requested_wavelength_min_nm=float(bounds[0]), requested_wavelength_max_nm=float(bounds[1]),
                                  wavelength_min_nm=float(wave[0]), wavelength_max_nm=float(wave[-1]),
                                  wavelength_count=len(wave), reject_flags=' '.join(reject_flags), backend=backend))
                for row in range(0, src.shape[0], tile_size):
                    for col in range(0, src.shape[1], tile_size):
                        window = (slice(row, row+tile_size), slice(col, col+tile_size))
                        start = time.perf_counter()
                        host = src.read_rrs(window, spectral)
                        flags = src.flags[window]
                        lat, lon = [v[window].astype('float32').filled(np.nan) for v in (src.latitude, src.longitude)]
                        invalid = (np.ma.getmaskarray(flags) | ((flags.data.astype('uint32') & mask) != 0)
                                   | ~np.isfinite(lat) | ~np.isfinite(lon)
                                   | (np.abs(lat) > 90) | (np.abs(lon) > 180))
                        timing['read_seconds'] += time.perf_counter()-start
                        start = time.perf_counter()
                        cube = xp.asarray(host)
                        result = xp.sum(cube * weights, axis=-1)
                        result = xp.where(xp.asarray(invalid) | ~xp.all(xp.isfinite(cube), axis=-1), xp.nan, result)
                        values = xp.asnumpy(result) if backend == 'cupy' else result
                        timing['compute_seconds'] += time.perf_counter()-start
                        timing['valid_pixels'] += int(np.count_nonzero(np.isfinite(values)))
                        start = time.perf_counter()
                        output[window] = values
                        for variable, data in zip(coordinates, (lat, lon)):
                            variable[window] = data
                        timing['write_seconds'] += time.perf_counter()-start
                        timing['tiles'] += 1
            os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    timing.update(total_seconds=time.perf_counter()-started, backend=backend, tile_size=tile_size,
                  wavelength_count=len(wave), wavelength_min_nm=float(wave[0]), wavelength_max_nm=float(wave[-1]))
    return timing


def process_pace(source, destination, *, backend='numpy', tile_size=128,
                 wavelength_range=(400., 700.), reject_flags=DEFAULT_FLAGS):
    """Reduce a PACE OCI L2 AOP cube on its native swath; see docs/pace.md."""
    return _process_swath(source, destination, reader=PaceSwath, backend=backend,
                          tile_size=tile_size, wavelength_range=wavelength_range,
                          reject_flags=reject_flags)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--backend', choices=('numpy', 'cupy'), default='numpy')
    parser.add_argument('--tile-size', type=int, default=128)
    parser.add_argument('--wavelength-range', nargs=2, type=float, default=(400., 700.))
    parser.add_argument('--reject-flags', nargs='+', default=DEFAULT_FLAGS)
    print(json.dumps(process_pace(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
