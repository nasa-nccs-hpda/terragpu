"""NASA OB.DAAC VIIRS L2 ocean-color swaths (not land, thermal or SDR products)."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from .pace import OceanColorSwath, DEFAULT_FLAGS, _netcdf, _process_swath


class ViirsSwath(OceanColorSwath):
    """Decode separate Rrs_<nm> variables and preserve native navigation/QA.

    Band labels are nominal wavelengths, not a hyperspectral sampling grid.
    Every variable's own packing and valid range are applied by netCDF4.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.dataset = _netcdf()(path)
        try:
            ds = self.dataset
            if getattr(ds, 'instrument', '') != 'VIIRS':
                raise ValueError('Expected VIIRS ocean-color data')
            geo, nav = ds.groups['geophysical_data'], ds.groups['navigation_data']
            selected = sorted((float(name[4:]), var) for name, var in geo.variables.items()
                              if re.fullmatch(r'Rrs_\d+(?:\.\d+)?', name))
            if len(selected) < 2:
                raise ValueError('Expected separate VIIRS Rrs wavelength variables')
            self.wavelengths = np.array([w for w, _ in selected])
            self.bands = [v for _, v in selected]
            self.flags = geo['l2_flags']
            self.latitude, self.longitude = nav['latitude'], nav['longitude']
            self.shape = self.flags.shape
            dims = ('number_of_lines', 'pixels_per_line')
            if (len(self.shape) != 2 or not np.all(np.diff(self.wavelengths) > 0)
                    or any(v.dimensions != dims or v.shape != self.shape
                           for v in [*self.bands, self.flags, self.latitude, self.longitude])):
                raise ValueError('Inconsistent VIIRS swath grids or wavelengths')
            if any(getattr(v, 'units', '') != 'sr^-1' for v in self.bands):
                raise ValueError('Expected Rrs in sr^-1')
            if not np.issubdtype(self.flags.dtype, np.integer):
                raise ValueError('Quality flags must be integers')
        except Exception:
            self.close()
            raise

    def read_rrs(self, window, spectral):
        return np.stack([v[window].astype('float32').filled(np.nan)
                         for v in self.bands[spectral]], axis=-1)


def process_viirs(source, destination, *, backend='numpy', tile_size=128,
                  wavelength_range=(400., 700.), reject_flags=DEFAULT_FLAGS):
    """Integrate sparse nominal Rrs samples / span; not a chlorophyll retrieval.

    Reuses the PACE tile engine, with VIIRS-specific variable discovery/decoding.
    This sparse-band summary is not spectrally equivalent to PACE's dense mean.
    """
    return _process_swath(source, destination, reader=ViirsSwath, backend=backend,
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
    print(json.dumps(process_viirs(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
