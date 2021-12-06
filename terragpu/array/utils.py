
__all__ = [
    "_get_band_locations"
]

def _get_band_locations(raster_bands: list, requested_bands: list):
    """
    Get list indices for band locations.
    """
    locations = []
    for b in requested_bands:
        try:
            locations.append(raster_bands.index(b.lower()))
        except ValueError:
            raise ValueError(f'{b} not in raster bands {raster_bands}')
    return locations
