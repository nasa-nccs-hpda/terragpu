import xarray as xr

__all__ = ["add_indices", "cs1", "dvi", "ndvi"]

CHUNKS = {'band': 1, 'x': 2048, 'y': 2048}

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

def cs1(raster):
    """
    Cloud detection index (CS1), CS1 := (3. * NIR1) / (Blue + Green + Red)
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with SI calculated
    """
    nir1, red, blue, green = _get_band_locations(
        raster.attrs['band_names'], ['nir1', 'red', 'blue', 'green'])
    index = (
        (3. * raster['band_data'][nir1, :, :]) /
        (raster['band_data'][blue, :, :] + raster['band_data'][green, :, :] \
            + raster['band_data'][red, :, :])
    ).compute()
    return index.expand_dims(dim="band", axis=0).fillna(0).chunk(chunks=CHUNKS)

def cs2(raster):
    """
    Cloud detection index (CS2), CS2 := (Blue + Green + Red + NIR1) / 4.
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with CS2 calculated
    """
    nir1, red, blue, green = _get_band_locations(
        raster.attrs['band_names'], ['nir1', 'red', 'blue', 'green'])
    index = (
        (raster['band_data'][blue, :, :] + raster['band_data'][green, :, :] \
            + raster['band_data'][red, :, :] + raster['band_data'][nir1, :, :])
        / 4.0
    ).compute()
    return index.expand_dims(dim="band", axis=0).fillna(0).chunk(chunks=CHUNKS)

def dvi(raster):
    """
    Difference Vegetation Index (DVI), DVI := NIR1 - Red
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with DVI calculated
    """
    nir1, red = _get_band_locations(
        raster.attrs['band_names'], ['nir1', 'red'])
    index = (
        raster['band_data'][nir1, :, :] - raster['band_data'][red, :, :]
    ).compute()
    return index.expand_dims(dim="band", axis=0).chunk(chunks=CHUNKS)

def dwi(raster):
    """
    Difference Water Index (DWI), DWI := factor * (Green - NIR1)
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with DWI calculated
    """
    nir1, green = _get_band_locations(
        raster.attrs['band_names'], ['nir1', 'green'])
    index = (
        raster['band_data'][green, :, :] - raster['band_data'][nir1, :, :]
    ).compute()
    return index.expand_dims(dim="band", axis=0).chunk(chunks=CHUNKS)

def fdi(raster):
    """
    Forest Discrimination Index (FDI), type int16
    8 band imagery: FDI := NIR2 - (RedEdge + Blue)
    4 band imagery: FDI := NIR1 - (Red + Blue)
    :param data: xarray or numpy array object in the form (c, h, w)
    :return: new band with FDI calculated
    """
    bands = ['blue', 'nir2', 'rededge']
    if not all(b in bands for b in raster.attrs['band_names']):
        bands = ['blue', 'nir1', 'red']
    blue, nir, red = _get_band_locations(
        raster.attrs['band_names'], bands)
    index = (
        raster['band_data'][nir, :, :] - \
            (raster['band_data'][red, :, :] + raster['band_data'][blue, :, :])
    ).compute()
    return index.expand_dims(dim="band", axis=0).chunk(chunks=CHUNKS)

def ndvi(raster):
    """
    Difference Vegetation Index (DVI), NDVI := (NIR - Red) / (NIR + RED)
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with DVI calculated
    """
    nir1, red = _get_band_locations(
        raster.attrs['band_names'], ['nir1', 'red'])
    index = (
        (raster['band_data'][nir1, :, :] - raster['band_data'][red, :, :]) /
        (raster['band_data'][nir1, :, :] + raster['band_data'][red, :, :])
    ).compute()
    return index.expand_dims(dim="band", axis=0).fillna(0).chunk(chunks=CHUNKS)

def ndwi(raster):
    """
    Normalized Difference Water Index (NDWI)
    NDWI := factor * (Green - NIR1) / (Green + NIR1)
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with SI calculated
    """
    nir1, green = _get_band_locations(
        raster.attrs['band_names'], ['nir1', 'green'])
    index = (
        (raster['band_data'][green, :, :] - raster['band_data'][nir1, :, :]) /
        (raster['band_data'][green, :, :] + raster['band_data'][nir1, :, :])
    ).compute()
    return index.expand_dims(dim="band", axis=0).fillna(0).chunk(chunks=CHUNKS)

def si(raster):
    """
    Shadow Index (SI), SI := (Blue * Green * Red) ** (1.0 / 3)
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with SI calculated
    """
    red, blue, green = _get_band_locations(
        raster.attrs['band_names'], ['red', 'blue', 'green'])
    index = (
        (raster['band_data'][blue, :, :] - raster['band_data'][green, :, :] /
            raster['band_data'][red, :, :]) ** (1.0/3.0)
    ).compute()
    return index.expand_dims(dim="band", axis=0).fillna(0).chunk(chunks=CHUNKS)


indices_mappings = {
    'cs1': cs1,
    'cs2': cs2,
    'dvi': dvi,
    'dwi': dwi,
    'fdi': fdi,
    'ndvi': ndvi,
    'ndwi': ndwi,
    'si': si
}

def get_indices(index_key):
    try:
        return indices_mappings[index_key]
    except KeyError:
        raise ValueError(f'Invalid indices mapping: {index_key}.')


def add_indices(raster, indices):
    """
    :param rastarr: xarray or numpy array object in the form (c, h, w)
    :param bands: list with strings of bands in the raster
    :param indices: indices to calculate and append to the raster
    :param factor: factor used for toa imagery
    :return: raster with updated bands list
    """
    nbands = len(raster.attrs['band_names'])  # get initial number of bands
    indices = [b.lower() for b in indices]  # lowercase indices list
    for index_id in indices:  # iterate over each new band
        
        # Counter for number of bands, increase metadata at concat
        indices_function  = get_indices(index_id)
        nbands += 1  # Counter for number of bands

        # Calculate band (indices)
        new_index = indices_function(raster)  # calculate the new index
        new_index.coords['band'] = [nbands]  # add band indices to raster
        new_index = new_index.to_dataset()  # move from array to dataset

        # Set metadata
        raster.attrs['band_names'].append(index_id)
        raster = xr.concat([raster, new_index], dim='band')

    return raster.chunk(chunks=CHUNKS)
