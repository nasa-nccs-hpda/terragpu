"""Compatibility adapters for legacy HLS Datasets containing band_data.

Band names remain explicit; callers must apply product-specific scale/offset
and QA masks before calculating indices.
"""
from . import wv_indices as _wv


def _array(raster):
    data = raster['band_data'].copy(deep=False)
    data.attrs = {**data.attrs, **raster.attrs}
    return data


def _adapt(name):
    def index(raster):
        return _wv.get_indices(name)(_array(raster))
    index.__name__ = name
    return index


indices_mappings = {name: _adapt(name) for name in _wv.indices_registry}
globals().update(indices_mappings)
__all__ = [*indices_mappings, 'get_indices', 'add_indices']


def get_indices(index_key):
    try:
        return indices_mappings[index_key.lower()]
    except KeyError:
        raise ValueError(f'Invalid indices mapping: {index_key}.') from None


def add_indices(raster, indices):
    if set(raster.data_vars) != {'band_data'}:
        raise ValueError('HLS adapter requires a Dataset containing only band_data')
    result = _wv.add_indices(_array(raster), indices)
    dataset = result.to_dataset(name='band_data')
    dataset.attrs = dict(result.attrs)
    return dataset
