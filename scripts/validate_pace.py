"""Independent float64 validation of TerraGPU PACE/VIIRS ocean-color output, including packing/QA.

Usage: python scripts/validate_pace.py INPUT.nc OUTPUT.nc
Reads raw packed samples, explicitly decodes metadata, and uses np.trapezoid
instead of the processor's float32 weight reduction. Does not import terragpu.
"""
import argparse
import json
import re

from netCDF4 import Dataset
import numpy as np


def validate(source, output):
    valid_count, maximum_error = 0, 0.
    with Dataset(source) as src, Dataset(output) as dst:
        geo = src.groups['geophysical_data']
        is_viirs = getattr(src, 'instrument', '') == 'VIIRS'
        if is_viirs:
            pairs = sorted((float(name[4:]), v) for name, v in geo.variables.items() if re.fullmatch(r'Rrs_\d+(?:\.\d+)?', name))
            all_bands = [v for _, v in pairs]
            rrs = all_bands[0]
            for v in all_bands:
                v.set_auto_maskandscale(False)
        else:
            rrs = geo['Rrs']
            rrs.set_auto_maskandscale(False)
        flags = src.groups['geophysical_data']['l2_flags']
        names = flags.flag_meanings.split()
        masks = [int(flags.flag_masks[names.index(name)]) & 0xffffffff for name in dst.reject_flags.split()]
        wave = (np.array([w for w, _ in pairs]) if is_viirs else
                np.asarray(src.groups['sensor_band_parameters']['wavelength_3d'][:], dtype='float64'))
        chosen = np.flatnonzero((wave >= dst.requested_wavelength_min_nm) & (wave <= dst.requested_wavelength_max_nm))
        wave = wave[chosen]
        assert len(wave) == dst.wavelength_count
        assert (wave[0], wave[-1]) == (dst.wavelength_min_nm, dst.wavelength_max_nm)
        assert rrs.shape[:2] == dst['mean_Rrs'].shape
        for row in range(0, rrs.shape[0], 64):
            window = (slice(row, row+64), slice(None))
            if is_viirs:
                selected = [all_bands[i] for i in chosen]
                samples = [v[window] for v in selected]
                valid = np.logical_and.reduce([(raw != v._FillValue) & (raw >= v.valid_min) & (raw <= v.valid_max)
                                                for raw, v in zip(samples, selected)])
                decoded = np.stack([raw.astype('float64') * float(v.scale_factor) + float(v.add_offset)
                                    for raw, v in zip(samples, selected)], axis=-1)
            else:
                raw = rrs[window + (slice(int(chosen[0]), int(chosen[-1])+1),)]
                valid = np.all((raw != rrs._FillValue) & (raw >= rrs.valid_min) & (raw <= rrs.valid_max), axis=-1)
                decoded = raw.astype('float64') * float(rrs.scale_factor) + float(rrs.add_offset)
            q = flags[window]
            valid &= ~np.ma.getmaskarray(q)
            for mask in masks:
                valid &= (q.data.astype('uint32') & mask) == 0
            for name in ('latitude', 'longitude'):
                coord = src.groups['navigation_data'][name][window].filled(np.nan)
                np.testing.assert_equal(dst[name][window].filled(np.nan), coord)
                valid &= np.isfinite(coord) & (np.abs(coord) <= (90 if name == 'latitude' else 180))
            expected = np.trapezoid(decoded, x=wave, axis=-1)/(wave[-1]-wave[0])
            valid &= np.isfinite(expected)
            expected[~valid] = np.nan
            actual = dst['mean_Rrs'][window].filled(np.nan)
            np.testing.assert_array_equal(np.isfinite(actual), valid)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=True)
            valid_count += int(valid.sum())
            if valid.any():
                maximum_error = max(maximum_error, float(np.max(np.abs(actual[valid]-expected[valid]))))
        assert valid_count > 0, 'No valid pixels: validation would be vacuous'
        return dict(valid_pixels=valid_count, max_absolute_error=maximum_error,
                    shape=list(rrs.shape[:2]), wavelength_count=len(wave),
                    coordinates_and_mask_match=True, rtol=1e-5, atol=1e-8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('output')
    args = parser.parse_args()
    print(json.dumps(validate(args.source, args.output), indent=2))
