# WorldView and VIIRS examples

## WorldView: archived ARD multispectral tiles

```bash
terragpu-data worldview --cache-dir data/worldview-example
python -m terragpu.worldview data/worldview-example/1040010025C68500.json data/worldview-indices.tif
python scripts/validate_worldview.py data/worldview-example/1040010025C68500.json data/worldview-indices.tif
```

The downloader fetches a pinned WorldView-3 sample over Addis Ababa from the
public Maxar ARD sample bucket: an eight-band analytic COG (~257 MB), cloud and
saturation masks, and STAC item/collection metadata. SHA-256 and sizes are
checked against the bundled descriptors, and verified files work offline. The
source collection labels its license **proprietary**. Public sample access is
not a redistribution grant; the repository contains descriptors and synthetic
tests, not vendor imagery. Source URLs and attribution are retained in the
sample manifest. See [vendor sample guidance](https://ard.maxar.com/docs/getting-started/getting-started/).

`process_worldview` supports local WorldView ARD STAC items with `ms_analytic`,
`cloud-mask-raster` and `ms-saturation-mask-raster` assets beside the item.
It takes band positions from the analytic asset's `eo:bands`, verifies its grid
against STAC, and applies the ARD reflectance scale (DN/10000). It computes NDVI
and green/NIR NDWI in spatial windows using NumPy or `--backend cupy`.

The processor respects internal raster validity masks and requires cloud class
1 (clear) and saturation class 0. Quality masks are aligned with nearest-neighbor
resampling; the sample cloud grid is coarser than the analytic grid. Water and
terrain shadow are not excluded by this example policy. Zero denominators and
invalid input become NaN. Output is an atomic, non-overwriting two-band float32
GeoTIFF with the analytic CRS/transform and source/QA tags.

Product specifications: [analytic bands and masks](https://developers.maxar.com/docs/specifications/ard-user-guide/ard-output-files/multispectral-image),
[saturation mask](https://developers.maxar.com/docs/specifications/ard-user-guide/ard-output-files/ms-pixel-saturation),
[reflectance scale](https://ard.maxar.com/docs/operations-support/faq/).
The ordering API for ARD was retired in January 2026; this adapter reads archived
files and does not depend on that API. Raw DN/IMD deliveries, pan-sharpened RGB,
SWIR/CAVIS, and arbitrary WorldView processing levels are not yet supported.

Local validation on 2026-09-11 used acquisition `1040010025C68500`, tile
`37/033311333031`, 2016-11-17. The 4251 × 4251 grid yielded 14,570,155 pixels
valid for both indices. Every output value and mask was checked against float64
raw-band ratios and separately reprojected QA; maximum absolute error was
1.251e-7. CRS and transform matched. This establishes workflow correctness, not
a general WorldView science validation or GPU speedup.

## VIIRS: NOAA-21 L2 ocean color

```bash
python -m pip install -e '.[data,viirs]'
terragpu-data nasa --short-name VIIRSJ2_L2_OC --version 2025.0 \
  --bbox -76 34 -74 36 --start 2024-06-01 --end 2024-06-03 \
  --limit 1 --output data/viirs-example --token-file /path/to/private-token
python -m terragpu.viirs data/viirs-example/JPSS2_VIIRS.20240601T173601.L2.OC.nc data/viirs-mean.nc
python scripts/validate_pace.py data/viirs-example/JPSS2_VIIRS.20240601T173601.L2.OC.nc data/viirs-mean.nc
```

The validator supports both PACE and VIIRS. Catalog ordering may change; use the
filename recorded in the manifest. This is a whole-swath download (~82 MB), not
a crop. The existing Earthdata token flow and verified offline cache apply.
See [NASA's VIIRS ocean-color collection directory](https://cmr.earthdata.nasa.gov/search/site/collections/directory/OB_CLOUD/gov.nasa.eosdis).

`ViirsSwath` reads separate `Rrs_<nm>` variables and each variable's packing,
valid range and fill, plus `l2_flags` and native 2-D latitude/longitude.
It shares the tiled CPU/CuPy reduction engine and named QA policy with PACE,
while retaining VIIRS-specific discovery and decoding. `--wavelength-range`,
`--reject-flags`, `--tile-size` and `--backend` work as in the [PACE example](pace.md).

The example's five nominal wavelengths are 411, 445, 489, 556 and 667 nm. The
trapezoid mean over these sparse samples is an arithmetic workload; it is not
spectrally equivalent to PACE's hyperspectral mean, an AVW product or a
chlorophyll retrieval. The output remains a swath NetCDF, without a fabricated
affine grid. This adapter targets NASA OB.DAAC L2 ocean color, not VIIRS surface
reflectance, nighttime lights, thermal, L1B or SDR products. Only NOAA-21 has
real-data validation so far.

Local validation used `JPSS2_VIIRS.20240601T173601.L2.OC.nc`, CMR
`G3798108024-OB_CLOUD`, collection version 2025.0 (file processing version R2025).
All 3216 × 3200 pixels were checked against independent raw-packed float64
decoding and integration: 907,436 valid pixels, identical masks/navigation, and
maximum absolute error 3.514e-9 sr^-1 (rtol=1e-5, atol=1e-8).

Both adapters have synthetic CPU tests and CUDA parity tests. GPU execution,
repeated performance measurements and multi-device scheduling await PRISM.
Single-device tiled processing remains the baseline; Dask should be compared
for collections and multiple GPUs rather than added to every small operation.
