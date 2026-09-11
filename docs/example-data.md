# Automatic example data

No PRISM filesystem paths are required for the public quickstart. Downloads are
explicit commands; importing TerraGPU never performs network I/O.

## No-login example

```bash
python -m pip install -e .
terragpu-example
# Or without reinstalling entry points:
python -m terragpu.example
# CUDA environment:
terragpu-example --backend cupy --output data/example-si-gpu.tif
```

This downloads Rasterio's 1.7 MB `RGB.byte.tif`, verifies a pinned SHA-256, processes
it in spatial windows, and checks output values, mask, CRS and transform against
an independent CPU reference. Download bytes are pinned to an immutable upstream
commit. Source URL, checksum, band order, units and license link are recorded in
`data/examples/sample-manifest.json`.

The three bands are display-scaled red/green/blue bytes. The operation is the
geometric mean of RGB (the library's SI arithmetic), **not NDVI or a calibrated
reflectance product**. There is no NIR band. This is an I/O/backend correctness
example, not a representative GPU performance benchmark or replacement for the
HLS/WorldView/PACE/AVIRIS scientific experiments.

The input cache works offline after its first download. Output TIFFs are not
overwritten: pass a new `--output` when rerunning. To download only:

```bash
terragpu-data sample --cache-dir data/examples
```

Source: [Rasterio example dataset](https://github.com/rasterio/rasterio/blob/57d9fda6c31c5595ea54262f905b43c5f8419e06/tests/data/RGB.byte.tif),
[upstream BSD license](https://github.com/rasterio/rasterio/blob/57d9fda6c31c5595ea54262f905b43c5f8419e06/LICENSE.txt).
No raster binaries are committed to this repository.

## NASA collections without PRISM paths

Install the optional downloader (not required for the RGB example):

```bash
python -m pip install -e '.[data]'
terragpu-data nasa --short-name HLSL30 --version 2.0 \
  --bbox -77.1 38.8 -76.9 39.0 --start 2024-06-01 --end 2024-07-01 \
  --limit 1 --search-only
```

Search requires no authentication. To download the discovered native granule,
remove `--search-only`, choose a new output directory and select a local login
strategy:

```bash
terragpu-data nasa --short-name HLSL30 --version 2.0 \
  --bbox -77.1 38.8 -76.9 39.0 --start 2024-06-01 --end 2024-07-01 \
  --limit 1 --output data/hls-l30-example --login-strategy interactive
```

For unattended runs use `--login-strategy environment` (the default), with an
Earthdata token or credentials already configured locally, or `netrc` for an
existing local configuration. TerraGPU passes `persist=False` and never writes
credentials into its manifest. Do not put credentials into GitHub or chat.
See [earthaccess authentication](https://earthaccess.readthedocs.io/en/stable/user/explanation/authenticate/).

Use `HLSS30` for the S30 counterpart. For PACE and public AVIRIS collections,
select their exact CMR collection short name/version and date/region using
[Earthdata Search](https://search.earthdata.nasa.gov/), then use the same command.
This downloads original native products and sidecars exposed as granule data
links; product-specific processing adapters are still being implemented.
WorldView requires an authorized public sample or licensed data source; this
command does not bypass access restrictions or provide a generic WorldView feed.

The region selects whole granules; it does not crop them. `--limit` defaults to
one granule (maximum ten), and a single spectral granule can still be large.
Downloads go to a staging directory and are published only after successful
completion. Completed output includes granule identifiers, exact query, client
version, file sizes and SHA-256 hashes. These hashes establish a reusable local
cache, not independent provider checksum verification. Repeating the same command
verifies and reuses the cache without network/authentication. A different query
requires a new output directory. CMR search results can change over time; retain
the downloaded manifest for the actual selected granules.

## Validation status

The RGB example has been downloaded and run locally. Anonymous HLS discovery was verified with earthaccess 0.19.0: the documented
query selected `HLS.L30.T18SUJ.2024158T154508.v2.0`
(CMR `G3110284222-LPCLOUD`). NASA download orchestration is covered by offline tests;
authenticated real-granule transfer still requires an Earthdata login and has
not been claimed as validated. GPU example parity awaits PRISM hardware.
