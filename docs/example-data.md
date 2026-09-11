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

For a password-free local handoff, generate an Earthdata **User Token** in
[Earthdata Login](https://urs.earthdata.nasa.gov/), save only the token in an
owner-readable file outside the repository, and pass `--token-file /path/to/token`.
The token must not include a `Bearer ` prefix, quotes or a shell assignment.
On POSIX the file must have mode 600. The token-file option uses the default
`environment` strategy, sets EARTHDATA_TOKEN only during login, and restores the
previous environment afterwards. Neither token contents nor its path enter the
download manifest. The authenticated Earthdata session remains in process for
the transfer. Token-file login should run from one controlling thread.

NASA documents a 60-day lifetime for User Tokens; replace expired tokens in the
same file. [Token management](https://urs.earthdata.nasa.gov/documentation/for_users/user_token).

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

## Native HLS NDVI

After downloading one HLS V2 granule, process its separate band COGs directly:

```bash
python -m terragpu.hls data/hls-l30-example data/hls-ndvi.tif
# On a configured CUDA system, choose --backend cupy.
```

The processor reads spatial windows (512 pixels by default), selecting B04/B05
for L30 or B04/B8A for S30. It applies the documented 0.0001 reflectance scale,
rejects -9999 reflectance fill and 255 QA fill, and masks Fmask bits 1–4
(cloud, adjacent cloud/shadow, shadow, snow). Water and every aerosol level are
retained; this is an explicit example policy, not a universal science filter.
Zero denominators become NaN, and NDVI is not clipped. Input grids must match.
Output is a compressed float32 GeoTIFF with source and QA policy tags.
Exactly one native V2 granule is required per directory, and existing outputs
are rejected. No Dask scheduler or intermediate multiband scene is required.
See the [HLS V2 user guide, sections 6.2 and 6.4](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf).

## Validation status

The RGB example has been downloaded and run locally. Anonymous HLS discovery was verified with earthaccess 0.19.0: the documented
query selected `HLS.L30.T18SUJ.2024158T154508.v2.0`
(CMR `G3110284222-LPCLOUD`). Authenticated transfer using an Earthdata User Token
was verified on 2026-09-11: 15 files, 242,178,772 bytes. Native L30 NDVI was then
processed on CPU and checked across the full 3660 × 3660 grid against an
independent float64 raw-DN ratio and independently decoded QA bits. Masks, CRS
and transform matched; values passed rtol=1e-5, atol=2e-6. The maximum absolute
error was 0.00038147 (ratios near zero denominators can be large). Only 74,378
pixels survived masking, so this scene is a workflow check rather than a
representative clear-scene benchmark. No GPU speedup is inferred from this run.
Synthetic tests cover L30/S30 band selection, individual QA bits, native fill,
zero denominators and mismatched grids. Real S30 and GPU parity await validation.
