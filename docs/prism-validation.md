# PRISM validation: first run and dataset request

The synthetic qualification run is ready. No imagery is needed for that run.
It has passed CPU testing locally; V100/H100 CUDA execution is awaiting PRISM.
Run within an allocated job, not on a login node. Select one V100 first and one
H100 next. Use the site's approved CUDA 12 environment for the common baseline;
do not change system drivers. Keep separate result directories per GPU/job.

```bash
git clone https://github.com/nasa-nccs-hpda/terragpu.git
cd terragpu
git checkout codex/geospatial-revival
# Activate a PRISM-compatible Python/CUDA environment first.
python -m pip install -e '.[parallel,test]'
# Install exactly one compatible CuPy distribution if absent; see requirements/README.md.
bash scripts/prism_benchmark.sh results/v100-first
# Repeat in an H100 allocation using results/h100-first.
```

Report the commit, job/GPU type, Python and CUDA environment/module names,
`run-status.txt`, `tests.txt`, `cupy-config.txt`, and benchmark JSON files.
A failed preflight is useful: retain its log rather than switching backends.
Send results in the task; do not upload proprietary imagery to GitHub.
This runner is single-GPU and does not require Dask-CUDA or the full RAPIDS stack.
Dask-CUDA qualification is a later multi-GPU milestone.

## Automatic data access

You no longer need to supply PRISM paths for the downloadable quickstart.
Run `python -m terragpu.example` on CPU or append `--backend cupy` on a GPU.
See [automatic example data](example-data.md) for NASA catalog discovery and
authenticated native downloads. The product matrix below remains the target
for scientific validation; the tiny RGB quickstart does not replace it.

## Scientific validation datasets

Start with full-resolution source products plus their metadata. No training
labels, annotations or model checkpoints are needed. Do not pre-convert PACE or
AVIRIS to TIFF merely to match the current reader.

| Priority | Collection | Minimum first delivery | Include |
|---|---|---|---|
| 1 | HLS v2 L30 and S30 | One granule of each, ideally the same tile/nearby dates with land, water, some cloud and clear pixels | Original band COGs, Fmask, product metadata. At minimum B02/B03/B04 plus L30 B05 or S30 B8A; retain all bands if readily available |
| 1 | WorldView-2 or WorldView-3 | One orthorectified multispectral scene; eight-band preferred, four-band acceptable | Original multispectral TIFF/NITF and XML/IMD sidecars, explicit sensor/band order, units/scaling, nodata, CRS; retain original full scene |
| 2 | PACE OCI | One Level-2 granule with the spectral Rrs array, not only derived chlorophyll/BGC scalars | Native NetCDF, wavelength coordinates, navigation latitude/longitude, quality flags, complete product name and processing version |
| 2 | AVIRIS-NG (or AVIRIS Classic) | One orthocorrected Level-2 surface-reflectance flightline | Native cube plus ENVI .hdr when applicable, wavelengths/FWHM, bad-band metadata, nodata, units and geolocation/ancillary files |

HLS band mapping and delivery conventions: [HLS data products](https://hls.gsfc.nasa.gov/data-products/).
PACE Rrs wavelengths vary by product layout: [NASA Earthdata PACE wavelength discussion](https://forum.earthdata.nasa.gov/viewtopic.php?t=5552).
AVIRIS-NG surface reflectance and headers: [ORNL DAAC product guide](https://daac.ornl.gov/AVIRIS/guides/AVIRIS-NG_L2_Reflectance.html).

One scene each is enough to start integration. After correctness passes, expand
to 5–10 representative scenes per product, a 10-date HLS stack, and one input
whose uncompressed working set exceeds the tested GPU's VRAM. Do not choose
large downloads solely to hit a disk-size target: working-set memory depends on
bands, dtype, temporaries and requested operations. Include variation in nodata,
cloud fraction, surface type and storage layout; record selection criteria.

For each scene, send a readable PRISM path, source/granule identifier, product
version, acquisition time, file names/sizes, band/variable names, and permission
to redistribute or a public download reference. A directory listing and metadata
header are enough initially. Source paths and restricted scene names can stay
in this task; public issues should contain reproducible public identifiers only.

## Readiness gates

1. Synthetic CUDA test suite passes on both GPU types without fallback.
2. Product adapter parity: masks exact; geospatial/spectral metadata preserved;
   floating outputs within declared tolerance.
3. Real-data windowed and Dask runs produce matching outputs and complete with
   measured peak RAM/VRAM. Current synthetic I/O timings are not paper results.
4. Freeze versions and manifests before repeated paper experiments.
