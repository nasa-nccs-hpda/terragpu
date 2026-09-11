# Automatic example downloads

- Added a pinned, SHA-256-verified public RGB example with offline cache reuse.
- Added terragpu-data and terragpu-example commands; download and process without
  PRISM paths or a login for the small sample.
- Added optional earthaccess discovery/downloads with explicit collection/version,
  region, dates, bounded granule counts and local provenance manifests.
- NASA downloads use locally configured authentication and never persist credentials.
- RGB sample is explicitly a display-byte arithmetic demo, not NDVI/reflectance.

# Geospatial scope update

- Removed model training/inference packages, ML/DL examples and notebooks,
  neural-network visualization helpers, and model dependencies.
- Replaced the legacy Raster skeleton with a context-managed TerraGPU class.
- Made Dask/distributed optional via the parallel extra. I/O defaults to numpy;
  request dask/dask-cupy explicitly. Removed dataframe/backend helpers tied to ML.
- Added single-device windowed spectral processing and an end-to-end I/O comparison.
- Removed references to the former raster package from source and legacy tests.
- Replaced model-heavy container recipe and publishing CI with CPU geospatial checks.

# Revival development — 2026.9.0.dev0

- Modern package metadata, current CPU dependencies, lightweight package import,
  core tests and wheel-build CI. Legacy model code remains unqualified.
- Explicit I/O backends: default dask now means CPU; select dask-cupy for GPU.
  Explicit CuPy requests fail instead of silently falling back.
- Lazy spectral indices, corrected FDI band selection and documented SI formula,
  float arithmetic and NaN for invalid/zero-denominator results. HLS no longer
  fills invalid values with zero or eagerly computes each index.
- Band metadata and source arrays are no longer mutated by append/write calls.
  Appending indices normalizes output band coordinates to 1-based integers.
- Broken dl_pipeline/rf_pipeline installed entry points removed pending migration;
  legacy module paths remain available. Added terragpu-benchmark.
- Synthetic benchmark and PRISM/paper roadmap; no GPU speedup claimed yet.

# Changelog

All notable changes to this project will be documented in this file. Any security or bug findings
should be documented as CVE or BUG respectively.

## Status: [Released]

## [2022.01] - TBD

TBD

## [2021.11] - 2021-11-01

### Added

- Included deep learning and machine learning classes.
- Adding the integration of external projects for additional features.
- Adding additional Dask capabilities for multi-GPU support.

### Changed

- Moved backend from TensorFlow to PyTorch.

## [2021.02] - 2021-02-19

### Added

- Included raster library and additional raster implementations.
- Preprocessing libraries included.

### Approved

Jordan A. Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov

## [2020.12] - 2020-12-14

### Added

- Deep Learning modules.
- New indices based on Wooten calculation.
- Random Forest General driver, GPU acceleration.

### Changed

- Added deeprsensing features.
- More xarray native functions for speed.

### Removed

- Projects, added Examples directory.

### Approved

Jordan A. Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov

## [0.0.1] - 2020-09-05

### Added

- CHANGELOG to the project.
- New indices based on Wooten calculation.
- Random Forest General driver.

### Changed

- Restructure, moved Raster to its own class.
- More xarray native functions for speed.

### Removed

- Reference to Weis modular code.

### Approved

Jordan A. Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
