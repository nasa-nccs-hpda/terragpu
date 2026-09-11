# TerraGPU

GPU-accelerated geospatial raster processing with CPU reference implementations.
TerraGPU focuses on spectral indices, explicit device residency, windowed I/O,
and optional distributed arrays. Model training and inference have been removed.
This is a development release; CUDA validation on PRISM is pending.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# Optional parallel processing and the full development test suite:
python -m pip install -e '.[parallel,test]'
python -m pytest
```

## Download and run an example

```bash
python -m terragpu.example
```

Downloads a checksum-verified 1.7 MB public RGB raster and validates windowed
processing. No login or PRISM paths are needed. Use `--backend cupy` in a CUDA
environment. See [automatic example data](docs/example-data.md) for caching,
offline use and optional NASA Earthdata downloads for HLS/PACE/AVIRIS.

## Process a large raster without Dask

```python
from terragpu.streaming import process_indices

process_indices(
    "scene.tif", "indices.tif",
    bands=["blue", "green", "red", "nir1"],  # actual source band order
    indices=["ndvi", "ndwi"], backend="cupy", tile_size=1024,
)
```

Use `backend="numpy"` on CPU. Window processing bounds application array sizes
by tile size; source blocks, GDAL cache and GPU allocator pools also affect memory.
The output must not already exist. Input nodata becomes NaN. Product QA masks
and radiometric conventions must be handled explicitly.

## Work with a raster

```python
from terragpu import Raster
from terragpu import io

with Raster("scene.tif", bands=["blue", "green", "red", "nir1"], backend="numpy") as scene:
    io.imsave(scene.index("ndvi"), "ndvi.tif")
```

For lazy chunked arrays choose `backend="dask"` or `"dask-cupy"` and install the
parallel extra. Default I/O uses the CPU NumPy path, not Dask. TerraGPU owns its
Raster class and has no dependency on the former raster wrapper package.

## Benchmarks and roadmap

```bash
terragpu-benchmark --backend numpy --size 2048 --output results/numpy.json
python -m terragpu.benchmark_io --size 2048 --repeat 3 --output results/io.json
```

See the [geospatial roadmap](docs/modernization-plan.md),
[execution-backend decision](docs/execution-backends.md),
[benchmark protocol](docs/benchmark-protocol.md), and
[GPU installation guide](requirements/README.md).
The paper plan targets PRISM V100/H100 with HLS, WorldView, PACE and AVIRIS.
Start with the [PRISM test and dataset checklist](docs/prism-validation.md).
Historical scripts in archives/ and generated HTML documentation are not the
supported API; current self-contained tests live in tests/.

## Authors

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
- Margaret Wooten, margaret.wooten@nasa.gov

## Contributors

- Andrew Weis, aweis1998@icloud.com
- Brian Lee, brianlee52@bren.ucsb.edu

## Installation

See the build [guide](requirements/README.md).

## Contributing

Please see our [guide for contributing to terragpu](CONTRIBUTING.md).

## References

Tutorials will be published under [Medium](https://medium.com/@jordan.caraballo/) for additional support
and development, including how to use the library or any upcoming releases.

Please consider citing this when using terragpu in a project. You can use the citation BibTeX:

```bibtex
@software{
  A_Caraballo-Vega_TerraGPU_2020,
  author = {A Caraballo-Vega, Jordan},
  doi = {10.5281/zenodo.5765917},
  license = {Apache-2.0},
  month = {7},
  title = {{TerraGPU}},
  url = {https://github.com/nasa-cisto-ai/terragpu},
  version = {2021.11},
  year = {2020}
}
```
