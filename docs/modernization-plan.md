# TerraGPU geospatial acceleration roadmap

Updated 2026-09-11 following the maintainer's scope change. Target hardware:
NCCS PRISM V100 and H100. Target collections: HLS, WorldView, PACE, VIIRS and AVIRIS,
plus a public satellite stereo pair for classical photogrammetry benchmarking.
Model training, inference, estimators and model-framework integration are removed
from scope and source. TerraGPU owns its Raster abstraction; the former raster
wrapper library is not a dependency.

## Review and implemented foundation

The original repository had undeclared dependencies, Python 3.7/CUDA 10.1
recipes, eager model imports, CPU Dask imports coupled to GPU availability,
implicit GPU selection, eager HLS index computation, incorrect FDI membership
and SI arithmetic, integer overflow, metadata mutation and unreproducible timing.

The foundation now has modern packaging, optional Dask, explicit CPU/GPU paths,
corrected lazy indices, georeferenced I/O tests, a context-managed Raster class,
single-device window processing, and resident/end-to-end benchmark harnesses.
Model packages and model-specific supporting assets are removed; generic color
helpers and historical geospatial material remain. Historical tests and generated
HTML are not evidence of supported behavior; tests/ is the current suite.

## Execution design

See [the backend decision](execution-backends.md). Keep Rasterio/rioxarray for
geospatial I/O, xarray for labeled metadata, NumPy for reference computation,
CuPy for GPU arrays. Dask is an optional chunk/distribution layer, not the source
of GPU acceleration. Compare it with direct streaming on identical inputs.
Current package versions resolved locally are recorded in the CPU constraints
file; that file is not a GPU lock or a minimum-version qualification.

V100 needs the CUDA 12 lane; use a shared supported software environment on
V100/H100 for controlled comparisons, then a separate latest-compatible H100
lane. Verify actual drivers and wheel architecture support inside allocations.
[NVIDIA architecture support](https://developer.nvidia.com/blog/?p=102958),
[CuPy installation](https://docs.cupy.dev/en/stable/install.html), and
[RAPIDS/Dask-CUDA compatibility](https://docs.rapids.ai/install/) govern selection.
Do not add the full RAPIDS meta-package when only Dask-CUDA is required.

## Milestones and gates

1. **Geospatial core:** implemented CPU reference, optional Dask, Raster class,
   windowed indices, package tests and benchmarks. Gate: base wheel runs without
   Dask or model packages; parallel tests pass; GPU parity on PRISM still pending.
2. **PRISM + product contracts:** qualify V100/H100 environments, record topology,
   driver/runtime and exact versions; obtain representative scene IDs/paths.
   Implement product-native ingestion, nodata/QA, scale/offset and spatial/spectral
   contracts. Gate: all four products pass metadata/numerical round trips and
   at least one larger-than-VRAM workload completes with measured peak memory.
3. **Geospatial kernels:** spectral reductions/normalization, band math, masking,
   focal operations with tested halos, and reprojection/resampling as independently
   validated workloads. Select order from real use cases and profiling. Never
   silently substitute a different spatial algorithm or radiometric convention.
4. **Performance:** compare direct streaming, resident arrays and Dask; optimize
   chunk/block alignment, fuse kernels where beneficial, evaluate pinned buffers,
   transfer overlap and allocation pools. Add multi-GPU only after single-device
   parity. Gate: repeated end-to-end improvements with unchanged scientific output.
5. **Paper/release:** freeze manifests, environment locks, code revision and
   experiment protocol; collect independent runs and ablations; generate figures
   from raw records; draft methods/results/limitations and archive artifacts.
   Updating dependencies alone is not the scientific contribution.

## Product/workload matrix

| Collection | Initial operations | Required metadata/contract |
|---|---|---|
| HLS | QA-masked reflectance, NDVI/NDWI, time reductions | L30/S30/version, band mapping, Fmask, scale/offset and common grid |
| WorldView | Multispectral indices, masks, focal operations | Sensor/band order, units, orthorectification/RPC status, redistribution terms |
| PACE | Wavelength selection, masking, spectral reductions | Instrument/product/level, groups, wavelengths, quality flags, swath versus grid |
| VIIRS | Ocean-color masking and spectral reductions | Platform/product/version, per-variable packing, wavelengths, flags and swath navigation |
| AVIRIS | Bad-band masking, spectral normalization/reductions | Generation/product, wavelength, bad-band flags, units, ENVI/NetCDF layout |

Public HLS L30, WorldView-3 ARD, PACE OCI and NOAA-21 VIIRS examples now have
local CPU metadata/numerical validation and download manifests. AVIRIS and
additional product variants remain pending. Access does not imply permission
to redistribute WorldView. PACE/VIIRS output retains native swath coordinates.
CUDA parity and product-specific scientific qualification remain open.

Focal means now have a tiled GeoTIFF path with explicit halos, finite-neighbor
boundary semantics and seam tests, including tiles smaller than the halo.
This is available independently of the resident focal benchmark; larger-than-VRAM
qualification still requires measured memory and real GPU runs.

## Publication experiment design

Hypothesis: controlling residency, working-set size and execution granularity
improves geospatial throughput across multispectral/hyperspectral workloads;
the best strategy depends on I/O, memory and arithmetic intensity. Include
negative results and the crossover where CPU/direct streaming beats scheduling.

Baselines: idiomatic NumPy/Rasterio, tuned CPU Dask, CuPy resident, windowed CuPy,
Dask-CuPy single-GPU, then available multi-GPU configurations. Match dtype,
preprocessing, output compression/layout and numerical tolerances. Separate
V100/H100 hardware comparisons on common software from latest-stack comparisons.

Record scene checksums/version, input/output sizes, GPU count/VRAM/interconnect,
CPU/RAM, storage, driver/runtime, environment, threads and revision. Separate
cold startup/JIT, disk cache conditions, graph construction, read, transfer,
compute and write. Synchronize CUDA; report raw samples, medians, dispersion and
confidence intervals across independent jobs. Use warmups, >=10 kernel repeats,
multiple end-to-end jobs, and randomized backend order. Measure peak RAM/VRAM,
transfer volume, pixels/s or spectra/s and scaling efficiency. Add energy only
with trustworthy telemetry. Masks must match exactly; report numerical error
alongside time. Current I/O benchmark is a preliminary smoke test, not this full
protocol.

Ablations: tile/chunk size; lazy versus resident versus streaming; fused versus
unfused kernels; transfers included/excluded; allocation pool; number of GPUs.
Figures: stage breakdown, throughput versus scene size, peak memory, numerical
error, controlled V100/H100 comparison, strong/weak scaling and crossover plots.
Paper sections: problem/related systems, geospatial contracts, execution design,
validation, experimental method, results/ablations, limitations, reproducibility.

## Next PRISM run

Use scripts/setup_prism_uv.sh and scripts/run_prism_paper.sh in the existing
allocation, as documented in prism-paper-runs.md. The user will run the jobs and
return result bundles; no SSH or scheduler configuration is needed here.
No remote job has been submitted and no GPU improvement is claimed locally.
