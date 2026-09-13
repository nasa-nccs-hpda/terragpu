# Storage and execution tradeoffs in GPU-accelerated geospatial raster workflows

Working manuscript scaffold, not a submission or completed results section.
Authors, affiliations, target venue and final scope require maintainer review.

## Abstract

To be written after the controlled experiment matrix is complete. Distinguish
resident-kernel, prepared-cache and native-input-to-output results. Do not use
the largest observed speedup as an unqualified headline.

## Research questions

1. When does a GPU improve the complete runtime of scientifically equivalent
   raster workflows compared with a CPU worker sweep?
2. How do storage location, tile size and reuse across spatial scales change
   the tradeoff between decoding, transfer, computation and output?
3. What memory demand and numerical error accompany each strategy?

## Contribution and related work

Describe TerraGPU's product-specific metadata/masking contracts and execution
strategies. Establish why the reusable software is useful alongside existing
Rasterio, SciPy, CuPy, Dask and Xarray-Spatial workflows. Compare native GPU I/O
capabilities with nvTIFF, KvikIO and relevant geospatial I/O work. Do not claim to
invent GPU GeoTIFF I/O. See docs/gpu-geotiff-gap.md for the initial source list.
A comparative capability table and pinned upstream baselines remain to be completed.

## Methods implemented

The publication suite starts from public WorldView ARD or a supplied georeferenced
GeoTIFF. The WorldView path performs CPU QA alignment and radiometric index
preparation inside each timed trial. The generic path starts from the supplied
GeoTIFF and does not include upstream product generation. Trials evaluate distinct
spatial scales using finite-neighbor means and variances, centered float64
moments, float32 outputs, and explicit halos and nodata semantics.

Three strategies are compared: separate passes per analysis; retaining each tile
across analyses; and initial conversion to an uncompressed halo cache followed
by reuse. CPU worker counts and tile sizes are experimental factors. CPU workers
run bounded threaded Rasterio reads and SciPy computation; the output writer and
GDAL codec are single threaded for every backend. GPU execution uses ordinary
CuPy host/device transfers and retains each tile during its requested analyses.

Wall time includes input decode, applicable preparation/packing, all analyses,
and LZW GeoTIFF output close. Downloads, validation and scratch deletion are
excluded. Validation compares all outputs to an independently centered float64
reference, exact finite masks, CRS, affine grid and band metadata. Unit tests
also compare to direct neighborhood calculations and whole-image references.

Case order is randomized with recorded seeds. Each invocation records warmups
and repeated trials, raw timings, environment, source hashes and memory samples.
Plots aggregate medians from separate invocations without pooling within-run
samples. The experimenter must document that repeated invocations came from
independent allocations; the plotting code cannot establish that fact.

## Results to populate

- Independent-job total-time distributions and best-tested-CPU comparisons.
- Sensitivity to tile size, worker count, query count and storage location.
- Direct versus packed/reused execution including all conversion/output costs.
- Numerical errors and sampled memory, including starting allocation baselines.
- Existing product-suite results, labeled separately by timing scope.

Exploratory GH200 results exist but are not the final controlled experiment set.
Native-pipeline GPU measurements introduced by this change are still pending.
No GDS performance result is verified. Stereo matching accuracy needs qualification
before treating that workload as scientifically useful photogrammetry.

## Limitations

The default example is small and does not establish larger-than-VRAM operation.
CPU worker tuning covers one specified implementation, not all competing CPU
algorithms. Filesystem caches are uncontrolled and writes have no fsync barrier.
RSS/device memory are sampled and may miss peaks; device usage can include other
processes and allocator retention affects baseline memory. Generalizing from one
GPU architecture or a few scenes requires additional evidence. Native swath
workflows retain their own geometry and are not covered by this affine spatial
pipeline. No GPU codec or physical direct-storage route is qualified here.

## Reproducibility and software availability

Archive an approved release, complete environment records, scene manifests and
raw/derived public metrics. Exclude credentials, proprietary imagery and private
system paths. Generate figures with the checked-in plotting command. Document
institutional review and dataset redistribution permissions where applicable.

## Contributions and AI assistance

Maintainers must confirm author contributions and accountability. Codex assisted
with implementation, tests, documentation and this scaffold. Verification includes
CPU regression tests, independent numerical references, recorded CI and user-run
PRISM qualification; final scientific interpretation and manuscript approval
remain with the human authors. Expand this disclosure for the selected venue.
