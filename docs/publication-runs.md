# Native-pipeline publication experiments on PRISM

This suite closes measurement gaps in the existing cache-only experiment. It is
an evaluation harness, not a claim that the paper is ready for submission.
No GDS installation is needed. The default uses the previously downloaded public
WorldView sample, or downloads it automatically; no Earthdata token is required.

## Install with uv inside your GPU allocation

Run from the repository root. The commands assume the allocation provides one GPU
and at least eight CPU cores to the job step. Use the CPU count granted by your
allocation if different; the benchmark honors affinity and SLURM_CPUS_PER_TASK.

```bash
git switch codex/geospatial-revival
git pull --ff-only
srun --ntasks=1 --cpus-per-task=8 bash scripts/setup_prism_publication.sh

export TERRAGPU_IO_ROOT=/lscratch/$USER/terragpu
export TERRAGPU_STORAGE_LABEL=nvme
srun --ntasks=1 --cpus-per-task=8 --export=ALL bash -ec '
  arch=$(uname -m)
  if [[ "$arch" == arm64 ]]; then arch=aarch64; fi
  source ".venv-publication-$arch/bin/activate"
  bash scripts/run_prism_publication.sh results/prism-publication-nvme-01
'

tar -czf prism-publication-nvme-01-results.tar.gz \
  -C results prism-publication-nvme-01
```

The separate environment uses Python 3.12, the existing CuPy/CUDA 12 lane,
SciPy, psutil and matplotlib. KvikIO is not required by this new suite.
Setup uses the architecture-correct uv executable configured by UV_BIN.

The default experiment is substantially longer than the earlier cache-only run:
it writes and validates real compressed GeoTIFF outputs for every repetition.
With eight CPUs available, it has 30 cases (four CPU worker counts plus GPU,
three strategies and one/three analyses), each with one warmup and five measured
trials. Allow several GB of scratch space for the sample; larger inputs require
space for the working cache, prepared indices and every query output. Temporary
payloads are removed after each validated trial. Logs, raw JSON, CSVs and figures
stay in the results directory. A failed run retains its partial JSON and logs.
Use a fresh results directory for each invocation.

## What the cases mean

Every trial starts from the original input. For the default WorldView example,
CPU QA alignment and NDVI/NDWI preparation are included in the timer. For
`--source`, timing starts from that supplied GeoTIFF, and processing that created
it is outside scope. All cases produce the same LZW-compressed float32 GeoTIFF
outputs, with identical bands, grid, masks and validation tolerances. Validation
requires exactly one distinct output per query and checks float32 storage, NaN
nodata, LZW compression and 256-by-256 tiled layout; numerical agreement alone
does not permit a cheaper output encoding in a comparison.

- `stream`: reread/decode and transfer tiles separately for each distinct analysis.
- `reuse`: read/transfer each tile once and retain it across the analyses.
- `packed`: decode and pack the raw halo cache first, then reuse cached tiles.
  Packing is included in total time, not charged to an imaginary earlier job.

Queries are finite-neighbor spatial means and variances. The default windows are
15/31 pixels for query one, 29/61 for query two, and 43/91 for query three: a
spatial-scale sensitivity experiment. Different queries produce different outputs.
The CPU runs the same centered-float64 calculations as the GPU. CPU workers
parallelize window reads and SciPy computation using a bounded thread pool. The
writer is serialized and GDAL compression uses one thread for every backend.
This is a worker sweep of a specified Rasterio/SciPy baseline; it does not claim
to be the fastest possible CPU implementation or an external package benchmark.

The timer includes native preparation, optional packing, raster reads, transfers,
all analyses and compressed output close. It excludes download/checksum discovery,
validation, warmups and deletion of scratch files. Queries share preprocessing
within a trial; fresh preprocessing is charged again in the next trial. There is
no KvikIO/GDS mode here because ordinary CuPy won the earlier working-cache test.

Cases run in seeded randomized order each repetition. Raw times and within-run
bootstrap summaries are retained. A single invocation is one experimental run,
not several independent scheduler jobs. Cache flushing and fsync durability are
not requested; these are warm-filesystem application timings, not physical disk
bandwidth measurements. Output hashes may differ with insignificant floating-point
rounding; metadata, finite masks and numerical tolerance are checked instead.

## Memory and scaling

A monitor samples process RSS and, for GPU cases, device-wide CUDA used memory
and this process's default CuPy memory-pool reservation every 10 ms. The reports
include starting usage, sampled peaks and sampling failures. Device usage includes
other GPU processes. Allocator reservation excludes non-pool CUDA allocations.
The same process runs several cases, so retained host/library allocations affect
RSS baselines. These measurements are not exact per-operation allocation peaks.
GPU pool caches are cleared before GPU trials, outside the measured interval.

To sweep tile sizes, worker counts, query counts or process a larger real scene,
pass extra arguments to the runner inside the same activated job step:

```bash
bash scripts/run_prism_publication.sh results/prism-publication-large-01 \
  --source /path/to/georeferenced.tif \
  --tiles 512 1024 2048 --workers 1 2 4 8 --query-counts 1 3 5
```

Do not replicate pixels and call that a new independent scene. Larger-than-VRAM
qualification requires a representative real input whose decoded size exceeds
VRAM, and a successful run with observed memory. This suite is tile bounded but
the default sample does not meet that qualification. Native swath PACE/VIIRS
remain in the existing five-product suite; they are not converted to affine
GeoTIFFs by this experiment.

## Independent runs and figures

Repeat the same matrix in at least three separately allocated jobs, with fresh
output directories and recorded seeds (TERRAGPU_SEED). Keep code, environment,
source, storage label and CPU allocation fixed. Three is a starting minimum;
collect more if variability prevents stable conclusions. To compare NVMe and
PanFS, run separate groups and retain the mount reports. Do not pool them.

Each run creates PNG/PDF plots and a CSV under `figures/`. To combine comparable
completed runs after downloading them:

```bash
python -m terragpu.publication_figures \
  results/prism-publication-nvme-01/publication.json \
  results/prism-publication-nvme-02/publication.json \
  results/prism-publication-nvme-03/publication.json \
  --output results/publication-nvme-figures
```

Aggregation validates matching source, code, environment, hardware, storage label
and case matrix. It also rejects nonfinite/negative memory values, peaks below
starting usage, missing GPU memory measurements in GPU cases and invalid recorded
numerical-validation evidence before creating figures. It summarizes run medians, not a pooled set of correlated
within-job repetitions. Points show each run median; memory plots report sampled
process RSS. CPU comparisons in the trial CSV use the best tested CPU case at the
same query count, rather than selecting only a convenient slow CPU tile size.
Inspect small numerical differences, memory baselines and all negative results.

## Submission gates still requiring evidence

The commands produce evidence, not automatic qualification. Before submission:
run the GPU tests on PRISM; collect independent runs and representative larger
scenes; compare relevant upstream tools under matched scientific conventions;
review stereo accuracy before including it; complete methods/related work,
figures, author review, applicable institutional clearance and archived release.
See paper/manuscript.md and docs/gpu-geotiff-gap.md. No tag, merge or publication
is performed by these scripts.

## Allocation provenance for new run series

New reports include a unique run identifier and the Slurm job, step and optional
array identifiers. Only these selected environment fields are captured; the
full environment and authentication variables are not exported. The figure
provenance retains the reported identifiers.

For multiple reports produced with allocation provenance, add
`--require-distinct-jobs` to the figure command. It requires at least two reports
with different run IDs and numeric Slurm job IDs, and rejects separate `srun`
steps from the same allocation. This checks reported allocation identity; it does
not prove statistical independence, eliminate shared-storage interference or
replace review of scheduler records.

Older reports and local runs without Slurm remain usable with the default figure
command. They cannot pass this optional allocation check. Do not invent or guess
missing job IDs. Keep the same benchmark revision throughout a run series rather
than upgrading mid-series solely to obtain this metadata.

## Diagnose native pipeline stages

Use a separate diagnostic run before choosing prefetching, transfer overlap or
codec changes. The optional `--profile-stages` records per-sample
`stage_timings` in `publication.json`: read/decode (including masks, scaling and
host padding), host-to-device transfer, computation, device-to-host transfer
(or CPU contiguous materialization), writer calls, output close, and packing
when selected. Native WorldView preparation remains the existing separate
`native_preparation_seconds` measurement.

With the publication environment active in an allocated PRISM GPU session:

```bash
git pull --ff-only
TERRAGPU_IO_ROOT=/lscratch/$USER/terragpu \
TERRAGPU_STORAGE_LABEL=nvme \
bash scripts/run_prism_publication.sh results/prism-publication-profile-01 \
  --profile-stages --workers 1 4 --query-counts 1 3 --strategies stream reuse
```

Request at least four CPU cores for that example; consult
[PRISM validation](prism-validation.md) for environment and device checks.
Keep completed throughput results intact. Profiling synchronizes the current
CUDA stream before and after each GPU stage, changes execution timing, and is
not a substitute for an unprofiled performance run. CPU worker durations may
overlap: their sums are aggregate worker time, not elapsed pipeline time.
Setup, scheduling and measurement overhead are not fully attributed. Writes
can be buffered until close; neither measurement implies an fsync durability
barrier or GDS. Plot aggregation rejects mixing profiled and unprofiled runs.
CUDA profiling tests are included in the runner but require PRISM qualification.

## Test the larger-data hypothesis

The controlled size sweep reuses the public WorldView sample without additional
credentials. It materializes 1x1, 2x2 and 4x4 repetitions of its QA-masked NDVI/NDWI
raster: approximately 18, 72 and 289 million pixels. These are physical GeoTIFFs,
not VRT aliases. All files and figures label the repeated extent as synthetic;
the seams and enlarged coordinates do not represent new observations.

With the publication environment active and at least eight allocated CPU cores:

```bash
git pull --ff-only
TERRAGPU_IO_ROOT=/lscratch/$USER/terragpu \
TERRAGPU_STORAGE_LABEL=nvme \
srun --ntasks=1 --cpus-per-task=8 bash scripts/run_prism_scaling.sh \
  results/prism-scaling-01 --scales 1 2 4 --tiles 1024 2048 \
  --workers 1 4 8 --query-counts 1 3

tar -czf prism-scaling-01-results.tar.gz -C results prism-scaling-01
```

Use the environment from `scripts/setup_prism_publication.sh`; see
[PRISM validation](prism-validation.md) for device/environment checks. Allow
roughly 40 GiB free scratch for this default sample and configuration; the
runner checks a conservative uncompressed input/output estimate before each
size. Generated scenes and trial outputs are deleted after use. Timing,
validation, environment and figure artifacts remain in the results directory.

This tests two separate effects: total data volume (scene size) and work per
GPU tile (tile size). It uses the reuse strategy for every backend, five measured
trials plus a warmup, and records CPU/GPU parity for every trial. WorldView QA/index
preparation and generation of the repeated scene are outside the timer. Timed
work starts at the generated GeoTIFF and ends after all compressed spatial-feature
GeoTIFFs close. Do not compare these totals directly with the earlier native
WorldView preparation-inclusive results. BigTIFF supports large outputs.

`sweep/scaling.csv` contains every configuration's raw-size throughput, median
runtime and sampled memory peaks. Per-size JSON retains individual trials and
provenance. `sweep/scaling.png` and PDF compare the best tested CPU and GPU at
each size and query count; `best-tested.json` identifies the selected worker/tile
configurations. This selection is descriptive and should be confirmed in fresh
allocations. Memory sampling and filesystem-cache limitations still apply.

Larger total data alone need not improve the GPU ratio when per-tile compute and
I/O both scale proportionally. This sweep can reveal that outcome as well as a
crossover. It does not establish larger-than-VRAM behavior or generalization to
new scenes. For the paper, confirm findings with independent larger imagery;
`--source /path/to/real.tif --scales 1` runs the same spatial-feature benchmark on
an existing real scene without repetition (still normalized to physical float32
GeoTIFF before timing). Additional scales such as 8 are opt-in and increase disk
space and runtime substantially.

### Direct bash launches and CPU allocation evidence

The publication and scaling reports record `execution.cpu_resources`: CPU
logical count, process affinity and allowlisted Slurm per-task/per-node CPU
variables. Missing values remain null; job CPU strings such as `8(x2)` remain
unaltered. These are separate observations, not a verified allocation size.
A shell can see all node CPUs or inherit stale scheduler variables. In particular,
`SLURM_CPUS_ON_NODE` is not a per-task worker budget.

Launching with `bash` does not automatically use every visible CPU: the explicit
`--workers 1 4 8` sweep runs separate configurations with those worker counts,
and the runner limits common numerical-library thread counts to one. Confirm
that the allocation permits the requested counts using `scontrol show job
"$SLURM_JOB_ID"`; the existing allocation/affinity guard cannot prove scheduler
entitlement when per-task information is absent. No additional `srun` step is
required by TerraGPU itself. See [PRISM validation](prism-validation.md).
