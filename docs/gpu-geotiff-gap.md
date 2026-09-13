# GPU GeoTIFF: contribution assessment

Checked 2026-09-13. A general claim to invent GPU TIFF/GeoTIFF reading and writing
is not supported by the existing landscape.

- NVIDIA nvTIFF already provides CUDA TIFF decode/encode, tiled and stripped
  inputs, and GeoTIFF metadata APIs. Its supported samples-per-pixel and codec
  combinations constrain which remote-sensing products can use it directly.
  https://docs.nvidia.com/cuda/nvtiff/
- The documented nvTIFF stream API includes host-memory input. GPU decompression
  therefore does not by itself establish a GPUDirect storage path.
  https://docs.nvidia.com/cuda/nvtiff/apireference.html
- Xarray-Spatial's current repository documents a native GeoTIFF/COG reader/writer
  with GPU options, as well as CPU/Dask/CuPy spatial computation. This needs a
  pinned-version capability and performance comparison before any uniqueness claim.
  Repository documentation is not our independent qualification of every feature.
  https://github.com/xarray-contrib/xarray-spatial
- cog3pio and GPU-native Zarr work are further related efforts; they should be
  included in the related-work review, not treated as evidence that no tools exist.
  https://github.com/weiji14/cog3pio
  https://talks.osgeo.org/foss4g-2025/talk/TSVGYJ/

A narrower research hypothesis is that geospatial metadata/mask contracts,
storage-aware tiling and tile reuse across useful analyses can improve complete
scientific workflows. Whether TerraGPU advances that hypothesis enough for a
methods paper depends on matched baselines and measured benefit. A wrapper around
KvikIO, a new raw cache format or dependency upgrades alone do not establish novelty.

A future device-buffer GeoTIFF path would need TIFF/BigTIFF parsing, byte-range
selection, predictor and codec support, metadata/mask/scale handling, validated
georeferencing and output interoperability. It should build on qualified codecs
rather than introduce an untested decoder. If pursued, first define one bounded
codec/layout contract and benchmark it against nvTIFF and relevant existing I/O.
Prove the underlying GDS path independently on the target mount. Retain explicit
CPU/fallback paths and report every host transfer and conversion cost.

The present implementation intentionally uses Rasterio for native decode/encode
and ordinary CuPy transfers. The publication suite tests the strongest measured
path to date. It neither claims native GPU GeoTIFF decoding nor depends on getting
GDS enabled at NCCS. No new codec is required to run the experiments.
