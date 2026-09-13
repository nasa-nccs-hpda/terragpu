# PACE OCI example

Install `terragpu[data,pace]` to download and process a native OCI Level-2 AOP
swath. The optional `pace` extra uses netCDF4; base raster imports do not require
it. NASA currently distributes the AOP collection as version 3.2:
[PACE release notes](https://pace.oceansciences.org/access_pace_data.htm).

```bash
terragpu-data nasa --short-name PACE_OCI_L2_AOP --version 3.2 \
  --bbox -76 34 -74 36 --start 2024-06-01 --end 2024-06-03 \
  --limit 1 --output data/pace-example --token-file /path/to/private-token
python -m terragpu.pace \
  data/pace-example/PACE_OCI.20240601T165222.L2.OC_AOP.V3_2.nc \
  data/pace-mean.nc
python scripts/validate_pace.py \
  data/pace-example/PACE_OCI.20240601T165222.L2.OC_AOP.V3_2.nc \
  data/pace-mean.nc
```

The catalog query selected this file during validation; use the filename in the
download manifest if catalog ordering changes. Bounding boxes select entire
swaths, not crops. This example downloaded 398,558,528 bytes. The verified local
cache and subsequent processing work offline. No PRISM path is needed.

## Operation and scientific scope

`PaceSwath` owns the grouped NetCDF handle. It reads `geophysical_data/Rrs`,
`geophysical_data/l2_flags`, `navigation_data/latitude` and `longitude`, and uses
`sensor_band_parameters/wavelength_3d` as the Rrs spectral coordinate. It checks
grid dimensions, units and increasing wavelengths. netCDF4 decodes packing and
masks fill and valid-range violations from variable metadata. In the example,
Rrs is packed int16 with scale 0.000002 and offset 0.05; ignoring the offset
would produce incorrect reflectance. See the
[netCDF4 mask/scale API](https://unidata.github.io/netcdf4-python/).

The reduction is the trapezoidal integral of Rrs divided by wavelength span,
using native samples inside 400–700 nm. There is no endpoint interpolation: this
file supplies 136 samples from 400 to 699 nm. The output records requested and
actual intervals. It requires all selected spectral samples and geolocation to
be valid, retains negative valid reflectances, and applies the named reject flags
`ATMFAIL LAND HIGLINT HILT HISATZEN STRAYLIGHT CLDICE HISOLZEN LOWLW NAVWARN
NAVFAIL PRODFAIL`. Flags are resolved from the file's metadata rather than fixed
bit positions. This is an explicit example policy, configurable with
`--reject-flags`, not a universal science filter. NASA explains the flag metadata
in its [ocean-color support forum](https://forum.earthdata.nasa.gov/viewtopic.php?t=3717).

This is a hyperspectral arithmetic and I/O workload, not an implementation of
NASA AVW, chlorophyll retrieval, atmospheric correction, or a validated new
science product. The result stays on the original swath with 2-D latitude and
longitude in compressed NetCDF. No regular affine map grid is invented.

## CPU/GPU execution and validation

Spatial tiles default to 128 × 128 pixels, retaining the selected spectral axis.
`--backend cupy` sends decoded tiles to one CUDA device; read/decode and NetCDF
write stay on CPU. Timings separate read/decode, compute including transfers, and
write calls; total also includes setup and final file flush. Compression buffers
can defer writing until close, so summed stage timings are not total runtime.
Output is staged and atomically renamed; existing files are rejected.

The simple streaming path avoids scheduler overhead for this single-file
workload. Dask should be evaluated for multiple swaths/devices after CPU/GPU
parity and repeated I/O measurements. A GPU kernel alone cannot remove NetCDF
decode costs. No Dask replacement or GPU speedup is established by this example.

Local real-data validation used
`PACE_OCI.20240601T165222.L2.OC_AOP.V3_2.nc`, CMR `G4194538633-OB_CLOUD`,
on 2026-09-11. Input shape is 1710 × 1272 × 172; 516,451 pixels survived the
example policy. `scripts/validate_pace.py` independently decodes raw packed data
in float64, integrates with NumPy trapezoid, and compares the entire output,
mask and coordinates (rtol=1e-5, atol=1e-8). The script requires nonzero valid
pixels. Synthetic tests cover packing/offset, nonuniform wavelengths, missing
spectral samples, flags including the signed high bit, invalid requests and
coordinates. CUDA parity tests await PRISM; these local results do not qualify
multi-GPU processing or paper performance claims.
