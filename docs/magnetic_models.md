# PFSS frequency bands and local magnetic grids

## PFSS field-line bands

`solar_toolkit.radio.fieldline_association.fieldline_band_segments` clips a
polyline of Carrington Cartesian coordinates in solar radii to an inclusive
frequency band. The caller specifies the mechanism, positive integer harmonic,
frequency bounds in MHz and PFSS source-surface radius.

- **Plasma frequency:** supply a positive Newkirk density multiplier. The code
  converts the band edges to spherical shells using the existing Newkirk model.
  It intersects the shells with the straight segments between traced points.
  One segment can enter and leave a shell twice.
- **Electron cyclotron frequency:** supply a magnetic-field magnitude in Gauss
  for each traced point. The code linearly interpolates the sampled magnitude
  between points. The band is based on the selected harmonic of the local
  electron cyclotron frequency, using 2.8 MHz per Gauss at the fundamental.

Missing coordinates or required field samples split the line. A zero-length
contact contributes no interval. Returned runs locate the contributing original
edges and fractional endpoints. These intervals describe where the model
frequency falls in the band. They do not establish emission, escape conditions,
or an association with an observed source.

For new PFSS runs, set `"sample_field_strength": true` in the existing compute
configuration to request magnitudes from the solver along the traced line.
The samples and their status travel with the completed PFSS bundle. Older
bundles can still be used for plasma bands. A cyclotron query reports missing
field samples for any line where they are unavailable. Sampling uses
the [public field-line interface](https://docs.sunpy.org/projects/sunkit-magex/en/stable/_modules/sunkit_magex/pfss/fieldline.html).

Generate a small bundle with the
[synthetic magnetic example](../Python/examples/public_api/README.md). From the
repository root on macOS, run:

```bash
./Apps/run.sh workflow pfss bands --bundle Local/magnetic-demo/pfss \
    --mechanism cyclotron --frequency-bounds-mhz 56 84 --harmonic 2 \
    --allowed-roots "$PWD/Local/magnetic-demo"
```

The command prints JSON to stdout. To write it to a new file, add
`--output` with a path under an allowed root and outside the verified bundle.
`--rss` supplies the outer radius when the bundle lacks it. For a plasma query
use `--mechanism plasma --newkirk-multiplier <positive-factor>`. On Windows,
pass the same arguments to `Apps/run.ps1`.

## SHARP CEA and local NLFFF grids

`solar_toolkit.modeling.nlfff.read_sharp_cea_boundary` reads explicit `Br`,
`Bt`, and `Bp` FITS files from the same SHARP record. Each component must be
identified by its `SEGMENT` card or a dot-separated filename token. The reader
checks HARP, time, CEA geometry, dimensions, magnetic units and an explicit
`QUALITY=0`. The result stores `(Bp, -Bt, Br)` in Gauss on the native pixel grid.
These components point west, north and radially outward. The CEA WCS describes
sample locations. The vector directions are local to each sampling point.
The returned mask identifies pixels with three finite components.
See the [HMI coordinate technical note](https://arxiv.org/abs/1309.2392).

`read_nlfff_grid` reads a computed, regular local volume. The public NPZ
contains the following entries:

| Name | Value |
| --- | --- |
| `volume` | Finite real array of shape `(3, nz, ny, nx)`, with west, north, outward components |
| `spacing` | Positive finite `(dx, dy, dz)` in the declared length unit |
| `reference` | Optional reference potential field, same shape as `volume` |
| `observed` | Optional observed lower boundary, shape `(3, ny, nx)` |
| `metadata` | Scalar Unicode string containing the JSON below |

```json
{
  "schema_version": 1,
  "frame": "local_west_north_radial",
  "array_axis_order": "component,z,y,x",
  "field_unit": "G",
  "length_unit": "Mm"
}
```

The reader converts compatible physical units to Gauss and Mm. The grid
contains local coordinate directions and spacings. A Carrington or
helioprojective registration needs separate information.

```bash
./Apps/run.sh workflow nlfff inspect-boundary \
    --br Local/magnetic-demo/synthetic.Br.fits \
    --bt Local/magnetic-demo/synthetic.Bt.fits \
    --bp Local/magnetic-demo/synthetic.Bp.fits --allowed-roots "$PWD/Local/magnetic-demo"
./Apps/run.sh workflow nlfff inspect-result \
    --result Local/magnetic-demo/local_field.npz --allowed-roots "$PWD/Local/magnetic-demo"
```

Both commands print a JSON summary. They read and validate their inputs without
solving a field. For a scientific assessment of an extrapolation, use the
solver's boundary, convergence and model diagnostics.
