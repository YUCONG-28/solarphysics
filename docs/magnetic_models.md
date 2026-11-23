# PFSS frequency bands and local magnetic grids

## Purpose and reader

Purpose: fix how a traced field line is clipped to a frequency band, and what a
local magnetic grid does and does not carry once it is read.

Reader: a caller deciding whether a returned band interval or a read boundary may
be used as evidence about an observation, and the reviewer checking that a
documentation change kept that boundary.

`solar_toolkit.radio.fieldline_association.fieldline_band_segments` clips a
polyline of Carrington Cartesian coordinates in solar radii to an inclusive
frequency band. The caller specifies the mechanism, positive integer harmonic,
frequency bounds in MHz and PFSS source-surface radius.

The one claim this guide does not support: that a returned interval is evidence
of emission or of an association with an observed source.
[Assumptions and validity](#assumptions-and-validity) states that boundary in
full.

Vocabulary: [terminology](terminology.md). File structure:
[writing-standard](writing-standard.md).

## Scope

Covered:

- Two band mechanisms on a traced PFSS line: plasma frequency from a Newkirk
  density multiplier, electron cyclotron frequency from supplied field
  magnitudes.
- PFSS bundles whose solver sampled the field strength along the line, and older
  bundles that support plasma bands only.
- Reading a SHARP CEA boundary from explicit `Br`, `Bt` and `Bp` files, and
  reading a computed local NLFFF volume.

Deliberately out of scope:

- Solving a field. The readers validate their inputs and do not extrapolate.
- Emission, escape conditions or source association.
- Registering a local grid in Carrington or helioprojective coordinates.
- A scientific assessment of an extrapolation.

## Definitions and notation

| Symbol or field | Unit, frame | Meaning |
| --- | --- | --- |
| polyline | Carrington Cartesian, solar radii | traced field line the band is clipped against |
| `frequency bounds` | MHz, inclusive | band the polyline is clipped to |
| `h` (`--harmonic`) | positive integer | harmonic of the local electron cyclotron frequency |
| Newkirk density multiplier | positive factor | scales the existing Newkirk model for the plasma mechanism |
| `--rss` | solar radii | outer radius, supplied when the bundle lacks it |
| `B` | Gauss | magnetic-field magnitude sampled at a traced point |
| `B_s` | Gauss | magnitude linearly interpolated at fractional position `s` along a segment |
| `\nu_{\mathrm{ce}}` | MHz | local electron cyclotron frequency |
| `Br`, `Bt`, `Bp` | Gauss, CEA pixel grid | SHARP FITS components read from one record |
| stored boundary | Gauss, west/north/outward | components of the read SHARP boundary |
| `volume` | field unit `G`, shape `(3, nz, ny, nx)` | local NLFFF grid, west, north, outward |
| `spacing` | `(dx, dy, dz)` in the declared length unit | local grid spacing |
| `metadata` | scalar JSON, `frame=local_west_north_radial`, `length_unit=Mm` | frame and units of the local grid |

## Method

### PFSS field-line bands

- **Plasma frequency:** supply a positive Newkirk density multiplier. The code
  converts the band edges to spherical shells using the existing Newkirk model.
  It intersects the shells with the straight segments between traced points.
  One segment can enter and leave a shell twice.
- **Electron cyclotron frequency:** supply a magnetic-field magnitude in Gauss
  for each traced point. The code linearly interpolates the sampled magnitude
  between points. The band is based on the selected harmonic of the local
  electron cyclotron frequency, using 2.8 MHz per Gauss at the fundamental.

The cyclotron relation is Equation 1 and the band test after interpolation is
Equation 2.

```math
\nu_{\mathrm{ce}}(B) = 2.8\,\mathrm{MHz\,G^{-1}}\cdot B \tag{1}
```

```math
\nu_{\min} \le h\,\nu_{\mathrm{ce}}(B_s) \le \nu_{\max},
\qquad B_s = (1-s)\,B_i + s\,B_{i+1} \tag{2}
```

Symbols and units are those of
[Definitions and notation](#definitions-and-notation); the plasma mechanism
keeps the Newkirk model's own shell geometry and adds no equation here.

Missing coordinates or required field samples split the line. A zero-length
contact contributes no interval. Returned runs locate the contributing original
edges and fractional endpoints. `[implemented]`

For new PFSS runs, set `"sample_field_strength": true` in the existing compute
configuration to request magnitudes from the solver along the traced line. The
samples and their status travel with the completed PFSS bundle. Older bundles
can still be used for plasma bands. A cyclotron query reports missing field
samples for any line where they are unavailable. Sampling uses the
[public field-line interface](https://docs.sunpy.org/projects/sunkit-magex/en/stable/_modules/sunkit_magex/pfss/fieldline.html).

### SHARP CEA and local NLFFF grids

`solar_toolkit.modeling.nlfff.read_sharp_cea_boundary` reads explicit `Br`,
`Bt`, and `Bp` FITS files from the same SHARP record. Each component must be
identified by its `SEGMENT` card or a dot-separated filename token. The reader
checks HARP, time, CEA geometry, dimensions, magnetic units and an explicit
`QUALITY=0`. `[implemented]`

The stored boundary is Equation 3.

```math
\mathbf{B}_{\mathrm{stored}} = \left(B_p,\,-B_t,\,B_r\right) \tag{3}
```

The result stores `(Bp, -Bt, Br)` in Gauss on the native pixel grid. These
components point west, north and radially outward. The CEA WCS describes sample
locations. The vector directions are local to each sampling point. The returned
mask identifies pixels with three finite components. See the [HMI coordinate
technical note](https://arxiv.org/abs/1309.2392).

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

The grid contains local coordinate directions and spacings.

## Assumptions and validity

Two conditions bound both mechanisms: the polyline is supplied in Carrington
Cartesian coordinates in solar radii, and the band the caller names is the
inclusive band the model frequency is compared against.

- These intervals describe where the model frequency falls in the band. They do
  not establish emission, escape conditions, or an association with an observed
  source. `[inferred]`
- The vector directions are local to each sampling point; the CEA WCS describes
  sample locations only.
- The grid carries local coordinate directions and spacings; a Carrington or
  helioprojective registration needs separate information.
- The reader converts compatible physical units to Gauss and Mm.

## Failure modes and uncertainty

| Trigger | Detection | Consequence | Retained record |
| --- | --- | --- | --- |
| Missing coordinates or required field samples | the traced line is split at the gap | no interval spans the gap | returned runs locate the contributing original edges and fractional endpoints |
| A segment only touches a spherical shell without passing through it | the contact has zero length | no interval is contributed | the original edges of the remaining runs |
| Cyclotron query on a bundle without sampled field strength | the query reports missing field samples per line | no cyclotron band for that line | the plasma band remains available on the same bundle |
| A SHARP component without `SEGMENT` card or dot-separated filename token | the component cannot be identified | the record is not read | the explicit `Br`, `Bt`, `Bp` file names |
| SHARP record without an explicit `QUALITY=0` | the quality check fails | no boundary is returned | HARP, time, geometry and unit checks |
| `--output` outside an allowed root or inside the verified bundle | the command rejects the path | no file is written | the allowed roots and the bundle location |
| Local grid in an incompatible length unit | the unit conversion cannot be completed | the grid is not converted | the `metadata` JSON with `length_unit` |

## Verification

Generate a small bundle with the
[synthetic magnetic example](../Python/examples/public_api/README.md). From the
repository root on macOS, run:

```bash
./Apps/run.sh workflow pfss bands --bundle Local/magnetic-demo/pfss \
    --mechanism cyclotron --frequency-bounds-mhz 56 84 --harmonic 2 \
    --allowed-roots "$PWD/Local/magnetic-demo"
```

The command prints JSON to stdout. To write it to a new file, add `--output`
with a path under an allowed root and outside the verified bundle. `--rss`
supplies the outer radius when the bundle lacks it. For a plasma query use
`--mechanism plasma --newkirk-multiplier <positive-factor>`. On Windows, pass
the same arguments to `Apps/run.ps1`.

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

## Code anchors

| Step | Module or entry point |
| --- | --- |
| Band clip on a traced line | `solar_toolkit.radio.fieldline_association.fieldline_band_segments`, `workflow pfss bands` |
| Field samples along the line | `"sample_field_strength": true` in the existing compute configuration |
| SHARP CEA boundary | `solar_toolkit.modeling.nlfff.read_sharp_cea_boundary`, `workflow nlfff inspect-boundary` |
| Local NLFFF grid | `read_nlfff_grid`, `workflow nlfff inspect-result` |
