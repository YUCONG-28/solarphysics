# Complete chronological image matching

## Purpose and reader

Purpose: fix how a displayed sequence is built from independently observed
streams, which frame each output row may reference, and what the resulting time
differences mean.

Reader: a user of `solar_toolkit.time.build_sequence_match_plan` and of the App v1
and Composer surfaces that consume its plan, plus the reviewer checking that a
documentation change kept the contract.

`solar_toolkit.time.build_sequence_match_plan` is an independently callable
library API. Application adapters use the same fixed plan for preview and export;
using the matcher does not require starting an application. See the
[synthetic sequence example](../Python/examples/public_api/sequence_matching_example.py)
for a complete example that reads no observation files.

The one claim this guide does not support: that a matched row is a coincident
physical observation. [Verification](#verification) states that boundary together
with the acceptance checks.

Vocabulary: [terminology](terminology.md). File structure:
[writing-standard](writing-standard.md).

## Scope

Covered:

- Apply channel, quality and explicit selection conditions before constructing a
  `TimeStream`. Each independent wavelength or frequency sequence is one stream.
- The user's original observed-UTC window, adjacent-layer matching, strict RR/LL
  pairing, persisted configuration and migration, and the acceptance workflow.

Deliberately out of scope:

- Clock offsets, exposure-midpoint transformations and light-time corrections do
  not widen the original-observation window.
- Spectrum sampling points do not compete with images for the hierarchy.
- Physical source identification and scientific simultaneity.

## Definitions and notation

| Symbol or field | Unit, frame | Meaning |
| --- | --- | --- |
| window | observed UTC, half-open | `[time_start, time_end)`, start included, end excluded |
| `N`, `M` | frame counts | original child frames and parent frames of one adjacent layer pair |
| `j_i` | parent frame index | the parent that original child frame `i` selects |
| effective matching clock | ns | separately stored clock for stage costs |
| `delta_ns` | ns | one edge's difference, parent minus child, Equation 2 |
| `selection.delta_ns` | ns | selected stream's effective clock minus output effective clock, Equation 3 |
| `\Delta_{\max}` | ns | largest absolute selection difference over output rows, Equation 4 |
| `original_index` | input position | supplied input position, which may be unordered |
| `sorted_index` | chronological position | position in chronological order |
| `within_tolerance` | boolean quality flag | tolerance is a flag, never a frame deletion |
| `nearest_distance_ns`, `is_nearest`, `coverage_forced` | edge fields | separate a nearest choice from a coverage-forced one |

| Field | Meaning |
| --- | --- |
| `plan.master_source_id`, `row.master_frame` | Smallest root stream and its selected frame |
| `plan.output_source_id`, `row.output_frame` | Largest final stream and its output frame |
| `plan.source_order`, `plan.parent_maps` | Complete hierarchy and mappings for original child frames |
| `row.time` | Output frame's real observed UTC |
| `row.match_time` | Output frame's separate effective matching clock |
| `row.parent_matches` | Exact adjacent-layer edges selected for this row |
| `selection.delta_ns` | Selected stream's effective clock minus output effective clock |
| `selection.within_tolerance` | Quality flag evaluated against that output-relative difference |

## Method

### Selection and hierarchy

The planner then filters every stream to the user's original observed-UTC window
`[time_start, time_end)`: start is included, end is excluded. Outside-window
frames never fill a gap.

Nonempty streams are sorted by increasing valid frame count, then stable source
ID. The smallest stream provides the **root master**. Each following stream is a
child of its immediate predecessor. The largest, last stream provides the
**output time axis**: every one of its frames supplies one output row and its
real observed UTC. Every valid in-window frame in every nonempty stream appears
at least once in the final rows.

Selected sorted indices and original observed times never decrease. Repeating the
current observation is allowed; returning to an earlier one is forbidden.
`original_index` preserves the supplied input position, which may be unordered;
`sorted_index` describes chronological order. Reuse does not create a new
independent observation.

### Nearest matching subject to full coverage

For adjacent parent and child layers, every original child selects one parent.
Parent indices may stay unchanged or advance by one; the mapping must cover every
parent. A dynamic program minimizes the sum of absolute parent-child
matching-time differences over these feasible mappings. That objective is
Equation 1. Each original child contributes once. Repetitions introduced when
later layers expand the sequence do not reweight an earlier layer's optimization.

```math
\min_{j_1 \ldots j_N} \sum_{i=1}^{N}
  \left| \tau^{\mathrm{match}}_{j_i}
       - \tau^{\mathrm{match}}_{i} \right|
\quad\text{subject to}\quad
j_{i+1} - j_i \in \{0, 1\},\quad
\{\, j_1, \ldots, j_N \,\} = \{\, 1, \ldots, M \,\} \tag{1}
```

Coverage takes priority over unconstrained nearest-neighbour selection.
Consequently, a selected parent is not always the absolute nearest candidate.
The edge records `nearest_distance_ns`, `is_nearest` and `coverage_forced` to make
that distinction inspectable. Equally near candidates are not deviations. Equal
transition costs consistently prefer advancing, so repeated runs are
deterministic. Equally long layers necessarily map one-to-one in chronological
order.

Output rows are expanded by following the adjacent maps backwards from the last
layer. Ancestors are not rematched directly to the output frame. Stage costs use
the separately stored effective matching clock; ordering and window eligibility
use original observation UTC. Each stage uses rolling costs and stored
backtracking decisions with `O(NM)` time and `O(NM)` backtracking space.

### Root, output and time differences

An edge's `delta_ns` is parent minus child. That edge difference is Equation 2.

```math
\Delta_{\mathrm{edge}} = \tau_{\mathrm{parent}} - \tau_{\mathrm{child}} \tag{2}
```

The maximum output difference is `max(abs(selection.delta_ns))` over all present
selections in all output rows; it is not the largest adjacent-edge difference.
That maximum is Equation 4, and the per-selection difference it is taken over is
Equation 3.

```math
\Delta_{\mathrm{sel}} = \tau^{\mathrm{match}}_{\mathrm{selected}}
  - \tau^{\mathrm{match}}_{\mathrm{output}} \tag{3}
```

```math
\Delta_{\max} = \max \left| \Delta_{\mathrm{sel}} \right| \tag{4}
```

Adjacent nearest choices can accumulate. For example, a synthetic output at 4
seconds may select an intermediate frame at 6 seconds, which selects a root at 10
seconds. Those adjacent differences are 2 and 4 seconds; the root differs from
output by 6 seconds. This does not imply time reversal during playback. Video
follows increasing output UTC, and every selected stream stays unchanged or
advances. Different coverage ranges can also require large, flagged boundary
differences when every in-window observation must be retained.

Coverage, omitted frames and reuse counts are recomputed from final output
references.

### Precision and exact image references

`normalize_time` retains the source text, source scale, true UTC with nine
fractional digits and continuous integer TAI nanoseconds. Naive scientific clocks
default to UTC; explicit time zones and TAI labels are converted. Legacy
`datetime` wrappers retain their return types and microsecond precision limit.
`TimeStamp.to_datetime()` returns naive UTC and truncates submicrosecond digits
without changing the exact stamp. It raises `ValueError` for a leap second, which
`datetime` cannot represent; retain the exact stamp for that observation. FITS
observation metadata takes precedence over filename fallbacks; original header
cards remain available in the locator.

Logical image identities include file version, image HDU and frequency/Stokes
plane. Duplicate paths do not add independent frames. Materialization loads the
exact planned HDU/plane and rejects source versions changed since indexing; it
does not perform another nearest-time lookup. Derived observation headers
normalize `DATE-OBS`, `T_OBS` where present, and `TIMESYS` consistently to UTC.

RR and LL originals are separately filtered to the window before synthesis.
Within the pairing tolerance, chronological one-to-one pairing first maximizes
compatible pairs, then minimizes their total absolute time difference. Original
polarization frames participate at most once, with compatible shapes, grids,
frequencies and units. Display matching can repeat a complete validated L+R pair,
and preserves both member timestamps.

### Calling the module and applications

```python
from solar_toolkit.time import build_sequence_match_plan

# streams contains the caller's already selected TimeStream objects.
plan = build_sequence_match_plan(
    streams,
    time_start=start_utc,
    time_end=end_utc,
)
for row in plan.rows:
    output_utc = row.time.utc
    selections = row.selections
    # Read these exact references, render, and encode in this row order.
```

Image Composer, App v1 synchronization, Source Map, Overlay, MUSER video and
AIA/radio sequences consume fixed plans. Browsing can move forwards or backwards;
export traverses the plan chronologically. Image selection windows are
independent of plotting shifts, browsing cursors and spectrum display windows.
Jet preserves the exposure midpoint and light-time clock as an explicit matching
transformation.

Window values, policy versions and input identities participate in fingerprints.
Changing a window rebuilds the plan even if the retained frame set happens to be
unchanged. Changed input files require a rescan. Preview and export must share
the same current fingerprint and exact references.

### Persisted configuration and migration

The plan structure remains schema **1**. Matching strategy **3** is
`hierarchical_monotone_full_coverage_minimum_absolute_delta`; strict RR/LL
pairing is strategy **2**. These algorithm versions are separate from
serialization versions.

Composer saves schema **2**, reads schema 1 with a migration notice, and records
`auto_min_frames_hierarchical` as its root policy. Former time, relative-index,
strict and manually selected-master settings migrate to chronological full
coverage. Files lacking window settings remain unbounded. App v1 provides
`set_time_window`; cursor navigation does not edit its saved window. Tolerances,
clock offsets, source times, layer order and migration notices remain
inspectable.

## Assumptions and validity

The plan is valid only under the conditions below; each one is a property of the
input window, of the supplied frames or of the cached state.

- An absent bound is open; two finite bounds must satisfy start before end.
- Empty streams display no data and do not enter the hierarchy. An all-empty
  window produces zero rows and cannot be exported.
- Equal timestamps retain distinct stable identities.
- Time tolerance is a quality flag; it never deletes a selected frame.
- Scientific operations requiring simultaneous observations must inspect quality
  flags rather than treating full display coverage as temporal coincidence.
- Invalid complete observation timestamps cannot silently become date-only
  midnight observations.
- Unsupported nonsingleton cube axes are rejected rather than counted as a single
  observation.
- Both members of an existing pair must be in the window; a midpoint check alone
  is insufficient.
- Without explicit scientific corrections, these are the corresponding real
  observation UTC differences.
- Cached strategy 1 or 2 matching plans are invalidated and rebuilt.

## Failure modes and uncertainty

| Trigger | Detection | Consequence | Retained record |
| --- | --- | --- | --- |
| Coverage forces a parent that is not the nearest candidate | `is_nearest` is false and `coverage_forced` is set | the mapping is not nearest-neighbour | `nearest_distance_ns` on the edge |
| Adjacent offsets accumulate across layers | `selection.delta_ns` exceeds tolerance while the root differs from the output | a large output-relative difference without time reversal | `row.parent_matches` and the quality flag |
| Coverage ranges differ between streams | flagged boundary differences above tolerance | every in-window observation is still retained | the retained frame and its flag |
| A leap second reaches `to_datetime()` | `ValueError` | the `datetime` view is unavailable | the exact `TimeStamp` |
| An incomplete observation timestamp is supplied | the timestamp is rejected, not turned into midnight | no date-only observation is created | the locator header cards |
| A source version changed since indexing | materialization rejects the file | no second nearest-time lookup is performed | the planned HDU/plane |
| A cached strategy 1 or 2 plan is present | the plan is invalidated | the plan is rebuilt | the migration notice |

## Verification

The generic radio workflow takes explicit scientific inputs and a new private
output directory. It supplies no event preset or default time window. After
installing both packages, run from the repository root; replace all placeholders:

```sh
Apps/run.sh workflow radio time-matching-acceptance \
  --aia-directory "<AIA-directory>" \
  --dart-directory "<DART-directory>" \
  --muser-directory "<MUSER-image-directory>" \
  --time-start "<inclusive-start-UTC>" \
  --time-end "<exclusive-end-UTC>" \
  --aia-waves <first-wavelength-A> <second-wavelength-A> \
  --frequencies <frequency-MHz> ... \
  --fov <xmin-arcsec> <xmax-arcsec> <ymin-arcsec> <ymax-arcsec> \
  --dart-mode RR+LL \
  --scenarios aia-dart aia-dart-muser \
  --output "<new-private-output-directory>"
```

Both UTC bounds, two AIA wavelengths, one to six radio frequencies, four FOV
coordinates and the output directory are required. `--dart-mode` selects `I` or
`RR+LL`; `--scenarios` selects `aia-dart`, `aia-dart-muser`, or both. The MUSER
directory is required only for a scenario containing MUSER. AIA input uses
`<AIA-directory>/<wavelength>/*.fits`; DART uses
`<DART-directory>/<frequency>MHz/{I,RR,LL}/*.fits` as applicable. MUSER input is
the supplied directory of combined image FITS. Its actual frequency planes are
selected within the adapter's 5 MHz band tolerance and remain recorded
explicitly.

The workflow re-reads source FITS, records rejected inputs, and saves resolved
settings, input manifests, fixed plans, row tables, independent coverage/order
checks, representative previews and full MP4 files. It checks window membership,
complete valid-frame coverage, monotone references, strict pair membership,
encoded references, presentation order and independently decoded frame count.
Frame locators, observed UTC and rendered-pixel checksums retain provenance.
Outputs and source observations remain outside public source; inputs are not
modified.

These checks establish software matching and rendering behavior, not physical
source identification or scientific simultaneity. Visual checks of representative
frames complement full metadata checks. When originals are unavailable, a saved
manifest can support metadata-only replay; label that scope separately and mark
new original-data/video acceptance as not executed. Historical videos or reports
do not establish acceptance of changed code.

## Code anchors

| Step | Module or entry point |
| --- | --- |
| Plan construction | `solar_toolkit.time.build_sequence_match_plan` |
| Time normalization and exact stamps | `normalize_time`, `TimeStamp.to_datetime()` |
| Plan consumers | Image Composer, App v1 synchronization, Source Map, Overlay, MUSER video, AIA/radio sequences |
| App v1 saved window | `set_time_window` |
| Acceptance workflow | `Apps/run.sh workflow radio time-matching-acceptance` |
