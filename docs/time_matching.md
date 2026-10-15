# Complete chronological image matching

`solar_toolkit.time.build_sequence_match_plan` is an independently callable
library API. Application adapters use the same fixed plan for preview and export;
using the matcher does not require starting an application. See the
[synthetic sequence example](../Python/examples/public_api/sequence_matching_example.py)
for a complete example that reads no observation files.

## Selection and hierarchy

Apply channel, quality and explicit selection conditions before constructing a
`TimeStream`. Each independent wavelength or frequency sequence is one stream.
The planner then filters every stream to the user's original observed-UTC window
`[time_start, time_end)`: start is included, end is excluded. An absent bound is
open; two finite bounds must satisfy start before end. Outside-window frames
never fill a gap. Clock offsets, exposure-midpoint transformations and light-time
corrections do not widen the original-observation window.

Nonempty streams are sorted by increasing valid frame count, then stable source
ID. The smallest stream provides the **root master**. Each following stream is a
child of its immediate predecessor. The largest, last stream provides the
**output time axis**: every one of its frames supplies one output row and its real
observed UTC. Every valid in-window frame in every nonempty stream appears at
least once in the final rows. Empty streams display no data and do not enter the
hierarchy. An all-empty window produces zero rows and cannot be exported.

Selected sorted indices and original observed times never decrease. Repeating
the current observation is allowed; returning to an earlier one is forbidden.
`original_index` preserves the supplied input position, which may be unordered;
`sorted_index` describes chronological order. Equal timestamps retain distinct
stable identities. Reuse does not create a new independent observation.

## Nearest matching subject to full coverage

For adjacent parent and child layers, every original child selects one parent.
Parent indices may stay unchanged or advance by one; the mapping must cover
every parent. A dynamic program minimizes the sum of absolute parent-child
matching-time differences over these feasible mappings. Each original child
contributes once. Repetitions introduced when later layers expand the sequence
do not reweight an earlier layer's optimization.

Coverage takes priority over unconstrained nearest-neighbour selection.
Consequently, a selected parent is not always the absolute nearest candidate.
The edge records `nearest_distance_ns`, `is_nearest` and `coverage_forced` to make
that distinction inspectable. Equally near candidates are not deviations. Equal
transition costs consistently prefer advancing, so repeated runs are
deterministic. Equally long layers necessarily map one-to-one in chronological
order. Time tolerance is a quality flag; it never deletes a selected frame.

Output rows are expanded by following the adjacent maps backwards from the last
layer. Ancestors are not rematched directly to the output frame. Stage costs use
the separately stored effective matching clock; ordering and window eligibility
use original observation UTC. Each stage uses rolling costs and stored
backtracking decisions with `O(NM)` time and `O(NM)` backtracking space.

## Root, output and time differences

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

An edge's `delta_ns` is parent minus child. The maximum output difference is
`max(abs(selection.delta_ns))` over all present selections in all output rows;
it is not the largest adjacent-edge difference. Without explicit scientific
corrections, these are the corresponding real observation UTC differences.

Adjacent nearest choices can accumulate. For example, a synthetic output at
4 seconds may select an intermediate frame at 6 seconds, which selects a root
at 10 seconds. Those adjacent differences are 2 and 4 seconds; the root differs
from output by 6 seconds. This does not imply time reversal during playback.
Video follows increasing output UTC, and every selected stream stays unchanged
or advances. Different coverage ranges can also require large, flagged boundary
differences when every in-window observation must be retained.

Coverage, omitted frames and reuse counts are recomputed from final output
references. Scientific operations requiring simultaneous observations must
inspect quality flags rather than treating full display coverage as temporal
coincidence.

## Precision and exact image references

`normalize_time` retains the source text, source scale, true UTC with nine
fractional digits and continuous integer TAI nanoseconds. Naive scientific clocks
default to UTC; explicit time zones and TAI labels are converted. Legacy
`datetime` wrappers retain their return types and microsecond precision limit.
Invalid complete observation timestamps cannot silently become date-only
midnight observations. FITS observation metadata takes precedence over filename
fallbacks; original header cards remain available in the locator.

Logical image identities include file version, image HDU and frequency/Stokes
plane. Unsupported nonsingleton cube axes are rejected rather than counted as a
single observation. Spectrum sampling points do not compete with images for the
hierarchy. Duplicate paths do not add independent frames. Materialization loads
the exact planned HDU/plane and rejects source versions changed since indexing;
it does not perform another nearest-time lookup. Derived observation headers
normalize `DATE-OBS`, `T_OBS` where present, and `TIMESYS` consistently to UTC.

RR and LL originals are separately filtered to the window before synthesis.
Within the pairing tolerance, chronological one-to-one pairing first maximizes
compatible pairs, then minimizes their total absolute time difference. Original
polarization frames participate at most once, with compatible shapes, grids,
frequencies and units. Both members of an existing pair must be in the window;
a midpoint check alone is insufficient. Display matching can repeat a complete
validated L+R pair, and preserves both member timestamps.

## Calling the module and applications

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
export traverses the plan chronologically. Image selection windows are independent
of plotting shifts, browsing cursors and spectrum display windows. Jet preserves
the exposure midpoint and light-time clock as an explicit matching transformation.

Window values, policy versions and input identities participate in fingerprints.
Changing a window rebuilds the plan even if the retained frame set happens to be
unchanged. Changed input files require a rescan. Preview and export must share
the same current fingerprint and exact references.

## Persisted configuration and migration

The plan structure remains schema **1**. Matching strategy **3** is
`hierarchical_monotone_full_coverage_minimum_absolute_delta`; strict RR/LL pairing
is strategy **2**. These algorithm versions are separate from serialization
versions. Cached strategy 1 or 2 matching plans are invalidated and rebuilt.

Composer saves schema **2**, reads schema 1 with a migration notice, and records
`auto_min_frames_hierarchical` as its root policy. Former time, relative-index,
strict and manually selected-master settings migrate to chronological full
coverage. Files lacking window settings remain unbounded. App v1 provides
`set_time_window`; cursor navigation does not edit its saved window. Tolerances,
clock offsets, source times, layer order and migration notices remain inspectable.

## Original-FITS and video acceptance

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
selected within the adapter's 5 MHz band tolerance and remain recorded explicitly.

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
