# Documentation writing standard

Purpose: fix how every tracked Markdown file in this repository is organized, what
each kind of document may claim, and which parts of a file are contract and may
not move.

Reader: anyone adding or revising documentation, and the reviewer who must decide
whether a documentation change is complete.

Vocabulary: [terminology](terminology.md). Behaviour and ownership rules:
[ARCHITECTURE.md](../ARCHITECTURE.md). Contribution procedure:
[CONTRIBUTING.md](../CONTRIBUTING.md).

## 1. Document classes

Every document belongs to exactly one class. The class decides the skeleton and
how far a revision may rewrite the body.

| Class | Purpose | Documents |
| --- | --- | --- |
| **S** scientific method | Explain how an observation is read, matched, fitted or compared, and what the result does and does not prove | `docs/time_matching.md`, `docs/foundations.md`, `docs/magnetic_models.md`, `docs/radio_comparison.md`, `docs/aia_radio_composite.md` |
| **C** architecture and contract | Explain ownership, boundaries, dependency direction, interfaces | `README.md`, `README.en.md`, `README.zh-CN.md`, `ARCHITECTURE.md`, `Apps/README.md`, `Apps/docs/architecture.md`, `Python/README.md`, `Python/docs/FUNCTION_MAP.md`, `environment/README.md` |
| **P** operating procedure | Explain how a person runs a routine, delivers a change, or recovers | `WORKFLOW_README.md`, `CONTRIBUTING.md`, `CONTRIBUTING.en.md`, `CONTRIBUTING.zh-CN.md`, `Python/CONTRIBUTING.md`, `Apps/docs/development.md`, `docs/migration.md`, example READMEs |
| **H** historical record | Preserve a dated decision, release, or audit as evidence | `Python/CHANGELOG.md`, `LICENSING.md` |

Also governed, with a shortened skeleton: the four `AGENTS.md` files,
`Apps/tests/fixtures/README.md`, and `.github` templates. `LICENSE` files are
license text and are never restructured.

## 2. Skeletons

### 2.1 Class S

1. **Purpose and reader** — what the guide specifies, and the single claim it
   does not support.
2. **Scope** — inputs covered; inputs deliberately out of scope.
3. **Definitions and notation** — a table of symbols, fields, flags with units and
   coordinate frame.
4. **Method** — prose plus numbered equations; one concept per subsection.
5. **Assumptions and validity** — when the method holds, and the error if it does
   not.
6. **Failure modes and uncertainty** — a table: trigger, detection, consequence,
   retained record.
7. **Verification** — how a user proves the run behaved as documented.
8. **Code anchors** — the module that implements each step.

### 2.2 Class C

1. **Purpose and boundary** — what this partition owns, where everything else
   belongs (two-column table).
2. **Layout** — path or interface table.
3. **Dependency directions** — arrow diagram; name the forbidden direction.
4. **Run and verify** — commands, with the environment requirement.
5. **Rules that must not be violated** — normative, each rule testable.
6. **Related documents** — links only, no restated explanation.

### 2.3 Class P

1. **Daily loop** 2. **Submit a reproducible change** 3. **Conflict and rollback**
4. **Troubleshooting** (symptom / cause / action table) 5. **Pre-delivery
checklist** 6. **Related documents**.

Existing step numbering that a reader may follow is preserved inside these
groups; do not renumber a command away from its step.

### 2.4 Class H and shortened forms

- Class H: never rewrite the body. Prepend a three-line status block —
  `> **Record type:**`, `> **Effective version:**`, `> **Status:**` — then leave
  the original content intact.
- `AGENTS.md` files: prepend a purpose/reader/precedence lead, keep every existing
  rule bullet verbatim, and close with a related-documents list. Rules may not be
  reworded, merged, or dropped.
- `.github` templates: keep the heading and the checklist items; a template is a
  form, not an essay.

## 3. Claim discipline

Tag every sentence that asserts a physical or numerical result:

| Tag | Meaning | Allowed when |
| --- | --- | --- |
| `[observed]` | about the data or image in hand | instrument, time and quantity are named |
| `[implemented]` | about what the code does | the code anchor is cited |
| `[inferred]` | an interpretation beyond the first two | the inference and its evidence are named |

Chinese documents use `[观测]`, `[实现]`, `[推断]`.

An untagged sentence may describe a field, a file, or a command — nothing else.
A negative boundary ("this does not prove co-location", "this is not an absolute
cross-instrument calibration") is as mandatory as a positive claim.

## 4. Equations and notation

Numbered display equations use a fenced block:

````markdown
```math
\Delta t = \tau_{\mathrm{obs}} - \tau_{\mathrm{ref}}
```
````

Introduce every symbol in the notation table first, name units in text
(`arcsec`, `MHz`, `SFU`, `s`), and distinguish measured from derived quantities
in the symbol name (`FWHM_observed`, `FWHM_intrinsic`).

## 5. One definition, one home

- Each concept is defined once; [terminology](terminology.md) records the English
  term, the Chinese term, and the document that owns the definition.
- Other documents link to that home instead of restating it.
- A restated definition is a defect: it drifts.

## 6. Anchors and links are contract

`Apps/tests/contracts/test_docs.py` resolves every relative link and every heading
fragment in all tracked Markdown. Renaming a heading therefore breaks a contract
unless every inbound reference is updated in the same change.

Headings with known inbound references — do not rename:

| Document | Anchors referenced elsewhere |
| --- | --- |
| `README.md` | `start-with-the-library`, `application-demonstration`, `repository-layout`, `contributing-and-maintenance`, `licensing-and-citation`, 中文 `从基础模块开始` |
| `README.en.md` | the same five previous anchors, each reachable as `README.md#<anchor>` |
| `CONTRIBUTING.md` | `prepare-the-environment`, `submit-a-reproducible-change`, `public-content-boundary`, 中文 `贡献指南`, `准备环境`, `提交可复现的修改`, `公开内容边界` |
| `Apps/README.md` | `install` |
| `Python/README.md` | `install`, `scientific-image-filenames`, `verify` |
| `docs/aia_radio_composite.md` | `输出格式` |
| `docs/migration.md` | `scientific-input-validation` |

Bilingual rules:

- English is canonical. `*.en.md` files carry the previous anchors and link to the
  canonical `*.md` heading, and they contain no ```bash or ```powershell
  installation block — installation has one maintained home.
- `README.md` and `README.zh-CN.md` must expose the same set of link targets;
  likewise `CONTRIBUTING.md` and `CONTRIBUTING.zh-CN.md`. Adding a link to one
  side requires adding it to the other.

## 7. Other contract strings

| File | Asserted content |
| --- | --- |
| `README.md`, `README.zh-CN.md` | start with `# Solar Physics Toolkit`; contain `(Python/README.md)` before `(Apps/README.md)`; contain `solarphysics_env_latest`; contain `create -n solarphysics_env_latest -c conda-forge python=3.14 pip`; no legacy distribution wording and no historical publication-manifest marker |
| `Apps/README.md` + `Apps/docs/architecture.md` + `Apps/docs/development.md` | together contain `solarphysics_env_latest`, `solarphysics_env`, `Python 3.14`, `run.sh`, `run.ps1`, `frontend app-v1`, `app-v1-preview`, `--config-file`, `paths.local.yaml`, `Local/`, `Auto`, `Light`, `Dark`, `.spapp.json`, `.spflow.json`, `MPL-2.0` |
| `docs/README.md` | links `../Python/README.md`, `../Python/docs/FUNCTION_MAP.md`, `../Apps/README.md`, `../CONTRIBUTING.md`, `../CONTRIBUTING.zh-CN.md`, `../environment/README.md`, `../Python/CHANGELOG.md`, `../LICENSING.md`, `../CITATION.cff`, `radio_comparison.md`, `time_matching.md`, `aia_radio_composite.md`, `migration.md` |
| `LICENSING.md` | links `LICENSE`, `Python/LICENSE`, `Apps/LICENSE`, `Apps/solar_apps/frontends/app_v1/LICENSE.md`, `Apps/LICENSES/GPL-3.0-only.txt`, `Apps/solar_apps/ui/media/NOTICE.txt` |
| all tracked Markdown | no drive-letter repository or toolchain path, no drive-letter user-profile path other than an explicit `<user>` or `<username>` placeholder, no private mail address, no historical publication-manifest filename pair, no test-count phrase |

If a required string conflicts with the new structure, keep the string, add a
footnote, and leave the test untouched.

Describe a forbidden token by its class, never by quoting it: the contract test
scans every tracked Markdown file, so a quoted token is itself an offence.

## 8. Verification

The contract suite imports `pytest`, `yaml`, and `tomllib`, so it needs the
Miniforge `solarphysics_env_latest` environment. On a machine without it, run the
text-only half with the same link and anchor rules the suite uses:

```console
python3 - <<'PY'
import re, pathlib, subprocess, collections
F = chr(96)
fence = re.compile("^(" + F + "{3,}|~{3,}).*?^\\1[^\\n]*$", re.M | re.S)
def prose(t): return fence.sub("", t)
def links(t):
    p = prose(t)
    inline = re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", p)
    refs = re.findall(r"^\s*\[[^\]\n]+\]:\s*(\S+)", p, re.M)
    return [v.split(' "', 1)[0].strip("<>") for v in inline + refs]
def anchors(t):
    s, c = set(), {}
    for h in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", prose(t), re.M):
        slug = re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-")
        n = c.get(slug, 0); c[slug] = n + 1
        s.add(slug + "-%d" % n if n else slug)
    s.update(re.findall(r"<a\s+(?:id|name)=[\"']([^\"']+)", t))
    return s
root = pathlib.Path(".")
files = [f for f in subprocess.run(["git", "ls-files", "*.md"], capture_output=True, text=True).stdout.split()]
bad = []
for f in files:
    p = root / f
    for link in links(p.read_text(encoding="utf-8")):
        if re.match(r"^[a-z]+:", link): continue
        path, _, frag = link.partition("#")
        target = (p.parent / path).resolve() if path else p.resolve()
        if not str(target).startswith(str(root.resolve())) or not target.exists():
            bad.append((f, link, "missing")); continue
        if frag and target.suffix == ".md" and frag not in anchors(target.read_text(encoding="utf-8")):
            bad.append((f, link, "missing anchor"))
print("broken:", len(bad))
for b in bad[:20]: print("  ", b)
PY
```

Run it before and after a change; the count must not increase, and for a
documentation-only change it must stay at zero.
