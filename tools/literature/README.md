# Literature Catalog Maintenance

This directory owns the executable literature-search and catalog-maintenance
workflow. `Paper/` remains a static evidence library: the only generated files
are `Paper/catalog/papers.json` and `Paper/catalog/papers.md`.

## Commands

Run an offline, read-only consistency check:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\literature\update_catalog.ps1 -Check
```

Refresh the deterministic catalog view without network searches:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\literature\update_catalog.ps1 -AsOfDate 2026-07-15 -SkipLiveSearch
```

Run the incremental arXiv and Crossref search by omitting
`-SkipLiveSearch`. Only results scored at relevance level A or B can enter the
catalog; lower-relevance search results are skipped. Search results must fall
inside the configured `lookback_days` window ending on `-AsOfDate`; future and
older results are rejected. Each source reports successful/failed query counts,
and a live run fails if both sources complete zero successful queries. The
public parameters are:

- `-AsOfDate yyyy-MM-dd`
- `-SkipLiveSearch`
- `-Check`
- `-CommitAndPush`
- `-GitRemote <name>`
- `-GitBranch <name>`

`-Check` never performs a live search or writes files. It validates the
35-field JSON schema, rejects duplicate citekeys, normalized titles, DOIs, arXiv IDs and ADS identities,
renders Markdown in memory, and compares it with the stored view.

Cross-platform equivalents are available without PowerShell:

```bash
python3 tools/literature/catalog_cli.py check
python3 tools/literature/catalog_cli.py render          # preview
python3 tools/literature/catalog_cli.py render --apply  # write
```

Non-empty `local_pdf_path` values must use `data://literature-catalog/` and
must be paired with a 64-hex SHA-256. Live candidates receive a stable citekey
only when they do not already have one; existing citekeys are never recomputed.

## Publication safety

`-CommitAndPush` is opt-in. It refuses every pre-existing staged path and every
changed path outside this exact allowlist:

- `Paper/catalog/papers.json`
- `Paper/catalog/papers.md`

The publisher uses explicit pathspecs; it does not use `git add -A` and does
not stage PDFs, documents, reports, logs, or tool files.

## Tests

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Invoke-Pester -Script .\tools\literature\tests\LiteratureCatalog.Tests.ps1 -EnableExit"
```

The tests cover path resolution, strict schema types and enums, duplicate
checks, scoring, metadata-aware merging, date-window filtering, source status,
deterministic rendering, read-only checking, the two-output contract, and the
Git publish allowlist. Write-path tests operate only on Pester's temporary test
drive.
