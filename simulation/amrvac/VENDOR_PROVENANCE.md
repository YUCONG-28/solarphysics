# MPI-AMRVAC vendor snapshot

`amrvac/amrvac/` is a source snapshot used by the project build adapter. It is
copied into the ignored `Local/amrvac/` runtime tree before compilation and
must not be compiled in place.

- Upstream: <https://github.com/amrvac/amrvac.git>
- Upstream commit: `d4bc82d808bea5bffda0f54de3546c4aa460a2ac`
- License: GNU GPL version 3; see `amrvac/LICENSE`
- Snapshot status: historical local deployment, not the latest upstream release
- Outer-repository policy: ordinary vendor files, with no nested `.git`

The imported worktree contained the following local differences relative to
the upstream commit:

- deleted `lib/makefile`;
- modified `tests/Makefile`;
- modified `tests/mhd/solar_atmosphere_2.5D/mod_usr.t`;
- added root `makefile` and `mod_usr.t`;
- added `tests/makefile`;
- added two legacy test run scripts.

These differences are preserved in the vendored files so a clean checkout
matches the locally validated build input. The original nested Git metadata is
kept only in the ignored local runtime archive and is not part of this
repository.
