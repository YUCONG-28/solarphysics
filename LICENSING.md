# License scope

The root [MIT license](LICENSE) covers self-owned root files and self-owned
content in `tools/`, `docs/` and `environment/`. Component licenses and explicit
third-party notices take precedence for their respective files. The root
license does not relicense those components.

| Component | License and controlling notice |
| --- | --- |
| Self-owned root files, `tools/`, `docs/`, `environment/` | [MIT](LICENSE) |
| Scientific library, tests and examples in `Python/` | [MIT](Python/LICENSE) |
| General application source in `Apps/`, except the components below | [MIT](Apps/LICENSE) |
| App 1.0 source in `Apps/solar_apps/frontends/app_v1/` | [GPL-3.0-only scope](Apps/solar_apps/frontends/app_v1/LICENSE.md), [full license](Apps/LICENSES/GPL-3.0-only.txt) |
| Bundled Mediabunny media asset | [MPL-2.0 and asset notice](Apps/solar_apps/ui/media/NOTICE.txt) |

Preserve existing copyright, license and attribution notices when copying or
redistributing files. Historical copyright notices remain in place even where
current project and citation metadata identify Y. Li, Shandong University.
Dependency packages, including PyQt6, retain their publishers' licenses; the
repository licenses do not replace dependency terms. Environment locks record
artifacts and do not grant rights to their upstream code.

The Python distribution declares MIT. The Apps distribution declares its
combined MIT, GPL-3.0-only and MPL-2.0 contents in its package metadata. Review
the relevant component notices for the files and distributions you use.
