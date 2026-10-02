# Documentation

| Task | Entry point |
| --- | --- |
| Select and compose scientific modules | [Library guide](../Python/README.md), [public API map](../Python/docs/FUNCTION_MAP.md) |
| Run deterministic API examples | [Public API examples](../Python/examples/public_api/README.md) |
| Read local EUVI FITS files | [STEREO example](../Python/examples/stereo/README.md) |
| Read, match and display synthetic radio data | [Radio example](../Python/examples/radio/README.md) |
| Plot a supplied SXR dataset | [SXR example](../Python/examples/sxr/README.md) |
| Run the application demonstration | [Application guide](../Apps/README.md) |
| Develop application adapters | [Apps development](../Apps/docs/development.md) |
| Inspect interface ownership | [Capability matrix](../Apps/docs/app-v1-capability-matrix.md), [interface map](../Apps/docs/app-v1-equivalence-audit.md) |
| Understand package and runtime boundaries | [Architecture](../ARCHITECTURE.md) |
| Verify and contribute code | [Development workflow](../WORKFLOW_README.md) |
| Recreate platform dependencies | [Environment guide](../environment/README.md) |
| Compose AIA/radio displays | [Composite guide](aia_radio_composite.md) |

The library supports Python 3.10 and later. Apps require Python 3.14.
An environment lock applies to its named platform; compatibility testing and
hash-verified dependency replay are separate checks.

Documentation describes interfaces, units, algorithm conditions and software
behavior. Observations, event parameters, research records and generated results
remain outside public source. A synchronized directory is not a substitute for
Git history; do not synchronize `.git` through a two-way file service.
