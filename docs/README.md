# Repository documentation

| Task | Entry point |
| --- | --- |
| Start the desktop application | [Root README](../README.md), [application guide](../Apps/README.md) |
| Use or test the scientific library | [Python guide](../Python/README.md) |
| Find canonical APIs and compatibility replacements | [Public function map](../Python/docs/FUNCTION_MAP.md) |
| Develop and verify application changes | [Apps development](../Apps/docs/development.md) |
| Inspect native capabilities and retained interfaces | [Capability matrix](../Apps/docs/app-v1-capability-matrix.md), [interface map](../Apps/docs/app-v1-equivalence-audit.md) |
| Run the STEREO EUVI example | [Example guide](../Python/examples/stereo/README.md) |
| Understand source and runtime ownership | [Architecture](../ARCHITECTURE.md) |
| Verify and publish a code change | [Development workflow](../WORKFLOW_README.md) |
| Maintain literature records | [Paper guide](../Paper/README.md), [catalog tools](../tools/literature/README.md) |
| Reproduce the environment | [Environment guide](../environment/README.md) |
| Follow the AIA/radio composite workflow | [Composite guide](aia_radio_composite.md) |

The library supports Python 3.10 and later. Apps and the default application
environment require Python 3.14. Environment locks belong to the named platform;
compatibility CI and a successful hash-verified fresh replay provide different
evidence. See the environment guide before claiming an exact environment.

The [App 1.0 integration plan](../Apps/docs/app-v1-integration-plan.md) records
the original development phases. Use the current guides above for supported
commands and interfaces. Execution logs, screenshots, and test receipts remain
in private runtime storage rather than in public documentation.

Keep application code in `Apps`, reusable code and examples in `Python`, static
literature evidence in `Paper`, and maintenance tools in `tools`. Configuration,
observations, notebook outputs, logs, and local literature stay outside tracked
source or under ignored `Local` paths.

File synchronization does not imply Git synchronization. When using a second
computer, compare file contents as well as branches; a synchronized directory
without `.git` has no independent commit history. Pause the affected file sync
while reorganizing source, then resume it and check for conflicts. Never copy
`.git` through a two-way file synchronization service.
