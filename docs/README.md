# Repository documentation

| Task | Entry point |
| --- | --- |
| Start the desktop application | [Root README](../README.md), [application guide](../Apps/README.md) |
| Use or test the scientific library | [Python guide](../Python/README.md) |
| Run the STEREO EUVI example | [Example guide](../Python/examples/stereo/README.md) |
| Understand source and runtime ownership | [Architecture](../ARCHITECTURE.md) |
| Verify and publish a code change | [Development workflow](../WORKFLOW_README.md) |
| Maintain literature records | [Paper guide](../Paper/README.md), [catalog tools](../tools/literature/README.md) |
| Reproduce the environment | [Environment guide](../environment/README.md) |
| Follow the AIA/radio composite workflow | [Composite guide](aia_radio_composite.md) |

Keep application code in `Apps`, reusable code and examples in `Python`, static
literature evidence in `Paper`, and maintenance tools in `tools`. Configuration,
observations, notebook outputs, logs, and local literature stay outside tracked
source or under ignored `Local` paths.

File synchronization does not imply Git synchronization. When using a second
computer, compare file contents as well as branches; a synchronized directory
without `.git` has no independent commit history. Pause the affected file sync
while reorganizing source, then resume it and check for conflicts. Never copy
`.git` through a two-way file synchronization service.
