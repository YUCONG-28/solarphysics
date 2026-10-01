# Apps architecture

`solar_apps` is an application layer over the reusable `solar_toolkit` library.
It is organized by responsibility rather than by launch script:

```text
solar_apps/
|-- cli/          command routing and compatibility aliases
|-- platform/     runtime layout, configuration, state, paths, processes
|-- ui/           theme and Web, Streamlit, Qt, and media adapters
|-- frontends/    App 1.0 and retained compatibility interfaces
`-- workflows/    AIA, HMI, radio, visualization, data, net, and X-ray flows
```

Frontend modules translate user intent into workflow calls. Workflow modules
coordinate scientific operations but leave reusable calculations in
`solar_toolkit`. Platform services are domain-neutral. The UI layer shares one
semantic design system without changing scientific normalization or exports.

Runtime files are resolved through `RuntimeLayout` and live below the ignored
repository-level `Local/` directory. Production code must not inject paths into
`sys.path` or infer repository roots with fixed parent indexes.

Private configuration layout version 2 is declared by
`apps.runtime_layout_version`; migrations retain only explicitly supported
settings. This version is independent of the per-frontend `StateStore` schema.

See the repository-level [architecture](../../ARCHITECTURE.md) for dependency
and privacy rules.
