# Architecture diagrams

[Service boundaries](../architecture.md).

## Sources

- [workspace.dsl](workspace.dsl): runtime elements and request/storage relationships.
- [Context](generated/c4/context.md) · [Containers](generated/c4/containers.md): generated C4 views.
- Data movement: [observations](flows/observations.md), [forecasts](flows/forecast-release.md), [verification](flows/verification.md), [training](flows/training.md).

Private backends execute inside their consumers. Four logical databases share
one PostgreSQL instance; the diagrams omit auxiliary deployment containers.

## Build and check

```bash
pnpm docs:architecture
pnpm docs:architecture:check
```

Requires Python 3.10+, Java 21+ and curl. Without Node, use
`python3 scripts/build-architecture [--check]`.
Structurizr is cached in `vendor/architecture-tools/`; version and SHA256 are
pinned in [architecture-tools.json](../../scripts/architecture-tools.json).

Edit DSL/flow sources, then regenerate C4 pages. Keep view keys stable; never
edit `generated/` by hand. Commit sources and generated pages together.
`--check` compares regenerated files; it does not validate architectural accuracy
or handwritten Mermaid syntax.
