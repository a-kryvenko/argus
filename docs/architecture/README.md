# Architecture

Argus separates observation ownership (Clio), forecast publication (Prophet),
impact assessment (Intelligence) and public access (API and Web). The
[architecture reference](../architecture.md) describes package boundaries,
database ownership and runtime constraints in detail.

## C4 model

[workspace.dsl](workspace.dsl) is the source of truth for runtime elements and
relationships. Structurizr generates two views from that model:

- [System context](generated/c4/context.md): users, API clients and providers.
- [Runtime containers](generated/c4/containers.md): HTTP and worker processes,
  owned databases, original-file storage and model artifacts.

Arrows represent requests or storage access, not response data. Libraries such as
`forecast-core` and `intelligence-core` execute inside their consuming processes;
they are not network services. Four logical databases share one PostgreSQL
instance. The container view focuses on domain services, not a complete deployment
inventory. Intelligence HTTP computes drag assessments without its worker database;
its worker currently records release integration stub results.

## Data flows

Flows are authored directly as Mermaid in Markdown. Arrows show data or processing
progression. Collection, generation, verification and training have separate
triggers. Public reads do not trigger collection, training or forecast generation.

- [Observation collection and preparation](flows/observations.md)
- [Independent product publication](flows/forecast-release.md)
- [Operational verification](flows/verification.md)
- [Training and artifact delivery](flows/training.md), covering the public
  AIA/Ridge pipeline; private training details remain in private repositories.

## Build and check

From the repository root:

```bash
pnpm docs:architecture
pnpm docs:architecture:check
```

Requirements: Python 3.10+, Java 21+ and curl for the initial download.
Without Node, run `python3 scripts/build-architecture` (add `--check` for
verification). No application databases, model artifacts or private repositories
are needed. The script was verified on Linux; it uses no platform-specific binary.

The first build downloads Structurizr **2026.09.19** into the ignored
`vendor/architecture-tools/` directory. Its version, URL and SHA256 are pinned in
[`scripts/architecture-tools.json`](../../scripts/architecture-tools.json).
Subsequent builds use that cached distribution and verify its hash again.

The build uses Structurizr's native Mermaid exporter and wraps the output in
Markdown under `generated/c4/`. Exported HTML labels are simplified to text with
line breaks so the diagrams do not require Mermaid's loose security mode.
GitHub or a Mermaid-capable Markdown preview renders all diagrams. Flow pages
are already readable source documents and need no generation step.

`--check` regenerates C4 pages in a temporary directory and compares bytes and
filenames, including missing and obsolete files, without modifying `generated/`.
The [diagram workflow](../../.github/workflows/architecture-docs.yml) runs the
same check on relevant pull requests and pushes to master. It checks generated
C4 consistency; it does not validate handwritten Mermaid syntax or architectural
accuracy.

## Editing rules

- Edit `workspace.dsl` for runtime elements and relationships. Keep view keys
  stable because they determine generated filenames.
- Edit `flows/*.md` for processing steps, data movement and failure paths.
  Link new flow pages from this overview.
- Preserve the distinction between request direction in C4 views and data
  direction in flows.
- Update diagrams with changes to service contracts, storage ownership or
  publication paths. Commit model changes and regenerated C4 pages together.
- Do not edit `generated/` by hand. When upgrading Structurizr, update its pinned
  version and hash and regenerate the C4 pages in the same change.

Tool reference: [Structurizr Mermaid export](https://docs.structurizr.com/export/mermaid).
