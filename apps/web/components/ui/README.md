# shadcn/ui components

Source: https://ui.shadcn.com/r/styles/new-york/ (MIT; see LICENSE.md).

Local adaptations:

- Imports resolve through the web app's `@/` alias.
- Radix portals use `dashboardPortal()` so the scoped dashboard theme applies.
- The sidebar skeleton has a deterministic width for React rendering purity.
- `useIsMobile` uses `useSyncExternalStore` for a stable server snapshot.

Tailwind 3 utilities and tokens are scoped to `.dashboard`. These components
currently belong to the dashboard; using them on public pages requires an
explicit styling decision. Keep the scope and portal handling when updating
components with the shadcn CLI.
