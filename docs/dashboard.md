# Dashboard

The public site and `/api/v1/public/*` endpoints remain public. The dashboard
has its own navigation and these pages:

- `/dashboard/login/`: username/password sign-in; no public registration.
- `/dashboard/observations/`: stored `measurement` rows, filterable by UTC date
  range and metric, with stable time sorting and server pagination.
- `/dashboard/observations/normalized/`: wide hourly `normalized_observation`
  rows, with date filters, sorting and pagination. These tables do not trigger
  collection or normalization.
- `/dashboard/users/`: create accounts, assign existing groups, block/reactivate
  users and reset passwords.
- `/dashboard/api-stats/`: aggregate statistics for all API routes.

## Setup

Apply database migrations before starting the updated API:

```sh
pnpm db:migrate
pnpm app:user create admin --group admins
```

The command prompts twice for a password (12–256 characters). Passwords are not
passed on the command line. Usernames are case-insensitive; supported characters
are letters, digits, `_`, `.`, `@`, and `-`. To reset a password:

```sh
pnpm app:user reset-password admin
```

Set API environment variables in `.env.local`, then restart the API:

```dotenv
# Production: exact frontend origin, no path. Multiple origins are comma-separated.
DASHBOARD_ORIGINS=https://argussun.com
DASHBOARD_COOKIE_SECURE=true
```

For local development over HTTP:

```dotenv
DASHBOARD_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
DASHBOARD_COOKIE_SECURE=false
```

Secure cookies default to enabled. Outside production Compose, allowed origins
default to the two local origins above. Production Compose defaults to
`https://${APP_NAME}`; set `DASHBOARD_ORIGINS` explicitly for additional origins.
After changing production environment files, recreate the API container (a
restart alone does not reload its environment):

```sh
cd /var/www
docker compose --env-file .env --env-file .env.local up -d --no-deps --force-recreate api
```

The frontend uses
the existing same-origin `/api/v1` proxy. No frontend secrets are required.

For a deployed installation, run in the API container after applying migrations:

```sh
docker compose exec api .venv/bin/python -m app.commands.manage_user create admin --group admins
```

## Access and sessions

The migration creates the `admins` group with `observations.read`, `users.manage`
and `api_stats.read` permissions. Users can belong to multiple groups. An active
user with no groups may sign in but cannot access protected data. Every dashboard
API endpoint checks the relevant permission on the server. Public endpoints and
the older contract-token mechanism are separate.

Opaque random sessions last 12 hours. Only their SHA-256 digests are stored in the
database. Cookies use HttpOnly, SameSite=Strict, Path=/ and Secure in production.
State-changing requests require a configured Origin, including login and logout.
Passwords use salted scrypt (N=32768, r=8, p=3), following an
[OWASP recommended configuration](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html#scrypt).
Session handling follows the [OWASP session guidance](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html).

Blocking users, changing groups or resetting passwords revokes their sessions.
Administrators cannot block themselves or remove their own admin membership;
updates also preserve an active administrator. The UI refreshes the session on
navigation, window focus and every minute; protected API reads always check it.
Dashboard API responses disable caching.

Login attempts are limited across API workers using database counters: ten
attempts per username and 100 per peer address in each 15-minute window, including
successful attempts. Behind a proxy, the peer limit can cover multiple users.
Counters contain hashed keys, not stored IP addresses, and old windows are
removed on subsequent login attempts. Expired sessions are removed on login.

## API statistics

Each worker buffers counters for up to 15 seconds, then merges them into
PostgreSQL hourly buckets. The buffer has a fixed key-count limit. Statistics
store route templates, HTTP method, status code, request count, total elapsed
milliseconds and latency histogram buckets. They do not store request/response
bodies, query strings, usernames, cookies or tokens. Unknown routes use one
`__unmatched__` label to bound cardinality. Dashboard/statistics requests count too.

The UI shows counts, 4xx/5xx totals, average duration, estimated p95 upper bounds,
status distribution and hourly activity for 24 hours, seven days or 30 days.
Periods include the current UTC hour. Durations measure application response
processing, excluding network transfer and streamed response bodies. The final
histogram bucket represents durations over 60 seconds. This is approximate
operational telemetry, not billing or an exact request audit.

Buckets older than 30 days are deleted by the flush loop. Graceful shutdown
attempts a final flush; crashes, database failures and a full buffer can lose
metrics. A failed metrics write is logged and does not fail public API requests.

## Verification

```sh
./scripts/test-python -q
pnpm --filter web check-types
pnpm --filter web build
```

The PostgreSQL integration check creates a disposable schema and exercises the
migration in both directions, login, cookie flags, Origin rejection, groups,
blocking, password resets, expiry, pagination, the public observation endpoint,
rate limits and persisted statistics. In this repository's split environments,
run with API database dependencies and the root test dependencies:

```sh
PYTHONPATH=apps/api:.venv/lib/python3.12/site-packages apps/api/.venv/bin/python apps/api/tests/integration/verify_dashboard.py
```

## Dashboard UI

The dashboard uses shadcn/ui's New York components, adapted from the official
[`dashboard-01` block](https://ui.shadcn.com/blocks), with Lucide icons and the
existing ECharts library. Component source and its MIT license are in
`apps/web/components/ui`. Configuration lives in `apps/web/components.json`.

The navigation is grouped into Analytics, Observations and Administration.
Add future risk panels as normal Next.js pages and register their required
permission in `app/dashboard/navigation.ts`; enforce that permission in the API
as well. Shared page headings, metric cards, loading/empty states and pagination
are in `app/dashboard/_components/presentation.tsx`. `ApiActivity` and
`TrafficChart` demonstrate reusable analytic panels without imposing a CRUD
framework or changing the backend API.

Tailwind 3 utilities are scoped to `.dashboard` and preflight is disabled to
preserve the public site's CSS. Dashboard theme variables and a local reset live
in `app/dashboard/dashboard.css`. Radix dialogs, tooltips and menus render inside
`#dashboard-portals` to retain scoped styles. Preserve this portal container when
adding components. Restart an existing development server after installing the
new PostCSS/Tailwind configuration.

The sidebar supports icon collapse, a mobile drawer and a persisted layout
preference. Forms use labeled inputs and dialogs; the existing session and
permission checks remain active. The overview and statistics page use real API
aggregates, with loading, empty and error states rather than demonstration data.

Browser checks mock the API so they never create real users or modify stored
observations. They exercise sign-in/out, permissions, filtering, pagination,
user editing, responsive navigation, chart rendering and public-style isolation:

```sh
pnpm --filter web exec playwright install chromium
pnpm --filter web test:dashboard
pnpm --filter web lint:dashboard
```

The tests reuse a development server on port 3000 or start one if none is running.
An existing Chromium executable can be selected with
`PLAYWRIGHT_CHROMIUM_EXECUTABLE`. API authorization is independently covered by
the Python integration tests described above.
