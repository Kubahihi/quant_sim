# Production readiness

## Deterministic build

The supported runtime is Python 3.12.13. Production dependencies are frozen in
`requirements.txt`, which is also the manifest installed by Streamlit Cloud.
CI and development tools are frozen in `requirements-dev.lock`.

```powershell
uv venv --python 3.12.13
uv pip sync requirements-dev.lock
.venv\Scripts\python.exe -m pytest -q -W error --cov=src --cov=ui --cov-fail-under=57
```

Change direct production constraints only in `requirements.in`, then regenerate
both locked manifests using the committed package-index cutoff:

```powershell
$cutoff = Get-Content .dependency-cutoff
uv pip compile requirements.in --python-version 3.12 --universal --exclude-newer $cutoff --output-file requirements.txt
uv pip compile requirements.in requirements-dev.txt --python-version 3.12 --universal --exclude-newer $cutoff --output-file requirements-dev.lock
```

Advance `.dependency-cutoff` to the current UTC timestamp only during an
intentional dependency refresh, then audit and test the newly resolved graph.
This prevents an unrelated release from changing lock regeneration after the
fact. Commit `requirements.in`, `.dependency-cutoff`, and both regenerated
manifests together. Select Python 3.12 explicitly in Streamlit Cloud's Advanced
settings before deployment.

## Production fail-closed requirements

Set `QUANT_SIM_ENV=production` either as an operating-system variable or as a
top-level Streamlit secret. The operating-system value takes precedence. In
this mode the application refuses unsafe fallbacks instead of silently
continuing with ephemeral storage. An ambiguous Streamlit server without an
explicit environment is treated as production by persistent storage layers.

Required shared database settings:

- `TURSO_DATABASE_URL`
- `TURSO_AUTH_TOKEN`

Required Streamlit `[storage]` settings:

- `STORAGE_BACKEND = "r2"`
- `R2_BUCKET`
- `R2_ENDPOINT_URL`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`

If the standalone Flask API is deployed, production also requires:

- authentication enabled;
- rate limiting enabled;
- debug mode disabled;
- no default user identity;
- no wildcard CORS origin. Set `API_CORS_ORIGINS` to explicit HTTPS origins,
  or keep CORS disabled.

## CI release gate

The GitHub Actions workflow performs:

1. full-history secret scanning with Gitleaks;
2. pull-request dependency review, blocking vulnerabilities from low severity;
3. installation of the pinned Python runtime and CI lock file;
4. compatibility and known-vulnerability audit of the installed environment;
5. exact-pin and production/development deployment parity validation;
6. regeneration comparison for both locked manifests;
7. bytecode compilation;
8. blocking Ruff correctness checks;
9. startup of a real headless Streamlit server and its HTTP health probe;
10. the complete warning-free test suite with a 57% coverage floor;
11. creation and immediate verification of a commit-bound release manifest.

The secret scanner uses only fingerprint-specific exceptions in
`.gitleaksignore`; broad rule or path exclusions are forbidden. The dependency
audit uses the exactly synchronized environment and fails on any known advisory
or incomplete package collection. `pip-audit` is itself pinned through
`requirements-dev.lock`, while Dependabot continues to propose weekly Python
and GitHub Actions updates.

For successful pushes to `main` and manual workflow runs, CI retains the
verified `quant-sim-release-<commit>` manifest for 90 days. It records the full
commit SHA, exact Python version, complete deployable-file inventory, individual
SHA-256 hashes and a canonical source-tree hash. Creation fails if the checkout
is dirty or if the build interpreter differs from `.python-version`. Pull
requests verify the same mechanism but do not retain an artifact.

CI also creates a GitHub artifact attestation for the retained manifest. This
cryptographically binds the downloaded file to the repository, workflow and
commit that produced it; the manifest's internal hashes alone are not treated
as trusted provenance.

Dependabot checks both Python packages and GitHub Actions weekly. A dependency
update is not release-ready until the regenerated lock files and CI checks pass.

Streamlit test auto-login is disabled unless all three conditions hold:
`QUANT_SIM_ENV=test`, `QUANT_SIM_TEST_AUTO_LOGIN=1`, and an active pytest test
context. A generic pytest environment marker alone cannot bypass authentication.

The API returns no-store, anti-sniffing, anti-framing and referrer-policy
headers. Production responses also include HSTS. Production request logs carry
only the request id, method, path, status and duration; request bodies,
credentials and authentication tokens are never logged.

Use `GET /api/v1/health` as a liveness probe and `GET /api/v1/ready` as a
readiness probe. Readiness verifies both the authentication database and the
configured storage backend, returns HTTP 503 when either is unavailable, and
never exposes exception details. Results are cached for 30 seconds so R2's
write/read/delete validation is not repeated on every monitoring request.

## Production preflight and restore drill

Run the configuration-only preflight before deployment:

```powershell
.venv\Scripts\python.exe scripts\production_preflight.py --config-only
```

After the deployed application has initialized its database schema, run the
live preflight. This verifies the Turso query/schema and performs R2's temporary
write/read/delete health cycle:

```powershell
.venv\Scripts\python.exe scripts\production_preflight.py
```

Export a standalone SQLite-compatible Turso backup and prove it can be restored
without modifying the source backup:

```powershell
.venv\Scripts\python.exe scripts\production_preflight.py `
  --restore-only --restore-drill C:\backups\quant-sim.db
```

The restore drill copies the backup into a temporary directory, opens the copy
read-only, runs SQLite integrity validation, and verifies the core auth and
Wharton tables. Output contains only stable status/reason codes, never secret
values or raw exception messages.

## Rollback procedure

Before every deployment, record the successful CI run, retain its release
manifest, and retain evidence of the live preflight and restore drill. The
rollback candidate is the most recent previously deployed commit for which all
three records exist.

1. Select that known-good full commit SHA; do not use a mutable branch name.
2. Check out the commit and obtain its matching
   `quant-sim-release-<commit>` artifact from the successful CI run.
3. Verify the downloaded artifact's GitHub provenance, substituting the actual
   repository owner and name:

   ```powershell
   gh attestation verify C:\release\release-manifest.json `
     --repo <owner>/<repository>
   ```

4. Verify the candidate's contents and exact commit before deployment:

   ```powershell
   .venv\Scripts\python.exe scripts\release_manifest.py verify `
     --manifest C:\release\release-manifest.json `
     --expected-commit <full-commit-sha>
   ```

5. Deploy exactly that commit with the existing production secrets. Do not
   rebuild or regenerate its dependency locks during rollback.
6. Require `/api/v1/health`, `/api/v1/ready`, and the live production preflight
   to pass, then verify a real authenticated session, WInS reconciliation and
   report export.
7. Record the incident, selected commit, manifest tree hash, deployment result
   and operator.

A code rollback does not roll back production data. Never restore the database
automatically merely because the application was rolled back. Database restore
requires a separate incident decision, a known-good exported backup, a passing
isolated restore drill and an explicit assessment of data loss and schema
compatibility. Prefer forward-compatible schema changes so the previous
application release can operate against the current database.

## Manual production checks still required

Before a release, run the live preflight and restore drill, then verify a real
authenticated Streamlit session, WInS reconciliation, report export and the
documented rollback procedure. A backup is not considered valid until the
isolated restore drill succeeds.
