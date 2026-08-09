## What changed

Describe the user-visible outcome and the reason for the change.

## Release checklist

- [ ] The change is focused and has automated tests where practical.
- [ ] CI passes, including the security and release-quality gates.
- [ ] No credentials, tokens, private keys, or production data are included.
- [ ] Dependency manifests and the lock/cutoff evidence were updated together, if dependencies changed.
- [ ] Configuration and database changes remain backward compatible, or the migration and rollback path is documented.
- [ ] Production-impacting changes were checked with preflight and restore-drill procedures where applicable.
- [ ] The release manifest, SBOM, license inventory, and rollback instructions remain accurate.

## Risk and rollback

State the main failure mode and the exact rollback action. Write `Not applicable` only for changes with no runtime impact.
