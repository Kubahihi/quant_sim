# GitHub repository ruleset

The repository contains the controls needed for protected reviews, but GitHub must enforce them in the repository settings. Create a branch ruleset named `Production branches` targeting the default branch (`main`) with the following configuration.

## Required rules

- Restrict deletions and block force pushes.
- Require a pull request before merging.
- Require at least one approval and dismiss stale approvals when new commits are pushed.
- Require review from Code Owners.
- Require all review conversations to be resolved.
- Require the branch to be up to date before merging.
- Require these status checks:
  - `CI / Security gate`
  - `CI / Release quality gate`
- Require linear history.
- Allow squash merge; disable merge commits for the protected branch.

Do not grant a standing bypass to application credentials or normal contributors. If an owner must use an emergency bypass, record the reason and follow it immediately with a reviewed pull request that restores the protected state.

## Verification

After enabling the ruleset, open a draft pull request that intentionally fails one test. Confirm that GitHub blocks merging until both required checks pass, an owner approves, and every conversation is resolved. Then dismiss the approval with a new commit and confirm that a fresh approval is required.

Repository files cannot prove that the server-side ruleset is enabled. Keep a screenshot or exported ruleset with the release evidence and recheck the configuration before each production release.
