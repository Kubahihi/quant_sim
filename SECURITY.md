# Security policy

## Supported version

Only the most recently deployed commit with a passing CI release gate is
supported. Older commits should be treated as unsupported unless they are the
documented rollback candidate for an active incident.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting flow under **Security →
Advisories → Report a vulnerability**. Do not put credentials, exploit details,
private data, or an unpatched vulnerability in a public issue.

If private reporting is unavailable, contact the repository owner through an
already established private channel. Include the affected commit, impact,
reproduction steps, and suggested mitigation, but never include a live secret.

The target is to acknowledge a report within three business days and provide a
status update within seven business days. Disclosure should be coordinated
until a fix or mitigation is available.

## Exposed credentials

Treat a suspected credential disclosure as an incident:

1. revoke or rotate the credential immediately;
2. replace it in Streamlit Cloud or the relevant provider, then redeploy;
3. inspect Git history, CI artifacts, application logs and provider audit logs;
4. run the full secret scan and production preflight;
5. document the exposure window, affected systems and remediation.

Deleting a secret from the latest commit is not sufficient if it appeared in
Git history. Never add a broad Gitleaks exception for a real credential.
