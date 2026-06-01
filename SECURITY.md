# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

## Reporting a Vulnerability

We take the security of Distributed LLM seriously. If you believe you have found a security vulnerability, please report it to us as described below.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to **security@distributed-llm.ai**.

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

To help us understand the issue, please include:

* Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit it

## Preferred Languages

We prefer all communications to be in English.

## Disclosure Policy

When we receive a security bug report, we will:

1. **Acknowledge** receipt within 48 hours
2. **Triage** the vulnerability within 5 business days (confirm, assess severity)
3. **Develop** a fix (timeline depends on severity):
   - Critical: 7 days
   - High: 14 days
   - Medium: 30 days
   - Low: 60 days
4. **Release** the fix and notify the reporter
5. **Credit** the reporter in the security advisory (unless they prefer anonymity)

We will coordinate with the reporter throughout this process and will not disclose the vulnerability publicly until a fix has been released.

## Bug Bounty

We currently do not offer a formal bug bounty program, but we deeply appreciate security researchers who responsibly disclose vulnerabilities. We will:

- Credit you in the security advisory
- Mention you in the release notes
- Send you DistLLM swag (stickers, t-shirt)

## Security Hall of Fame

We maintain a hall of fame for security researchers who have helped improve DistLLM's security. If you report a valid vulnerability, you'll be added to this list (with your permission).

## Scope

The following are in scope for security reports:

- Authentication/authorization bypasses
- Remote code execution
- SQL injection, command injection, SSRF
- Cross-site scripting (XSS)
- Information disclosure (API keys, internal paths)
- Denial of service (resource exhaustion)
- Cryptographic weaknesses
- Supply chain vulnerabilities

The following are out of scope:

- Social engineering
- Physical attacks
- Denial of service via volumetric attacks
- Issues in third-party dependencies (report to them directly)

## Comments on this Policy

If you have suggestions on how this process could be improved, please submit a pull request or contact us at security@distributed-llm.ai.
