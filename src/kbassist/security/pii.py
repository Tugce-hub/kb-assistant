"""Secret and PII redaction.

Applied at three points: before indexing (nothing sensitive is embedded or
stored), on the user's question before it is written to the audit log, and on
the model's answer before it is posted to Slack.

Validators (Luhn, TCKN checksum, IBAN mod-97) keep false positives low so that
ordinary numbers in docs/code are not mangled.
"""

from __future__ import annotations

import re
from typing import Callable

# --- validators -------------------------------------------------------------


def _luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _tckn_ok(s: str) -> bool:
    """Turkish national ID (T.C. Kimlik No) checksum."""
    if len(s) != 11 or not s.isdigit() or s[0] == "0":
        return False
    d = [int(c) for c in s]
    d10 = ((sum(d[0:9:2]) * 7) - sum(d[1:8:2])) % 10
    d11 = sum(d[:10]) % 10
    return d[9] == d10 and d[10] == d11


def _iban_ok(s: str) -> bool:
    s = s.replace(" ", "").upper()
    if len(s) < 15:
        return False
    rearranged = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in rearranged)
    return int(num) % 97 == 1


# --- patterns ---------------------------------------------------------------

_Rule = tuple[str, re.Pattern[str], Callable[[str], bool] | None]

RULES: list[_Rule] = [
    # Secrets first, so that e.g. a token containing digits is not half-matched as a card.
    ("AWS_ACCESS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), None),
    ("GITHUB_TOKEN", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b"), None),
    ("SLACK_TOKEN", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), None),
    ("ANTHROPIC_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"), None),
    ("OPENAI_STYLE_KEY", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}\b"), None),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"), None),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), None),
    # key=value style secrets: keep the key name, drop the value.
    (
        "SECRET_ASSIGNMENT",
        re.compile(r"(?i)\b((?:api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"])([^'\"\s]{8,})(['\"])"),
        None,
    ),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?\b"), _iban_ok),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), _luhn_ok),
    ("TCKN", re.compile(r"\b[1-9]\d{10}\b"), _tckn_ok),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    ("PHONE_TR", re.compile(r"(?<!\d)(?:\+90|0)\s?5\d{2}\s?\d{3}\s?\d{2}\s?\d{2}(?!\d)"), None),
]

# Addresses that appear in docs as examples and are safe to keep.
EMAIL_ALLOWLIST = re.compile(r"@(?:example\.(?:com|org|net)|users\.noreply\.github\.com)$", re.I)


def redact(text: str) -> tuple[str, int]:
    """Return (redacted_text, number_of_redactions)."""
    count = 0

    for label, pattern, validator in RULES:

        def sub(m: re.Match[str], label: str = label, validator=validator) -> str:
            nonlocal count
            value = m.group(0)
            if label == "SECRET_ASSIGNMENT":
                count += 1
                return f"{m.group(1)}[REDACTED:SECRET]{m.group(3)}"
            if label == "EMAIL" and EMAIL_ALLOWLIST.search(value):
                return value
            if validator is not None and not validator(value.replace(" ", "").replace("-", "")):
                return value
            count += 1
            return f"[REDACTED:{label}]"

        text = pattern.sub(sub, text)
    return text, count


def redact_text(text: str) -> str:
    return redact(text)[0]
