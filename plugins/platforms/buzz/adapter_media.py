"""Relay media and attachment helpers shared by the Buzz platform adapter.

Buzz-hosted media is private to the community: inbound text keeps same-relay media URLs out of the agent
context until they are authenticated and localised, and outbound attachment names are normalised before they
reach the cache or the agent context. These helpers are pure functions over their arguments, so they live in
their own topical sibling module — ``adapter.py`` is already over its code-health line cap and may only shrink.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

# Inbound attachments download only after the gates pass and must match their declared NIP-94 size + SHA-256.
_MAX_ATTACHMENT_FILENAME_BYTES = 120
# Buzz-hosted media is private to the community: same-relay URLs must be authenticated + localised for vision.
_MEDIA_URL_PATTERN = r"https?://[^\s<>\[\]()]+/media/[0-9a-f]{64}(?:\.[a-z0-9]{1,10})?(?:\?[^\s<>\[\]()]*)?"
_MARKDOWN_MEDIA_RE = re.compile(
    rf"!\[(?P<alt>[^\]]*)\]\(\s*(?P<url>{_MEDIA_URL_PATTERN})(?:\s+[\"'][^\"']*[\"'])?\s*\)", re.IGNORECASE
)
_BARE_MEDIA_RE = re.compile(_MEDIA_URL_PATTERN, re.IGNORECASE)
_MEDIA_PATH_RE = re.compile(r"^/media/(?P<sha>[0-9a-f]{64})(?:\.[a-z0-9]{1,10})?/?$", re.IGNORECASE)


def _safe_attachment_filename(value: str) -> str:
    """Return a basename that is safe for cache files and agent context."""
    name = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(c for c in name if ord(c) >= 32 and c != "\x7f").strip()
    if name in {"", ".", ".."}:
        return "attachment.bin"
    suffix = Path(name).suffix if len(Path(name).suffix.encode("utf-8")) <= 20 else ""
    stem = name[:-len(suffix)] if suffix else name
    byte_budget = _MAX_ATTACHMENT_FILENAME_BYTES - len(suffix.encode("utf-8"))
    safe_stem = stem.encode("utf-8")[:byte_budget].decode("utf-8", errors="ignore").rstrip(" .")
    return f"{safe_stem or 'attachment'}{suffix}"


def _attachment_origin(value: str) -> Optional[tuple[str, int]]:
    """Normalize a configured host/URL to an exact HTTPS-equivalent origin."""
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = urlsplit(raw if "://" in raw else f"//{raw}")
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port or 443
    except ValueError:
        return None
    if (parsed.scheme and parsed.scheme not in {"https", "wss"}) or not host:
        return None
    return host, port


def _effective_port(parsed) -> Optional[int]:
    try:
        if parsed.port is not None:
            return parsed.port
    except ValueError:
        return None
    return {"https": 443, "wss": 443, "http": 80, "ws": 80}.get(parsed.scheme)


def _is_relay_media_url(url: str, relay_url: str) -> bool:
    """Return whether *url* is a Buzz media object on the configured relay."""
    candidate, relay = urlsplit(url), urlsplit(relay_url)
    return bool(
        candidate.scheme in ("http", "https") and candidate.hostname and relay.hostname
        and candidate.hostname.lower() == relay.hostname.lower()
        and _effective_port(candidate) == _effective_port(relay) and _MEDIA_PATH_RE.fullmatch(candidate.path)
    )


def _find_relay_media_refs(text: str, relay_url: str) -> tuple[list[str], list[tuple[int, int, str]]]:
    """Find same-relay media URLs and their safe text replacements."""
    urls: list[str] = []
    replacements: list[tuple[int, int, str]] = []
    markdown_spans: list[tuple[int, int]] = []
    for match in _MARKDOWN_MEDIA_RE.finditer(text):
        url = match.group("url")
        if not _is_relay_media_url(url, relay_url):
            continue
        markdown_spans.append(match.span())
        replacements.append((*match.span(), match.group("alt").strip()))
        if url not in urls:
            urls.append(url)
    for match in _BARE_MEDIA_RE.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in markdown_spans):
            continue
        url = match.group(0)
        if not _is_relay_media_url(url, relay_url):
            continue
        replacements.append((*match.span(), ""))
        if url not in urls:
            urls.append(url)
    return urls, replacements


def _replace_media_refs(text: str, replacements: list[tuple[int, int, str]]) -> str:
    for start, end, replacement in sorted(replacements, reverse=True):
        text = f"{text[:start]}{replacement}{text[end:]}"
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", text)).strip()
