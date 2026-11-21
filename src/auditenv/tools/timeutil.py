"""ISO-8601 timestamp helpers (always with explicit UTC offsets)."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

_OFFSET_RE = re.compile(r"^([+-])(\d{2}):(\d{2})$")


def tz_from_offset(offset: str) -> timezone:
    m = _OFFSET_RE.match(offset)
    if not m:
        raise ValueError("bad offset")
    sign = 1 if m.group(1) == "+" else -1
    return timezone(sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3))))


def offset_minutes(offset: str) -> int:
    m = _OFFSET_RE.match(offset)
    if not m:
        raise ValueError("bad offset")
    sign = 1 if m.group(1) == "+" else -1
    return sign * (int(m.group(2)) * 60 + int(m.group(3)))


def parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        raise ValueError("timestamp without offset")
    return dt


def fmt_ts(dt: datetime, tz: timezone) -> str:
    return dt.astimezone(tz).isoformat(timespec="seconds")


def local_date(ts: str, offset: str) -> date:
    """Calendar date of a timestamp in the given fixed-offset timezone."""
    return parse_ts(ts).astimezone(tz_from_offset(offset)).date()
