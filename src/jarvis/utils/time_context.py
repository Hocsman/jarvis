"""Format the current time for injection into the LLM system context.

Prefers the user's local timezone (derived from location) so the assistant can
answer "what time is it?" in the form the user expects, instead of UTC.
"""

from datetime import datetime, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover - Python < 3.9
    ZoneInfo = None  # type: ignore[assignment]
    ZoneInfoNotFoundError = Exception  # type: ignore[assignment,misc]


# Day and month names written out rather than taken from strftime.
#
# `%A` and `%B` follow the machine's C locale, so on a French Mac the
# English template below produced "samedi, août 01, 2026 at 22:10" —
# French words inside an English sentence, silently, in a prompt. These
# are not a language pattern: they are an output format, like %Y-%m-%d.
_WEEKDAYS = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday",
)
_MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)


def _spell(moment: datetime) -> str:
    """`Saturday, August 01, 2026 at 22:10`, whatever the locale says."""
    return (
        f"{_WEEKDAYS[moment.weekday()]}, {_MONTHS[moment.month - 1]} "
        f"{moment.day:02d}, {moment.year} at {moment.hour:02d}:{moment.minute:02d}"
    )


def format_time_context(
    tz_name: Optional[str] = None,
    *,
    now_utc: Optional[datetime] = None,
) -> str:
    """Return a human-readable string describing the current time.

    Resolution order:
    1. ``tz_name`` via ``zoneinfo`` (when GeoIP exposes an IANA zone).
    2. The OS local timezone (``datetime.astimezone()``).
    3. UTC, as a last resort when neither zone can be named.
    """
    now = now_utc if now_utc is not None else datetime.now(timezone.utc)

    if tz_name and ZoneInfo is not None:
        try:
            local = now.astimezone(ZoneInfo(tz_name))
            if local.tzname():
                return f"{_spell(local)} {local.tzname()}"
        except (ZoneInfoNotFoundError, KeyError, ValueError):
            pass

    try:
        system_local = now.astimezone()
        if system_local.tzname():
            return f"{_spell(system_local)} {system_local.tzname()}"
    except (ValueError, OSError):
        pass

    return f"{_spell(now)} UTC"


def format_reference_moment(now_local: datetime) -> str:
    """The anchor an extractor resolves a spoken time against.

    Carries the weekday, because without it "jeudi" cannot be resolved at
    all — and with it, resolving it is one step.

    Carries no timezone, neither name nor abbreviation. `%Z` gives `IST`
    for three different zones and no numeric offset, so a model asked to
    convert from it would be guessing. The code converts; the model reads
    a wall clock and hands one back.
    """
    return (
        f"Reference moment: {_spell(now_local)}\n"
        f"Reference moment, machine-readable: "
        f"{now_local.strftime('%Y-%m-%dT%H:%M')}"
    )


def _read_system_timezone() -> str:
    """The machine's IANA zone name, or empty.

    Deliberately the machine rather than GeoIP, which is what the reply
    engine prefers for its own display string. There, being wrong is
    cosmetic. Here it decides when a reminder fires, and GeoIP reports
    the traffic's exit node — wrong under a VPN, and cached for as long
    as the location cache holds.
    """
    import os
    from pathlib import Path

    try:
        link = Path("/etc/localtime")
        if link.is_symlink():
            parts = Path(os.readlink(link)).parts
            if "zoneinfo" in parts:
                return "/".join(parts[parts.index("zoneinfo") + 1:])
    except Exception:
        pass

    env = (os.environ.get("TZ") or "").strip()
    if env and ZoneInfo is not None:
        try:
            ZoneInfo(env)
            return env
        except Exception:
            pass
    return ""


def local_timezone_name() -> str:
    """The zone a reminder is resolved and fired in, or empty.

    Empty is honest rather than a guess: the instant is then computed at
    the current local offset, which is naive about a daylight-saving
    change far enough ahead.
    """
    name = _read_system_timezone()
    if name:
        return name

    try:
        from .location import get_location_context_with_timezone

        _, tz_name = get_location_context_with_timezone()
        if tz_name and ZoneInfo is not None:
            ZoneInfo(tz_name)
            return tz_name
    except Exception:
        pass
    return ""
