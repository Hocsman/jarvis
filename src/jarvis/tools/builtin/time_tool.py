"""Time tool for looking up the current time and date in any city or timezone."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # type: ignore
    except ImportError:
        ZoneInfo = None  # type: ignore
        ZoneInfoNotFoundError = Exception  # type: ignore

import requests

from ...debug import debug_log
from ...utils.location import get_location_context_with_timezone
from ...utils.time_context import format_time_context
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult

# Open-Meteo geocoding returns an IANA timezone field per result (the
# same free, key-less endpoint getWeather already uses for its lookups).
_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Cache of lowercased location string -> (iana_zone, display_name).
# Only successful lookups are cached to avoid poisoning on transient outages.
_geocode_cache: Dict[str, Tuple[str, str]] = {}
_MAX_GEOCODE_CACHE_SIZE = 256

# Bare universal zero-offset zones: UTC, GMT, Zulu, Universal, Z.
_ROOT_TIMEZONES = frozenset({"UTC", "GMT", "ZULU", "UNIVERSAL", "Z"})

# A zone path with slashes ("Europe/Athens", "America/New_York", "Etc/GMT+5").
# The slash disambiguates zone paths from named places without a network lookup.
_IANA_ZONE_RE = re.compile(r"^[A-Za-z0-9_+-]+(?:/[A-Za-z0-9_+-]+)+$")


def clear_geocode_cache() -> None:
    """Clear the in-memory geocoding cache."""
    _geocode_cache.clear()


def _is_valid_iana_zone(value: str) -> bool:
    """Return True when ``value`` is a timezone name zoneinfo can load."""
    if ZoneInfo is None:
        return False
    clean = value.strip()
    if clean.upper() in _ROOT_TIMEZONES or _IANA_ZONE_RE.match(clean):
        target = "UTC" if clean.upper() in {"ZULU", "Z"} else clean
        try:
            ZoneInfo(target)
            return True
        except (ZoneInfoNotFoundError, ValueError):
            return False
    return False


def _geocode_timezone(location: str, timeout_sec: float) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a place name to its IANA timezone via Open-Meteo.

    Returns ``(iana_zone, display_name)``. When no place matches, returns
    ``(None, None)``. When a place matches but lacks a timezone, returns
    ``(None, display_name)``.
    """
    key = location.strip().lower()
    cached = _geocode_cache.get(key)
    if cached is not None:
        return cached

    params = {
        "name": location.strip(),
        "count": 1,
        "language": "en",
        "format": "json",
    }
    geo_response = requests.get(
        _GEOCODING_URL,
        params=params,
        timeout=timeout_sec,
        allow_redirects=False,
    )
    geo_response.raise_for_status()
    geo_data = geo_response.json()
    results = geo_data.get("results") or []
    if not results:
        return None, None

    place = results[0]
    name = place.get("name") or location.strip()
    admin1 = place.get("admin1") or ""
    country = place.get("country") or ""

    parts = [name]
    if admin1 and admin1 != name:
        parts.append(admin1)
    if country and country != name and country != admin1:
        parts.append(country)
    display = ", ".join(parts)

    tz = place.get("timezone")
    if tz:
        if len(_geocode_cache) >= _MAX_GEOCODE_CACHE_SIZE:
            _geocode_cache.pop(next(iter(_geocode_cache)))
        _geocode_cache[key] = (tz, display)
    return tz, display


class TimeTool(Tool):
    """Tool for getting the current time/date in any location or timezone."""

    def risk_for(self, args: Optional[Dict[str, Any]] = None) -> str:
        """Looks at the world without changing it."""
        from ..policy import RISK_READ

        return RISK_READ

    @property
    def name(self) -> str:
        return "getTime"

    @property
    def description(self) -> str:
        return (
            "Current time and date in a specific city, country, or timezone "
            "(e.g. 'what time is it in London?', 'what's the date in Tokyo?'). "
            "Use for time/date questions about a place other than the user's "
            "own location: their local time is already in the assistant's "
            "context and needs no tool. NOT for weather; that is getWeather."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "OPTIONAL. City name, country, or IANA timezone "
                        "(e.g. 'Thessaloniki', 'Japan', 'Europe/Athens', 'UTC'). "
                        "Set it when the user names a place. If omitted, "
                        "returns the user's local time: which is already in "
                        "the assistant's context."
                    ),
                }
            },
            "required": [],
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Get the current time for the requested place (or the user's own)."""
        context.user_print("🕐 Checking time...")

        timeout_sec = 8.0
        if context.cfg is not None:
            timeout_sec = float(getattr(context.cfg, "llm_tools_timeout_sec", 8.0))

        location_str = ""
        if args and isinstance(args, dict):
            raw_location = args.get("location")
            location_str = str(raw_location).strip() if raw_location else ""

        debug_log(f"getTime: checking '{location_str or 'local'}'", "tools")

        try:
            if location_str:
                if _is_valid_iana_zone(location_str):
                    upper = location_str.strip().upper()
                    if upper in _ROOT_TIMEZONES:
                        tz_name = "UTC" if upper in {"ZULU", "Z"} else upper
                        display = tz_name
                    else:
                        tz_name = location_str.strip()
                        display = tz_name
                else:
                    tz_name, display = _geocode_timezone(location_str, timeout_sec)
                    if display is None:
                        return ToolExecutionResult(
                            success=False,
                            reply_text=(
                                f"Could not find location '{location_str}'. "
                                "Try a different city name or spelling."
                            ),
                        )
                    if not tz_name or not _is_valid_iana_zone(tz_name):
                        return ToolExecutionResult(
                            success=False,
                            reply_text=(
                                f"Could not determine the timezone for "
                                f"'{display}'."
                            ),
                        )

                time_str = format_time_context(tz_name)
                short_name = display.split(",")[0].strip()
                context.user_print(f"✅ Current time in {short_name}: {time_str}")
                return ToolExecutionResult(
                    success=True,
                    reply_text=f"Current time in {display}: {time_str}",
                )

            # No location: the user's own local time. Prefer the GeoIP zone
            # when available and enabled; format_time_context falls back to OS zone.
            tz_name: Optional[str] = None
            location_enabled = (
                getattr(context.cfg, "location_enabled", True)
                if context.cfg is not None else True
            )
            if context.cfg is not None and location_enabled:
                try:
                    _, tz_name = get_location_context_with_timezone(
                        config_ip=getattr(context.cfg, "location_ip_address", None),
                        auto_detect=getattr(context.cfg, "location_auto_detect", True),
                        resolve_cgnat_public_ip=getattr(
                            context.cfg, "location_cgnat_resolve_public_ip", True
                        ),
                        location_cache_minutes=getattr(
                            context.cfg, "location_cache_minutes", 60
                        ),
                    )
                except Exception as e:
                    debug_log(f"getTime: local tz lookup failed: {e}", "tools")
                    tz_name = None

            time_str = format_time_context(tz_name)
            context.user_print(f"✅ Current time: {time_str}")
            return ToolExecutionResult(
                success=True,
                reply_text=f"Current time: {time_str}",
            )

        except requests.exceptions.Timeout:
            debug_log("getTime: time request timed out", "tools")
            context.user_print("⚠️ Time service timeout.")
            return ToolExecutionResult(
                success=False,
                reply_text="Time service is taking too long to respond. Please try again.",
            )
        except requests.exceptions.RequestException as e:
            debug_log(f"getTime: time request failed: {e}", "tools")
            context.user_print("⚠️ Time service unavailable.")
            return ToolExecutionResult(
                success=False,
                reply_text="Time service is temporarily unavailable. Please try again later.",
            )
        except Exception as e:
            debug_log(f"getTime: time error: {e}", "tools")
            context.user_print("⚠️ Error getting time.")
            return ToolExecutionResult(
                success=False,
                reply_text=f"Error getting time: {e}",
            )
