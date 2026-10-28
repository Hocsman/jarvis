"""Weather tool implementation using Open-Meteo API (free, no API key required)."""

import re
import unicodedata

import requests
from typing import Dict, Any, Optional, Tuple
from ...debug import debug_log
from ...llm import get_auxiliary_backend
from ...memory.graph import fold_for_search
from ...utils.location import get_location_info
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


# Sentinel strings an LLM extractor may emit to mean "no place mentioned".
# Matched case-insensitively as whole-value comparisons, not substrings.
_NO_PLACE_SENTINELS = frozenset({
    "none", "null", "no", "no place", "no location",
    "n/a", "na", "unknown", "unspecified",
})


def _fold_name(name: str) -> str:
    """Case and accents dropped, whitespace runs reduced to one space, so two
    spellings of a name compare equal."""
    return " ".join(fold_for_search(name).split())


def _assistant_names(cfg) -> frozenset:
    """The names the assistant answers to, folded for comparison.

    That is the wake word, which is also the name the persona goes by, and
    its aliases, the spellings the speech recogniser produces for it.
    Values that are not strings are skipped, so a config object without a
    wake word simply yields no names.
    """
    raw = [getattr(cfg, "wake_word", None)]
    aliases = getattr(cfg, "wake_aliases", None)
    if isinstance(aliases, (list, tuple, set, frozenset)):
        raw.extend(aliases)
    names = (_fold_name(n) for n in raw if isinstance(n, str))
    return frozenset(n for n in names if n)


def _extract_place_from_user_text(text: str, cfg) -> Optional[str]:
    """Ask a small LLM to pull a place name out of the user's utterance.

    Used as a last-ditch fallback when the tool-calling LLM didn't fill the
    ``location`` argument AND GeoIP auto-detect is unavailable. Small chat
    models (e.g. gemma4:e2b) regularly fail to propagate a city into tool
    args even when the user literally just said one — pulling the place
    straight from the user's text sidesteps that weakness so the user
    doesn't have to keep repeating themselves.

    Returns ``None`` when no place is named, the call fails, or the
    extractor gives back something that doesn't look like a place. A reply
    equal to the assistant's own name (wake word or alias) counts as no
    place: "Yuba, what's the weather" is the user addressing the assistant,
    and geocoding "Yuba" would answer for Yuba City.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    if cfg is None:
        return None

    model = (
        getattr(cfg, "tool_router_model", "")
        or getattr(cfg, "intent_judge_model", "")
        or getattr(cfg, "llm_chat_model", "")
    )
    if not model:
        return None

    sys_prompt = (
        "You extract a single place name from a user's utterance so a weather "
        "tool can look it up. Reply with ONLY the place name (city, town, or "
        "country), with no punctuation, quotes, or explanation. If the user "
        "did not name any place, reply with exactly: none"
    )
    user_prompt = f"User utterance: {text}\n\nPlace:"

    try:
        resp = get_auxiliary_backend(cfg, model).direct(
            model, sys_prompt, user_prompt,
            timeout_sec=float(getattr(cfg, "llm_tools_timeout_sec", 8.0)),
        )
    except Exception as e:
        debug_log(f"    ⚠️ place extraction failed: {e}", "tools")
        return None

    if not resp or not isinstance(resp, str):
        return None

    # Strip punctuation and quotes the extractor might wrap around the name.
    place = resp.strip().strip("'\"`*.,:;!?()[]{}<>").split("\n", 1)[0].strip()
    if not place:
        return None
    if place.lower() in _NO_PLACE_SENTINELS:
        return None
    if _fold_name(place) in _assistant_names(cfg):
        debug_log(
            f"    ⚠️ place extractor answered with the assistant's own name "
            f"('{place}'); treating it as no place",
            "tools",
        )
        return None
    # Reject multi-sentence or overly long replies — those are almost always
    # the model explaining ("the user did not name a place") instead of
    # answering. Place names are at most a handful of words (e.g. "New York",
    # "Stratford-upon-Avon", "São Paulo"), so 5 words is a generous cap.
    if len(place) > 60 or "." in place or len(place.split()) > 5:
        return None
    return place


_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# The Unicode name of a Latin letter built on plain ones: "LATIN SMALL LETTER
# O WITH STROKE", "LATIN SMALL LETTER DOTLESS I", "LATIN SMALL LIGATURE OE".
_LATIN_BUILT_ON = re.compile(
    r"^LATIN (?:SMALL|CAPITAL) (?:LETTER|LIGATURE) (?:DOTLESS )?([A-Z]{1,2})(?: WITH .+)?$"
)


def _base_letters(c: str) -> str:
    """The plain letters a Latin letter is built on, for those NFKD keeps whole.

    ``ø`` gives "o", ``ı`` "i", ``ł`` "l", ``œ`` "oe": a keyboard or a
    transcript without them still names the same place. They are read from
    the character's Unicode name, so no table of letters is kept, and
    letters of their own such as ``ð`` or ``þ`` (ETH, THORN) stay as they are.
    """
    if c.isascii():
        return c
    match = _LATIN_BUILT_ON.match(unicodedata.name(c, ""))
    return match.group(1).lower() if match else c


def _name_words(name: str) -> list:
    """A place name as the words that identify it.

    Case and accents are folded away, letters with a stroke or without a dot
    become the letter they are built on, and every punctuation mark becomes
    a space: "Saint-Denis" and "saint denis", "São Paulo" and "Sao Paulo",
    "Tromsø" and "Tromso" give the same words. Punctuation is recognised by
    its Unicode category, which holds in every script without a list of
    characters.
    """
    folded = fold_for_search(name or "")
    spaced = "".join(
        " " if unicodedata.category(c).startswith("P") else _base_letters(c)
        for c in folded
    )
    return spaced.split()


def _name_match(asked: str, found: str) -> Optional[str]:
    """How the geocoder's name relates to the name asked for.

    ``"exact"`` when they are the same words, ``"leading"`` when one is the
    other's leading whole words ("New York City" and "New York", "Paris 13e"
    and "Paris 13e Arrondissement"), ``None`` otherwise. A piece of a word
    never counts: Open-Meteo's fuzzy search answers "Bagnus" with
    "Bagnusch", which is another place.
    """
    asked_words, found_words = _name_words(asked), _name_words(found)
    if not asked_words or not found_words:
        return None
    if asked_words == found_words:
        return "exact"
    shorter = min(len(asked_words), len(found_words))
    if asked_words[:shorter] == found_words[:shorter]:
        return "leading"
    return None


def _geocode_first(name: str, language: str) -> Optional[Dict[str, Any]]:
    """Open-Meteo's first result for ``name`` searched in ``language``.

    Only the first result is ever asked for. Further down the list sit
    places that merely share a name: searched in French, "London" gives
    Londres (UK) first and London, Ontario second.
    """
    resp = requests.get(
        _GEOCODE_URL,
        params={"name": name, "count": 1, "language": language, "format": "json"},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    return results[0] if results else None


def _find_place(
    name: str, language: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """The place ``name`` designates, and the nearest one the geocoder offered.

    Open-Meteo matches names in the language it is asked for rather than
    only translating its answer: "Londres" searched in English is a town in
    Argentina, searched in French it is London. So the search runs in the
    utterance's language first, then in English. The first result whose name
    is exactly the asked name wins; failing that, the first that shares its
    leading words; failing that, none. The first result seen is returned
    alongside, so a refusal can say which place was passed over.
    """
    languages = [language, "en"] if language and language != "en" else ["en"]
    nearest = None
    leading = None
    for lang in languages:
        found = _geocode_first(name, lang)
        if not found:
            continue
        nearest = nearest or found
        match = _name_match(name, found.get("name", ""))
        debug_log(
            f"    🔎 '{name}' searched in '{lang}' → '{found.get('name', '')}': "
            f"{match or 'not the same name'}",
            "tools",
        )
        if match == "exact":
            return found, nearest
        if match == "leading" and leading is None:
            leading = found
    return leading, nearest


def _place_parts(town: str, region: str, country: str) -> list:
    """Town, region and country, each once.

    A part carrying the same name as one before it is dropped: Tbilisi's
    region is Tbilisi, Monaco's region and country are both Monaco.
    """
    parts = []
    for part in (town, region, country):
        part = (part or "").strip()
        if part and all(_name_words(part) != _name_words(kept) for kept in parts):
            parts.append(part)
    return parts


def _place_display(place: Dict[str, Any]) -> str:
    """Town, region and country, so a wrong place is heard at once."""
    return ", ".join(_place_parts(
        place.get("name") or "", place.get("admin1") or "", place.get("country") or "",
    ))


def _utterance_language(context) -> Optional[str]:
    """The language Whisper heard the request in, or ``None`` off the voice path."""
    language = getattr(context, "language", None)
    if not isinstance(language, str) or not language.strip():
        return None
    return language.strip().lower()


def _home_city(context) -> str:
    """The home city the user declared in config, or ``""``."""
    return str(getattr(getattr(context, "cfg", None), "weather_city", "") or "").strip()


# WMO Weather interpretation codes
# https://open-meteo.com/en/docs
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherTool(Tool):
    """Tool for getting current weather using Open-Meteo API."""

    def risk_for(self, args):
        """Looks at the world without changing it."""
        from ..policy import RISK_READ

        return RISK_READ

    @property
    def name(self) -> str:
        return "getWeather"

    @property
    def description(self) -> str:
        return (
            "Weather only (current + forecast). NOT for time-of-day, date, or "
            "location questions — those are already in the assistant's context. "
            "Use for ANY weather question: now, later today, tomorrow, this week. "
            "Call with {} — user location is auto-detected. Do NOT ask the user "
            "where they are or request a city; just call this tool with empty args."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "OPTIONAL. City name or location (e.g., 'London', 'New York', 'Tokyo'). Only set this if the user explicitly named a place different from their own location. If omitted, the tool auto-uses the user's current detected location — never ask the user for this argument."
                }
            },
            "required": []
        }

    def _get_user_location(self, context: ToolContext) -> Optional[Dict[str, Any]]:
        """Get user's current location from config/auto-detection.

        Returns dict with 'lat', 'lon', and 'display_name' keys, or None if unavailable.
        """
        try:
            location_info = get_location_info(
                config_ip=getattr(context.cfg, 'location_ip_address', None),
                auto_detect=getattr(context.cfg, 'location_auto_detect', True),
                resolve_cgnat_public_ip=getattr(context.cfg, 'location_cgnat_resolve_public_ip', True),
                location_cache_minutes=getattr(context.cfg, 'location_cache_minutes', 60),
            )

            if "error" in location_info:
                debug_log(f"    ⚠️ location detection failed: {location_info.get('error')}", "tools")
                return None

            # Use coordinates directly (avoids geocoding issues with district names)
            lat = location_info.get("latitude")
            lon = location_info.get("longitude")
            if lat is None or lon is None:
                return None

            display_parts = _place_parts(
                location_info.get("city") or "",
                location_info.get("region") or "",
                location_info.get("country") or "",
            )
            display_name = ", ".join(display_parts) if display_parts else "your location"

            return {"lat": lat, "lon": lon, "display_name": display_name}
        except Exception as e:
            debug_log(f"    ⚠️ location detection error: {e}", "tools")
            return None

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Get current weather for a location."""
        context.user_print("🌤️ Checking weather...")

        try:
            # Get location from args, or fall back to user's detected location
            location_str = ""
            if args and isinstance(args, dict):
                raw_location = args.get("location")
                # Handle None values (LLM may pass location: null/None)
                location_str = str(raw_location).strip() if raw_location else ""

            # Determine coordinates and display name
            lat: Optional[float] = None
            lon: Optional[float] = None
            location_display: str = ""

            # Where the name to geocode came from, for the debug log: the
            # caller's argument, the user's utterance, or the home city.
            place_source = "argument"

            if not location_str:
                # No location provided - try auto-detected coordinates first.
                user_loc = self._get_user_location(context)
                if user_loc:
                    lat = user_loc["lat"]
                    lon = user_loc["lon"]
                    location_display = user_loc["display_name"]
                    debug_log(
                        f"    📍 using detected location: {location_display} ({lat}, {lon})",
                        "tools",
                    )
                else:
                    # Auto-detect failed. Last resort: scrape a place name from
                    # the user's current utterance. Small tool-calling models
                    # often drop the city from tool args even when the user
                    # just said one, so doing this on the tool side stops the
                    # "I need it for London" → "please tell me which city"
                    # ping-pong loop.
                    user_text = getattr(context, "redacted_text", "") or ""
                    cfg = getattr(context, "cfg", None)
                    extracted = _extract_place_from_user_text(user_text, cfg)
                    if extracted:
                        debug_log(
                            f"    📍 auto-detect unavailable; extracted place from user text: '{extracted}'",
                            "tools",
                        )
                        location_str = extracted
                        place_source = "user text"
                    else:
                        # Still nothing: fall back to the home city the user
                        # declared in config. Geo-IP needs a local GeoLite2
                        # database that many installs don't have, and a failed
                        # tool result is precisely what tempts the model into
                        # inventing a plausible forecast — so a city the user
                        # has already told us about beats asking again.
                        home_city = _home_city(context)
                        if home_city:
                            debug_log(
                                f"    📍 auto-detect unavailable; using configured "
                                f"weather_city: '{home_city}'",
                                "tools",
                            )
                            location_str = home_city
                            place_source = "weather_city"
                        else:
                            # Auto-detect genuinely failed, the user didn't name
                            # a place, and no home city is configured. Asking is
                            # the right move.
                            return ToolExecutionResult(
                                success=False,
                                reply_text=(
                                    "I couldn't auto-detect your location. "
                                    "Please tell me which city to check the weather for."
                                ),
                            )

            # Set when the named place was not found and the home city stands
            # in for it: the result opens with it so the reply says so.
            not_found_note = ""

            if location_str:
                debug_log(
                    f"    🌤️ geocoding location: '{location_str}' (from {place_source})",
                    "tools",
                )
                # Place names come back in the language they were found in:
                # the user's when the search in it matched, English
                # otherwise. The rest of the result stays English for the LLM.
                language = _utterance_language(context)
                place, nearest = _find_place(location_str, language)

                if place is None:
                    nearest_name = (
                        f" (the geocoder's nearest name is {_place_display(nearest)})"
                        if nearest else ""
                    )
                    home_city = _home_city(context)
                    # The name that failed may be the home city itself, from
                    # the config or said by the user: it is not tried twice.
                    failed_home = bool(home_city) and (
                        _name_words(home_city) == _name_words(location_str)
                    )
                    home_place = None
                    if home_city and not failed_home:
                        home_place, _ = _find_place(home_city, language)

                    if home_place is None:
                        if failed_home:
                            debug_log(f"    ⚠️ configured home city '{home_city}' not found", "tools")
                            refusal = (
                                f"The user's configured home city '{location_str}' was "
                                f"not found under that name{nearest_name}. Tell the user, "
                                f"and ask which city to check."
                            )
                        elif home_city:
                            debug_log(
                                f"    ⚠️ '{location_str}' not found, nor home city '{home_city}'",
                                "tools",
                            )
                            refusal = (
                                f"Could not find a place named '{location_str}'{nearest_name}, "
                                f"nor the user's configured home city '{home_city}'. "
                                f"Ask the user which city they meant."
                            )
                        else:
                            debug_log(f"    ⚠️ '{location_str}' not found, no home city set", "tools")
                            refusal = (
                                f"Could not find a place named '{location_str}'{nearest_name}. "
                                f"Ask the user which city they meant."
                            )
                        context.user_print(f"⚠️ No place named '{location_str}'.")
                        return ToolExecutionResult(success=False, reply_text=refusal)

                    debug_log(
                        f"    ⚠️ '{location_str}' not found; using home city '{home_city}'",
                        "tools",
                    )
                    context.user_print(f"⚠️ No place named '{location_str}'; using {home_city}.")
                    not_found_note = (
                        f"No place named '{location_str}' was found{nearest_name}, so this "
                        f"is the weather for the user's home city instead. Tell the user "
                        f"'{location_str}' was not found."
                    )
                    place = home_place

                lat = place["latitude"]
                lon = place["longitude"]
                location_display = _place_display(place)

                debug_log(f"    📍 resolved to {location_display} ({lat}, {lon})", "tools")

            # Step 2: Get current weather + forecast
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,wind_gusts_10m",
                "hourly": "temperature_2m,weather_code",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                "forecast_days": 7,
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "timezone": "auto"
            }

            weather_response = requests.get(weather_url, params=weather_params, timeout=10)
            weather_response.raise_for_status()
            weather_data = weather_response.json()

            current = weather_data.get("current", {})
            if not current:
                return ToolExecutionResult(
                    success=False,
                    reply_text=f"Weather data temporarily unavailable for {location_display}."
                )

            # Extract current weather values
            temp_c = current.get("temperature_2m")
            feels_like_c = current.get("apparent_temperature")
            humidity = current.get("relative_humidity_2m")
            weather_code = current.get("weather_code", 0)
            wind_speed = current.get("wind_speed_10m")
            wind_gusts = current.get("wind_gusts_10m")

            # Convert to Fahrenheit as well
            temp_f = round(temp_c * 9/5 + 32, 1) if temp_c is not None else None
            feels_like_f = round(feels_like_c * 9/5 + 32, 1) if feels_like_c is not None else None

            # Get weather description
            weather_desc = WMO_CODES.get(weather_code, "Unknown conditions")

            # Build response text — current conditions
            lines = [not_found_note, ""] if not_found_note else []
            lines.append(f"Current weather in {location_display}:")
            # Town and region, said aloud, are what let the user hear a wrong
            # place: left to itself the reply names the town alone.
            display_parts = location_display.split(", ")
            if len(display_parts) >= 2:
                spoken_place = ", ".join(display_parts[:2])
                lines.append(
                    f'Say the place as "{spoken_place}" in the reply, so the '
                    f"user can tell at once if it is the wrong place."
                )
            lines += [
                f"",
                f"Conditions: {weather_desc}",
            ]

            if temp_c is not None:
                lines.append(f"Temperature: {temp_c}°C ({temp_f}°F)")

            if feels_like_c is not None and feels_like_c != temp_c:
                lines.append(f"Feels like: {feels_like_c}°C ({feels_like_f}°F)")

            if humidity is not None:
                lines.append(f"Humidity: {humidity}%")

            if wind_speed is not None:
                wind_info = f"Wind: {wind_speed} km/h"
                if wind_gusts and wind_gusts > wind_speed:
                    wind_info += f" (gusts up to {wind_gusts} km/h)"
                lines.append(wind_info)

            # Append today's hourly forecast (remaining hours)
            hourly = weather_data.get("hourly", {})
            hourly_times = hourly.get("time", [])
            hourly_temps = hourly.get("temperature_2m", [])
            hourly_codes = hourly.get("weather_code", [])

            if hourly_times and hourly_temps:
                # Get current hour from the current time field
                current_time = current.get("time", "")
                current_hour_str = current_time[11:13] if len(current_time) >= 13 else ""
                current_hour = int(current_hour_str) if current_hour_str.isdigit() else 0
                today_prefix = current_time[:10] if len(current_time) >= 10 else ""

                hourly_lines = []
                for i, t in enumerate(hourly_times):
                    if not t.startswith(today_prefix):
                        continue
                    hour_str = t[11:13] if len(t) >= 13 else ""
                    hour = int(hour_str) if hour_str.isdigit() else -1
                    # Show every 3 hours from now onwards
                    if hour > current_hour and hour % 3 == 0 and i < len(hourly_temps) and i < len(hourly_codes):
                        desc = WMO_CODES.get(hourly_codes[i], "")
                        hourly_lines.append(f"  {hour:02d}:00 — {hourly_temps[i]}°C, {desc}")

                if hourly_lines:
                    lines.append("")
                    lines.append("Today's forecast (upcoming hours):")
                    lines.extend(hourly_lines)

            # Append daily forecast
            daily = weather_data.get("daily", {})
            daily_dates = daily.get("time", [])
            daily_codes = daily.get("weather_code", [])
            daily_max = daily.get("temperature_2m_max", [])
            daily_min = daily.get("temperature_2m_min", [])

            if daily_dates and daily_max and daily_min:
                lines.append("")
                lines.append("7-day forecast:")
                for i, date_str in enumerate(daily_dates):
                    if i < len(daily_max) and i < len(daily_min) and i < len(daily_codes):
                        desc = WMO_CODES.get(daily_codes[i], "")
                        lines.append(f"  {date_str}: {daily_min[i]}–{daily_max[i]}°C, {desc}")

            reply_text = "\n".join(lines)

            debug_log(f"    ✅ weather retrieved: {weather_desc}, {temp_c}°C", "tools")
            # Use first part of location_display for concise output
            short_name = location_display.split(",")[0].strip()
            context.user_print(f"✅ Weather for {short_name}: {weather_desc}, {temp_c}°C")

            return ToolExecutionResult(success=True, reply_text=reply_text)

        except requests.exceptions.Timeout:
            debug_log("weather request timed out", "tools")
            context.user_print("⚠️ Weather service timeout.")
            return ToolExecutionResult(
                success=False,
                reply_text="Weather service is taking too long to respond. Please try again."
            )
        except requests.exceptions.RequestException as e:
            debug_log(f"weather request failed: {e}", "tools")
            context.user_print("⚠️ Weather service unavailable.")
            return ToolExecutionResult(
                success=False,
                reply_text="Weather service is temporarily unavailable. Please try again later."
            )
        except Exception as e:
            debug_log(f"weather error: {e}", "tools")
            context.user_print("⚠️ Error getting weather.")
            return ToolExecutionResult(
                success=False,
                reply_text=f"Error getting weather: {e}"
            )
