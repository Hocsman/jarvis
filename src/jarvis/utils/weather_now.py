"""Lightweight current-weather fetch for the dashboard.

Standalone from the ``getWeather`` agent tool (which needs a
ToolContext + agent plumbing). This hits the free Open-Meteo API
directly — geocode a city name, then read its current conditions —
and returns a flat dict the dashboard bridge can push to the page.

No API key required. Returns ``None`` on any failure so the caller
can keep the previous values instead of blanking the card.
"""

from __future__ import annotations

from typing import Dict, Optional

import requests

from ..debug import debug_log


# WMO weather interpretation codes -> (English description, emoji).
# https://open-meteo.com/en/docs (weather_code)
_WMO_EN = {
    0:  ("Clear sky", "☀️"),
    1:  ("Mainly clear", "🌤️"),
    2:  ("Partly cloudy", "⛅"),
    3:  ("Overcast", "☁️"),
    45: ("Foggy", "🌫️"),
    48: ("Depositing rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    53: ("Drizzle", "🌦️"),
    55: ("Dense drizzle", "🌧️"),
    61: ("Slight rain", "🌦️"),
    63: ("Rain", "🌧️"),
    65: ("Heavy rain", "🌧️"),
    66: ("Freezing rain", "🌧️"),
    67: ("Heavy freezing rain", "🌧️"),
    71: ("Slight snow", "🌨️"),
    73: ("Snow", "🌨️"),
    75: ("Heavy snow", "❄️"),
    77: ("Snow grains", "🌨️"),
    80: ("Light showers", "🌦️"),
    81: ("Showers", "🌧️"),
    82: ("Violent showers", "⛈️"),
    85: ("Snow showers", "🌨️"),
    86: ("Heavy snow showers", "❄️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with hail", "⛈️"),
    99: ("Violent thunderstorm with hail", "⛈️"),
}

# French descriptions for francophone locales
_WMO_FR = {
    0:  ("ciel dégagé", "☀️"),
    1:  ("plutôt dégagé", "🌤️"),
    2:  ("partiellement nuageux", "⛅"),
    3:  ("couvert", "☁️"),
    45: ("brouillard", "🌫️"),
    48: ("brouillard givrant", "🌫️"),
    51: ("bruine légère", "🌦️"),
    53: ("bruine", "🌦️"),
    55: ("bruine dense", "🌧️"),
    61: ("pluie légère", "🌦️"),
    63: ("pluie", "🌧️"),
    65: ("forte pluie", "🌧️"),
    66: ("pluie verglaçante", "🌧️"),
    67: ("forte pluie verglaçante", "🌧️"),
    71: ("neige légère", "🌨️"),
    73: ("neige", "🌨️"),
    75: ("forte neige", "❄️"),
    77: ("grains de neige", "🌨️"),
    80: ("averses légères", "🌦️"),
    81: ("averses", "🌧️"),
    82: ("fortes averses", "⛈️"),
    85: ("averses de neige", "🌨️"),
    86: ("fortes averses de neige", "❄️"),
    95: ("orage", "⛈️"),
    96: ("orage avec grêle", "⛈️"),
    99: ("violent orage avec grêle", "⛈️"),
}


def _describe(code: Optional[int], language: str = "en") -> tuple[str, str]:
    table = _WMO_FR if (language or "").strip().lower().startswith("fr") else _WMO_EN
    if isinstance(code, (int, float)) and int(code) in table:
        return table[int(code)]
    return ("", "🌡️")


def fetch_weather_summary(
    city: Optional[str] = None,
    language: Optional[str] = None,
    timeout: float = 8.0,
) -> Optional[Dict]:
    """Return current weather for ``city`` as a flat dict, or ``None``.

    Keys: ``temp, loc, desc, icon, hum, wind, feels`` — exactly what the
    dashboard's ``weatherUpdated`` handler consumes.
    """
    if not city or not str(city).strip():
        return None
    city = str(city).strip()

    lang = (language or "en").strip().lower()
    lang_code = lang[:2] if len(lang) >= 2 else "en"

    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": lang_code, "format": "json"},
            timeout=timeout,
        )
        geo.raise_for_status()
        results = (geo.json() or {}).get("results") or []
        if not results:
            debug_log(f"weather_now: no geocode result for {city!r}", "tools")
            return None
        r0 = results[0]
        lat, lon = r0.get("latitude"), r0.get("longitude")
        loc_name = r0.get("name") or city
        cc = r0.get("country_code") or ""
        loc_label = f"{loc_name}, {cc}" if cc else loc_name

        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,"
                           "apparent_temperature,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=timeout,
        )
        wx.raise_for_status()
        cur = (wx.json() or {}).get("current") or {}
        desc, icon = _describe(cur.get("weather_code"), language=lang)
        return {
            "temp": round(float(cur.get("temperature_2m", 0)), 1),
            "loc": loc_label,
            "desc": desc,
            "icon": icon,
            "hum": round(float(cur.get("relative_humidity_2m", 0))),
            "wind": round(float(cur.get("wind_speed_10m", 0))),
            "feels": round(float(cur.get("apparent_temperature", 0))),
        }
    except Exception as exc:
        debug_log(f"weather_now: fetch failed for {city!r}: {exc}", "tools")
        return None
