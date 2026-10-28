"""The weather reply says which place the weather is for.

Field trace, 28/10/2026: the speech recogniser heard "Bagnus", the geocoder
answered with Bagnusch, a mountain village in Graubünden, and the reply gave
its weather as if it were the user's own town. Nothing in the reply gave the
mistake away: it named no region and no country.

Two contracts, against the real chat model with only Open-Meteo faked. The
tool itself runs, so the reply reads the result the tool really produces:

- the reply names the region as well as the town, so a wrong place is heard
  at once ("à Vannes, en Bretagne");
- when the name asked for is no place and the home city stands in, the reply
  says the asked name was not found and names the home city it answers for.

They take ``real_model_config`` (sandboxed memory, location off), with
``weather_city`` set to a town of the eval's own: the user's settings supply
the provider and the models, nothing that describes the user.

Run: ./scripts/run_evals.sh -k weather_names_the_place
"""

from __future__ import annotations

import dataclasses
import re
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import requests


def _place(name, admin1, country, lat, lon):
    return {
        "name": name, "admin1": admin1, "country": country,
        "latitude": lat, "longitude": lon,
    }


VANNES = _place("Vannes", "Bretagne", "France", 47.66, -2.76)
BAGNUSCH = _place("Bagnusch", "Canton des Grisons", "Suisse", 46.62, 9.54)


def _fold(text: str) -> str:
    from jarvis.memory.graph import fold_for_search
    return fold_for_search(text or "")


class _FakeOpenMeteo:
    """Answers the weather module's requests; geocodes by name in any language."""

    def __init__(self, places):
        self.places = {_fold(k): v for k, v in places.items()}
        self.forecasts = []

    def get(self, url, params=None, timeout=None):
        params = params or {}
        resp = Mock()
        resp.raise_for_status = Mock()
        if "geocoding" in url:
            found = self.places.get(_fold(params.get("name", "")))
            resp.json.return_value = {"results": [found]} if found else {}
        else:
            self.forecasts.append((params.get("latitude"), params.get("longitude")))
            resp.json.return_value = {
                "current": {
                    "time": "2026-10-28T14:00",
                    "temperature_2m": 13.0,
                    "apparent_temperature": 11.0,
                    "relative_humidity_2m": 80,
                    "weather_code": 61,
                    "wind_speed_10m": 18.0,
                    "wind_gusts_10m": 30.0,
                },
            }
        return resp

    def forecast_was_for(self, place):
        return (place["latitude"], place["longitude"]) in self.forecasts


def reply_with_fake_geocoder(cfg, text, places):
    """One full French voice turn with Open-Meteo faked. Returns (reply, fake)."""
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database
    from jarvis.reply.engine import run_reply_engine
    import jarvis.tools.builtin.weather as weather

    fake = _FakeOpenMeteo(places)
    db = Database(":memory:", sqlite_vss_path=None)
    try:
        with patch.object(
            weather, "requests",
            SimpleNamespace(get=fake.get, exceptions=requests.exceptions),
        ), patch(
            "jarvis.reply.engine.extract_search_params_for_memory",
            return_value={"keywords": []},
        ):
            reply = run_reply_engine(
                db=db, cfg=cfg, tts=None, text=text,
                dialogue_memory=DialogueMemory(inactivity_timeout=300, max_interactions=20),
                language="fr",
            )
    finally:
        db.close()
    return reply or "", fake


def names_the_region(reply: str) -> bool:
    return "bretagne" in _fold(reply)


def says_the_name_was_not_found(reply: str) -> bool:
    folded = _fold(reply)
    return bool(re.search(r"\bbagnus\b", folded)) and "vannes" in folded


@pytest.mark.eval
class TestWeatherNamesThePlace:
    # Sampled: whether a reply names the region is a habit of the model, not
    # a certainty, and a single pass would say nothing either way.
    ATTEMPTS = 5

    def test_the_reply_names_the_region(self, real_model_config):
        cfg = dataclasses.replace(real_model_config, weather_city="")
        misses = []
        for attempt in range(1, self.ATTEMPTS + 1):
            reply, fake = reply_with_fake_geocoder(
                cfg, "quel temps fait-il à Vannes ?", {"Vannes": VANNES},
            )
            assert fake.forecast_was_for(VANNES), f"the tool did not run for Vannes: {reply!r}"
            hit = names_the_region(reply)
            print(f"   attempt {attempt}: {'✅' if hit else '❌'} {reply[:90]!r}")
            if not hit:
                misses.append(reply)

        print(f"\n📊 Region named: {self.ATTEMPTS - len(misses)}/{self.ATTEMPTS}")
        assert not misses, f"replies that name no region: {misses!r}"

    def test_a_name_that_is_no_place_is_said_not_found(self, real_model_config):
        cfg = dataclasses.replace(real_model_config, weather_city="Vannes")
        misses = []
        for attempt in range(1, self.ATTEMPTS + 1):
            reply, fake = reply_with_fake_geocoder(
                cfg, "quel temps fait-il à Bagnus ?",
                {"Bagnus": BAGNUSCH, "Vannes": VANNES},
            )
            assert not fake.forecast_was_for(BAGNUSCH), (
                f"weather was fetched for Bagnusch, a place that is not Bagnus: {reply!r}"
            )
            hit = says_the_name_was_not_found(reply)
            print(f"   attempt {attempt}: {'✅' if hit else '❌'} {reply[:90]!r}")
            if not hit:
                misses.append(reply)

        print(f"\n📊 'Bagnus' said not found, Vannes named: "
              f"{self.ATTEMPTS - len(misses)}/{self.ATTEMPTS}")
        assert not misses, f"replies that hide the missing place: {misses!r}"
