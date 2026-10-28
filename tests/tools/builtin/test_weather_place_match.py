"""The place getWeather answers for must be the place that was asked for.

Field trace, 28/10/2026: the speech recogniser heard "Bagnus" for Bagneux,
the model passed ``{"location": "Bagnus"}``, Open-Meteo's fuzzy search
returned "Bagnusch" (a village in Graubünden, 1,575 m) and Yuba gave that
village's weather as if it were the user's own town. The geocoder's first result is
only kept when its name is the name asked for: the same words once case,
accents and punctuation are set aside, or one name being the other's leading
whole words ("New York City" / "New York"). Otherwise the tool falls back to
the configured home city, saying the asked place was not found, or asks.

The search runs in the language the utterance was spoken in first, then in
English: Open-Meteo matches names in the requested language, so a French
"Londres" searched in English lands on Londres, Catamarca, Argentina.

Open-Meteo is faked here from responses measured on the real API, so nothing
leaves the machine.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.builtin.weather import WeatherTool


def _place(name, admin1, country, lat, lon):
    return {
        "name": name, "admin1": admin1, "country": country,
        "latitude": lat, "longitude": lon,
    }


BAGNUSCH_FR = _place("Bagnusch", "Canton des Grisons", "Suisse", 46.62, 9.54)
BAGNUSCH_EN = _place("Bagnusch", "Grisons", "Switzerland", 46.62, 9.54)
BAGNEUX_FR = _place("Bagneux", "Île-de-France", "France", 48.80, 2.31)
BAGNEUX_EN = _place("Bagneux", "Île-de-France Region", "France", 48.80, 2.31)
LONDON_UK_FR = _place("Londres", "Angleterre", "Royaume-Uni", 51.51, -0.13)
LONDON_UK_EN = _place("London", "England", "United Kingdom", 51.51, -0.13)
LONDON_ONTARIO_FR = _place("London", "Ontario", "Canada", 42.98, -81.23)
LONDRES_ARGENTINA_EN = _place("Londres", "Catamarca", "Argentina", -27.71, -67.13)


class FakeOpenMeteo:
    """Stands in for ``requests.get`` against both Open-Meteo endpoints.

    ``places`` maps ``(name, language)`` to the results the geocoder gives,
    best first; only as many as the request's ``count`` come back, as on the
    real API. Every forecast request is recorded by its coordinates, which is
    what tells which place the weather was fetched for.
    """

    def __init__(self, places):
        self.places = places
        self.forecasts = []

    def get(self, url, params=None, timeout=None):
        params = params or {}
        resp = Mock()
        resp.raise_for_status = Mock()
        if "geocoding" in url:
            found = self.places.get((params.get("name"), params.get("language")), [])
            count = int(params.get("count", 10))
            resp.json.return_value = {"results": found[:count]} if found else {}
        else:
            self.forecasts.append((params.get("latitude"), params.get("longitude")))
            resp.json.return_value = {
                "current": {
                    "time": "2026-10-28T14:00",
                    "temperature_2m": 12.0,
                    "apparent_temperature": 11.0,
                    "relative_humidity_2m": 70,
                    "weather_code": 3,
                    "wind_speed_10m": 10.0,
                    "wind_gusts_10m": 15.0,
                },
            }
        return resp

    def forecast_was_for(self, place):
        return (place["latitude"], place["longitude"]) in self.forecasts


def _context(language=None, weather_city="", user_text=""):
    ctx = Mock(spec=ToolContext)
    ctx.user_print = Mock()
    ctx.cfg = Mock()
    ctx.redacted_text = user_text
    ctx.language = language
    # Empty model config keeps the LLM place extractor out of the way.
    ctx.cfg.llm_chat_model = ""
    ctx.cfg.tool_router_model = ""
    ctx.cfg.intent_judge_model = ""
    ctx.cfg.weather_city = weather_city
    return ctx


def _run(fake, args, ctx):
    with patch.object(WeatherTool, "_get_user_location", return_value=None), \
         patch("requests.get", side_effect=fake.get):
        return WeatherTool().run(args, ctx)


class TestAMisheardNameIsNotAnotherPlace:
    def test_falls_back_to_the_home_city_and_says_the_name_was_not_found(self):
        fake = FakeOpenMeteo({
            ("Bagnus", "fr"): [BAGNUSCH_FR],
            ("Bagnus", "en"): [BAGNUSCH_EN],
            ("Bagneux", "fr"): [BAGNEUX_FR],
            ("Bagneux", "en"): [BAGNEUX_EN],
        })

        result = _run(fake, {"location": "Bagnus"},
                      _context(language="fr", weather_city="Bagneux"))

        assert result.success is True, result.reply_text
        assert not fake.forecast_was_for(BAGNUSCH_FR)
        assert fake.forecast_was_for(BAGNEUX_FR)
        assert "Bagnus'" in result.reply_text, "the asked name must be reported as not found"
        assert "Current weather in Bagneux" in result.reply_text

    def test_asks_which_city_when_no_home_city_is_configured(self):
        fake = FakeOpenMeteo({
            ("Bagnus", "fr"): [BAGNUSCH_FR],
            ("Bagnus", "en"): [BAGNUSCH_EN],
        })

        result = _run(fake, {"location": "Bagnus"}, _context(language="fr"))

        assert result.success is False
        assert fake.forecasts == []
        assert "Bagnus" in result.reply_text
        assert "city" in result.reply_text.lower()

    def test_a_place_pulled_from_the_utterance_passes_the_same_rule(self):
        fake = FakeOpenMeteo({
            ("Bagnus", "fr"): [BAGNUSCH_FR],
            ("Bagnus", "en"): [BAGNUSCH_EN],
            ("Bagneux", "fr"): [BAGNEUX_FR],
        })
        ctx = _context(language="fr", weather_city="Bagneux",
                       user_text="quel temps fait-il à Bagnus")

        with patch("src.jarvis.tools.builtin.weather._extract_place_from_user_text",
                   return_value="Bagnus"):
            result = _run(fake, {}, ctx)

        assert result.success is True, result.reply_text
        assert not fake.forecast_was_for(BAGNUSCH_FR)
        assert fake.forecast_was_for(BAGNEUX_FR)

    def test_a_home_city_whose_name_does_not_match_is_not_used(self):
        fake = FakeOpenMeteo({
            ("Bagnus", "fr"): [BAGNUSCH_FR],
            ("Bagnus", "en"): [BAGNUSCH_EN],
        })

        result = _run(fake, {}, _context(language="fr", weather_city="Bagnus"))

        assert result.success is False
        assert fake.forecasts == []

    def test_a_name_nobody_has_falls_back_to_the_home_city(self):
        fake = FakeOpenMeteo({("Bagneux", "en"): [BAGNEUX_EN]})

        result = _run(fake, {"location": "Xqzvbn"}, _context(weather_city="Bagneux"))

        assert result.success is True, result.reply_text
        assert fake.forecast_was_for(BAGNEUX_EN)
        assert "Xqzvbn" in result.reply_text


class TestTheSearchSpeaksTheUsersLanguage:
    def test_a_name_in_the_users_language_finds_the_place_meant(self):
        """Searched in English, "Londres" is a town of 3,000 in Argentina."""
        fake = FakeOpenMeteo({
            ("Londres", "fr"): [LONDON_UK_FR],
            ("Londres", "en"): [LONDRES_ARGENTINA_EN],
        })

        result = _run(fake, {"location": "Londres"}, _context(language="fr"))

        assert result.success is True, result.reply_text
        assert fake.forecast_was_for(LONDON_UK_FR)
        assert not fake.forecast_was_for(LONDRES_ARGENTINA_EN)

    def test_an_english_name_from_a_french_speaker_is_found_in_english(self):
        """In French, "London" puts Londres (UK) first and London, Ontario
        second: only the geocoder's first result is ever judged, so Canada is
        never reached for, and the English search settles it."""
        fake = FakeOpenMeteo({
            ("London", "fr"): [LONDON_UK_FR, LONDON_ONTARIO_FR],
            ("London", "en"): [LONDON_UK_EN],
        })

        result = _run(fake, {"location": "London"}, _context(language="fr"))

        assert result.success is True, result.reply_text
        assert fake.forecast_was_for(LONDON_UK_EN)
        assert not fake.forecast_was_for(LONDON_ONTARIO_FR)

    def test_an_exact_name_in_english_wins_over_leading_words_in_the_users_language(self):
        airfield = _place("St Petersburg Gorskaya", "Saint-Pétersbourg", "Russie", 60.03, 29.96)
        city = _place("St Petersburg", "St. Petersburg", "Russia", 59.94, 30.31)
        fake = FakeOpenMeteo({
            ("St Petersburg", "fr"): [airfield],
            ("St Petersburg", "en"): [city],
        })

        result = _run(fake, {"location": "St Petersburg"}, _context(language="fr"))

        assert result.success is True, result.reply_text
        assert fake.forecast_was_for(city)
        assert not fake.forecast_was_for(airfield)

    def test_without_a_detected_language_the_search_is_in_english(self):
        fake = FakeOpenMeteo({("London", "en"): [LONDON_UK_EN]})

        result = _run(fake, {"location": "London"}, _context(language=None))

        assert result.success is True, result.reply_text
        assert fake.forecast_was_for(LONDON_UK_EN)


class TestWhatCountsAsTheSameName:
    def _accepts(self, asked, found_name):
        found = _place(found_name, "Somewhere", "Someland", 10.0, 20.0)
        fake = FakeOpenMeteo({(asked, "en"): [found]})
        result = _run(fake, {"location": asked}, _context())
        return result.success and fake.forecast_was_for(found)

    def test_case_accents_and_punctuation_are_set_aside(self):
        assert self._accepts("sao paulo", "São Paulo")
        assert self._accepts("Saint Denis", "Saint-Denis")
        assert self._accepts("L'Haÿ-les-Roses", "L’Hay les Roses")

    def test_one_name_may_be_the_others_leading_words(self):
        assert self._accepts("New York City", "New York")
        assert self._accepts("Paris 13e", "Paris 13e Arrondissement")
        assert self._accepts("Paris, France", "Paris")

    def test_a_piece_of_a_word_is_not_the_same_name(self):
        assert not self._accepts("Bagnus", "Bagnusch")

    def test_trailing_words_are_not_the_same_name(self):
        assert not self._accepts("York", "New York")


class TestThePlaceCanBeHeard:
    def test_the_result_names_town_region_and_country(self):
        fake = FakeOpenMeteo({("Bagneux", "fr"): [BAGNEUX_FR]})

        result = _run(fake, {"location": "Bagneux"}, _context(language="fr"))

        assert result.success is True, result.reply_text
        assert "Bagneux, Île-de-France, France" in result.reply_text

    def test_the_result_asks_for_town_and_region_to_be_said(self):
        """Left to itself the reply names the town alone, and a wrong town
        with the right-sounding name goes unheard."""
        fake = FakeOpenMeteo({("Bagneux", "fr"): [BAGNEUX_FR]})

        result = _run(fake, {"location": "Bagneux"}, _context(language="fr"))

        assert '"Bagneux, Île-de-France"' in result.reply_text

    def test_a_detected_location_is_said_with_its_region_too(self):
        ctx = _context()
        fake = FakeOpenMeteo({})
        with patch.object(WeatherTool, "_get_user_location", return_value={
            "lat": 43.30, "lon": 5.37,
            "display_name": "Marseille, Provence-Alpes-Côte d'Azur, France",
        }), patch("requests.get", side_effect=fake.get):
            result = WeatherTool().run({}, ctx)

        assert result.success is True, result.reply_text
        assert '"Marseille, Provence-Alpes-Côte d\'Azur"' in result.reply_text
