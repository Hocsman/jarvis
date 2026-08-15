"""The configured home city is the last fallback before giving up.

Geo-IP detection needs a local GeoLite2 database that many installs don't
have, so ``getWeather`` with no explicit location used to fail outright and
ask which city to check. When the user has already declared a city in
config (``weather_city``, also used by the dashboard's weather card), the
tool should use it instead of asking — a failed tool result is what tempts
the model into inventing a plausible-looking forecast.

Order of resolution: explicit argument → detected location → a place named
in the user's own utterance → configured home city → ask.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.builtin.weather import WeatherTool


def _weather_response(temp):
    """Open-Meteo forecast response in the shape the tool reads."""
    resp = Mock()
    resp.json.return_value = {
        "current": {
            "temperature_2m": temp,
            "relative_humidity_2m": 50,
            "apparent_temperature": temp,
            "weather_code": 0,
            "wind_speed_10m": 5.0,
            "wind_gusts_10m": 8.0,
        },
        "hourly": {"time": [], "temperature_2m": [], "weather_code": []},
    }
    resp.raise_for_status = Mock()
    return resp


def _context(weather_city="", user_text=""):
    ctx = Mock(spec=ToolContext)
    ctx.user_print = Mock()
    ctx.cfg = Mock()
    ctx.redacted_text = user_text
    # Empty model config short-circuits the LLM-backed place extractor.
    ctx.cfg.ollama_base_url = ""
    ctx.cfg.ollama_chat_model = ""
    ctx.cfg.llm_chat_model = ""
    ctx.cfg.tool_router_model = ""
    ctx.cfg.intent_judge_model = ""
    ctx.cfg.weather_city = weather_city
    return ctx


class TestHomeCityFallback:
    def setup_method(self):
        self.tool = WeatherTool()

    @patch.object(WeatherTool, "_get_user_location", return_value=None)
    @patch("requests.get")
    def test_uses_configured_city_when_detection_fails(self, mock_get, _loc):
        geo = Mock()
        geo.json.return_value = {"results": [{
            "latitude": 48.79, "longitude": 2.31, "name": "Bagneux",
            "country": "France",
        }]}
        geo.raise_for_status = Mock()
        mock_get.side_effect = [geo, _weather_response(20.0)]

        result = self.tool.run({}, _context(weather_city="Bagneux"))

        assert result.success is True, result.reply_text
        # The geocoding request must have asked for the configured city.
        geocode_params = mock_get.call_args_list[0].kwargs.get("params", {})
        assert geocode_params.get("name") == "Bagneux"

    @patch.object(WeatherTool, "_get_user_location", return_value=None)
    def test_still_asks_when_no_city_configured(self, _loc):
        result = self.tool.run({}, _context(weather_city=""))
        assert result.success is False
        assert "city" in result.reply_text.lower()

    @patch.object(WeatherTool, "_get_user_location", return_value=None)
    @patch("requests.get")
    def test_explicit_argument_wins_over_configured_city(self, mock_get, _loc):
        geo = Mock()
        geo.json.return_value = {"results": [{
            "latitude": 45.76, "longitude": 4.84, "name": "Lyon",
            "country": "France",
        }]}
        geo.raise_for_status = Mock()
        mock_get.side_effect = [geo, _weather_response(22.0)]

        self.tool.run({"location": "Lyon"}, _context(weather_city="Bagneux"))

        geocode_params = mock_get.call_args_list[0].kwargs.get("params", {})
        assert geocode_params.get("name") == "Lyon"

    @patch.object(WeatherTool, "_get_user_location")
    @patch("requests.get")
    def test_detected_location_wins_over_configured_city(self, mock_get, mock_loc):
        mock_loc.return_value = {"lat": 43.30, "lon": 5.37, "display_name": "Marseille, France"}
        mock_get.return_value = _weather_response(25.0)

        result = self.tool.run({}, _context(weather_city="Bagneux"))

        assert result.success is True
        assert "Marseille" in result.reply_text
