"""Tests for the getTime tool."""

from unittest.mock import Mock, patch

import requests

from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.builtin.time_tool import TimeTool, clear_geocode_cache
from src.jarvis.tools.policy import RISK_READ
from src.jarvis.tools.registry import BUILTIN_TOOLS
from src.jarvis.tools.types import ToolExecutionResult


class TestTimeTool:
    """Test the getTime tool behaviour."""

    def setup_method(self):
        clear_geocode_cache()
        self.tool = TimeTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()
        self.context.cfg = Mock()
        self.context.cfg.llm_tools_timeout_sec = 8.0
        self.context.cfg.location_enabled = True
        self.context.redacted_text = ""

    def test_tool_properties(self):
        """Tool metadata: camelCase name, time-focused description, optional location."""
        assert self.tool.name == "getTime"
        assert "time" in self.tool.description.lower()
        assert self.tool.inputSchema["type"] == "object"
        assert "location" in self.tool.inputSchema["properties"]
        assert self.tool.inputSchema["required"] == []

    def test_risk_classification(self):
        """Reading the time looks at the world without changing it."""
        assert self.tool.risk_for({}) == RISK_READ
        assert self.tool.risk_for({"location": "Tokyo"}) == RISK_READ

    def test_registered_in_builtin_tools(self):
        """The router and reply loop must be able to select getTime."""
        assert "getTime" in BUILTIN_TOOLS
        assert isinstance(BUILTIN_TOOLS["getTime"], TimeTool)

    @patch("src.jarvis.tools.builtin.time_tool.format_time_context",
           return_value="Tuesday, July 01, 2025 at 13:00 EEST")
    @patch("requests.get")
    def test_run_with_city_success(self, mock_get, mock_fmt):
        """A named city is geocoded and its IANA timezone drives the answer."""
        geo_response = Mock()
        geo_response.status_code = 200
        geo_response.json.return_value = {
            "results": [{
                "name": "Thessaloniki",
                "admin1": "Central Macedonia",
                "country": "Greece",
                "timezone": "Europe/Athens",
            }]
        }
        geo_response.raise_for_status = Mock()
        mock_get.return_value = geo_response

        result = self.tool.run({"location": "Thessaloniki"}, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Thessaloniki" in result.reply_text
        assert "Tuesday, July 01, 2025 at 13:00 EEST" in result.reply_text
        # The geocoded zone must be what gets formatted, never the raw city
        # name and never a guess.
        mock_fmt.assert_called_once_with("Europe/Athens")

    @patch("src.jarvis.tools.builtin.time_tool.format_time_context",
           return_value="Tuesday, July 01, 2025 at 13:00 EEST")
    @patch("requests.get")
    def test_run_with_iana_zone_direct(self, mock_get, mock_fmt):
        """A bare IANA zone in the location arg needs no network call."""
        result = self.tool.run({"location": "Europe/Athens"}, self.context)

        assert result.success is True
        assert "Europe/Athens" in result.reply_text
        mock_fmt.assert_called_once_with("Europe/Athens")
        mock_get.assert_not_called()

    @patch("src.jarvis.tools.builtin.time_tool.format_time_context",
           return_value="Tuesday, July 01, 2025 at 10:00 UTC")
    @patch("requests.get")
    def test_run_with_root_timezones_direct(self, mock_get, mock_fmt):
        """Root timezones like UTC and GMT resolve offline without network requests."""
        for zone in ("UTC", "GMT", "zulu"):
            result = self.tool.run({"location": zone}, self.context)
            assert result.success is True
            mock_get.assert_not_called()

    @patch("src.jarvis.tools.builtin.time_tool.format_time_context",
           return_value="Tuesday, July 01, 2025 at 15:00")
    @patch("requests.get")
    def test_run_with_posix_zone_containing_digits(self, mock_get, mock_fmt):
        """POSIX/IANA zones with digits resolve directly without geocoding."""
        result = self.tool.run({"location": "Etc/GMT+5"}, self.context)

        assert result.success is True
        assert "Etc/GMT+5" in result.reply_text
        mock_fmt.assert_called_once_with("Etc/GMT+5")
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_run_unknown_zone_shape_geocodes(self, mock_get):
        """A slash-y string that is NOT a valid zone must geocode, not error."""
        geo_response = Mock()
        geo_response.status_code = 200
        geo_response.json.return_value = {
            "results": [{
                "name": "Atlantis",
                "country": "Atlantis",
                "timezone": "Atlantic/Reykjavik",
            }]
        }
        geo_response.raise_for_status = Mock()
        mock_get.return_value = geo_response

        result = self.tool.run({"location": "Not/A_Zone"}, self.context)

        assert result.success is True
        assert "Atlantis" in result.reply_text
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_run_location_not_found(self, mock_get):
        """Unknown places are reported honestly, not guessed at."""
        geo_response = Mock()
        geo_response.status_code = 200
        geo_response.json.return_value = {"results": []}
        geo_response.raise_for_status = Mock()
        mock_get.return_value = geo_response

        result = self.tool.run({"location": "Nonexistent Place XYZ"}, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "could not find" in result.reply_text.lower()

    @patch("requests.get")
    def test_run_geocode_result_without_timezone(self, mock_get):
        """A geocode hit without a timezone field must fail cleanly."""
        geo_response = Mock()
        geo_response.status_code = 200
        geo_response.json.return_value = {
            "results": [{"name": "Somewhere", "country": "Nowhere"}]
        }
        geo_response.raise_for_status = Mock()
        mock_get.return_value = geo_response

        result = self.tool.run({"location": "Somewhere"}, self.context)

        assert result.success is False
        assert "timezone" in result.reply_text.lower()

    @patch("requests.get")
    def test_run_geocode_result_with_unresolvable_timezone_fails_cleanly(self, mock_get):
        """A geocode hit with an unrecognised timezone fails rather than confabulating local time."""
        geo_response = Mock()
        geo_response.status_code = 200
        geo_response.json.return_value = {
            "results": [{
                "name": "Faketown",
                "country": "Nowhere",
                "timezone": "Invalid/Unsupported_Zone",
            }]
        }
        geo_response.raise_for_status = Mock()
        mock_get.return_value = geo_response

        result = self.tool.run({"location": "Faketown"}, self.context)

        assert result.success is False
        assert "could not determine the timezone" in result.reply_text.lower()

    @patch("requests.get")
    def test_geocode_caching_avoids_repeated_requests(self, mock_get):
        """Multiple lookups for the same city use the cache and skip the network."""
        geo_response = Mock()
        geo_response.status_code = 200
        geo_response.json.return_value = {
            "results": [{
                "name": "Paris",
                "country": "France",
                "timezone": "Europe/Paris",
            }]
        }
        geo_response.raise_for_status = Mock()
        mock_get.return_value = geo_response

        res1 = self.tool.run({"location": "Paris"}, self.context)
        res2 = self.tool.run({"location": "paris"}, self.context)

        assert res1.success is True
        assert res2.success is True
        mock_get.assert_called_once()

    @patch("src.jarvis.tools.builtin.time_tool.format_time_context",
           return_value="Tuesday, July 01, 2025 at 13:00 EEST")
    @patch("src.jarvis.tools.builtin.time_tool.get_location_context_with_timezone",
           return_value=("some location context", "Europe/Istanbul"))
    def test_run_no_location_uses_geoip_zone(self, mock_loc, mock_fmt):
        """No location: prefer the user's GeoIP zone when one is known."""
        result = self.tool.run(None, self.context)

        assert result.success is True
        mock_fmt.assert_called_once_with("Europe/Istanbul")

    @patch("src.jarvis.tools.builtin.time_tool.format_time_context",
           return_value="Tuesday, July 01, 2025 at 13:00 EEST")
    @patch("src.jarvis.tools.builtin.time_tool.get_location_context_with_timezone")
    def test_run_no_location_with_location_disabled_skips_geoip(self, mock_loc, mock_fmt):
        """When location is disabled in settings, skip GeoIP and use OS timezone directly."""
        self.context.cfg.location_enabled = False
        result = self.tool.run({}, self.context)

        assert result.success is True
        mock_loc.assert_not_called()
        mock_fmt.assert_called_once_with(None)

    @patch("requests.get")
    def test_run_network_timeout(self, mock_get):
        """Network timeouts degrade to a clear message, not a crash."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        result = self.tool.run({"location": "London"}, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "taking too long" in result.reply_text.lower()

    @patch("requests.get")
    def test_run_network_error(self, mock_get):
        """Network failures degrade to a clear message, not a crash."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        result = self.tool.run({"location": "London"}, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "unavailable" in result.reply_text.lower()
