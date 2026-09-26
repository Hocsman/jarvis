"""Tests for dashboard weather and privacy guards."""

import json
from unittest.mock import patch, MagicMock
import pytest

from jarvis.utils.weather_now import fetch_weather_summary, _describe
from desktop_app.dashboard.bridge import DashboardBridge


@pytest.fixture(autouse=True)
def qapp():
    from PyQt6.QtCore import QCoreApplication
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    yield app


class TestWeatherNow:
    """Tests for lightweight current weather utility."""

    @pytest.mark.unit
    def test_returns_none_when_city_empty_or_none(self):
        """Must return None without making any HTTP requests when city is empty."""
        with patch("requests.get") as mock_get:
            assert fetch_weather_summary(None) is None
            assert fetch_weather_summary("") is None
            assert fetch_weather_summary("   ") is None
            mock_get.assert_not_called()

    @pytest.mark.unit
    def test_uses_english_by_default(self):
        """Weather description and geocoding language should default to English."""
        mock_geo = MagicMock()
        mock_geo.json.return_value = {
            "results": [{"latitude": 51.5, "longitude": -0.1, "name": "London", "country_code": "GB"}]
        }
        mock_geo.raise_for_status = MagicMock()

        mock_wx = MagicMock()
        mock_wx.json.return_value = {
            "current": {
                "temperature_2m": 18.2,
                "relative_humidity_2m": 65,
                "apparent_temperature": 17.5,
                "weather_code": 0,
                "wind_speed_10m": 12.0,
            }
        }
        mock_wx.raise_for_status = MagicMock()

        with patch("requests.get", side_effect=[mock_geo, mock_wx]) as mock_get:
            res = fetch_weather_summary("London", language="en")
            assert res is not None
            assert res["loc"] == "London, GB"
            assert res["desc"] == "Clear sky"
            assert res["icon"] == "☀️"
            # Verify geocoding param used language=en
            geo_call_params = mock_get.call_args_list[0][1]["params"]
            assert geo_call_params["language"] == "en"

    @pytest.mark.unit
    def test_uses_french_when_requested(self):
        """Weather description and geocoding should use French when language is fr."""
        mock_geo = MagicMock()
        mock_geo.json.return_value = {
            "results": [{"latitude": 48.85, "longitude": 2.35, "name": "Paris", "country_code": "FR"}]
        }
        mock_geo.raise_for_status = MagicMock()

        mock_wx = MagicMock()
        mock_wx.json.return_value = {
            "current": {
                "temperature_2m": 20.0,
                "relative_humidity_2m": 50,
                "apparent_temperature": 19.0,
                "weather_code": 0,
                "wind_speed_10m": 10.0,
            }
        }
        mock_wx.raise_for_status = MagicMock()

        with patch("requests.get", side_effect=[mock_geo, mock_wx]) as mock_get:
            res = fetch_weather_summary("Paris", language="fr")
            assert res is not None
            assert res["desc"] == "ciel dégagé"
            geo_call_params = mock_get.call_args_list[0][1]["params"]
            assert geo_call_params["language"] == "fr"


class TestDashboardWeatherPrivacyGuard:
    """Tests for privacy guards in dashboard weather polling."""

    @pytest.mark.unit
    def test_no_weather_poll_when_location_disabled(self):
        """When location_enabled is False, weather timer is not started and no fetch occurs."""
        fake_cfg = MagicMock()
        fake_cfg.location_enabled = False
        fake_cfg.weather_city = "Paris"

        with patch("desktop_app.dashboard.bridge.fetch_weather_summary") as mock_fetch:
            bridge = DashboardBridge(submit_fn=None, cfg=fake_cfg)
            assert not bridge._weather_timer.isActive()
            bridge.ready()
            mock_fetch.assert_not_called()

    @pytest.mark.unit
    def test_no_weather_poll_when_city_empty_and_no_geoip(self):
        """When weather_city is empty and GeoIP yields nothing, do not fall back to Paris."""
        fake_cfg = MagicMock()
        fake_cfg.location_enabled = True
        fake_cfg.weather_city = ""

        with patch("desktop_app.dashboard.bridge.get_location_info", return_value=None):
            with patch("desktop_app.dashboard.bridge.fetch_weather_summary") as mock_fetch:
                bridge = DashboardBridge(submit_fn=None, cfg=fake_cfg)
                assert not bridge._weather_timer.isActive()
                bridge.ready()
                mock_fetch.assert_not_called()

    @pytest.mark.unit
    def test_weather_polls_when_city_configured(self):
        """When weather_city is explicitly configured and location is enabled, timer starts."""
        fake_cfg = MagicMock()
        fake_cfg.location_enabled = True
        fake_cfg.weather_city = "Berlin"
        fake_cfg.response_language = "de"

        with patch("desktop_app.dashboard.bridge.fetch_weather_summary") as mock_fetch:
            bridge = DashboardBridge(submit_fn=None, cfg=fake_cfg)
            assert bridge._weather_timer.isActive()
            assert bridge._weather_city == "Berlin"
            assert bridge._language == "de"
