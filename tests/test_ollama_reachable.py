"""Tests for ``is_ollama_reachable``: the probe that decides whether bare
auxiliary model pins survive on a remote-provider config.

The probe is a bare TCP connect — an HTTP(S)_PROXY must never decide
whether a *local* server is up, and the settings parser calls this on
every ``debug_log`` reload, so repeated reads must not block on network
I/O. Tests mock at the socket boundary.
"""

from __future__ import annotations

import socket

import pytest

import src.jarvis.config as config


class _FakeSocket:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _clear_reachability_cache(monkeypatch):
    monkeypatch.setattr(config, "_ollama_reachable_cache", {})


def test_reachable_when_the_tcp_connect_succeeds(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _FakeSocket())

    assert config.is_ollama_reachable("http://127.0.0.1:11434") is True


def test_unreachable_when_the_connection_is_refused(monkeypatch):
    def _refused(*a, **k):
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(socket, "create_connection", _refused)

    assert config.is_ollama_reachable("http://127.0.0.1:11434") is False


def test_unreachable_when_the_host_drops_packets(monkeypatch):
    def _timeout(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(socket, "create_connection", _timeout)

    assert config.is_ollama_reachable("http://10.255.255.1:11434") is False


def test_host_and_port_come_from_the_base_url(monkeypatch):
    seen = {}

    def _connect(address, timeout):
        seen["address"] = address
        seen["timeout"] = timeout
        return _FakeSocket()

    monkeypatch.setattr(socket, "create_connection", _connect)

    assert config.is_ollama_reachable("http://192.168.1.42:9999", timeout=0.25) is True
    assert seen["address"] == ("192.168.1.42", 9999)
    assert seen["timeout"] == 0.25


def test_default_port_is_ollamas(monkeypatch):
    seen = {}

    def _connect(address, timeout):
        seen["address"] = address
        return _FakeSocket()

    monkeypatch.setattr(socket, "create_connection", _connect)

    config.is_ollama_reachable("http://127.0.0.1")
    assert seen["address"] == ("127.0.0.1", 11434)


def test_the_probe_issues_no_http_request(monkeypatch):
    """A proxy answer is not a reachability answer: no requests call at all."""
    import requests

    def _boom(*a, **k):
        raise AssertionError("the probe must not issue HTTP requests")

    monkeypatch.setattr(requests, "get", _boom)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _FakeSocket())

    assert config.is_ollama_reachable() is True


def test_a_second_call_within_the_ttl_does_not_reprobe(monkeypatch):
    probes = []
    monkeypatch.setattr(
        socket, "create_connection",
        lambda *a, **k: probes.append(1) or _FakeSocket(),
    )

    assert config.is_ollama_reachable() is True
    assert config.is_ollama_reachable() is True
    assert len(probes) == 1


def test_the_cache_expires_after_the_ttl(monkeypatch):
    probes = []
    monkeypatch.setattr(
        socket, "create_connection",
        lambda *a, **k: probes.append(1) or _FakeSocket(),
    )
    monkeypatch.setattr(config, "_OLLAMA_REACHABILITY_TTL_SEC", -1.0)

    config.is_ollama_reachable()
    config.is_ollama_reachable()
    assert len(probes) == 2
