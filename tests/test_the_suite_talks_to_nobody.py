"""The suite talks to nobody.

A unit test that reaches for the network measures the network: a model
server that answers on one address family and not the other, a resolver's
answer for ``example.com``, a geocoding service's opinion of a Mock's repr.
The guard in ``conftest.py`` refuses every connection to a host beyond this
machine, and every connection to Ollama's port on this one, before it is
made; these tests hold it to that, and check it leaves alone what a test
serves itself.
"""

from __future__ import annotations

import ipaddress
import socket
import time

import pytest


@pytest.mark.unit
def test_a_host_beyond_this_machine_is_refused_at_once():
    sock = socket.socket()
    sock.settimeout(5)
    started = time.monotonic()
    try:
        with pytest.raises(OSError):
            sock.connect(("93.184.216.34", 80))
    finally:
        sock.close()
    assert time.monotonic() - started < 0.5, "refused, not timed out"


@pytest.mark.unit
def test_a_name_beyond_this_machine_is_not_even_resolved():
    with pytest.raises(OSError):
        socket.getaddrinfo("example.com", 443)


@pytest.mark.unit
@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
def test_the_model_server_s_port_is_refused_on_this_machine_too(host):
    with pytest.raises(OSError):
        socket.getaddrinfo(host, 11434)
    try:
        sock = socket.socket(socket.AF_INET6 if host == "::1" else socket.AF_INET)
    except OSError:
        pytest.skip(f"this machine has no socket family for {host}")
    sock.settimeout(5)
    try:
        with pytest.raises(OSError):
            sock.connect((host, 11434))
    finally:
        sock.close()


@pytest.mark.unit
def test_a_server_a_test_runs_itself_is_reachable():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        client = socket.create_connection(("127.0.0.1", port), timeout=2)
        client.close()
    finally:
        server.close()


@pytest.mark.unit
def test_public_dns_answers_for_the_world_without_a_resolver(public_dns):
    infos = socket.getaddrinfo("example.com", 443)

    assert infos, "a name beyond this machine gets an answer"
    assert all(ipaddress.ip_address(info[4][0]).is_global for info in infos), infos
    assert all(info[4][0] == public_dns for info in infos)


@pytest.mark.unit
def test_public_dns_leaves_loopback_names_to_the_machine(public_dns):
    infos = socket.getaddrinfo("localhost", 80)

    assert infos
    assert all(ipaddress.ip_address(info[4][0]).is_loopback for info in infos), infos
