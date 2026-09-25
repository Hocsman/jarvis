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
import os
import socket

import pytest

# The guard's own words, so a refusal by the operating system (a real
# connection that nobody answered) cannot pass for the guard's.
GUARD = "talks to nobody"


@pytest.mark.unit
def test_a_host_beyond_this_machine_is_refused_by_the_guard(network_refusals):
    sock = socket.socket()
    try:
        with pytest.raises(ConnectionRefusedError, match=GUARD):
            sock.connect(("93.184.216.34", 80))
    finally:
        sock.close()
    assert network_refusals == [("socket.connect", "93.184.216.34", 80)]


@pytest.mark.unit
def test_a_datagram_beyond_this_machine_is_refused_too():
    """A resolver asked for the machine's public address is asked over UDP,
    with no connect to refuse."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(ConnectionRefusedError, match=GUARD):
            sock.sendto(b"\x00", ("208.67.222.222", 53))
    finally:
        sock.close()


@pytest.mark.unit
def test_a_name_beyond_this_machine_is_not_even_resolved():
    with pytest.raises(ConnectionRefusedError, match=GUARD):
        socket.getaddrinfo("example.com", 443)


@pytest.mark.unit
@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
def test_the_model_servers_port_is_refused_on_this_machine_too(host):
    with pytest.raises(ConnectionRefusedError, match=GUARD):
        socket.getaddrinfo(host, 11434)
    try:
        sock = socket.socket(socket.AF_INET6 if host == "::1" else socket.AF_INET)
    except OSError:
        pytest.skip(f"this machine has no socket family for {host}")
    try:
        with pytest.raises(ConnectionRefusedError, match=GUARD):
            sock.connect((host, 11434))
    finally:
        sock.close()


@pytest.mark.unit
def test_only_the_performance_suite_may_want_the_model_server():
    from conftest import _the_model_server_is_wanted

    assert _the_model_server_is_wanted(["tests/performance/"])
    assert _the_model_server_is_wanted(["tests\\performance\\test_pipeline_timings.py"])
    assert not _the_model_server_is_wanted(["tests"])
    assert not _the_model_server_is_wanted([])
    assert not _the_model_server_is_wanted(["tests/test_performance_of_something.py"])


@pytest.mark.unit
def test_no_proxy_can_carry_a_request_past_the_guard():
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY"):
        assert name not in os.environ and name.lower() not in os.environ, name
    assert os.environ.get("NO_PROXY") == "*"


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


@pytest.mark.unit
def test_public_dns_leaves_numeric_addresses_as_they_are(public_dns):
    infos = socket.getaddrinfo("10.0.0.1", 80)

    assert infos
    assert all(info[4][0] == "10.0.0.1" for info in infos), infos
