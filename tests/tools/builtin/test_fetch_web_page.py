"""Tests for fetch web page tool."""

import pytest
from unittest.mock import Mock, patch
import requests

from src.jarvis.tools.builtin import fetch_web_page
from src.jarvis.tools.builtin.fetch_web_page import FetchWebPageTool
from src.jarvis.tools.builtin.web_search import _MAX_REDIRECTS
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.types import ToolExecutionResult


def _make_response_mock(**attrs) -> Mock:
    """Build a Mock that doubles as both the requests response and a context
    manager (the production code uses ``with requests.get(...) as resp`` so
    the connection is released deterministically).
    """
    resp = Mock(**attrs)
    resp.__enter__ = Mock(return_value=resp)
    resp.__exit__ = Mock(return_value=False)
    # The body is streamed under a byte ceiling rather than read whole,
    # so the double has to stream too: a mock that only answers
    # `.content` measures a shape production no longer has.
    corps = attrs.get("content")
    if corps is None:
        texte = attrs.get("text")
        corps = texte.encode("utf-8") if isinstance(texte, str) else b""
    resp.iter_content = Mock(return_value=iter([corps]))
    if "encoding" not in attrs:
        resp.encoding = "utf-8"
    return resp


class TestFetchWebPageTool:
    """Test fetch web page tool functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = FetchWebPageTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()

    def test_tool_properties(self):
        """Test tool metadata properties."""
        assert self.tool.name == "fetchWebPage"
        assert "fetch" in self.tool.description.lower()
        assert self.tool.inputSchema["type"] == "object"
        assert "url" in self.tool.inputSchema["required"]

    def test_run_no_args(self):
        """Test fetch web page with no arguments."""
        result = self.tool.run(None, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "url" in result.reply_text.lower()

    def test_run_empty_url(self):
        """Test fetch web page with empty URL."""
        args = {"url": ""}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "url" in result.reply_text.lower()

    @pytest.mark.usefixtures("public_dns")
    @patch('requests.get')
    def test_run_success(self, mock_get):
        """Test successful web page fetch."""
        mock_response = _make_response_mock(
            status_code=200,
            text='<html><head><title>Test</title></head><body><p>Content</p></body></html>',
            content=b'<html><head><title>Test</title></head><body><p>Content</p></body></html>',
            headers={'content-type': 'text/html'},
            raise_for_status=Mock(),
        )
        mock_get.return_value = mock_response

        args = {"url": "https://example.com"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "example.com" in result.reply_text
        self.context.user_print.assert_called()

    @pytest.mark.usefixtures("public_dns")
    @patch('requests.get')
    def test_run_success_without_beautifulsoup(self, mock_get):
        """Test successful web page fetch without BeautifulSoup."""
        mock_response = _make_response_mock(
            status_code=200,
            text='<html><body>Raw content</body></html>',
            content=b'<html><body>Raw content</body></html>',
            headers={'content-type': 'text/html'},
            raise_for_status=Mock(),
        )
        mock_get.return_value = mock_response

        with patch('builtins.__import__', side_effect=ImportError):
            args = {"url": "https://example.com"}
            result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Raw Content" in result.reply_text

    @pytest.mark.usefixtures("public_dns")
    @patch('requests.get')
    def test_run_http_error(self, mock_get):
        """Test fetch web page with HTTP error."""
        mock_response = _make_response_mock(status_code=404)
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        args = {"url": "https://example.com/notfound"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "Failed to fetch page" in result.reply_text

    @pytest.mark.usefixtures("public_dns")
    @patch('requests.get')
    def test_run_request_error(self, mock_get):
        """Test fetch web page with network error."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        args = {"url": "https://example.com"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "Failed to fetch page" in result.reply_text

    def test_run_invalid_url(self):
        """Test fetch web page with invalid URL."""
        args = {"url": "not-a-url"}
        result = self.tool.run(args, self.context)
        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        # "not-a-url" becomes "https://not-a-url", whose DNS does not
        # resolve, so the public-web guard turns it away before any
        # request is made. What matters is that it is refused and says
        # so, not which of the two wordings it used.
        texte = result.reply_text.lower()
        assert ("failed" in texte or "error" in texte
                or "not on the public web" in texte)

    @pytest.mark.usefixtures("public_dns")
    @patch('requests.get')
    def test_run_with_links_extraction(self, mock_get):
        """Test fetch web page including link extraction when include_links=True."""
        html = (
            '<html><head><title>Links Page</title></head>'
            '<body><p>Intro</p>'
            '<a href="/relative">Relative Link</a>'
            '<a href="https://absolute.test/page">Absolute Link</a>'
            '<a href="mailto:test@example.com">Mail</a>'
            '</body></html>'
        )
        mock_response = _make_response_mock(
            status_code=200,
            text=html,
            content=html.encode(),
            raise_for_status=Mock(),
        )
        mock_get.return_value = mock_response

        args = {"url": "https://example.com", "include_links": True}
        result = self.tool.run(args, self.context)
        assert result.success is True
        assert isinstance(result, ToolExecutionResult)
        assert "Links found on page" in result.reply_text
        # relative link should be resolved to absolute
        assert "https://example.com/relative" in result.reply_text
        assert "absolute.test" in result.reply_text


def _page(*pieces: bytes) -> Mock:
    """An ordinary page, streamed in the given pieces."""
    body = b"".join(pieces)
    resp = _make_response_mock(
        status_code=200, content=body, headers={'content-type': 'text/html'},
        raise_for_status=Mock(),
    )
    resp.iter_content = Mock(return_value=iter(pieces))
    return resp


class _StreamOnly:
    """A response whose body can only be streamed. `requests` reads the
    body whole before handing a response back unless it was asked for a
    stream (`get_like_requests` does the same), and this double fails the
    test the moment that happens."""

    is_redirect = False
    is_permanent_redirect = False
    status_code = 200
    encoding = "utf-8"
    headers = {"content-type": "text/html"}

    def __init__(self, *pieces: bytes):
        self._pieces = pieces
        self.served = 0

    @property
    def content(self):
        raise AssertionError("the body was read whole")

    @property
    def text(self):
        raise AssertionError("the body was read whole")

    def iter_content(self, chunk_size=8192):
        for piece in self._pieces:
            self.served += 1
            yield piece

    def raise_for_status(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Clock:
    """A clock the test moves by hand."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


@pytest.mark.usefixtures("public_dns")
class TestFetchWebPageReadsTheWebCarefully:
    """The body arrives as a stream, under the byte ceiling and one wall
    clock; a page is reported under the address it was finally read from;
    the redirect walk stops where `webSearch` stops and reads no hop."""

    def setup_method(self):
        self.tool = FetchWebPageTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()

    def test_the_body_is_read_as_a_stream_never_whole(self, get_like_requests):
        page = _StreamOnly(b"<html><head><title>Streamed</title></head>",
                           b"<body><p>arrived in pieces</p></body></html>")

        with patch('requests.get', side_effect=get_like_requests(page)):
            result = self.tool.run({"url": "https://example.com"}, self.context)

        assert result.success is True, result.reply_text
        assert "arrived in pieces" in result.reply_text

    def test_a_page_is_reported_under_the_address_it_was_read_from(self, hop, get_like_requests):
        page = _page(b'<html><head><title>Final</title></head><body><p>Content here</p>'
                     b'<a href="/about">About the site</a></body></html>')

        with patch('requests.get',
                   side_effect=get_like_requests(hop("https://other.example/final"), page)):
            result = self.tool.run(
                {"url": "https://example.com/go", "include_links": True}, self.context,
            )

        assert result.success is True, result.reply_text
        assert "**URL:** https://other.example/final" in result.reply_text
        assert "https://other.example/about" in result.reply_text
        assert "https://example.com/about" not in result.reply_text

    def test_a_relative_redirect_is_taken_from_the_address_it_came_from(self, hop, get_like_requests):
        page = _page(b'<html><body><p>Content here</p>'
                     b'<a href="/about">About the site</a></body></html>')

        with patch('requests.get', side_effect=get_like_requests(hop("/final"), page)) as get:
            result = self.tool.run(
                {"url": "https://example.com/go", "include_links": True}, self.context,
            )

        assert result.success is True, result.reply_text
        assert get.call_args_list[1].args[0] == "https://example.com/final"
        assert "**URL:** https://example.com/final" in result.reply_text
        assert "https://example.com/about" in result.reply_text

    def test_the_raw_page_is_reported_under_the_address_it_was_read_from_too(self, hop, get_like_requests):
        page = _page(b'<html><body>Raw content</body></html>')

        with patch('requests.get',
                   side_effect=get_like_requests(hop("https://other.example/final"), page)), \
             patch('builtins.__import__', side_effect=ImportError):
            result = self.tool.run({"url": "https://example.com/go"}, self.context)

        assert result.success is True, result.reply_text
        assert "Raw Content" in result.reply_text
        assert "**URL:** https://other.example/final" in result.reply_text
        assert "https://example.com/go" not in result.reply_text

    def test_a_redirect_that_names_no_address_is_a_failure_not_a_page(self, hop, get_like_requests):
        """A 302 with an empty Location is a redirect to `requests`. Its body
        is a server's boilerplate at best, and a page with nothing in it
        marked success at worst."""
        nowhere = hop("")

        with patch('requests.get', side_effect=get_like_requests(nowhere)) as get:
            result = self.tool.run({"url": "https://example.com/go"}, self.context)

        assert result.success is False
        assert "nowhere" in result.reply_text, result.reply_text
        assert get.call_count == 1
        assert nowhere.closed

    def test_a_chain_within_the_cap_is_followed_and_no_hop_is_read(self, hop, get_like_requests):
        """The floor: the walk must not stop short of the cap. Every hop is
        asked for as a stream, closed before the next request and never
        read; the doubles fail the test otherwise."""
        hops = [hop(f"https://example.com/hop{i}") for i in range(_MAX_REDIRECTS)]
        page = _page(b'<html><body><p>The page at the end</p></body></html>')

        with patch('requests.get', side_effect=get_like_requests(*hops, page)) as get:
            result = self.tool.run({"url": "https://example.com/start"}, self.context)

        assert result.success is True, result.reply_text
        assert "The page at the end" in result.reply_text
        assert get.call_count == _MAX_REDIRECTS + 1
        assert all(each.closed for each in hops)

    def test_the_hop_cap_is_the_one_web_search_uses(self, hop, get_like_requests):
        hops = [hop(f"https://example.com/hop{i}") for i in range(_MAX_REDIRECTS + 5)]

        with patch('requests.get', side_effect=get_like_requests(*hops)) as get:
            result = self.tool.run({"url": "https://example.com/start"}, self.context)

        assert result.success is False
        assert "too many" in result.reply_text
        assert get.call_count == _MAX_REDIRECTS + 1

    def test_a_body_that_drips_is_cut_when_the_clock_runs_out(self, get_like_requests):
        """`timeout` bounds the connect and each read, never the fetch: a
        server that sends a byte every few seconds never trips it. The
        wall clock does, and what arrived by then is the page."""
        clock = _Clock()
        budget = fetch_web_page._FETCH_WALL_CLOCK_SEC
        pieces = [f"<p>piece {i}</p>".encode() for i in range(100)]

        class _Dripping(_StreamOnly):
            def iter_content(self, chunk_size=8192):
                for piece in super().iter_content(chunk_size):
                    clock.now += budget / 3
                    yield piece

        page = _Dripping(*pieces)
        with patch('requests.get', side_effect=get_like_requests(page)), \
             patch.object(fetch_web_page, "time", clock):
            result = self.tool.run({"url": "https://example.com"}, self.context)

        assert result.success is True, result.reply_text
        assert "piece 0" in result.reply_text
        assert "piece 99" not in result.reply_text
        assert page.served < len(pieces), "the whole drip was waited for"

    def test_a_walk_that_outlives_the_clock_stops_before_the_next_hop(self, hop, get_like_requests):
        clock = _Clock()
        budget = fetch_web_page._FETCH_WALL_CLOCK_SEC
        serve = get_like_requests(hop("https://example.com/slow"), hop("https://example.com/slower"))

        def slow_get(url, **kwargs):
            clock.now += budget + 1
            return serve(url, **kwargs)

        with patch('requests.get', side_effect=slow_get) as get, \
             patch.object(fetch_web_page, "time", clock):
            result = self.tool.run({"url": "https://example.com/start"}, self.context)

        assert result.success is False
        assert "too long" in result.reply_text, result.reply_text
        assert get.call_count == 1
