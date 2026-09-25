"""Fetch web page tool implementation for extracting content from URLs."""

import time

import requests
from typing import Dict, Any, Optional
from urllib.parse import urljoin
from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult

# One guard, one hop cap and one byte ceiling, not two of each: `webSearch`
# holds them and this tool imports them, since a copy would be the one
# that drifts.
from .web_search import _MAX_FETCH_BYTES, _MAX_REDIRECTS, _is_public_url

# The whole fetch, redirect walk and body together, under one wall clock.
# `timeout` on a request bounds the connect and each read, never the
# fetch: a server that sends a byte every few seconds never trips it. A
# hop that would start past the clock is not made; a body still arriving
# when it passes is cut there, as the byte ceiling cuts it.
_FETCH_WALL_CLOCK_SEC = 30.0
_REQUEST_TIMEOUT_SEC = 15.0


class FetchWebPageTool(Tool):
    """Tool for fetching and extracting content from web pages."""

    def risk_for(self, args):
        """Looks at the world without changing it."""
        from ..policy import RISK_READ

        return RISK_READ

    @property
    def name(self) -> str:
        return "fetchWebPage"

    @property
    def description(self) -> str:
        return "Fetch and extract text content from a web page URL."

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch content from"},
                "include_links": {"type": "boolean", "description": "Whether to include links found on the page"}
            },
            "required": ["url"]
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Fetch and extract content from a web page."""
        context.user_print("🌐 Fetching page content…")
        try:
            if not (args and isinstance(args, dict)):
                return ToolExecutionResult(success=False, reply_text="fetchWebPage requires a JSON object with 'url'.")
            url = str(args.get("url", "")).strip()
            include_links = bool(args.get("include_links", False))
            if not url:
                return ToolExecutionResult(success=False, reply_text="fetchWebPage requires a valid 'url'.")
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            # The URL came out of a model that has been reading the web
            # all turn, so a page it fetched a moment ago can propose the
            # next address. And this tool is `lecture`, so it is free by
            # default: nothing stands between that sentence and the
            # request except this line.
            if not _is_public_url(url):
                debug_log(f"fetchWebPage refused non-public {url}", "web")
                return ToolExecutionResult(
                    success=False,
                    reply_text=(
                        "That address is not on the public web, so it was not "
                        "fetched. Tell the user plainly and do not try a "
                        "variant of it."
                    ),
                )
            debug_log(f"fetchWebPage: fetching {url}", "web")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            # Redirects are followed by hand so every hop is checked: the
            # first address can be public and the second not, which is the
            # ordinary shape of this attack. The cap is `webSearch`'s, for
            # the same reason: each hop costs a request, and a chain that
            # long is a loop or a trap. Every request asks for a stream, so
            # a hop's body is never read and the page's arrives as the loop
            # below reads it.
            deadline = time.monotonic() + _FETCH_WALL_CLOCK_SEC
            current, response = url, None
            for _ in range(_MAX_REDIRECTS + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    debug_log(f"fetchWebPage: out of time before {current}", "web")
                    return ToolExecutionResult(
                        success=False,
                        reply_text="That page took too long to reach; nothing was read.",
                    )
                response = requests.get(current, headers=headers,
                                        timeout=min(_REQUEST_TIMEOUT_SEC, remaining),
                                        allow_redirects=False, stream=True)
                # `is True` rather than truthiness: on a real response
                # these are bools, and anything else — a stub, a mock, a
                # library that grew a property — must read as "not a
                # redirect" rather than sending the request round again.
                redirige = (getattr(response, "is_redirect", False) is True
                            or getattr(response, "is_permanent_redirect", False) is True)
                if not redirige:
                    break
                suivant = response.headers.get("Location", "")
                response.close()
                if not suivant:
                    # A redirect to `requests` (the header is there), a
                    # page to nobody: its body is boilerplate at best.
                    debug_log(f"fetchWebPage: {current} redirected nowhere", "web")
                    return ToolExecutionResult(
                        success=False,
                        reply_text="That page redirected nowhere; nothing was read.",
                    )
                suivant = urljoin(current, suivant)
                if not _is_public_url(suivant):
                    debug_log(f"fetchWebPage refused redirect to {suivant}", "web")
                    return ToolExecutionResult(
                        success=False,
                        reply_text=(
                            "That page redirected somewhere off the public web, "
                            "so nothing was read. Say so plainly."
                        ),
                    )
                debug_log(f"fetchWebPage: redirected to {suivant}", "web")
                current = suivant
            else:
                debug_log(f"fetchWebPage: gave up after {_MAX_REDIRECTS} redirects, "
                          f"last at {current}", "web")
                return ToolExecutionResult(
                    success=False,
                    reply_text="That page redirected too many times; nothing was read.",
                )
            # The address the page was read from, which a redirect may have
            # moved: the reply names it and relative links join against it.
            final_url = current

            # ``with`` releases the connection back to the pool deterministically
            # even if BeautifulSoup or the link extraction raises midway.
            with response:
                response.raise_for_status()
                # Streamed under the ceiling `webSearch` uses, shared rather
                # than copied for the reason `_is_public_url` is: the two
                # tools fetch the same web with the same trust in it. What
                # an unbounded read costs is not this tool: the daemon holds
                # the reminder thread and the routine runner, so an
                # exhausted process takes promises down with it.
                morceaux: list[bytes] = []
                lus = 0
                for morceau in response.iter_content(chunk_size=8192):
                    if not morceau:
                        continue
                    morceaux.append(morceau)
                    lus += len(morceau)
                    if lus >= _MAX_FETCH_BYTES:
                        debug_log(
                            f"fetchWebPage: page truncated at {lus} bytes", "tools")
                        break
                    if time.monotonic() >= deadline:
                        debug_log(
                            f"fetchWebPage: page cut at {lus} bytes, out of time", "web")
                        break
                response_content = b"".join(morceaux)
                response_text = response_content.decode(
                    response.encoding or "utf-8", "replace")
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response_content, 'html.parser')
                for script in soup(["script", "style", "meta", "link", "noscript"]):
                    script.decompose()
                title = ""
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.get_text().strip()
                text_content = soup.get_text()
                lines = []
                for line in text_content.split('\n'):
                    cleaned_line = line.strip()
                    if cleaned_line and len(cleaned_line) > 3:
                        lines.append(cleaned_line)
                seen_lines = set()
                unique_lines = []
                for line in lines:
                    if line not in seen_lines:
                        unique_lines.append(line)
                        seen_lines.add(line)
                content = '\n'.join(unique_lines[:500])
                links_section = ""
                if include_links:
                    links = []
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '').strip()
                        link_text = link.get_text().strip()
                        if href and link_text and len(link_text) > 3:
                            if href.startswith('/'):
                                href = urljoin(final_url, href)
                            elif not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                                continue
                            links.append(f"• {link_text}: {href}")
                    if links:
                        links_section = f"\n\n**Links found on page:**\n" + '\n'.join(links[:20])
                reply_parts = []
                if title:
                    reply_parts.append(f"**Title:** {title}")
                reply_parts.append(f"**URL:** {final_url}")
                reply_parts.append(f"**Content:**\n{content}")
                if links_section:
                    reply_parts.append(links_section)
                reply_text = '\n\n'.join(reply_parts)
                max_chars = 50_000
                if len(reply_text) > max_chars:
                    reply_text = f"[Truncated to {max_chars} chars]\n\n" + reply_text[:max_chars]
                debug_log(f"fetchWebPage: extracted {len(content)} chars of content", "web")
                context.user_print("✅ Page content fetched.")
                return ToolExecutionResult(success=True, reply_text=reply_text)
            except ImportError:
                text = response_text[:10000]
                reply_text = f"**URL:** {final_url}\n**Raw Content:**\n{text}"
                debug_log("fetchWebPage: BeautifulSoup not available, returning raw text", "web")
                context.user_print("✅ Page content fetched (raw).")
                return ToolExecutionResult(success=True, reply_text=reply_text)
        except requests.exceptions.RequestException as e:
            debug_log(f"fetchWebPage: request failed: {e}", "web")
            context.user_print("⚠️ Failed to fetch page.")
            return ToolExecutionResult(success=False, reply_text=f"Failed to fetch page: {e}")
        except Exception as e:  # pragma: no cover (safety net)
            debug_log(f"fetchWebPage: error: {e}", "web")
            context.user_print("⚠️ Error fetching page.")
            return ToolExecutionResult(success=False, reply_text=f"Error fetching page: {e}")
