"""Host-side tools for the event crawler RLM.

These run on the host (not in the WASM sandbox) because they need
network access, a browser, or external APIs.
"""

import base64
import json
import os
import urllib.error
import urllib.request


def browse(url: str) -> str:
    """Fetch a web page and return its raw HTML.

    Makes a simple HTTP GET request and returns the full HTML response.
    Does NOT execute JavaScript — only works with server-rendered pages.

    Args:
        url: The full URL to fetch (e.g. "https://example.com/event").

    Returns:
        The raw HTML string of the page. On error, returns a string
        starting with "Error:" describing what went wrong.
    """
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"Error: HTTP {e.code} fetching {url}"
    except urllib.error.URLError as e:
        return f"Error: {e.reason} fetching {url}"
    except TimeoutError:
        return f"Error: Timeout fetching {url}"


def perplexity_search(
    query: str,
    model: str = "sonar-pro",
    search_domain_filter: list[str] | None = None,
    search_recency_filter: str | None = None,
    search_context_size: str = "high",
) -> str:
    """Search the web using Perplexity AI and return an answer with citations.

    Makes a request to the Perplexity Sonar API which performs a live web
    search, synthesizes results, and returns an answer grounded in sources.

    Args:
        query: The search query — be specific for best results.
            Good: "John Doe CTO of Acme Corp background and achievements"
            Bad: "who is John Doe"
        model: Perplexity model to use.
            - "sonar": Lightweight, cost-effective search ($1/1M tokens).
              Best for simple factual lookups.
            - "sonar-pro": Advanced search with 2x more sources ($3/$15
              per 1M in/out tokens). Better for complex queries.
            - "sonar-reasoning-pro": Chain-of-thought reasoning for
              multi-step analysis ($2/$8 per 1M in/out tokens).
        search_domain_filter: Restrict search to specific domains (max 20).
            Allowlist example: ["linkedin.com", "crunchbase.com"]
            Denylist example: ["-reddit.com", "-pinterest.com"]
            Cannot mix allow and deny in one request. Use bare domain
            names without https:// or www. prefixes.
        search_recency_filter: Filter results by how recently they were
            published. One of: "hour", "day", "week", "month", "year".
            Use "month" or "week" for recent event information.
        search_context_size: How much search context to retrieve and
            include in the response. Affects quality and cost.
            - "low": Least context, cheapest ($5/1K requests).
            - "medium": Balanced ($8/1K requests).
            - "high": Most context, best quality ($12/1K requests).

    Returns:
        The search answer text followed by source citations. Each citation
        includes the source title and URL. Returns an error string if the
        API call fails.
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        return "Error: PERPLEXITY_API_KEY environment variable not set"

    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "web_search_options": {"search_context_size": search_context_size},
    }
    if search_domain_filter:
        body["search_domain_filter"] = search_domain_filter
    if search_recency_filter:
        body["search_recency_filter"] = search_recency_filter

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        "https://api.perplexity.ai/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        return f"Error: Perplexity API returned HTTP {e.code}: {body_text}"
    except urllib.error.URLError as e:
        return f"Error: {e.reason}"
    except TimeoutError:
        return "Error: Perplexity API request timed out"

    answer = result["choices"][0]["message"]["content"]

    citations = result.get("citations") or []
    if citations:
        answer += "\n\nSources:\n"
        for i, url in enumerate(citations, 1):
            answer += f"  [{i}] {url}\n"

    search_results = result.get("search_results") or []
    if search_results:
        answer += "\n\nSearch Results:\n"
        for sr in search_results:
            title = sr.get("title", "")
            url = sr.get("url", "")
            answer += f"  - {title}: {url}\n"

    return answer


def screenshot(url: str, full_page: bool = False) -> str:
    """Take a screenshot of a web page and return it as a base64 data URI.

    Opens the URL in a headless Chromium browser (via Playwright),
    waits for the page to load, and captures a PNG screenshot. Returns
    a data URI string that can be passed directly to predict() with a
    dspy.Image typed field for visual analysis.

    Use this to see what a page actually looks like — attendee avatars,
    sponsor logos, visual layouts, content hidden behind JavaScript
    rendering, hover states, etc.

    Args:
        url: The full URL to screenshot (e.g. "https://example.com/event").
        full_page: If True, capture the entire scrollable page. If False
            (default), capture only the visible viewport (1280x720).

    Returns:
        A base64 PNG data URI string ("data:image/png;base64,...") that
        can be used as a dspy.Image input to predict(). On error, returns
        a string starting with "Error:".
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "Error: playwright not installed (pip install playwright)"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(url, wait_until="networkidle", timeout=30000)
            png_bytes = page.screenshot(full_page=full_page)
            browser.close()
    except Exception as e:
        return f"Error: Failed to screenshot {url}: {e}"

    b64 = base64.b64encode(png_bytes).decode()
    return f"data:image/png;base64,{b64}"
