from predict_rlm import Skill

from .tools import browse, perplexity_search, screenshot

web_crawl_skill = Skill(
    name="web-crawl",
    instructions="""How to crawl web pages and search for information.

## Browsing pages

Call browse(url) to fetch any web page. Returns the raw HTML string.
Does not execute JavaScript.

    html = await browse("https://example.com/event")
    print(html[:500])  # preview

## Screenshots

Call screenshot(url) to capture a visual screenshot of a page. This
DOES execute JavaScript and renders the page fully. Returns a base64
PNG data URI that you can pass to predict() with dspy.Image:

    img_uri = await screenshot("https://example.com/event")
    result = await predict(
        "image: dspy.Image -> attendees: list[str], details: str",
        instructions="Look at this screenshot of an event page. List every attendee name visible, including names shown on avatars or hover tooltips.",
        image=img_uri,
    )

Use screenshot() when you need to see:
- Visually rendered content (avatars, logos, images)
- JavaScript-rendered content not in the raw HTML
- Full-page layout to understand page structure

Use full_page=True to capture the entire scrollable page:

    img_uri = await screenshot("https://example.com/event", full_page=True)

## Parsing HTML

Use the html.parser module from the standard library to extract text
and links from the raw HTML:

    from html.parser import HTMLParser

Or use predict() to extract structured data directly from the HTML —
the sub-LM can read and understand HTML:

    result = await predict(
        "html: str -> links: list[str]",
        instructions="Extract all <a href> URLs from this HTML page that point to subpages about speakers, schedule, or attendees.",
        html=page_html,
    )

To extract links with regex (quick and dirty):

    import re
    links = re.findall(r'href=["\\'](https?://[^"\\'>]+)', html)

## Extracting structured data with predict()

Use predict() to extract information from page content. You can pass
raw HTML directly — the sub-LM understands HTML structure:

    speakers = await predict(
        "html: str -> names: list[str], titles: list[str], topics: list[str]",
        instructions="Extract speaker names, job titles, and talk topics from this event page HTML.",
        html=page_html,
    )

For attendee lists:

    attendees = await predict(
        "html: str -> names: list[str], affiliations: list[str]",
        instructions="Extract attendee names and their company/organization affiliations from this HTML.",
        html=page_html,
    )

## Enriching with Perplexity search

Call perplexity_search() to find background information on people:

    # Basic search
    info = await perplexity_search("Jane Smith VP Engineering at TechCorp")

    # With domain filtering for professional profiles
    info = await perplexity_search(
        "Jane Smith VP Engineering TechCorp",
        search_domain_filter=["linkedin.com", "crunchbase.com"],
    )

    # Recent information only
    info = await perplexity_search(
        "TechConf 2025 speakers",
        search_recency_filter="month",
    )

    # Higher quality search for important queries
    info = await perplexity_search(
        "Jane Smith career background and achievements",
        model="sonar-pro",
        search_context_size="medium",
    )

Focus enrichment on the most prominent speakers (keynotes, headliners)
to control API costs. Use the default "sonar" model and "low" context
for most lookups.

## Writing the report

Write the final markdown report to /sandbox/output/report/summary.md:

    import os
    os.makedirs("/sandbox/output/report", exist_ok=True)

    with open("/sandbox/output/report/summary.md", "w") as f:
        f.write(report_markdown)
""",
    tools={
        "browse": browse,
        "perplexity_search": perplexity_search,
        "screenshot": screenshot,
    },
)

__all__ = ["web_crawl_skill"]
