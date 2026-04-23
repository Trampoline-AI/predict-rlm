import dspy

from predict_rlm import File

from .schema import EventOntology


class CrawlEvent(dspy.Signature):
    """Analyze an event page to produce an executive summary.

    IMPORTANT: Focus exclusively on the specific event at the given URL.
    Do NOT include information from other events, past editions, or the
    broader organization. Every fact in the report must come from this
    event's page or from Perplexity research about people attending
    THIS event.

    1. **Browse the event page** using browse(url). It returns raw HTML.
       Parse it to understand the event: name, date, location, theme, and
       overall structure. Print a summary of what you found.

    2. **Take a screenshot** of the event page using screenshot(url).
       Pass the screenshot to predict() with a dspy.Image field to visually
       inspect the page — look for attendee avatars, sponsor logos, and any
       content rendered by JavaScript that may not appear in the raw HTML.

    3. **Extract attendees from the HTML** — event platforms embed attendee
       info in the raw HTML even when it's only visible on hover. Look for
       avatar elements, tooltip attributes (title, aria-label, alt, data-*),
       and nearby text nodes that contain attendee names. Use predict() on
       the raw HTML to find every person name hidden in attributes or
       markup. Also take a screenshot(url, full_page=True) of the event
       page and use predict() with dspy.Image to visually identify attendee
       names on avatars.

    4. **Extract structured information** using predict() on the page's
       content. For speakers, extract: name, title, organization,
       talk topic, session time, bio snippet. For attendee lists, extract
       names and affiliations.

    5. **Research every attendee with Perplexity** — for each attendee
       name found, call perplexity_search() to find their current role,
       company, domain of expertise, and notable achievements. Use
       asyncio.gather() to run searches in parallel batches. Also enrich
       speakers the same way. When searching, include the event's city
       or country in the query — attendees almost always live and work
       near the event location, so this disambiguates common names.

    6. **Cluster attendees by domain/expertise** — based on the research
       results, group attendees into meaningful clusters. Examples:
       "AI/ML Engineers", "Founders & CTOs", "Researchers / Academia",
       "Product & Design", "Investors / VCs", "Data Scientists", etc.
       Let the actual data drive the clusters — don't force categories.
       For each cluster, list members with a one-line description.

    7. **Write the executive summary** as a markdown file to
       /sandbox/output/report/summary.md. Structure the report with
       sections matching the ontology categories. Include:
       - Event overview (name, date, location, theme, expected audience)
       - Speakers & presentations (who, what they'll present, their background)
       - Attendees by cluster (grouped by domain/expertise, each person
         with role, company, and why they're relevant)
       - Schedule highlights (key sessions, timing)
       - Sponsors & organizers (who's backing the event)
    """

    url: str = dspy.InputField(desc="URL of the event page to analyze")
    ontology: EventOntology = dspy.InputField(
        desc="Categories of information to extract — each becomes a report section"
    )
    report: File = dspy.OutputField(
        desc="Markdown executive summary file written to /sandbox/output/report/"
    )
