"""EventCrawler — RLM service for analyzing event pages and producing summaries.

Usage::

    crawler = EventCrawler(sub_lm="openai/gpt-5.1")
    result = await crawler.aforward(url="https://example.com/conference")
    # result.report is a File pointing to the generated markdown
"""

import dspy

from predict_rlm import PredictRLM

from .schema import EventOntology
from .signature import CrawlEvent
from .skills import web_crawl_skill


class EventCrawler(dspy.Module):

    def __init__(
        self,
        sub_lm: dspy.LM | str | None = None,
        max_iterations: int = 40,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.sub_lm = sub_lm
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.debug = debug

    async def aforward(
        self,
        url: str,
        ontology: EventOntology | None = None,
    ):
        ontology = ontology or EventOntology()
        predictor = PredictRLM(
            CrawlEvent,
            sub_lm=self.sub_lm,
            skills=[web_crawl_skill],
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            debug=self.debug,
        )
        return await predictor.acall(url=url, ontology=ontology)
