from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from src.state.states import FinalState
from src.helpers.reel_scrapper import Scraper
from src.helpers import video_explainer, report_generator, pdf_converter
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_scraper = Scraper()


def _scrape_node(state: FinalState) -> dict:
    logger.info("▶️ Node: scrape")
    return _scraper.scrape(state["url"])


def _explain_node(state: FinalState) -> dict:
    logger.info("▶️ Node: explain")
    explanation, question = video_explainer.describe(dict(state))
    return {"video_explanation": explanation, "question": question}


def _report_node(state: FinalState) -> dict:
    logger.info("▶️ Node: report")
    return {"report": report_generator.generate(dict(state))}


def _pdf_node(state: FinalState) -> dict:
    logger.info("▶️ Node: pdf")
    return {"notes_pdf_path": pdf_converter.convert(dict(state))}


def build_graph():
    """Build and compile the agentic reel pipeline graph."""

    graph = StateGraph(FinalState)

    graph.add_node("scrape", _scrape_node)
    graph.add_node("explain", _explain_node)
    graph.add_node("report", _report_node)
    graph.add_node("pdf", _pdf_node)

    graph.add_edge(START, "scrape")
    graph.add_edge("scrape", "explain")
    graph.add_edge("explain", "report")
    graph.add_edge("report", "pdf")
    graph.add_edge("pdf", END)

    return graph.compile()
