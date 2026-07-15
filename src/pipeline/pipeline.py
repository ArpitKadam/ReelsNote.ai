from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from src.state.states import FinalState
from src.helpers.reel_scrapper import Scraper
from src.helpers import video_explainer, report_generator
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_scraper = Scraper()  # Ensure Whisper is loaded at most once per process.


def _scrape_node(state: FinalState) -> dict:
    logger.info("▶️ Node: scrape")
    return _scraper.scrape(state["url"])


def _explain_node(state: FinalState) -> dict:
    logger.info("▶️ Node: explain")
    return {"video_explanation": video_explainer.describe(dict(state))}


def _report_node(state: FinalState) -> dict:
    logger.info("▶️ Node: report")
    return {"report": report_generator.generate(dict(state))}


def build_graph():
    """Build and compile the agentic reel pipeline graph."""
    
    graph = StateGraph(FinalState)

    graph.add_node("scrape", _scrape_node)
    graph.add_node("explain", _explain_node)
    graph.add_node("report", _report_node)

    graph.add_edge(START, "scrape")
    graph.add_edge("scrape", "explain")
    graph.add_edge("explain", "report")
    graph.add_edge("report", END)

    return graph.compile()
