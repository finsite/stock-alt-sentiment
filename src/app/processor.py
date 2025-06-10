"""Processor module for alternative sentiment analysis (e.g., Reddit, Stocktwits, Twitter)."""

from typing import Any

from textblob import TextBlob

from app.utils.setup_logger import setup_logger
from app.utils.types import validate_dict

# Initialize module logger
logger = setup_logger(__name__)


def process(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Process a batch of messages for sentiment analysis.

    Each message must include a 'text' field. Sentiment polarity is computed,
    and both 'sentiment_score' and 'sentiment_label' are added to the output.

    Parameters
    ----------
    payloads : list[dict[str, Any]]
        List of incoming messages to process.

    Returns
    -------
    list[dict[str, Any]]
        Enriched messages with sentiment data.
    """
    results: list[dict[str, Any]] = []

    for item in payloads:
        if not validate_dict(item, ["text"]):
            logger.warning("⚠️ Skipping message: missing required 'text' field: %s", item)
            continue

        text = item.get("text", "")
        try:
            blob = TextBlob(str(text))  # Ensure proper string input
            polarity = blob.sentiment.polarity  # type: ignore[attr-defined]

            label: str
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"

            item["sentiment_score"] = polarity
            item["sentiment_label"] = label
            results.append(item)
        except Exception as e:
            logger.exception("❌ Sentiment processing failed for text: %s | Error: %s", text, e)

    return results
