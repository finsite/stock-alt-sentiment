"""Repo-specific configuration for stock-alt-geolocation."""

from app.config_shared import *


def get_poller_name() -> str:
    """Return the name of the poller for this service."""
    return get_config_value("POLLER_NAME", "stock_alt_geolocation")


def get_rabbitmq_queue() -> str:
    """Return the RabbitMQ queue name for this poller."""
    return get_config_value("RABBITMQ_QUEUE", "stock_alt_geolocation_queue")


def get_dlq_name() -> str:
    """Return the Dead Letter Queue (DLQ) name for this poller."""
    return get_config_value("DLQ_NAME", "stock_alt_geolocation_dlq")
