"""Application logging and error capture.

Streamlit shows a traceback in the browser when a script raises. That is right
for development and wrong for production, where it leaks file paths and code
to whoever hit the error. `report_error` writes the detail to the log, where
operators can find it, and returns a short reference the person can quote.

Logs go to stderr so whatever runs the process (systemd, Docker, the platform
log drain) collects them. Set POTHOLE_LOG_LEVEL to change verbosity.
"""

from __future__ import annotations

import logging
import os
import secrets
import sys

LOGGER_NAME = "divot"
_DEFAULT_LEVEL = "INFO"
_configured = False


def configure_logging() -> logging.Logger:
    """Attach a stderr handler once and return the application logger."""

    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    if _configured:
        return logger

    level = os.getenv("POTHOLE_LOG_LEVEL", _DEFAULT_LEVEL).upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level, logging.INFO))
    # The root logger already has Streamlit's handler; without this the same
    # line is printed twice.
    logger.propagate = False
    _configured = True
    return logger


def get_logger() -> logging.Logger:
    """Return the application logger, configuring it on first use."""

    return configure_logging()


def report_error(message: str, error: BaseException) -> str:
    """Log an exception with a reference id and return that id.

    Show the id to the person who hit the error; keep the traceback in the log.
    """

    reference = secrets.token_hex(4)
    # exc_info takes the exception explicitly rather than reading the
    # current one, so this also works when called outside an except block.
    get_logger().error("%s [ref=%s]: %s", message, reference, error, exc_info=error)
    return reference


def log_auth_event(event: str, username: str, **fields: object) -> None:
    """Record a sign-in related event.

    Usernames are recorded because an auth trail without a subject is not much
    of a trail. Passwords never are.
    """

    detail = " ".join(f"{key}={value}" for key, value in fields.items())
    get_logger().info("auth %s user=%s %s", event, username or "<blank>", detail)
