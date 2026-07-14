from __future__ import annotations
import logging
import sys

logging_done = False


class ColorFormatter(logging.Formatter):
    """Logging formatter that colorizes console output by log level.

    Each log record is formatted using the same timestamp and message layout,
    with ANSI escape sequences applied to improve readability in terminals.
    """
    
    # ANSI Escape Sequences for Colors
    cyan = "\x1b[36m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    log_format = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
    date_format = "%H:%M:%S"

    LEVEL_COLORS = {
        logging.DEBUG: cyan,
        logging.INFO: blue,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, self.reset)
        formatter = logging.Formatter(
            fmt=f"{color}{self.log_format}{self.reset}",
            datefmt=self.date_format
        )
        return formatter.format(record)


def configure_logging(level=logging.INFO):
    """Configure the root logger for console output.

    Initializes a single stdout handler with the project's color formatter.
    Subsequent calls are ignored to prevent duplicate handlers from being
    attached to the root logger.
    """

    global logging_done

    if logging_done:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter())
    root_logger.addHandler(console_handler)

    logging_done = True


def get_logger(name: str, level=logging.INFO):
    """Return a named logger.

    Ensures the application's logging system is configured before returning
    the requested logger instance.
    """
    
    configure_logging(level)
    return logging.getLogger(name)

if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")