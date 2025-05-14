import logging
import sys
from app.config import LOG_LEVEL

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

def setup_logging():
    """
    Configure the logging system with a simple, readable format
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVELS.get(LOG_LEVEL, logging.INFO))
    
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)
    
    # Create console handler with a standard formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    
    root_logger.addHandler(console_handler)
    
    # Create application logger
    app_logger = logging.getLogger("statista-api")
    app_logger.setLevel(LOG_LEVELS.get(LOG_LEVEL, logging.INFO))
    
    return app_logger 