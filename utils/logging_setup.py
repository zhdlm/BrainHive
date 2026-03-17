import logging
from pathlib import Path
from datetime import datetime

def setup_logging():

    #Set path to save logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logfile = log_dir / f"Log_{datetime.now():%Y%m%d_%H%M%S}.log"

    # Create logger instance & set level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # File handler (DEBUG and above)
    formatter_file = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(lineno)d | %(message)s"
    )
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter_file)

    # Stream handler (INFO and above)
    formatter_console = logging.Formatter(
        "%(levelname)s | %(name)s | %(funcName)s | %(lineno)d | %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter_console)

    #Add logger to stream and file
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    #Show DEBUG level only for the project packages
    logging.getLogger("src").setLevel(logging.DEBUG)
    logging.getLogger("utils").setLevel(logging.DEBUG)
    logging.getLogger("scripts").setLevel(logging.INFO) 

    return logfile