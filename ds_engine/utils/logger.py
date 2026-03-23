"""
ds_engine.utils.logger
======================

Centralized logging engine for DSEngine.
Uses a two-phase logging approach: logs to stdout until the output directory 
is created, then writes retroactively and continuously to run.log as well.
"""

import logging
import sys
import os
from datetime import datetime
from structlog import get_logger as _get_logger

# We will use standard logging library configured properly.
# structlog is not explicitly required in requirements.txt, so we'll 
# use python's built-in logging.

_FORMAT = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# A global buffer to hold messages before set_output_dir is called
class BufferedHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.buffer = []

    def emit(self, record):
        self.buffer.append(self.format(record))


_buffer_handler = BufferedHandler()
formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
_buffer_handler.setFormatter(formatter)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(formatter)

# Root logger setup
root_logger = logging.getLogger('ds_engine')
root_logger.setLevel(logging.INFO)
root_logger.addHandler(_buffer_handler)
root_logger.addHandler(_stream_handler)

_output_dir_set = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name."""
    # name is often the module name. For output consistency, strip 'ds_engine.' 
    # to shorten if desired, but we'll use name directly.
    return logging.getLogger(name)


def set_output_dir(output_dir: str) -> None:
    """Set the output directory for logging and flush the buffer to run.log.
    
    Args:
        output_dir (str): Absolute or relative path to the experiment's output directory.
    """
    global _output_dir_set
    if _output_dir_set:
        return

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'run.log')
    
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Flush existing buffer
    for msg in _buffer_handler.buffer:
        file_handler.stream.write(msg + '\n')
    file_handler.stream.flush()
    
    # Attach file handler and remove buffer
    root_logger.addHandler(file_handler)
    root_logger.removeHandler(_buffer_handler)
    
    _output_dir_set = True


def log_step_start(name: str) -> None:
    """Format and log the start of a step. (Stdout handled by pipeline_runner directly).
    
    Args:
        name (str): The name of the step.
    """
    logger = get_logger('step_runner')
    logger.info(f"Starting step: {name}")


def log_step_end(name: str, duration_seconds: float) -> None:
    """Format and log the completion of a step.
    
    Args:
        name (str): The name of the step.
        duration_seconds (float): Execution time.
    """
    logger = get_logger('step_runner')
    logger.info(f"Completed step: {name} in {duration_seconds:.2f}s")


def log_warning(message: str) -> None:
    """Log a warning message globally.
    
    Args:
        message (str): The warning to log.
    """
    logger = get_logger('ds_engine')
    logger.warning(message)


def log_error(message: str) -> None:
    """Log an error message globally.
    
    Args:
        message (str): The error to log.
    """
    logger = get_logger('ds_engine')
    logger.error(message)
