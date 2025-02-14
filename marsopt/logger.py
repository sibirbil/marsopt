import logging
import sys
from datetime import datetime
from typing import Optional

# ANSI color codes
COLORS = {
    'grey': '\033[38;21m',
    'blue': '\033[34m',
    'yellow': '\033[33m',
    'red': '\033[31m',
    'bold_red': '\033[31;1m',
    'green': '\033[32m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'reset': '\033[0m'
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    LEVEL_COLORS = {
        logging.DEBUG: COLORS['grey'],
        logging.INFO: COLORS['green'],
        logging.WARNING: COLORS['yellow'],
        logging.ERROR: COLORS['red'],
        logging.CRITICAL: COLORS['bold_red']
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Save original levelname
        original_levelname = record.levelname
        # Add color to the levelname
        color = self.LEVEL_COLORS.get(record.levelno, COLORS['white'])
        record.levelname = f"{color}{original_levelname}{COLORS['reset']}"
        
        # Add colors to trial information if present
        if hasattr(record, 'trial_info'):
            record.trial_info = f"{COLORS['cyan']}{record.trial_info}{COLORS['reset']}"
        if hasattr(record, 'params'):
            record.params = f"{COLORS['blue']}{record.params}{COLORS['reset']}"
        if hasattr(record, 'objective'):
            record.objective = f"{COLORS['yellow']}{record.objective}{COLORS['reset']}"
        if hasattr(record, 'time'):
            record.time = f"{COLORS['grey']}{record.time}{COLORS['reset']}"
            
        return super().format(record)

def setup_logger(
    name: str = "MARSOpt",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure the logger with color support
    
    Parameters
    ----------
    name : str
        Name of the logger (default: "MARSOpt")
    level : int
        Logging level (default: logging.INFO)
    log_file : Optional[str]
        Path to log file. If provided, logs will be written to this file
    format_string : Optional[str]
        Custom format string for log messages. If None, uses default format
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | "
            "%(trial_info)s %(params)s -> objective=%(objective)s "
            "(time=%(time)s)"
        )
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        ColoredFormatter(
            fmt=format_string,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(console_handler)
    
    # File handler if log_file is provided (without colors)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                fmt=format_string,
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(file_handler)
    
    return logger

class OptimizationLogger:
    """Helper class to log optimization progress"""
    
    def __init__(
        self,
        name: str = "MARSOpt",
        level: int = logging.INFO,
        log_file: Optional[str] = None
    ):
        self.logger = setup_logger(name=name, level=level, log_file=log_file)
    
    def log_trial(
        self,
        iteration: int,
        n_trials: int,
        params: dict,
        objective: float,
        time: float,
        level: int = logging.INFO
    ) -> None:
        """
        Log a trial's results
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        n_trials : int
            Total number of trials
        params : dict
            Dictionary of parameters and their values
        objective : float
            Objective function value
        time : float
            Trial execution time
        level : int
            Logging level (default: logging.INFO)
        """
        # Format parameters string
        params_str = ", ".join([
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in params.items()
        ])
        
        # Create extra fields for the log record
        extra = {
            'trial_info': f"[Trial {iteration+1}/{n_trials}]",
            'params': params_str,
            'objective': f"{objective:.4f}",
            'time': f"{time:.2f}s"
        }
        
        self.logger.log(level, "", extra=extra)