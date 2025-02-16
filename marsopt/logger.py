import logging
import sys
from typing import Optional, Dict

# ANSI color codes for console output
COLORS = {
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "green": "\033[32m",
    "bold_white": "\033[1;37m",
    "reset": "\033[0m",
}


class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter that removes the log level and adds timestamps.

    This formatter ensures that logs appear with only the timestamp and message,
    without the log level (e.g., INFO, DEBUG).
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            The formatted log message.
        """
        return f"{self.formatTime(record, self.datefmt)} | {record.getMessage()}"


def setup_logger(name: str = "MARSOpt", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up and configure a logger.

    The logger prints messages to the console and optionally writes them to a file.
    It filters out INFO-level messages and applies color formatting.

    Parameters
    ----------
    name : str, optional
        The name of the logger (default is "MARSOpt").
    log_file : str, optional
        The file path to write logs to. If None, logging to a file is disabled.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        ColoredFormatter(fmt="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(console_handler)

    # File handler if log_file is provided (without colors)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)

    return logger


class OptimizationLogger:
    """
    Helper class for logging optimization progress.

    This class provides structured logging for optimization trials, 
    including parameters, objective function values, and execution time.

    Parameters
    ----------
    name : str, optional
        Name of the logger (default is "MARSOpt").
    log_file : str, optional
        Path to the log file. If None, logs are not written to a file.
    """

    def __init__(self, name: str = "MARSOpt", log_file: Optional[str] = None):
        self.logger = setup_logger(name=name, log_file=log_file)

    def log_trial(
        self,
        iteration: int,
        params: Dict[str, float],
        objective: float,
        time: float,
        best_iteration: int,
        best_value: float,
    ) -> None:
        """
        Logs the results of an optimization trial in a structured and readable format.

        This function logs key details of an optimization iteration, 
        including hyperparameters, objective function values, and the best result so far.

        Parameters
        ----------
        iteration : int
            The current trial number.
        params : Dict[str, float]
            The dictionary of hyperparameters used in the trial.
        objective : float
            The objective function value obtained in the trial.
        time : float
            The time taken for the trial (in seconds).
        best_iteration : int
            The iteration number that yielded the best objective value so far.
        best_value : float
            The best objective function value observed so far.

        Returns
        -------
        None
            Logs the information to the configured logger.
        """

        # Format parameters with colors
        params_str = "\n    ".join(
            [
                (
                    f"{COLORS['blue']}{k}{COLORS['reset']}: {v:.4f}"
                    if isinstance(v, float)
                    else f"{COLORS['blue']}{k}{COLORS['reset']}: {v}"
                )
                for k, v in params.items()
            ]
        )

        # Construct the log message
        log_message = (
            f"{COLORS['cyan']}Trial {iteration}{COLORS['reset']} | Elapsed Time: {time:.2f}s\n"
            f"{COLORS['yellow']}Objective: {objective:.4f}{COLORS['reset']}\n"
            f"Parameters:\n    {params_str}\n"
            f"{COLORS['green']}Best Iteration: {best_iteration}  Best Value: {best_value:.4f}{COLORS['reset']}\n"
        )

        self.logger.info(log_message)
