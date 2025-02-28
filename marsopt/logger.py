import logging
import sys
from typing import Optional, Dict


class SimpleFormatter(logging.Formatter):
    """
    Custom logging formatter that logs messages in a single-line format.
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
        return f"[I {self.formatTime(record, self.datefmt)}, {record.msecs:.0f}] {record.getMessage()}"


def setup_logger(name: str = "marsopt", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up and configure a logger.

    The logger prints messages to the console and optionally writes them to a file.

    Parameters
    ----------
    name : str, optional
        The name of the logger (default is "marsopt").
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

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        SimpleFormatter(fmt="%(asctime)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(console_handler)

    # File handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            SimpleFormatter(fmt="%(asctime)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)

    return logger


class OptimizationLogger:
    """
    Helper class for logging optimization progress.

    Parameters
    ----------
    name : str, optional
        Name of the logger (default is "marsopt").
    log_file : str, optional
        Path to the log file. If None, logs are not written to a file.
    """

    def __init__(self, name: str = "marsopt", log_file: Optional[str] = None):
        self.logger = setup_logger(name=name, log_file=log_file)

    def log_start(self, n_trials: int) -> None:
        """
        Logs the start of the optimization process.

        Parameters
        ----------
        n_trials : int
            The total number of trials for optimization.

        Returns
        -------
        None
        """
        # Format search space as a dictionary-like string

        log_message = (
            f"Optimization started with {n_trials} trials."
        )

        self.logger.info(log_message)

    def log_trial(
        self,
        iteration: int,
        variables: Dict[str, float],
        objective: float,
        best_iteration: int,
        best_value: float,
    ) -> None:
        """
        Logs the results of an optimization trial in a single-line format.

        Parameters
        ----------
        iteration : int
            The current trial number.
        variables : Dict[str, float]
            The dictionary of variables used in the trial.
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
        """
        # Format variables in a single-line dictionary format
        var_str = ", ".join(f"'{k}': {v}" if isinstance(v, float) else f"'{k}': {v}" for k, v in variables.items())

        # Construct the log message in a single line
        log_message = (
            f"Trial {iteration} finished with value: {objective} and variables: {{{var_str}}}. "
            f"Best is trial {best_iteration} with value: {best_value}."
        )

        self.logger.info(log_message)
