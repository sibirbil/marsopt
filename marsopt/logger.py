import logging
import sys
from datetime import datetime
from typing import Optional

# ANSI color codes
COLORS = {
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "green": "\033[32m",
    "bold_white": "\033[1;37m",
    "reset": "\033[0m",
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter that removes INFO and adds colors."""

    def format(self, record: logging.LogRecord) -> str:
        return f"{self.formatTime(record, self.datefmt)} | {record.getMessage()}"  # 🔥 HATA DÜZELTİLDİ


def setup_logger(name: str = "MARSOpt", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up and configure the logger without INFO and with colors.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(fmt="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)

    # File handler if log_file is provided (without colors)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(fmt="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

    return logger


class OptimizationLogger:
    """Helper class to log optimization progress"""

    def __init__(self, name: str = "MARSOpt", log_file: Optional[str] = None):
        self.logger = setup_logger(name=name, log_file=log_file)

    def log_trial(
        self,
        iteration: int,
        params: dict,
        objective: float,
        time: float,
        best_iteration: int,
        best_value: float,
    ) -> None:
        """
        Logs the results of an optimization trial in a structured and readable format with colors.
        """

        # Format parameters with color
        params_str = "\n    ".join(
            [f"{COLORS['blue']}{k}{COLORS['reset']}: {v:.4f}" if isinstance(v, float) else f"{COLORS['blue']}{k}{COLORS['reset']}: {v}"
             for k, v in params.items()]
        )

        # Log mesajını belirlenen formata uygun oluştur
        log_message = (
            f"{COLORS['cyan']}Trial {iteration}{COLORS['reset']} | Elapsed Time: {time:.2f}s\n"
            f"{COLORS['yellow']}Objective: {objective:.4f}{COLORS['reset']}\n"
            f"Parameters:\n    {params_str}\n"
            f"{COLORS['green']}Best Iteration: {best_iteration}  Best Value: {best_value:.4f}{COLORS['reset']}\n"
        )

        self.logger.info(log_message)
