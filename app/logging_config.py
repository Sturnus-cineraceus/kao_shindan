"""Shared, idempotent application logging configuration."""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    path = Path(os.getenv("LOG_FILE", "/app/logs/app.log"))
    max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))
    backups = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    root = logging.getLogger()
    signature = (str(path), max_bytes, backups)
    if getattr(root, "_kao_log_signature", None) != signature:
        for handler in root.handlers[:]:
            if getattr(handler, "_kao_handler", False):
                root.removeHandler(handler)
                handler.close()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        console = logging.StreamHandler()
        console._kao_handler = True
        console.setFormatter(formatter)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
        file_handler._kao_handler = True
        file_handler.setFormatter(formatter)
        root.addHandler(console)
        root.addHandler(file_handler)
        root._kao_log_signature = signature
    root.setLevel(level)
    return logging.getLogger("app")
