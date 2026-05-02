import logging
from src.config import AUDIT_LOG


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(AUDIT_LOG)
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    return logger
