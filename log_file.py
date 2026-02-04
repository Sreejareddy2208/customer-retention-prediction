import logging
import os

def setup_logging(script_name):
    log_dir = r"C:\Users\SREEJA REDDY\OneDrive\Attachments\Desktop\customer churn\logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    log_file = os.path.join(log_dir, f"{script_name}.log")

    handler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    logger.propagate = False

    return logger
