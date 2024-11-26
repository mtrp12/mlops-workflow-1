import os
import sys
import logging
from datetime import datetime
from src.logger.custom_logger import CustomLogger

LOG_FILE = f"mlops-workflow-1-{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="==> %(asctime)s - %(name)s - %(levelname)7s -  %(filename)s:%(lineno)d - %(message)s",
    level=logging.DEBUG
)

logging.setLoggerClass(CustomLogger)


if __name__ == "__main__":
    pass