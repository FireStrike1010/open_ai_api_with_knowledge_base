from fastapi import Request
from loguru import logger
import os


def get_state(request: Request):
    return request.app.state

class LoggerManager(logger.__class__):
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)  # Ensure log folder exists
        log_file_path = os.path.join(self.log_dir, "{time:YYYY-MM-DD}.log")
        logger.add(
            log_file_path,
            rotation="1 day",  
            retention="7 days",  
            compression="zip",
            level="INFO")
        self.logger = logger