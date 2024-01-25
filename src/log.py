import sys
import logging
import os

def setup_logger(
    filename: str,
    loggername: str = "my_logger",
):
    if os.path.exists(filename):
        os.remove(filename)

    class LogStream(object):
        def __init__(self, logger: logging.Logger, log_level=logging.INFO):
            self.logger = logger
            self.log_level = log_level

        def write(self, msg):
            if msg.rstrip() != "":
                self.logger.log(self.log_level, msg.rstrip())

        def flush(self):
            for handler in self.logger.handlers:
                handler.flush()

    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    sys.stdout = LogStream(logger, logging.INFO)
    sys.stderr = LogStream(logger, logging.ERROR)

loggername = "my_logger"
filename = "output.log"

setup_logger(filename, loggername)

import time
for i in range(10):
    print(i)
    time.sleep(1)

1/0