import logging
from logging.handlers import RotatingFileHandler

# *Logging Levels:
# *DEBUG: Detailed information, typically of interest only when diagnosing problems.
# *INFO: Confirmation that things are working as expected.
# *WARNING: An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.
# *ERROR: Due to a more serious problem, the software has not been able to perform some function.
# *CRITICAL: A serious error, indicating that the program itself may be unable to continue running.


# !USE logger.exception() instead of logger.ERROR() to get the Traceback of the error.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s')


file_handler = RotatingFileHandler('decathlon_forecast_pipeline.log' , mode='a', maxBytes=10*1024*1024, 
                                        backupCount=10, encoding=None, delay=0)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)