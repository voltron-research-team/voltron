import logging
import sys

def setup_logging(debug, log_file=None):
    log_level = logging.DEBUG if debug else logging.INFO

    if log_file:
        logging.basicConfig(level=log_level,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_file),
                                logging.StreamHandler(sys.stdout)
                            ])
    else:
        logging.basicConfig(level=log_level,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.StreamHandler(sys.stdout)
                            ])

    logging.info(f"Logging is set up with mode: {'DEBUG' if debug else 'INFO'}")