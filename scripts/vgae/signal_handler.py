import signal
import logging
import sys

# Global variables to track stop signals
stop_training = False

def signal_handler(sig, frame):
    global stop_training
    logging.info("Stopping training gracefully...")
    stop_training = True

def register_signal_handlers():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
