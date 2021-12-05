import logging
from time import time
from datetime import datetime


class DuplicateFilter(logging.Filter):
    def filter(self, record):
        current_log = (record.module, record.levelno, record.msg)
        if current_log != getattr(self, "last_log", None):
            self.last_log = current_log
            return True
        return False


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print(f'Elapsed time is {time() - start} ms')
        return result
    return wrapper


def logger(func):
    def wrapper(*args, **kwargs):
        print('_' * 25)
        print(f'Run on: {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
        print(func.__name__)
        func(*args, **kwargs)
        print('_' * 25)
    return wrapper

