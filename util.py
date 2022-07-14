""" Utilities """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Logging
# =======

import logging
from colorlog import ColoredFormatter

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
#    datefmt='%H:%M:%S.%f',
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('rn')
log.setLevel(logging.DEBUG)
log.handlers = []       # No duplicated handlers
log.propagate = False   # workaround for duplicated logs in ipython
log.addHandler(ch)

logging.addLevelName(logging.INFO + 1, 'INFOV')
def _infov(self, msg, *args, **kwargs):
    self.log(logging.INFO + 1, msg, *args, **kwargs)

logging.Logger.infov = _infov


import time
from datetime import timedelta

def start_time_measure(message=None):
    if message:
        print(message)
    return time.monotonic()

def end_time_measure(start_time, print_prefix=None):
    end_time = time.monotonic()
    if print_prefix:
        print(print_prefix + str(timedelta(seconds=end_time - start_time)))
    return end_time