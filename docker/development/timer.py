#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Benjamin Milde'

#http://www.huyng.com/posts/python-performance-analysis/
import time

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        self.secs = self.end_time - self.start_time
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % self.msecs)

    def current_secs(self):
        self.stop()
        return self.secs
