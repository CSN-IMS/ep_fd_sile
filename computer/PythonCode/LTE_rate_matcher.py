#!/usr/bin/env python3

import sys
import time
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import math

class LTE_rate_matcher(Py_Module):

    def lte_rm(self, x, y):
        Frames = self.n_frames

        for frame in range(Frames):
            y[frame] = x[frame][self.buf_ind-1]
        return 0

    def __init__(self, N, M, buf_ind):
        Py_Module.__init__(self)
        self.name = "py_LTE_rate_matcher"
        self.N = N
        self.M = M
        self.buf_ind = np.array(buf_ind)

        t_rm = self.create_task("rate_match")

        s_x = self.create_socket_in(t_rm, "x", N, np.int32)
        s_y = self.create_socket_out(t_rm, "y", M, np.int32)

        self.create_codelet(t_rm, lambda slf, lsk, fid: slf.lte_rm(lsk[s_x], lsk[s_y]))