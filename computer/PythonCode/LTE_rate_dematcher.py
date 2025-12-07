#!/usr/bin/env python3

import sys
import time
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import math

class LTE_rate_dematcher(Py_Module):

    def lte_rd(self, x, y):
        Frames = self.n_frames

        for frame in range(Frames):
            Cbuf_ = np.pad(x[frame], (self.cyc_shift, self.M-self.N-self.cyc_shift))
            y[frame] = Cbuf_[self.buf_ind-1]
        return 0

    def __init__(self, N, M, buf_ind, cyc_shift):
        Py_Module.__init__(self)
        self.name = "py_LTE_rate_dematcher"
        self.N = N
        self.M = M
        self.buf_ind = np.array(buf_ind)
        self.cyc_shift = cyc_shift

        t_rm = self.create_task("rate_dematch")

        s_x = self.create_socket_in(t_rm, "x", N, np.float32)
        s_y = self.create_socket_out(t_rm, "y", M, np.float32)

        self.create_codelet(t_rm, lambda slf, lsk, fid: slf.lte_rd(lsk[s_x], lsk[s_y]))