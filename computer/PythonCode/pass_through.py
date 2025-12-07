#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
from config import *
import numpy as np

class PassThrough(Py_Module):
    def nop(self, IN, CP, OUT):
        Frames = np.size(IN, 0)
        for frame in range(Frames):
            OUT[frame] = IN[frame]
        return 0

    def __init__(self, N):
        Py_Module.__init__(self)
        self.name = "py_passthrough"
        self.N = N

        t_nop = self.create_task("add_noise")

        s_nop_in = self.create_socket_in(t_nop, "X_N", N, np.float32)
        s_nop_sigma = self.create_socket_in(t_nop, "CP", 1, np.float32)
        s_nop_out = self.create_socket_out(t_nop, "Y_N", N, np.float32)

        self.create_codelet(t_nop, lambda slf, lsk, fid: slf.nop(lsk[s_nop_in], lsk[s_nop_sigma], lsk[s_nop_out]))