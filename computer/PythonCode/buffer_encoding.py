#!/usr/bin/env python3

import sys
import time
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import math

class BufferEncoding(Py_Module):

    def buffer_encoding(self, x, y):
        Frames = self.n_frames

        for frame in range(Frames):
            sys_n = x[frame][0:2*self.K:2]
            par_n = x[frame][1:2*self.K:2]
            tail_sys_n = x[frame][2*self.K:2*self.K+self.tail_len:2]
            tail_par_n = x[frame][1+2*self.K:2*self.K+self.tail_len:2]
            par_i = x[frame][2*self.K+self.tail_len:3*self.K+self.tail_len]
            tail_sys_i = x[frame][3*self.K+self.tail_len::2]
            tail_par_i = x[frame][1+3*self.K+self.tail_len::2]

            y[frame] = np.concatenate((sys_n, tail_sys_n, par_n, tail_par_n, tail_sys_i, par_i, tail_par_i))
        return 0

    def __init__(self, K, tail_len):
        Py_Module.__init__(self)
        self.name = "py_buffer_encoding"
        self.K = K
        self.tail_len = tail_len

        t_benc = self.create_task("buf_enc")

        s_x = self.create_socket_in(t_benc, "x", 3*K+2*tail_len, np.float32)
        s_y = self.create_socket_out(t_benc, "y", 3*K+2*tail_len, np.float32)

        self.create_codelet(t_benc, lambda slf, lsk, fid: slf.buffer_encoding(lsk[s_x], lsk[s_y]))