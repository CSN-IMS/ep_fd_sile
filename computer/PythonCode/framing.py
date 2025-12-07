#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
from config import *
import numpy as np

class Framing(Py_Module):

    def add_frame(self, IN, OUT):
        Frames = np.size(IN, 0)
        for frame in range(Frames):
            OUT[frame][0:self.frame_size] = IN[frame][self.N-self.frame_size:self.N]
            OUT[frame][self.frame_size:self.frame_size+self.N] = IN[frame][0:self.N]

        return 0

    def remove_frame(self, IN, OUT):
        Frames = np.size(IN, 0)
        for frame in range(Frames):
            OUT[frame] = IN[frame][self.frame_size:self.frame_size+self.N]

        return 0

    def __init__(self, N, frame_size):
        Py_Module.__init__(self)
        self.name = "py_framing"
        self.N = N
        self.frame_size = frame_size

        t_frm = self.create_task("frame")

        s_frm_in = self.create_socket_in(t_frm, "IN", N, np.float32)
        s_frm_out = self.create_socket_out(t_frm, "OUT", N+frame_size, np.float32)

        self.create_codelet(t_frm, lambda slf, lsk, fid: slf.add_frame(lsk[s_frm_in], lsk[s_frm_out]))

        t_dfrm = self.create_task("deframe")

        s_dfrm_in = self.create_socket_in(t_dfrm, "IN", N+frame_size, np.float32)
        s_dfrm_out = self.create_socket_out(t_dfrm, "OUT", N, np.float32)

        self.create_codelet(t_dfrm, lambda slf, lsk, fid: slf.remove_frame(lsk[s_dfrm_in], lsk[s_dfrm_out]))