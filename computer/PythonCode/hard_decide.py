#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
from config import *
import numpy as np

class HardDecide(Py_Module):

    def hard_decide(self, LLR, bit):
        bit_array = np.rint(np.sign(LLR)*(-0.5) +0.5).astype(np.int32)
        Frames = np.size(LLR, 0)
        for frame in range(Frames):
            bit[frame] = bit_array[frame]
                    
        return 0

    def __init__(self, N):
        Py_Module.__init__(self)
        self.name = "py_hard_decide"
        self.N = N

        t_hd = self.create_task("decide")

        s_LLR = self.create_socket_in(t_hd, "LLR", N, np.float32)
        s_bit = self.create_socket_out(t_hd, "bit", N, np.int32)

        self.create_codelet(t_hd, lambda slf, lsk, fid: slf.hard_decide(lsk[s_LLR], lsk[s_bit]))