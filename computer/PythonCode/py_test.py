#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
from config import *
import numpy as np

class PyTest(Py_Module):

    def py_test(self, OutN):
        Frames = np.size(OutN, 0)
        for frame in range(Frames):
            for n in range(self.N):
                OutN[frame][n] = 0
            
            self.count += self.N
            if self.count > len(self.f_list)-self.N:
                self.toggle_done()
        return 0

    def __init__(self, N):
        Py_Module.__init__(self)
        self.name = "py_source_float"
        self.N = N
        self.count = 0

        print(self.n_frames)
        
        t_srcf = self.create_task("generate")

        s_out = self.create_socket_out(t_srcf, "OutN", N, np.float32)

        self.create_codelet(t_srcf, lambda slf, lsk, fid: slf.source_float(lsk[s_out]))