#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import take_closest
import time

class Block3(Py_Module):    
    def block3(self, x_e, v_e, Le):
        Frames = self.n_frames

        for frame in range(Frames):
            # print(v_e)
            # print(x_e)
            # time.sleep(1)
            Le_d_kq = 2*np.sqrt(2)*x_e[frame][:]/v_e[frame]
            Le_d_kq = np.reshape(Le_d_kq, (self.K,self.Q))
            Le_d_kq = np.flip(Le_d_kq, axis=1)
            Le[frame] = np.reshape(Le_d_kq, self.N)            
        return 0

    def __init__(self, N, Q):
        Py_Module.__init__(self)
        self.name = "py_Block3_4QAM"
        self.N = N
        self.Q = Q
        self.K = N // Q
        self.X = 2**Q

        t_b3 = self.create_task("decode")

        s_x_e = self.create_socket_in(t_b3, "x_e", 2*N//Q, np.float32)
        s_v_e = self.create_socket_in(t_b3, "v_e", 1, np.float32)
        s_Le = self.create_socket_out(t_b3, "Le", N, np.float32)

        self.create_codelet(t_b3, lambda slf, lsk, fid: slf.block3(lsk[s_x_e], lsk[s_v_e], lsk[s_Le]))