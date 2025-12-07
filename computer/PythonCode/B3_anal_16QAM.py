#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import itertools
import time

class Block3(Py_Module):

    def block3(self, x_e, v_e, Le):
        Frames = self.n_frames

        for frame in range(Frames):
            x_e_c = np.empty(self.K, dtype=np.complex_)
            x_e_c.real[:] = x_e[frame][::2]
            x_e_c.imag[:] = x_e[frame][1::2]

            mult_const = 4*self.d/v_e[frame]

            Le_d_kq = np.empty((self.K * self.Q))
            Le_d_kq[3::4] = mult_const*(2*self.d -np.abs(x_e_c.imag[:]))
            Le_d_kq[2::4] = mult_const*(2*self.d -np.abs(x_e_c.real[:]))

            if_bigger_d_1 = x_e_c.imag > 2*self.d
            if_lesser_d_1 = x_e_c.imag < -2*self.d
            Le_d1_bigger = if_bigger_d_1 * (x_e_c.imag -2*self.d)
            Le_d1_lesser = if_lesser_d_1 * (x_e_c.imag +2*self.d)
            Le_d_1 = x_e_c.imag + Le_d1_bigger + Le_d1_lesser
            Le_d_kq[1::4] = mult_const*Le_d_1

            if_bigger_d_3 = x_e_c.real > 2*self.d
            if_lesser_d_3 = x_e_c.real < -2*self.d
            Le_d3_bigger = if_bigger_d_3 * (x_e_c.real -2*self.d)
            Le_d3_lesser = if_lesser_d_3 * (x_e_c.real +2*self.d)
            Le_d_3 = x_e_c.real + Le_d3_bigger + Le_d3_lesser
            Le_d_kq[0::4] = mult_const*Le_d_3

            Le[frame] = np.reshape(Le_d_kq, self.N)
        return 0

    def __init__(self, N, Q):
        Py_Module.__init__(self)
        self.name = "py_Block3_16QAM"
        self.N = N
        self.Q = Q
        self.K = N // Q
        self.X = 2**Q
        self.d = 1/np.sqrt(10)

        t_b3 = self.create_task("decode")

        s_x_e = self.create_socket_in(t_b3, "x_e", 2*N//Q, np.float32)
        s_v_e = self.create_socket_in(t_b3, "v_e", 1, np.float32)
        s_Le = self.create_socket_out(t_b3, "Le", N, np.float32)

        self.create_codelet(t_b3, lambda slf, lsk, fid: slf.block3(lsk[s_x_e], lsk[s_v_e], lsk[s_Le]))