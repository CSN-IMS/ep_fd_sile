#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import config
import time

class Block3(Py_Module):

    def __hard_decide(self, x_e):
        d1 = np.abs(x_e.imag[:]) > np.abs(x_e.real[:])
        d2 = x_e.real[:] < 0
        d3 = x_e.imag[:] < 0
        m = 4*d3[:] + 2*d2[:] + 1*d1[:]
        return m

    def __create_LUT(self, X_value):
        LUT_8psk = np.zeros((3, 8), dtype=np.complex_)
        diffSign = np.zeros((3,8))
        for q in range(3):
            for i in range(8):
                LUT_8psk[q][i] = 2*(1-2*((i&(1<<q))>0))*(X_value[i] - X_value[self.bin_refl_8PSK[i][q]])

        return LUT_8psk

    def block3(self, x_e, v_e, Le):
        Frames = self.n_frames

        for frame in range(Frames):
            x_e_c = np.empty(self.K, dtype=np.complex_)
            x_e_c.real[:] = x_e[frame][::2]
            x_e_c.imag[:] = x_e[frame][1::2]

            m = self.__hard_decide(x_e_c)
            da1 = self.LUT_8PSK[0][m[:]]
            da2 = self.LUT_8PSK[1][m[:]]
            da3 = self.LUT_8PSK[2][m[:]]

            Le_d_kq = np.empty((self.K * self.Q))
            Le_d_kq[0::3] = (x_e_c.real[:]*da1.real[:] + x_e_c.imag[:]*da1.imag[:]) / v_e[frame]
            Le_d_kq[1::3] = (x_e_c.real[:]*da2.real[:] + x_e_c.imag[:]*da2.imag[:]) / v_e[frame]
            Le_d_kq[2::3] = (x_e_c.real[:]*da3.real[:] + x_e_c.imag[:]*da3.imag[:]) / v_e[frame]

            Le[frame] = np.reshape(Le_d_kq, self.N)
        return 0

    def __init__(self, N, Q, X_value = config.X_value_8PSK_2):
        Py_Module.__init__(self)
        self.name = "py_Block3_8PSK"
        self.N = N
        self.Q = Q
        self.K = N // Q
        self.X = 2**Q
        self.bin_refl_8PSK = [[1, 3, 4], [0, 3, 4], [3, 1, 6], [2, 1, 6], [5, 7, 0], [4, 7, 0], [7, 5, 2], [6, 5, 2]]
        self.LUT_8PSK = self.__create_LUT(X_value)

        t_b3 = self.create_task("decode")

        s_x_e = self.create_socket_in(t_b3, "x_e", 2*N//Q, np.float32)
        s_v_e = self.create_socket_in(t_b3, "v_e", 1, np.float32)
        s_Le = self.create_socket_out(t_b3, "Le", N, np.float32)

        self.create_codelet(t_b3, lambda slf, lsk, fid: slf.block3(lsk[s_x_e], lsk[s_v_e], lsk[s_Le]))