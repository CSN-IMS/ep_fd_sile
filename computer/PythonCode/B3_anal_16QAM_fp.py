#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
from fxpmath import Fxp
import take_closest
import tables_anal
import time

class Block3(Py_Module):

    def get_vinv(self, SNR):
        return tables_anal.table_B3_16QAM_fp[take_closest.take_closest(tables_anal.SNR_16QAM, SNR)]

    def block3(self, x_e, v_e, Le):
        Frames = self.n_frames

        for frame in range(Frames):
            # n_int = 16
            # n_frac = 6
            x_fp = Fxp(x_e[frame], signed=True, n_word=8, n_frac=5, rounding='floor').get_val()
            # x_fp = Fxp(x_e[frame], signed=True, n_word=1+n_int+n_frac, n_frac=n_frac, rounding='floor').get_val()
            # print(Fxp(v_e, signed=False, n_word=8, n_frac=7, rounding="floor").raw().astype(np.uint8))
            # print(Fxp(x_e[frame], signed=True, n_word=8, n_frac=5, rounding='floor').raw().astype(np.uint8))
            # time.sleep(1)
            x_e_c = np.empty(self.K, dtype=np.complex_)
            x_e_c.real[:] = x_fp[::2]
            x_e_c.imag[:] = x_fp[1::2]

            v_e_fp = Fxp(v_e, signed=False, n_word=8, n_frac=7, rounding="floor").get_val()
            mult_const = self.get_vinv(10*np.log10(v_e_fp)).get_val()
            # print(self.get_vinv(10*np.log10(v_e[frame])).raw().astype(np.uint8))
            # time.sleep(1)

            Le_d_kq = np.empty((self.K * self.Q))
            Le_d_kq[3::4] = mult_const*Fxp((Fxp(2*self.d, signed=True, n_word=8, n_frac=5, rounding='floor').get_val() -np.abs(x_e_c.imag[:])), signed=True, n_word=8, n_frac=5, rounding='floor').get_val()
            Le_d_kq[2::4] = mult_const*Fxp((Fxp(2*self.d, signed=True, n_word=8, n_frac=5, rounding='floor').get_val() -np.abs(x_e_c.real[:])), signed=True, n_word=8, n_frac=5, rounding='floor').get_val()
            # print(Fxp(mult_const, signed=False, n_word=8, n_frac=4).raw().astype(np.uint8))
            # print(Fxp((2*self.d -np.abs(x_e_c.real[:])), signed=True, n_word=8, n_frac=5, rounding='floor').raw().astype(np.uint8))


            if_bigger_d_1 = x_e_c.imag > 2*self.d
            if_lesser_d_1 = x_e_c.imag < -2*self.d
            Le_d1_bigger = Fxp(if_bigger_d_1 * 2*(x_e_c.imag -Fxp(self.d, signed=True, n_word=8, n_frac=5, rounding='floor').get_val()), signed=True, n_word=8, n_frac=4, rounding='floor').get_val()
            Le_d1_lesser = Fxp(if_lesser_d_1 * 2*(x_e_c.imag +Fxp(self.d, signed=True, n_word=8, n_frac=5, rounding='floor').get_val()), signed=True, n_word=8, n_frac=4, rounding='floor').get_val()
            Le_d_1 = Le_d1_bigger + Le_d1_lesser + (1-if_bigger_d_1)*(1-if_lesser_d_1)*x_e_c.imag
            Le_d_kq[1::4] = mult_const*Fxp(Le_d_1, signed=True, n_word=8, n_frac=4, rounding='floor').get_val()
            # print(Fxp(x_e_c.imag[0], signed=True, n_word=8, n_frac=5, rounding='floor').raw().astype(np.uint8))
            # print(self.get_vinv(10*np.log10(v_e[frame])).raw().astype(np.uint8))

            if_bigger_d_3 = x_e_c.real > Fxp(2*self.d, signed=True, n_word=8, n_frac=5, rounding='floor').get_val()
            if_lesser_d_3 = x_e_c.real < Fxp(-2*self.d, signed=True, n_word=8, n_frac=5, rounding='floor').get_val()
            # print(Fxp(if_lesser_d_3 * (x_e_c.real +2*self.d), signed=True, n_word=8, n_frac=5, rounding='floor'))
            Le_d3_bigger = Fxp(if_bigger_d_3 * 2*(x_e_c.real -Fxp(self.d, signed=True, n_word=8, n_frac=5, rounding='floor').get_val()), signed=True, n_word=8, n_frac=4, rounding='floor').get_val()
            Le_d3_lesser = Fxp(if_lesser_d_3 * 2*(x_e_c.real +Fxp(self.d, signed=True, n_word=8, n_frac=5, rounding='floor').get_val()), signed=True, n_word=8, n_frac=4, rounding='floor').get_val()
            Le_d_3 = Le_d3_bigger + Le_d3_lesser + (1-if_bigger_d_3)*(1-if_lesser_d_3)*x_e_c.real
            Le_d_kq[0::4] = mult_const*Fxp(Le_d_3, signed=True, n_word=8, n_frac=4, rounding='floor').get_val()
            # print(Fxp(x_e_c.real[0], signed=True, n_word=8, n_frac=5, rounding='floor').raw().astype(np.uint8))
            # print(Fxp(Le_d_3, signed=True, n_word=8, n_frac=4, rounding='floor').raw().astype(np.uint8))

            Le_fp = Fxp(Le_d_kq, signed=True, n_word=8, n_frac=3, rounding='floor').get_val()
            # print(Fxp(Le_d_kq, signed=True, n_word=8, n_frac=3, rounding='floor').raw().astype(np.uint8))
            # time.sleep(1)
            Le[frame] = np.reshape(Le_fp, self.N)
        return 0

    def __init__(self, N, Q):
        Py_Module.__init__(self)
        self.name = "py_Block3_16QAM"
        self.N = N
        self.Q = Q
        self.K = N // Q
        self.X = 2**Q
        self.d = Fxp(1/np.sqrt(10), signed=False, n_word=8, n_frac=9, rounding='floor').get_val()
        t_b3 = self.create_task("decode")

        s_x_e = self.create_socket_in(t_b3, "x_e", 2*N//Q, np.float32)
        s_v_e = self.create_socket_in(t_b3, "v_e", 1, np.float32)
        s_Le = self.create_socket_out(t_b3, "Le", N, np.float32)

        self.create_codelet(t_b3, lambda slf, lsk, fid: slf.block3(lsk[s_x_e], lsk[s_v_e], lsk[s_Le]))