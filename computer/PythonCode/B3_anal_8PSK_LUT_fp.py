#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import config
from fxpmath import Fxp
import take_closest
import tables_anal
import time

class Block3(Py_Module):

    def get_vinv(self, SNR):
         return tables_anal.table_B3_8PSK_fp[take_closest.take_closest(tables_anal.SNR_QPSK, SNR)]

    def __hard_decide(self, x_e):
        d1 = np.abs(x_e.imag[:]) > np.abs(x_e.real[:])
        d2 = x_e.real[:] < 0
        d3 = x_e.imag[:] < 0
        m = 4*d3[:] + 2*d2[:] + 1*d1[:]
        return m

    def __create_LUT(self, X_value):
        LUT_8psk = np.zeros((3, 8), dtype=np.complex_)
        for q in range(3):
            for i in range(8):
                LUT_8psk[q][i] = 2*(1-2*((i&(1<<q))>0))*(X_value[i] - X_value[self.bin_refl_8PSK[i][q]])

        LUT_8psk_fp = Fxp(LUT_8psk, signed=True, n_word=4, n_frac=1)
        return LUT_8psk_fp.get_val()

    def block3(self, x_e, v_e, Le):
        Frames = self.n_frames

        for frame in range(Frames):
            # n_int=16
            # n_frac=3
            x_e_c = np.empty(self.K, dtype=np.complex_)
            x_e_c.real[:] = x_e[frame][::2]
            x_e_c.imag[:] = x_e[frame][1::2]

            x_e_c_fp = Fxp(x_e_c, signed=True, n_word=8, n_frac=5, rounding="floor")
            # print(Fxp(x_e[frame], signed=True, n_word=8, n_frac=5, rounding="floor").raw().astype(np.uint8))
            # x_e_c_fp = Fxp(x_e_c, signed=True, n_word=1+n_int+n_frac, n_frac=n_frac, rounding="floor")

            m = self.__hard_decide(x_e_c_fp.get_val())
            da1 = self.LUT_8PSK[0][m[:]]
            da2 = self.LUT_8PSK[1][m[:]]
            da3 = self.LUT_8PSK[2][m[:]]

            v_e_fp = Fxp(v_e, signed=False, n_word=8, n_frac=7)
            v_inv_fp = self.get_vinv(10*np.log10(v_e_fp.get_val()))

            # xmult1r_fp = Fxp(x_e_c_fp.get_val().real[:]*da1.real[:], signed=True, n_word=10, n_frac=5, rounding="floor")
            # xmult2r_fp = Fxp(x_e_c_fp.get_val().real[:]*da2.real[:], signed=True, n_word=10, n_frac=5, rounding="floor")
            # xmult3r_fp = Fxp(x_e_c_fp.get_val().real[:]*da3.real[:], signed=True, n_word=10, n_frac=5, rounding="floor")

            # xmult1i_fp = Fxp(x_e_c_fp.get_val().imag[:]*da1.imag[:], signed=True, n_word=10, n_frac=5, rounding="floor")
            # xmult2i_fp = Fxp(x_e_c_fp.get_val().imag[:]*da2.imag[:], signed=True, n_word=10, n_frac=5, rounding="floor")
            # xmult3i_fp = Fxp(x_e_c_fp.get_val().imag[:]*da3.imag[:], signed=True, n_word=10, n_frac=5, rounding="floor")

            # xadd1_fp = Fxp(xmult1r_fp.get_val()[:]+xmult1i_fp.get_val()[:], signed=True, n_word=10, n_frac=5, rounding="floor")
            # xadd2_fp = Fxp(xmult2r_fp.get_val()[:]+xmult2i_fp.get_val()[:], signed=True, n_word=10, n_frac=5, rounding="floor")
            # xadd3_fp = Fxp(xmult3r_fp.get_val()[:]+xmult3i_fp.get_val()[:], signed=True, n_word=10, n_frac=5, rounding="floor")

            xmult1r_fp = Fxp(x_e_c_fp.get_val().real[:]*da1.real[:], signed=True, n_word=8, n_frac=4, rounding="floor")
            xmult2r_fp = Fxp(x_e_c_fp.get_val().real[:]*da2.real[:], signed=True, n_word=8, n_frac=4, rounding="floor")
            xmult3r_fp = Fxp(x_e_c_fp.get_val().real[:]*da3.real[:], signed=True, n_word=8, n_frac=4, rounding="floor")

            xmult1i_fp = Fxp(x_e_c_fp.get_val().imag[:]*da1.imag[:], signed=True, n_word=8, n_frac=4, rounding="floor")
            xmult2i_fp = Fxp(x_e_c_fp.get_val().imag[:]*da2.imag[:], signed=True, n_word=8, n_frac=4, rounding="floor")
            xmult3i_fp = Fxp(x_e_c_fp.get_val().imag[:]*da3.imag[:], signed=True, n_word=8, n_frac=4, rounding="floor")

            xadd1_fp = Fxp(xmult1r_fp.get_val()[:]+xmult1i_fp.get_val()[:], signed=True, n_word=8, n_frac=4, rounding="floor")
            xadd2_fp = Fxp(xmult2r_fp.get_val()[:]+xmult2i_fp.get_val()[:], signed=True, n_word=8, n_frac=4, rounding="floor")
            xadd3_fp = Fxp(xmult3r_fp.get_val()[:]+xmult3i_fp.get_val()[:], signed=True, n_word=8, n_frac=4, rounding="floor")

            Le_d_kq = np.empty((self.K * self.Q))
            Le_d_kq[0::3] = xadd1_fp.get_val() * v_inv_fp.get_val()
            Le_d_kq[1::3] = xadd2_fp.get_val() * v_inv_fp.get_val()
            Le_d_kq[2::3] = xadd3_fp.get_val() * v_inv_fp.get_val()

            # print(Fxp(Le_d_kq, signed=True, n_word=8, n_frac=3, rounding="floor").raw().astype(np.uint8))
            
            Le_fp = Fxp(np.reshape(Le_d_kq, self.N), signed=True, n_word=8, n_frac=3, rounding="floor")

            Le[frame] = Le_fp.get_val()
            # print(Le[frame])
            # time.sleep(1)
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