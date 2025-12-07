#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
from fxpmath import Fxp

class Block5(Py_Module):

    def block5(self, x_e, v_e, mi_d, gamma_d, x_d_F, v_d):
        Frames = self.n_frames

        for frame in range(Frames):
            x_fp = Fxp(x_e[frame], signed=True, n_word=8, n_frac=5)
            mi_fp = Fxp(mi_d[frame], signed=True, n_word=8, n_frac=5)
            v_fp = Fxp(v_e[frame], signed=False, n_word=self.n_word, n_frac=7)
            gamma_fp = Fxp(gamma_d[frame], signed=False, n_word=self.n_word, n_frac=7)

            if v_fp <= gamma_fp + 1e-3:
                v_xb_star = gamma_fp
                x_kb_star = mi_fp
            else:
                v_xb_star = (v_fp * gamma_fp) / (v_fp - gamma_fp)
                x_kb_star = (mi_fp[:] * v_fp - x_fp[:] * gamma_fp) / (v_fp - gamma_fp)

            v_xb_star = Fxp(v_xb_star, signed=False, n_word=self.n_word, n_frac=7)
            x_kb_star = Fxp(x_kb_star, signed=True, n_word=self.n_word, n_frac=5)

            v_d[frame] = v_xb_star.get_val()

            x_kb_st_aux = x_kb_star.get_val()
            x_kb_st = np.empty(self.K, dtype = np.complex_)
            x_kb_st.real = x_kb_st_aux[::2]
            x_kb_st.imag = x_kb_st_aux[1::2]
            x_d_F_ = np.fft.fft(x_kb_st, self.K)/np.sqrt(self.K)
            x_d_F[frame][::2] = x_d_F_.real[:]
            x_d_F[frame][1::2] = x_d_F_.imag[:]
        return 0

    def __init__(self, N, Q):
        Py_Module.__init__(self)
        self.name = "py_Block5"
        self.N = N
        self.Q = Q
        self.K = N // Q
        self.n_word = 8

        t_b5 = self.create_task("decode")

        s_x_e = self.create_socket_in(t_b5, "x_e", 2*N//Q, np.float32)
        s_v_e = self.create_socket_in(t_b5, "v_e", 1, np.float32)
        s_mi_d = self.create_socket_in(t_b5, "mi_d", 2*N//Q, np.float32)
        s_gamma_d = self.create_socket_in(t_b5, "gamma_d", 1, np.float32)
        s_x_d_F = self.create_socket_out(t_b5, "x_d_F", 2*N//Q, np.float32)
        s_v_d = self.create_socket_out(t_b5, "v_d", 1, np.float32)

        self.create_codelet(t_b5, lambda slf, lsk, fid: slf.block5(lsk[s_x_e], lsk[s_v_e], lsk[s_mi_d], lsk[s_gamma_d], lsk[s_x_d_F], lsk[s_v_d]))