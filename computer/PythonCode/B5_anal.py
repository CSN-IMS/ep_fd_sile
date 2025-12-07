#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import math
import time
from fxpmath import Fxp

class Block5(Py_Module):

    def block5(self, x_e, v_e, mi_d, Cep, x_d_F, v_d):
        Frames = self.n_frames

        for frame in range(Frames):
            x_e_c = np.empty(self.K, dtype = np.complex_)
            mi_d_c = np.empty(self.K, dtype = np.complex_)            
            x_e_c.real = x_e[frame][::2]
            x_e_c.imag = x_e[frame][1::2]
            mi_d_c.real = mi_d[frame][::2]
            mi_d_c.imag = mi_d[frame][1::2]

            v_xb_star = v_e*Cep[frame]
            x_kb_star = np.empty(self.K, dtype = np.complex_)
            x_kb_star[:] = mi_d_c[:]*(1+Cep[frame]) -x_e_c[:]*Cep[frame]

            v_d[frame] = v_xb_star
            
            # x_aux = Fxp(np.zeros(2*self.K), signed=True, n_word=8, n_frac=5, rounding="floor")
            # x_aux[::2] = x_kb_star.real[:]
            # x_aux[1::2] = x_kb_star.imag[:]
            # print(Fxp(x_aux, signed=True, n_word=8, n_frac=5, rounding="floor").raw().astype(np.uint8))
            # time.sleep(1)

            x_d_F_ = np.fft.fft(x_kb_star, self.K)/np.sqrt(self.K)
            x_d_F[frame][::2] = x_d_F_.real[:]
            x_d_F[frame][1::2] = x_d_F_.imag[:]
        return 0

    def __init__(self, N, Q):
        Py_Module.__init__(self)
        self.name = "py_Block5"
        self.N = N
        self.Q = Q
        self.K = N // Q

        t_b5 = self.create_task("decode")

        s_x_e = self.create_socket_in(t_b5, "x_e", 2*N//Q, np.float32)
        s_v_e = self.create_socket_in(t_b5, "v_e", 1, np.float32)
        s_mi_d = self.create_socket_in(t_b5, "mi_d", 2*N//Q, np.float32)
        s_Cep = self.create_socket_in(t_b5, "Cep", 1, np.float32)
        s_x_d_F = self.create_socket_out(t_b5, "x_d_F", 2*N//Q, np.float32)
        s_v_d = self.create_socket_out(t_b5, "v_d", 1, np.float32)

        self.create_codelet(t_b5, lambda slf, lsk, fid: slf.block5(lsk[s_x_e], lsk[s_v_e], lsk[s_mi_d], lsk[s_Cep], lsk[s_x_d_F], lsk[s_v_d]))