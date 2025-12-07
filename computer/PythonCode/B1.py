#!/usr/bin/env python3

import sys
import time
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import time

class Block1(Py_Module):

    def block1(self, y_F, h, Phi, Psi, x_d_F, v_d, x_e, v_e):
        Frames = self.n_frames

        h_k_b_F = np.empty(self.K, dtype=np.complex_)
        h_k_b_F.real[:] = h[0][::2]
        h_k_b_F.imag[:] = h[0][1::2]

        h_F = np.atleast_2d(h_k_b_F).T

        if not self.is_init:
            x_d_F_in = np.zeros(x_d_F.shape, x_d_F.dtype)
            v_d_in = np.ones(v_d.shape, v_d.dtype)
            self.is_init = 1
        else:
            x_d_F_in = x_d_F
            v_d_in = v_d

        for frame in range(Frames):
            y_F_c = np.empty(self.K, dtype=np.complex_)
            Phi_c = np.empty((1, self.K), dtype=np.complex_)
            Psi_ = np.empty(self.K, dtype=np.float32)
            x_d_F_c = np.empty(self.K, dtype=np.complex_)

            y_F_c.real[:] = y_F[frame][::2]
            y_F_c.imag[:] = y_F[frame][1::2]
            Phi_c.real[0][:] = Phi[frame][::2]
            Phi_c.imag[0][:] = Phi[frame][1::2]
            Psi_[:] = Psi[frame][:]
            x_d_F_c.real[:] = x_d_F_in[frame][::2]
            x_d_F_c.imag[:] = x_d_F_in[frame][1::2]
            v_d_ = v_d_in[frame][0] #initialization
            
            c_MMSE = np.divide(Phi_c, 1 + v_d_ * Psi_)

            xi_temp = np.diag(np.dot(np.conj(h_F), c_MMSE))
            xi_xb = np.real(np.mean(xi_temp))
            xi_inv = 1/xi_xb

            v_e[frame] = max(xi_inv - v_d_, 1e-8)

            f_MMSE_kb = np.multiply(c_MMSE, xi_inv)

            x_e_F = np.zeros(self.K, dtype=np.complex_)
            x_e_F[:] = x_d_F_c[:] + np.conjugate(f_MMSE_kb) * (y_F_c[:] - h_k_b_F[:] * x_d_F_c[:])

            x_e_c = np.fft.ifft(x_e_F, self.K)*np.sqrt(self.K)

            x_e[frame][::2] = x_e_c.real[:]
            x_e[frame][1::2] = x_e_c.imag[:]

        return 0
    
    def rst(self):
        self.is_init = 0

    def __init__(self, N, Q, k_PAM=1):
        Py_Module.__init__(self)
        self.name = "py_Block1"
        self.N = N
        self.Q = Q
        self.K = N // Q
        self.k_PAM = k_PAM
        self.is_init = 0

        t_b1 = self.create_task("decode")

        s_v_d = self.create_socket_in(t_b1, "v_d", 1, np.float32)
        s_v_e = self.create_socket_out(t_b1, "v_e", 1, np.float32)
        s_h = self.create_socket_in(t_b1, "h", 2*N//Q, np.float32)
        s_y_F = self.create_socket_in(t_b1, "y_F", 2*N//Q, np.float32)
        s_Phi = self.create_socket_in(t_b1, "Phi", 2*N//Q, np.float32)
        s_Psi = self.create_socket_in(t_b1, "Psi", N//Q, np.float32)
        s_x_d_F = self.create_socket_in(t_b1, "x_d_F", 2*N//Q, np.float32)
        s_x_e = self.create_socket_out(t_b1, "x_e", 2*N//Q, np.float32)

        self.create_codelet(t_b1, lambda slf, lsk, fid: slf.block1(lsk[s_y_F], lsk[s_h], lsk[s_Phi], lsk[s_Psi],lsk[s_x_d_F], lsk[s_v_d], lsk[s_x_e], lsk[s_v_e]))