#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import time

class Block0(Py_Module):

    def block0(self, y, Var, h, y_F, Phi, Psi):
        Frames = self.n_frames
        var_wb = Var[0][0]
        h_k_b_F = np.empty(self.K, dtype=np.complex_)
        h_k_b_F.real[:] = h[0][::2]
        h_k_b_F.imag[:] = h[0][1::2]

        h_F = np.atleast_2d(h_k_b_F).T

        for frame in range(Frames):
            y_c = np.empty(self.K, dtype=np.complex_)
            y_c.real[:] = y[frame][::2]
            y_c.imag[:] = y[frame][1::2]
            
            y_F_c = np.fft.fft(y_c, self.K) / self.K
            Phi_c = np.multiply(h_F.T, (self.K*self.k_PAM)/var_wb)
            Psi_c = np.diag(np.dot(np.conj(h_F), Phi_c))

            y_F[frame][::2] = y_F_c.real[:]
            y_F[frame][1::2] = y_F_c.imag[:]
            Phi[frame][::2] = Phi_c.real[0][:]
            Phi[frame][1::2] = Phi_c.imag[0][:]
            Psi[frame] = Psi_c.real
            
        return 0

    def __init__(self, N, Q, k_PAM=1):
        Py_Module.__init__(self)
        self.name = "py_Block0"
        self.N = N
        self.Q = Q
        self.K = N // Q
        self.k_PAM = k_PAM

        t_b0 = self.create_task("decode")

        s_var = self.create_socket_in(t_b0, "Var", 1, np.float32)
        s_h = self.create_socket_in(t_b0, "h", 2*N//Q, np.float32)
        s_y = self.create_socket_in(t_b0, "y", 2*N//Q, np.float32)
        s_y_F = self.create_socket_out(t_b0, "y_F", 2*N//Q, np.float32)
        s_Phi = self.create_socket_out(t_b0, "Phi", 2*N//Q, np.float32)
        s_Psi = self.create_socket_out(t_b0, "Psi", N//Q, np.float32)

        self.create_codelet(t_b0, lambda slf, lsk, fid: slf.block0(lsk[s_y], lsk[s_var], lsk[s_h], lsk[s_y_F], lsk[s_Phi], lsk[s_Psi]))