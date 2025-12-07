#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import maxstar
import time

class Block4(Py_Module):

    def block4(self, log_D_, mi_d, gamma_d):
        Frames = self.n_frames
        
        for frame in range(Frames):
            log_D_kb_ = np.reshape(log_D_[frame], (self.K, self.X))
            log_D_kb_norm = np.zeros(self.K, dtype = np.float32)
            for k in range(self.K):
                log_D_kb_norm[k] = maxstar.maxstar(log_D_kb_[k])
            log_D_kb = (np.subtract(log_D_kb_.T, log_D_kb_norm)).T

            # D_kb_ = np.exp(log_D_[frame])
            # D_kb_ = np.reshape(D_kb_, (self.K, self.X))
            # D_kb_sum = np.sum(D_kb_, axis=1)
            # D_kb = (np.divide(D_kb_.T, D_kb_sum)).T

            D_kb = np.exp(log_D_kb)

            mi_kb_d = np.zeros(self.K, dtype = np.complex_)
            gamma_xb_d = 0
            for k in range(self.K):
                sum_a2d = 0
                for a in range(self.X):
                    mi_kb_d[k]+=self.X_value[a]*D_kb[k][a]
                    sum_a2d+=pow(abs(self.X_value[a]),2)*D_kb[k][a]
                gamma_xb_d+=sum_a2d-pow(abs(mi_kb_d[k]),2)

            gamma_d[frame] = gamma_xb_d/self.K

            mi_d[frame][::2] = mi_kb_d.real[:]
            mi_d[frame][1::2] = mi_kb_d.imag[:]
        
        return 0

    def __init__(self, N, Q, X_value):
        Py_Module.__init__(self)
        self.name = "py_Block4"
        self.N = N
        self.Q = Q
        self.K = N // Q
        self.X = 2**Q
        self.X_value = X_value

        t_b4 = self.create_task("decode")

        s_log_D_ = self.create_socket_in(t_b4, "log_D_", self.K*self.X, np.float32)
        s_mi_d = self.create_socket_out(t_b4, "mi_d", 2*N//Q, np.float32)
        s_gamma_d = self.create_socket_out(t_b4, "gamma_d", 1, np.float32)

        self.create_codelet(t_b4, lambda slf, lsk, fid: slf.block4(lsk[s_log_D_], lsk[s_mi_d], lsk[s_gamma_d]))