#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import maxstar as ms
import itertools

class Block3(Py_Module):

    def block3(self, La, log_D_, Le):
        Frames = self.n_frames

        for frame in range(Frames):
            La_d_bkq = np.zeros((self.K, self.Q))
            for k in range(self.K):
                for q in range(self.Q):
                    La_d_bkq[k][q] = La[frame][k*self.Q+q]

            Le_d_kq = np.empty((self.K,self.Q))
            for k in range(self.K):
                for q in range(self.Q):
                    array_dq0 = np.empty(self.X//2)
                    array_dq1 = np.empty(self.X//2)
                    dq0_cnt = 0
                    dq1_cnt = 0
                    for a in range(self.X):
                        if a & (1 << (self.Q-1-q)):
                            array_dq1[dq1_cnt] = log_D_[frame][k*self.X+a]
                            dq1_cnt+=1
                        else:
                            array_dq0[dq0_cnt] = log_D_[frame][k*self.X+a]
                            dq0_cnt+=1

                    Le_d_kq[k][self.Q-1-q] = ms.maxstar(array_dq0) - ms.maxstar(array_dq1) - La_d_bkq[k][q]

            for k in range(self.K):
                for q in range(self.Q):
                    Le[frame][k*self.Q+q] = Le_d_kq[k][q]

        return 0

    def __init__(self, N, Q):
        Py_Module.__init__(self)
        self.name = "py_Block3"
        self.N = N
        self.Q = Q
        self.K = N // Q
        self.X = 2**Q

        self.dq1_table = np.zeros((self.Q, self.X))
        self.dq0_table = np.ones((self.Q, self.X))
        for q in range(self.Q):
            for a in range(self.X):
                if a & (1 << (self.Q-1-q)):
                    self.dq1_table[q][a] = 1
                    self.dq0_table[q][a] = 0

        t_b3 = self.create_task("decode")

        s_La = self.create_socket_in(t_b3, "La", N, np.float32)
        s_log_D_ = self.create_socket_in(t_b3, "log_D_", self.K*self.X, np.float32)
        s_Le = self.create_socket_out(t_b3, "Le", N, np.float32)

        self.create_codelet(t_b3, lambda slf, lsk, fid: slf.block3(lsk[s_La], lsk[s_log_D_], lsk[s_Le]))