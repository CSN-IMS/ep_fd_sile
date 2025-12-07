#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import time
import itertools

LOG_LIM = 25

class Block3(Py_Module):

    def block3(self, La, log_D_, Le):
        Frames = self.n_frames

        # prevent log saturation
        log_D_[log_D_>LOG_LIM] = LOG_LIM
        log_D_[log_D_<-LOG_LIM] = -LOG_LIM

        for frame in range(Frames):
            log_D_kb_ = np.reshape(log_D_[frame], (self.K, self.X))

            # K*Q, X
            tuple_log_D_kb_0, tuple_dq0 = zip(*list(itertools.product(log_D_kb_, self.dq0_table)))
            np_log_D_kb_0 = np.asarray(tuple_log_D_kb_0)
            np_dq0 = np.asarray(tuple_dq0)
            mult_table_dq0 = np.multiply(np_dq0, np_log_D_kb_0)

            tuple_log_D_kb_1, tuple_dq1 = zip(*list(itertools.product(log_D_kb_, self.dq1_table)))
            np_log_D_kb_1 = np.asarray(tuple_log_D_kb_1)
            np_dq1 = np.asarray(tuple_dq1)
            mult_table_dq1 = np.multiply(np_dq1, np_log_D_kb_1)

            max_dq0 = np.empty((self.N))
            max_dq1 = np.empty((self.N))
            for n in range(self.N):
                aux_array = mult_table_dq1[n]
                aux_array = aux_array[aux_array != 0]
                max_dq1[n] = np.max(aux_array)

                aux_array = mult_table_dq0[n]
                aux_array = aux_array[aux_array != 0]
                max_dq0[n] = np.max(aux_array)

            Le_d_kq = np.empty((self.K*self.Q))
            Le_d_kq[:] = (max_dq0[:] - max_dq1[:]) - La[frame][:]
            Le_d_kq = np.reshape(Le_d_kq, (self.K,self.Q))
            Le_d_kq = np.flip(Le_d_kq, axis=1)

            Le[frame] = np.reshape(Le_d_kq, self.N)
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