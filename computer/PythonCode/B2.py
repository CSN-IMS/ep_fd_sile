#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import itertools
import time

# LOG_LIM = 25

class Block2(Py_Module):

    def block2(self, x_e, v_e, log_P, log_D_):
        Frames = self.n_frames

        for frame in range(Frames):
            x_e_c = np.empty(self.K, dtype=np.complex_)
            x_e_c.real[:] = x_e[frame][::2]
            x_e_c.imag[:] = x_e[frame][1::2]

            SNR_post_eq = 1/(self.k_PAM*v_e[frame])

            tuple_x_e_c, tuple_X_value = zip(*list(itertools.product(x_e_c, self.X_value)))
            np_x_e_c = np.asarray(tuple_x_e_c)
            np_X_value = np.asarray(tuple_X_value)

            log_D_aux = np.empty((1, self.K*self.X), np.float32)

            log_D_aux[0][:] = -pow(abs(np_x_e_c[:] - np_X_value[:]),2)*SNR_post_eq + log_P[frame][:]

            # # prevent log saturation
            # log_D_aux[log_D_>LOG_LIM] = LOG_LIM
            # log_D_aux[log_D_<-LOG_LIM] = -LOG_LIM
            # print(log_D_aux)
            # time.sleep(1)

            log_D_[frame] = log_D_aux

        return 0

    def __init__(self, N, Q, X_value, k_PAM=1):
        Py_Module.__init__(self)
        self.name = "py_Block2"
        self.N = N
        self.Q = Q
        self.K = N // Q
        self.X = 2**Q
        self.X_value = X_value
        self.k_PAM = k_PAM

        t_b2 = self.create_task("decode")

        s_v_e = self.create_socket_in(t_b2, "v_e", 1, np.float32)
        s_x_e = self.create_socket_in(t_b2, "x_e", 2*N//Q, np.float32)
        s_log_P = self.create_socket_in(t_b2, "log_P", self.K*self.X, np.float32)
        s_log_D_ = self.create_socket_out(t_b2, "log_D_", self.K*self.X, np.float32)

        self.create_codelet(t_b2, lambda slf, lsk, fid: slf.block2(lsk[s_x_e], lsk[s_v_e], lsk[s_log_P], lsk[s_log_D_]))