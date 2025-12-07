#!/usr/bin/env python3

import sys
import time
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import math

class interlacer(Py_Module):

    # input is enc_n of size 2*K+tail_len followed by enc_i of size K+tail_len
    # output is enc_n interlaced with enc_i of size 3*K followed by tail_n and tail_i
    def interlace(self, x_n, x_i, y):
        Frames = self.n_frames

        for frame in range(Frames):
            msg_n = x_n[frame][0:2*self.K]
            tail_n = x_n[frame][2*self.K:]
            msg_i = x_i[frame][0:2*self.K]
            tail_i = x_i[frame][2*self.K:]

            msg_nD = np.reshape(msg_n, (2,self.K), 'F')
            msg_iD = np.reshape(msg_i, (2,self.K), 'F')

            msg_3 = np.empty((3, self.K), dtype=np.int32)
            msg_3[0] = msg_nD[0]
            msg_3[1] = msg_nD[1]
            msg_3[2] = msg_iD[1]

            msg = np.reshape(msg_3, (3*self.K), 'F')
            y[frame] = np.concatenate((msg, tail_n, tail_i))
        return 0

    def deinterlace(self, x, y):
        Frames = self.n_frames

        for frame in range(Frames):
            msg = x[frame][:3*self.K]
            tail_n = x[frame][3*self.K:3*self.K+self.tail_len]
            tail_i = x[frame][3*self.K+self.tail_len:]
            msg_3 = np.reshape(msg, (3, self.K), 'F')
            msg_nD = np.empty((2, self.K), dtype=np.float32)
            msg_iD = np.empty((1, self.K), dtype=np.float32)
            msg_nD[0] = msg_3[0]
            msg_nD[1] = msg_3[1]
            msg_iD[0] = msg_3[2]
            msg_n = np.reshape(msg_nD, (2*self.K), 'F')
            msg_i = np.reshape(msg_iD, (1*self.K), 'F')

            y[frame] = np.concatenate((msg_n, tail_n, msg_i, tail_i))
        return 0

    def __init__(self, K, tail_len):
        Py_Module.__init__(self)
        self.name = "py_interlacer"
        self.K = K
        self.tail_len = tail_len

        t_int = self.create_task("interlace")

        s_x_n = self.create_socket_in(t_int, "x_n", 2*K+tail_len, np.int32)
        s_x_i = self.create_socket_in(t_int, "x_i", 2*K+tail_len, np.int32)
        s_y = self.create_socket_out(t_int, "y", 3*K+2*tail_len, np.int32)

        self.create_codelet(t_int, lambda slf, lsk, fid: slf.interlace(lsk[s_x_n], lsk[s_x_i], lsk[s_y]))

        t_deint = self.create_task("deinterlace")

        s_x = self.create_socket_in(t_deint, "x", 3*K+2*tail_len, np.float32)
        s_y2 = self.create_socket_out(t_deint, "y", 3*K+2*tail_len, np.float32)

        self.create_codelet(t_deint, lambda slf, lsk, fid: slf.deinterlace(lsk[s_x], lsk[s_y2]))