#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
from config import *
import numpy as np
import time

class Monitor_SER(Py_Module):

    def check_errors(self, U, V, done):
        Symbols = self.K//self.Q
        Frames = np.size(U, 0)
        for frame in range(Frames):
            U_ = np.reshape(U[frame], (Symbols, self.Q))
            V_ = np.reshape(V[frame], (Symbols, self.Q))

            # print(U_)
            # print(V_)
            # time.sleep(1)

            diff = (U_[:] != V_[:])
            # if(np.sum(np.any(diff, axis=1))):
            #     print(diff)
            #     print(U_)
            #     print(V_)
            #     time.sleep(1)
            self.se += np.sum(np.any(diff, axis=1))
            self.n_symbols += Symbols
        
        if (self.se >= self.max_se) or ((self.max_n_symbols != 0) and (self.n_symbols >= self.max_n_symbols)):
            self.se_old = self.se
            self.n_symbols_old = self.n_symbols
            self.se = 0
            self.n_symbols = 0
            self.toggle_done()
            done[0][0] = 1
            self.done_flag = 1

        return 0
    
    def rst(self):
        self.se = 0
        self.n_symbols = 0
        self.done_flag = 0
    
    def is_done(self):
        return self.done_flag == 1

    def get_ser(self):
        if self.n_symbols_old == 0:
            return 0
        
        return self.se_old / self.n_symbols_old

    def get_n_analyzed_sym(self):
        return self.n_symbols_old

    def __init__(self, K, Q, max_se, max_n_symbols = 0):
        Py_Module.__init__(self)
        self.name = "py_monitor_SER"
        self.K = K
        self.Q = Q
        self.max_se = max_se
        self.max_n_symbols = max_n_symbols
        self.se = 0
        self.n_symbols = 0
        self.se_old = 0
        self.n_symbols_old = 0
        self.done_flag = 0

        t_ce = self.create_task("check_errors")

        s_U = self.create_socket_in(t_ce, "U", K, np.int32)
        s_V = self.create_socket_in(t_ce, "V", K, np.int32)
        s_done = self.create_socket_out(t_ce, "done", 1, np.int32)

        self.create_codelet(t_ce, lambda slf, lsk, fid: slf.check_errors(lsk[s_U], lsk[s_V], lsk[s_done]))