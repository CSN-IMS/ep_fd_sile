#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import tanh
from fxpmath import Fxp
import time
import take_closest
import tables_anal

class Block4(Py_Module):

    def get_Cep(self, SNR):
         return tables_anal.Cep_QPSK_fp[take_closest.take_closest(tables_anal.SNR_QPSK, SNR)]

    def block4(self, Le, v_e, mi_d, Cep):
        Frames = self.n_frames
        
        for frame in range(Frames):
            # print(Le[frame])
            # time.sleep(1)
            p_d_kb_ = Fxp(self.tanh(Le[frame]/2), signed=True, n_word=8, n_frac=7, rounding="floor")

            mi_kb_d_real = self.d4.get_val()*p_d_kb_[1::self.Q].get_val()
            mi_kb_d_imag = self.d4.get_val()*p_d_kb_[0::self.Q].get_val()

            v_fp = Fxp(v_e[frame], signed=False, n_word=8, n_frac=7, rounding='floor').get_val()
            Cep[frame] = self.get_Cep(10*np.log10(v_fp))

            mi_d[frame][::2] = mi_kb_d_real[:]
            mi_d[frame][1::2] = mi_kb_d_imag[:]

        return 0

    def __init__(self, N, Q, ntanh = 0):
        Py_Module.__init__(self)
        self.name = "py_Block4"
        self.N = N
        self.Q = Q
        self.K = N // Q
        self.X = 2**Q
        self.d4 = Fxp(np.sqrt(2)/2, signed=False, n_word=8, n_frac=8, rounding="floor")

        if(ntanh == 0):
            self.tanh = np.tanh
        elif(ntanh == 1):
            self.tanh = tanh.tanh_segm1
        elif(ntanh == 2):
            self.tanh = tanh.tanh_segm2
        elif(ntanh == 3):
            self.tanh = tanh.tanh_segm3
        else:
            print("ERROR: wrong number for the TANH function on B4_anal_4QAM")

        t_b4 = self.create_task("decode")

        s_Le = self.create_socket_in(t_b4, "Le", N, np.float32)
        s_v_e = self.create_socket_in(t_b4, "v_e", 1, np.float32)
        s_mi_d = self.create_socket_out(t_b4, "mi_d", 2*N//Q, np.float32)
        s_Cep = self.create_socket_out(t_b4, "Cep", 1, np.float32)

        self.create_codelet(t_b4, lambda slf, lsk, fid: slf.block4(lsk[s_Le], lsk[s_v_e], lsk[s_mi_d], lsk[s_Cep]))