#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
import tanh
import time

class Block4(Py_Module):

    def gamma_av_int(self, complex_var_noise_dB):
        nbseg = 5
        segtype = np.array([0, 2, 1, 1, 0])
        sat = np.array([0, np.nan, np.nan, np.nan, 1])
        limvedB = np.array([-np.inf, -18, -13, -5, 20, np.inf])
        poly = [np.nan, [-0.80683, -19.2096, -136.1351], [5.7009e-05, 0.0036282, 0.071024, 0.4425], [-3.2697e-07, 1.8046e-05, -2.9344e-04, -7.2388e-04, 0.06583, 0.46924], np.nan]
        vedB = complex_var_noise_dB
        int_gamma = 1
        for n in range(nbseg):
            if ((vedB < limvedB[n+1]) & (vedB >= limvedB[n])):
                if segtype[n] == 0:
                    int_gamma = sat[n]
                elif segtype[n] == 1:
                    int_gamma = np.polyval(poly[n], vedB)
                elif segtype[n] == 2:
                    aux = np.polyval(poly[n], vedB)
                    int_gamma = np.power(10, (aux/10))
                else:
                    print('ERROR: Segment type not yet defined')

        return int_gamma

    def block4(self, Le, v_e, mi_d, gamma_d):
        Frames = self.n_frames
        
        for frame in range(Frames):
            p_d_kb_ = self.tanh(Le[frame]/2)

            mi_kb_d_real = (self.a8+self.b8*p_d_kb_[0::self.Q])*p_d_kb_[1::self.Q]
            mi_kb_d_imag = (self.a8-self.b8*p_d_kb_[0::self.Q])*p_d_kb_[2::self.Q]

            gamma_d[frame] = self.gamma_av_int(10*np.log10(v_e[frame]))

            mi_d[frame][::2] = mi_kb_d_real[:]
            mi_d[frame][1::2] = mi_kb_d_imag[:]

        return 0

    def __init__(self, N, Q, ntanh=0):
        Py_Module.__init__(self)
        self.name = "py_Block4"
        self.N = N
        self.Q = Q
        self.K = N // Q
        self.X = 2**Q
        self.a8 = np.sqrt((2+np.sqrt(2))/8)
        self.b8 = np.sqrt((2-np.sqrt(2))/8)

        if(ntanh == 0):
            self.tanh = np.tanh
        elif(ntanh == 1):
            self.tanh = tanh.tanh_segm1
        elif(ntanh == 2):
            self.tanh = tanh.tanh_segm2
        elif(ntanh == 3):
            self.tanh = tanh.tanh_segm3
        else:
            print("ERROR: wrong number for the TANH function on B4_anal_8PSK")

        t_b4 = self.create_task("decode")

        s_Le = self.create_socket_in(t_b4, "Le", N, np.float32)
        s_v_e = self.create_socket_in(t_b4, "v_e", 1, np.float32)
        s_mi_d = self.create_socket_out(t_b4, "mi_d", 2*N//Q, np.float32)
        s_gamma_d = self.create_socket_out(t_b4, "gamma_d", 1, np.float32)

        self.create_codelet(t_b4, lambda slf, lsk, fid: slf.block4(lsk[s_Le], lsk[s_v_e], lsk[s_mi_d], lsk[s_gamma_d]))