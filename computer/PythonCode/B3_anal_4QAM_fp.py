#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
import numpy as np
from fxpmath import Fxp
import take_closest
import tables_anal
import time

class Block3(Py_Module):
    # def vinvd4_table(self, v):
    #     return self.vinvd4[take_closest.take_closest(self.v_list, v)]

    def get_vinvd4(self, SNR):
         return tables_anal.table_B3_QPSK_fp[take_closest.take_closest(tables_anal.SNR_QPSK, SNR)]
    
    def block3(self, x_e, v_e, Le):
        Frames = self.n_frames

        for frame in range(Frames):
            # print(np.max(np.abs(x_e)))
            # n_int = -1
            # n_frac = 16
            x_fp = Fxp(x_e[frame], signed=True, n_word=8, n_frac=5, rounding='floor')
            # x_fp = Fxp(x_e[frame], signed=True, n_word=n_int+n_frac+1, n_frac=n_frac, rounding='floor')

            Le_fp = x_fp.get_val()[:]*self.get_vinvd4(10*np.log10(v_e[frame])).get_val()
            
            # print(Fxp(v_e, signed=False, n_word=8, n_frac=7).raw().astype(np.uint8))
            # print(Fxp(x_e[frame], signed=True, n_word=8, n_frac=5, rounding="floor").raw().astype(np.uint8))
            # print(self.get_vinvd4(10*np.log10(v_e[frame])).raw().astype(np.uint8))
            # self.get_vinvd4(10*np.log10(v_e[frame])).info(verbose=3)
            Le_fp = Fxp(Le_fp, signed=True, n_word=8, n_frac=3, rounding='floor')
            # print(Le_fp.raw().astype(np.uint8))
            # Le_fp.info(verbose=3)
            Le_d_kq = Le_fp.get_val()
            Le_d_kq = np.reshape(Le_d_kq, (self.K,self.Q))
            Le_d_kq = np.flip(Le_d_kq, axis=1)
            Le[frame] = np.reshape(Le_d_kq, self.N)
            # print(Le[frame])
            # time.sleep(1)
        return 0

    def __init__(self, N, Q):
        Py_Module.__init__(self)
        self.name = "py_Block3_4QAM"
        self.N = N
        self.Q = Q
        self.K = N // Q
        self.X = 2**Q
        self.d4 = Fxp(2*np.sqrt(2), signed=False, n_word=8, n_frac=6, rounding='floor')

        # # self.v_list = [0.05049022, 0.05887056, 0.06882072, 0.08059306, 0.09460089, 0.11141268, 0.13176292, 0.15662958, 0.18735346, 0.22578493, 0.27446678, 0.33681095, 0.4171298, 0.5202645, 0.6506417, 0.81141806, 1.0057392, 1.2417837, 1.540515, 1.94495, 2.5343635]
        # self.v_list = [0.078125, 0.09375,  0.109375, 0.125, 0.15625, 0.171875, 0.21875, 0.265625, 0.328125, 0.40625, 0.515625, 0.640625, 0.796875, 1., 1.234375, 1.53125, 1.9375, 2.53125]
        # self.inv_v_list = [19.80581586, 16.98641902, 14.53050767, 12.40801627, 10.57072507, 8.97563904,  7.58938858,  6.38449008,  5.33750484,  4.42899356, 3.64342818,  2.96902461,  2.39733531,  1.92209924,  1.53694422, 1.23241033,  0.99429355,  0.80529322,  0.64913357,  0.51415203, 0.39457639]
        # # self.inv_v_ap = Fxp(self.inv_v_list, signed = False, n_word=8, n_frac=4)
        # self.inv_v_ap = [19.75, 16.875, 14.5, 12.375, 10.5, 8.875, 7.5, 6.375, 5.25, 4.375, 3.625, 2.875, 2.375, 1.875, 1.5, 1.125, 0.875, 0.75, 0.625, 0.5, 0.375]
        # # self.vinvd4 = Fxp([31.875, 31.875, 31.875, 31.875, 29.875, 25.375, 21.375, 18., 15., 12.5, 10.25, 8.375, 6.75, 5.375, 4.25, 3.375, 2.75, 2.25, 1.75, 1.375, 1.], n_word=8)
        # self.vinvd4 = Fxp([31.875, 29.875, 25.375, 21.375, 18., 15., 12.5, 10.25, 8.375, 6.75, 5.375, 4.25, 3.375, 2.75, 2.25, 1.75, 1.375, 1.], signed=False, n_word=8, n_frac=3)

        t_b3 = self.create_task("decode")

        s_x_e = self.create_socket_in(t_b3, "x_e", 2*N//Q, np.float32)
        s_v_e = self.create_socket_in(t_b3, "v_e", 1, np.float32)
        s_Le = self.create_socket_out(t_b3, "Le", N, np.float32)

        self.create_codelet(t_b3, lambda slf, lsk, fid: slf.block3(lsk[s_x_e], lsk[s_v_e], lsk[s_Le]))