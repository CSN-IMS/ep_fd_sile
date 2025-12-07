#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
from config import *
import numpy as np
from fxpmath import Fxp
import socket
import time

class PynqConnect(Py_Module):
    def pynq_s0(self, x_e, v_e, Le):
        Frames = self.n_frames
        for frame in range(Frames):
            # print(Fxp(v_e[frame], signed=False, n_word=8, n_frac=7, rounding="floor").raw().astype(np.uint8))
            # print(Fxp(x_e[frame], signed=True, n_word=8, n_frac=5, rounding="floor").raw().astype(np.uint8))
            # print("------------------------------------------------------")
            ve = v_e[frame][0]
            # print(Fxp(x_e[frame], signed=True, n_word=8, n_frac=5, rounding="floor").get_val())
            data_send = Fxp(ve, signed=False, n_word=8, n_frac=7, rounding="floor").raw().astype(np.uint8).tobytes() +np.array([0, 0, 96], dtype=np.uint8).tobytes()
            data_send = data_send + Fxp(x_e[frame], signed=True, n_word=8, n_frac=5, rounding="floor").raw().astype(np.uint8).tobytes()
            # data_send = Fxp(x_e[frame], signed=True, n_word=8, n_frac=5, rounding="floor").raw().astype(np.uint8).tobytes()
            self.a.sendall(data_send)

            data_recv = self.a.recv(self.data_size_0)
            # print(np.frombuffer(data_recv, dtype=np.uint8))
            buf_data = np.frombuffer(data_recv[:], dtype=np.int8)

            Le_d_kq = Fxp(buf_data, raw=True, signed=True, n_word=8, n_frac=3, rounding="floor")
            # mi_d = Fxp(np.frombuffer(data_recv, dtype=np.int8), raw=True, signed=True, n_word=8, n_frac=5)
            # x_star = Fxp(np.frombuffer(data_recv, dtype=np.int8), raw=True, signed=True, n_word=8, n_frac=5)
            # print(Le_d_kq.raw().astype(np.uint8))
            # time.sleep(1)
            Le_d_kq = Le_d_kq.get_val()
            # print(Le_d_kq)
            # print(buf_data)
            # time.sleep(1)
            # Le_d_kq = np.reshape(Le_d_kq, (self.K,self.Q))
            # Le_d_kq = np.flip(Le_d_kq, axis=1)
            Le[frame] = np.reshape(Le_d_kq, self.N)
            
            # print(Le[frame])
            # OUT[frame] = Fxp(np.frombuffer(data_recv, dtype=np.uint8), raw=True, signed=True, n_word=8, n_frac=5).get_val()
            # time.sleep(1)
        return 0
    
    def pynq_s1(self, x_e, v_e, x_d_F, v_d):
        Frames = self.n_frames
        for frame in range(Frames):
            # print(Fxp(v_e[frame], signed=False, n_word=8, n_frac=7, rounding="floor").raw().astype(np.uint8))
            # print(Fxp(x_e[frame], signed=True, n_word=8, n_frac=5, rounding="floor").raw().astype(np.uint8))
            # print("------------------------------------------------------")
            ve = v_e[frame][0]
            data_send = Fxp(ve, signed=False, n_word=8, n_frac=7, rounding="floor").raw().astype(np.uint8).tobytes()+np.array([0, 0, 224], dtype=np.uint8).tobytes()
            data_send = data_send + Fxp(x_e[frame], signed=True, n_word=8, n_frac=5, rounding="floor").raw().astype(np.uint8).tobytes()
            # data_send = Fxp(x_e[frame], signed=True, n_word=8, n_frac=5).raw().astype(np.uint8).tobytes()
            self.a.sendall(data_send)

            data_recv = self.a.recv(self.data_size_1)
            # print(np.frombuffer(data_recv, dtype=np.uint8))
            buf_data = np.frombuffer(data_recv[:], dtype=np.int8)
            buf_datau = np.frombuffer(data_recv[:], dtype=np.uint8)
            v_star_fp = Fxp(buf_datau[0], raw=True, signed=False, n_word=8, n_frac=7, rounding="floor").get_val()
            x_star_fp = Fxp(buf_data[4:], raw=True, signed=True, n_word=8, n_frac=5, rounding="floor")
            x_star = x_star_fp.get_val()
            # print(v_star.raw().astype(np.uint8))
            # print(x_star_fp.raw().astype(np.uint8))
            x_kb_star = np.empty(self.K, dtype = np.complex_)
            x_kb_star.real = x_star[::2]
            x_kb_star.imag = x_star[1::2]
            # print(Fxp(buf_data[4:], raw=True, signed=True, n_word=8, n_frac=5, rounding="floor").raw().astype(np.uint8))
            # time.sleep(1)
            # print(x_kb_star)

            if(v_star_fp == 0):
                v_star_fp = 0.0078125
            
            v_d[frame] = v_star_fp
            x_d_F_ = np.fft.fft(x_kb_star, self.K)/np.sqrt(self.K)
            x_d_F[frame][::2] = x_d_F_.real[:]
            x_d_F[frame][1::2] = x_d_F_.imag[:]
            # print(v_star.raw().astype(np.uint8))

            # print(x_kb_star)
            # print(v_d[frame])
            # print(x_d_F[frame])
            # print(OUT[frame])
            # OUT[frame] = Fxp(np.frombuffer(data_recv, dtype=np.uint8), raw=True, signed=True, n_word=8, n_frac=5).get_val()
            # time.sleep(1)
        return 0

    def __init__(self, N, Q):
        Py_Module.__init__(self)
        self.name = "py_pynq_connect"
        self.N = N
        self.data_size = 2*N//Q+4
        self.data_size_0 = N
        self.data_size_1 = 2*N//Q+4
        self.Q = Q
        self.K = N // Q
        self.X = 2**Q

        self.a = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        t = ('192.168.2.99', 910) # 916
        self.a.connect(t)
        data_send = self.data_size.to_bytes(2, byteorder='big')
        self.a.sendall(data_send)
        data_send = self.data_size_0.to_bytes(2, byteorder='big')
        self.a.sendall(data_send)
        data_send = self.data_size_1.to_bytes(2, byteorder='big')
        self.a.sendall(data_send)
        
        t_s0 = self.create_task("decode_s0")
        t_s1 = self.create_task("decode_s1")

        s_x_e_s0 = self.create_socket_in(t_s0, "x_e", 2*N//Q, np.float32)
        s_v_e_s0 = self.create_socket_in(t_s0, "v_e", 1, np.float32)
        s_Le_s0 = self.create_socket_out(t_s0, "Le", N, np.float32)

        s_x_e_s1 = self.create_socket_in(t_s1, "x_e", 2*N//Q, np.float32)
        s_v_e_s1 = self.create_socket_in(t_s1, "v_e", 1, np.float32)
        s_x_d_F_s1 = self.create_socket_out(t_s1, "x_d_F", 2*N//Q, np.float32)
        s_v_d_s1 = self.create_socket_out(t_s1, "v_d", 1, np.float32)

        self.create_codelet(t_s0, lambda slf, lsk, fid: slf.pynq_s0(lsk[s_x_e_s0], lsk[s_v_e_s0], lsk[s_Le_s0]))

        self.create_codelet(t_s1, lambda slf, lsk, fid: slf.pynq_s1(lsk[s_x_e_s1], lsk[s_v_e_s1], lsk[s_x_d_F_s1], lsk[s_v_d_s1]))