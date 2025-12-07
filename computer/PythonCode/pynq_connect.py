#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
from config import *
import numpy as np
import socket
import time

class PynqConnect(Py_Module):
    def pynq(self, IN, OUT):
        Frames = np.size(IN, 0)
        for frame in range(Frames):
            data_send = IN[frame].tobytes()
            self.a.sendall(data_send)

            data_recv = self.a.recv(self.data_size)
            OUT[frame] = np.frombuffer(data_recv, dtype=np.float32)
        return 0

    def __init__(self, N):
        Py_Module.__init__(self)
        self.name = "py_pynq_connect"
        self.N = N
        self.data_size = N*4

        self.a = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        t = ('192.168.2.99', 917)
        self.a.connect(t)
        data_send = self.data_size.to_bytes(2, byteorder='big')
        self.a.sendall(data_send)
        

        t_pynq = self.create_task("connect")

        s_pynq_in = self.create_socket_in(t_pynq, "X_N", N, np.float32)
        s_pynq_out = self.create_socket_out(t_pynq, "Y_N", N, np.float32)

        self.create_codelet(t_pynq, lambda slf, lsk, fid: slf.pynq(lsk[s_pynq_in], lsk[s_pynq_out]))