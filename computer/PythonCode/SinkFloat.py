#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
from py_aff3ct.module.py_module import Py_Module
from config import *
import numpy as np

class SinkFloat(Py_Module):

    def sink_float(self, InN):
        file1 = open(self.file_name, "a")
        for frame in range(np.size(InN, 0)):
            for n in range(np.size(InN, 1)):
                file1.write(str(InN[frame][n])+"\n")

        file1.close()
        return 0

    def __init__(self, N, file_name):
        Py_Module.__init__(self)
        self.name = "py_sink_float"

        self.file_name = file_name
        
        t_sinkf = self.create_task("append_file")

        s_in = self.create_socket_in(t_sinkf, "InN", N, np.float32)

        self.create_codelet(t_sinkf, lambda slf, lsk, fid: slf.sink_float(lsk[s_in]))