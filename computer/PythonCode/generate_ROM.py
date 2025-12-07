import numpy as np
from fxpmath import Fxp
import tables_anal
import take_closest

def get_vinvd4(SNR):
    return tables_anal.table_B3_QPSK_fp[take_closest.take_closest(tables_anal.SNR_QPSK, SNR)]

def get_Cep(SNR):
    return tables_anal.Cep_8PSK_fp[take_closest.take_closest(tables_anal.SNR_8PSK, SNR)]

def get_vinv(SNR):
    return tables_anal.table_B3_8PSK_fp[take_closest.take_closest(tables_anal.SNR_8PSK, SNR)]
    # return tables_anal.table_B3_16QAM_fp[take_closest.take_closest(tables_anal.SNR_16QAM, SNR)]

var = np.linspace(0, 2, 257)

for i in range(256):
    # print("TO_UNSIGNED(", get_vinv(10*np.log10(var[i])).raw().astype(np.uint8), ", 8), ")


    print("TO_UNSIGNED(", get_vinv(10*np.log10(i/2**7)).raw().astype(np.uint8), ", 8), ")
    # Cep[frame] = self.get_Cep(10*np.log10(v_fp))


    # print("TO_UNSIGNED(", get_vinvd4(10*np.log10(var[i])).raw().astype(np.uint8), ", 8), ")
    # print("TO_UNSIGNED(", Fxp(2*np.sqrt(2)/var[i], signed=False, n_word=8, n_frac=3).raw().astype(np.uint8), ", 8), ")