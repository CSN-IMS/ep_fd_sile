import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
sys.path.insert(0, '../pyaf/build/lib')
import py_aff3ct as aff3ct
import pyaf
import config
import pynq_BPSK
import pynq_4QAM
import pynq_8PSK
import pynq_16QAM
import pass_through

# return [N, Q, X_value, Kb, k_PAM, cstl, matcher_ind, dematcher_ind, cyc_shift, pynq]
def get_mcs_config(mcs):
    if(mcs == 1):
        N = 256 # Kd
        Q = 2
        X_value = config.X_value_4QAM
        Kb = 128
        k_PAM = 1
        cstl = aff3ct.tools.constellation.Constellation_user("../pyaf/py_aff3ct/lib/aff3ct/conf/mod/4QAM_GRAY.mod")
        matcher_ind = config.matcher_ind_396_256
        dematcher_ind = config.dematcher_ind_256_396
        cyc_shift = config.cyc_shift_MCS1
        pynq = pynq_4QAM.PynqConnect
    elif(mcs == 2):
        N = 256 # Kd
        Q = 2
        X_value = config.X_value_4QAM
        Kb = 192
        k_PAM = 1
        cstl = aff3ct.tools.constellation.Constellation_user("../pyaf/py_aff3ct/lib/aff3ct/conf/mod/4QAM_GRAY.mod")
        matcher_ind = config.matcher_ind_588_256
        dematcher_ind = config.dematcher_ind_256_588
        cyc_shift = config.cyc_shift_MCS2
        pynq = pynq_4QAM.PynqConnect
    elif(mcs == 3):
        N = 384
        Q = 3
        X_value = config.X_value_8PSK_2
        Kb = 192
        k_PAM = 1
        cstl = aff3ct.tools.constellation.Constellation_user("../pyaf/py_aff3ct/lib/aff3ct/conf/mod/8PSK_GRAY_2.mod")
        matcher_ind = config.matcher_ind_588_384
        dematcher_ind = config.dematcher_ind_384_588
        cyc_shift = config.cyc_shift_MCS3
        pynq = pynq_8PSK.PynqConnect
    elif(mcs == 4):
        N = 384
        Q = 3
        X_value = config.X_value_8PSK_2
        Kb = 288
        k_PAM = 1
        cstl = aff3ct.tools.constellation.Constellation_user("../pyaf/py_aff3ct/lib/aff3ct/conf/mod/8PSK_GRAY_2.mod")
        matcher_ind = config.matcher_ind_876_384
        dematcher_ind = config.dematcher_ind_384_876
        cyc_shift = config.cyc_shift_MCS4
        pynq = pynq_8PSK.PynqConnect
    elif(mcs == 5):
        N = 512
        Q = 4
        X_value = config.X_value_16QAM
        Kb = 256
        k_PAM = 1
        cstl = aff3ct.tools.constellation.Constellation_user("../pyaf/py_aff3ct/lib/aff3ct/conf/mod/16QAM_GRAY_2.mod")
        matcher_ind = config.matcher_ind_780_512
        dematcher_ind = config.dematcher_ind_512_780
        cyc_shift = config.cyc_shift_MCS5
        pynq = pynq_16QAM.PynqConnect
    elif(mcs == 6):
        N = 512
        Q = 4
        X_value = config.X_value_16QAM
        Kb = 384
        k_PAM = 1
        cstl = aff3ct.tools.constellation.Constellation_user("../pyaf/py_aff3ct/lib/aff3ct/conf/mod/16QAM_GRAY_2.mod")
        matcher_ind = config.matcher_ind_1164_512
        dematcher_ind = config.dematcher_ind_512_1164
        cyc_shift = config.cyc_shift_MCS6
        pynq = pynq_16QAM.PynqConnect
    elif(mcs == 9):
        N = 128 # Kd
        Q = 1
        X_value = config.X_value_BPSK
        Kb = 64
        k_PAM = 2
        cstl = aff3ct.tools.constellation.Constellation_user("../pyaf/py_aff3ct/lib/aff3ct/conf/mod/BPSK.mod")
        matcher_ind = config.matcher_ind_204_128
        dematcher_ind = config.dematcher_ind_128_204
        cyc_shift = config.cyc_shift_MCS9
        pynq = pynq_BPSK.PynqConnect
    elif(mcs == 10):
        N = 128 # Kd
        Q = 1
        X_value = config.X_value_BPSK
        Kb = 96
        k_PAM = 2
        cstl = aff3ct.tools.constellation.Constellation_user("../pyaf/py_aff3ct/lib/aff3ct/conf/mod/BPSK.mod")
        matcher_ind = config.matcher_ind_300_128
        dematcher_ind = config.dematcher_ind_128_300
        cyc_shift = config.cyc_shift_MCS10
        pynq = pynq_BPSK.PynqConnect
    else:
        print("ERROR: MCS ", mcs, " not yet defined!")

    return [N, Q, X_value, Kb, cstl, matcher_ind, dematcher_ind, cyc_shift, pynq]

# return [chn h_channel]
def get_chn_config(channel_name, Nch):
    if(channel_name == "proakisc"):
        chn = pyaf.Channel_proakis_c(Nch)
        h_channel = config.h_proakisc_128
    elif(channel_name == "exp5"):
        chn = pyaf.Channel_EXPn(Nch, 5)
        h_channel = config.h_exp5_128
    elif(channel_name == "exp7"):
        chn = pyaf.Channel_EXPn(Nch, 7)
        h_channel = config.h_exp7_128
    elif(channel_name == "equ5"):
        chn = pyaf.Channel_EQUn(Nch, 5)
        h_channel = config.h_equ5_128
    elif(channel_name == "equ7"):
        chn = pyaf.Channel_EQUn(Nch, 7)
        h_channel = config.h_equ7_128
    elif(channel_name == "awgn"):
        chn = pass_through.PassThrough(Nch)
        h_channel = config.h_awgn_128

    return [chn, h_channel]