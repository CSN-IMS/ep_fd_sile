#!/usr/bin/env python3

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
sys.path.insert(0, '../pyaf/build/lib')

import numpy as np
import math
import time
import py_aff3ct as aff3ct
import pyaf
import B0
import B1
import B2
import B3
import B3_max
import B4
import B5
import framing
import LTE_rate_matcher
import LTE_rate_dematcher
import interlacer
import buffer_encoding
import Monitor_SER
import hard_decide
import config
import config_MCS

# python3 test_anal_ser.py mcs_number channel_name self_iterations
# ex:
# python3 test_anal_ser.py 1 proakisc 1

mcs_number = int(sys.argv[1])
channel_name = sys.argv[2]
S = int(sys.argv[3])

[N, Q, X_value, Kb, cstl, matcher_ind, dematcher_ind, cyc_shift, B3_cons, B4_cons] = config_MCS.get_mcs_config(mcs_number)
ntanh = 3 # 0 = exact tanh in B4, else number of segments up to 3

FS = 32
K=128
Ns = N//Q
tail=6
M=3*Kb+2*tail

FE = 100
maxF = 10000000

esn0_min =0
esn0_max = 100.1
esn0_step = 1
max_se = 1000
max_sym = 100000000

target_ser = 1e-5

esn0 = np.arange(esn0_min,esn0_max,esn0_step)
sigma_vals = 1/(math.sqrt(2) * 10 ** (esn0 / 20))
var_vals = 1/(10 ** (esn0 / 10))

src = aff3ct.module.source.Source_random_fast(N)
mdm = aff3ct.module.modem.Modem_generic(N, cstl)
frm = framing.Framing(Ns*2, FS)
[chn_p, channel_h] = config_MCS.get_chn_config(channel_name, Ns*2+FS)
gen = aff3ct.tools.Gaussian_noise_generator_implem.FAST
chn = aff3ct.module.channel.Channel_AWGN_LLR(Ns*2+FS, gen)
b0 = B0.Block0(N, Q)
b1 = B1.Block1(N, Q)
b2 = B2.Block2(N, Q, X_value)
b3 = B3_max.Block3(N, Q)
b4 = B4.Block4(N, Q, X_value)
b5 = B5.Block5(N, Q)
swi = aff3ct.module.switcher.Switcher(2, Ns*(2**Q), np.float32)
cnt = aff3ct.module.iterator.Iterator(S)
LaZero = np.zeros((1,N), np.float32)
x_d_F = np.zeros((1,2*Ns), np.float32)
v_d = np.ones((1,1), np.float32)
log_P = np.zeros((1,Ns*(2**Q)), np.float32)
hd = hard_decide.HardDecide(N)
mnt = Monitor_SER.Monitor_SER(N, Q, max_se, max_sym)

mnt["check_errors::U"] = src["generate::U_K"]
mdm["modulate::X_N1"] = src["generate::U_K"]
frm["frame::IN"] = mdm["modulate::X_N2"]
chn_p["add_noise::X_N"] = frm["frame::OUT"]
chn["add_noise::X_N"] = chn_p["add_noise::Y_N"]
frm["deframe::IN"] = chn["add_noise::Y_N"]
b0["decode::y"] = frm["deframe::OUT"]
b1["decode::y_F"] = b0["decode::y_F"]
b1["decode::Phi"] = b0["decode::Phi"]
b1["decode::Psi"] = b0["decode::Psi"]
# b1["decode::x_d_F"] = x_d_F
# b1["decode::v_d"] = v_d
b1["decode::x_d_F"] = b5["decode::x_d_F"]
b1["decode::v_d"] = b5["decode::v_d"]

b2["decode::x_e"] = b1["decode::x_e"]
b2["decode::v_e"] = b1["decode::v_e"]
b2["decode::log_P"] = log_P
b3["decode::La"] = LaZero
swi["commute::data "] = b2["decode::log_D_"]
b3["decode::log_D_"] = swi["commute::data1"]
swi["select::data0"] = swi["commute::data0"]
cnt["iterate"] = swi["select::status"]
swi["commute::ctrl "] = cnt["iterate::out"]
b4["decode::log_D_"] = swi["select::data"]
b5["decode::x_e"] = b1["decode::x_e"]
b5["decode::v_e"] = b1["decode::v_e"]
b5["decode::mi_d"] = b4["decode::mi_d"]
b5["decode::gamma_d"] = b4["decode::gamma_d"]
hd["decide::LLR"] = b3["decode::Le"]
mnt["check_errors::V"] = hd["decide::bit"]

sigma = np.ndarray(shape = chn["add_noise::CP"][:].shape, dtype = np.float32)
chn["add_noise::CP"] = sigma
chn_p["add_noise::CP"] = sigma
b0["decode::h"] = channel_h
b1["decode::h"] = channel_h

Z = np.zeros  (shape = swi["select::data1"][:].shape, dtype = np.float32)
swi["select::data1"] = Z

var = np.ndarray(shape = b0["decode::Var"][:].shape, dtype = np.float32)
b0["decode::Var"] = var

# seq  = aff3ct.tools.sequence.Sequence([src["generate"], b5["decode"]])

ber = np.zeros(len(esn0))
fer = np.zeros(len(esn0))
ser = np.zeros(len(esn0))
total_sym = np.zeros(len(esn0), dtype = np.int32)
# seq.export_dot('test.dot')
# for lt in seq.get_tasks_per_types():
#     for t in lt:
#         t.debug = True
#         t.set_debug_limit(10)

# l_tasks = seq.get_tasks_per_types()
# for lt in l_tasks:
#     for t in lt:
#         t.stats = True

print(" Es/NO (dB) | Frame number |    Symbols   |    SER   |  Tpt (Mbps)|  Time (s)")
print("------------|--------------|--------------|----------|------------|-----------")
for i in range(len(sigma_vals)):
	sigma[:] = sigma_vals[i]
	var[:] = var_vals[i]

	t = time.time()
	# seq.exec()
	while not mnt.is_done():
		cnt["iterate"].exec()
		src["generate"].exec()
		mdm["modulate"].exec()
		frm["frame"].exec()
		chn_p["add_noise"].exec()
		chn["add_noise"].exec()
		frm["deframe"].exec()
		b0["decode"].exec()
		b1.rst()
		for s in range(S):
			b1["decode"].exec()
			b2["decode"].exec()
			swi["commute"].exec()
			swi["select"].exec()
			cnt["iterate"].exec()
			b4["decode"].exec()
			b5["decode"].exec()
		b1["decode"].exec()
		b2["decode"].exec()
		swi["commute"].exec()
		b3["decode"].exec()
		hd["decide"].exec()
		mnt["check_errors"].exec()

	elapsed = time.time() - t

	# total_fra = mnt.get_n_analyzed_fra()
	# ber[i] = mnt.get_ber()
	# fer[i] = mnt.get_fer()

	total_sym[i] = mnt.get_n_analyzed_sym()
	ser[i] = mnt.get_ser()

	total_fra = (total_sym[i]/N)

	tpt = total_fra * K * 1e-6/elapsed

	print("%11.2f | %12d | %12d | %7.2e | %10.2f | %10.2f"%(esn0[i], total_fra, total_sym[i], ser[i], tpt, elapsed))

	src.reset()
	cnt.reset()
	mnt.reset()
	mnt.rst()

	# for m in seq.get_modules_reset():
	# 	m.reset()

	if(ser[i] < target_ser):
		break

# seq.show_stats()