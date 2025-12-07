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
import config
import config_MCS

mcs_number = int(sys.argv[1])
channel_name = sys.argv[2]
S = int(sys.argv[3])

[N, Q, X_value, Kb, cstl, matcher_ind, dematcher_ind, cyc_shift, B3_cons, B4_cons] = config_MCS.get_mcs_config(mcs_number)

Ns = N//Q
FS = 32

K=128
tail=6
M=3*Kb+2*tail

FE = 100
maxF = 5000000

esn0_min = -5
esn0_max = 100.1
esn0_step = 0.5
max_se = 128*10000
max_sym = 128*100

target_ber = 1e-5

esn0 = np.arange(esn0_min,esn0_max,esn0_step)
sigma_vals = 1/(math.sqrt(2) * 10 ** (esn0 / 20))
var_vals = 1/(10 ** (esn0 / 10))

tau = 5
deltadB = 5
alpha = math.sqrt(10**(deltadB/10)/(1+10**(deltadB/10)))

src = aff3ct.module.source.Source_random_fast(Kb)
enc_n = aff3ct.module.encoder.Encoder_RSC_generic_sys(Kb, 2*Kb+tail, False, [11, 13])
enc_i = aff3ct.module.encoder.Encoder_RSC_generic_sys(Kb, 2*Kb+tail, False, [11, 13])
itl_core = aff3ct.tools.interleaver_core.Interleaver_core_LTE(Kb)
itl_bit  = aff3ct.module.interleaver.Interleaver_int32(itl_core)
itl_llr  = aff3ct.module.interleaver.Interleaver_float(itl_core)
inter = interlacer.interlacer(Kb, tail)
rm = LTE_rate_matcher.LTE_rate_matcher(M, N, matcher_ind)
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
rd = LTE_rate_dematcher.LTE_rate_dematcher(N, M, dematcher_ind, cyc_shift)
trellis_n = enc_n.get_trellis()
trellis_i = enc_i.get_trellis()
benc = buffer_encoding.BufferEncoding(Kb, tail)
dec_n = aff3ct.module.decoder.Decoder_RSC_BCJR_seq_std(Kb, trellis_n, True)
dec_i = aff3ct.module.decoder.Decoder_RSC_BCJR_seq_std(Kb, trellis_i, True)
dec = aff3ct.module.decoder.Decoder_turbo_std(Kb,M,8,dec_n,dec_i,itl_llr, True)
mnt = aff3ct.module.monitor.Monitor_BFER_AR(Kb, FE, maxF)


mnt["check_errors::U"] = src["generate::U_K"]
enc_n["encode::U_K"] = src["generate::U_K"]
itl_bit["interleave::nat"] = src["generate::U_K"]
enc_i["encode::U_K"] = itl_bit["interleave::itl"]
inter["interlace::x_n"] = enc_n["encode::X_N"]
inter["interlace::x_i"] = enc_i["encode::X_N"]
rm["rate_match::x"] = inter["interlace::y"]
mdm["modulate::X_N1"] = rm["rate_match::y"]
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
rd["rate_dematch::x"] = b3["decode::Le"]
inter["deinterlace::x"] = rd["rate_dematch::y"]
benc["buf_enc::x"] = inter["deinterlace::y"]
dec["decode_siho::Y_N"] = benc["buf_enc::y"]
mnt["check_errors::V"] = dec["decode_siho ::V_K"]

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

print(" Es/NO (dB) | Frame number |    BER   |    FER   |  Tpt (Mbps)|  Time (s)")
print("------------|--------------|----------|----------|------------|-----------")
for i in range(len(sigma_vals)):
	sigma[:] = sigma_vals[i]
	var[:] = var_vals[i]

	t = time.time()
	# seq.exec()
	while not mnt.is_done():
		cnt["iterate"].exec()
		src["generate"].exec()
		enc_n["encode"].exec()
		itl_bit["interleave"].exec()
		enc_i["encode"].exec()
		inter["interlace"].exec()
		rm["rate_match"].exec()
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
		rd["rate_dematch"].exec()
		inter["deinterlace"].exec()
		benc["buf_enc"].exec()
		dec["decode_siho"].exec()
		mnt["check_errors"].exec()

	elapsed = time.time() - t

	total_fra = mnt.get_n_analyzed_fra()
	ber[i] = mnt.get_ber()
	fer[i] = mnt.get_fer()

	tpt = total_fra * K * 1e-6/elapsed

	print("%11.2f | %12d | %7.2e | %7.2e | %10.2f | %10.2f"%(esn0[i], total_fra, ber[i], fer[i], tpt, elapsed))

	src.reset()
	cnt.reset()
	mnt.reset()

	# for m in seq.get_modules_reset():
	# 	m.reset()

	if(ber[i] < target_ber):
		break

# seq.show_stats()