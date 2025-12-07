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
import B1_fp
import B1
import B3_anal_4QAM
import B3_anal_4QAM_fp
import B3_anal_8PSK_LUT
import B3_anal_8PSK_LUT_fp
import B3_anal_16QAM
import B3_anal_16QAM_fp
import B4_anal_4QAM
import B4_anal_4QAM_fp
import B4_anal_8PSK
import B4_anal_8PSK_fp
import B4_anal_16QAM
import B4_anal_16QAM_fp
import B5_anal_fp_fact
import framing
import LTE_rate_matcher
import LTE_rate_dematcher
import interlacer
import buffer_encoding
import config
import config_MCS

# python3 test_anal_fp.py channel_name self_iterations
# ex:
# python3 test_anal_fp.py proakisc 1

channel_name = sys.argv[1]
S = int(sys.argv[2])

[N_1, Q_1, X_value_1, Kb_1, cstl_1, matcher_ind_1, dematcher_ind_1, cyc_shift_1, B3_cons_1, B4_cons_1] = config_MCS.get_mcs_config(1, fp=1)
[N_2, Q_2, X_value_2, Kb_2, cstl_2, matcher_ind_2, dematcher_ind_2, cyc_shift_2, B3_cons_2, B4_cons_2] = config_MCS.get_mcs_config(2, fp=1)
[N_3, Q_3, X_value_3, Kb_3, cstl_3, matcher_ind_3, dematcher_ind_3, cyc_shift_3, B3_cons_3, B4_cons_3] = config_MCS.get_mcs_config(3, fp=1)
[N_4, Q_4, X_value_4, Kb_4, cstl_4, matcher_ind_4, dematcher_ind_4, cyc_shift_4, B3_cons_4, B4_cons_4] = config_MCS.get_mcs_config(4, fp=1)
[N_5, Q_5, X_value_5, Kb_5, cstl_5, matcher_ind_5, dematcher_ind_5, cyc_shift_5, B3_cons_5, B4_cons_5] = config_MCS.get_mcs_config(5, fp=1)
[N_6, Q_6, X_value_6, Kb_6, cstl_6, matcher_ind_6, dematcher_ind_6, cyc_shift_6, B3_cons_6, B4_cons_6] = config_MCS.get_mcs_config(6, fp=1)

ntanh = 3 # 0 = exact tanh in B4, else number of segments up to 3


FS = 32
K=128
Ns_1 = N_1//Q_1
Ns_2 = N_2//Q_2
Ns_3 = N_3//Q_3
Ns_4 = N_4//Q_4
Ns_5 = N_5//Q_5
Ns_6 = N_6//Q_6
tail=6
M_1=3*Kb_1+2*tail
M_2=3*Kb_2+2*tail
M_3=3*Kb_3+2*tail
M_4=3*Kb_4+2*tail
M_5=3*Kb_5+2*tail
M_6=3*Kb_6+2*tail

FE = 100
maxF = 5000000

esn0_min = 0
esn0_max = 100.1
esn0_step = 1
max_se = 128*10000
max_sym = 128*100

target_ber = 1e-5

esn0 = np.arange(esn0_min,esn0_max,esn0_step)
sigma_vals = 1/(math.sqrt(2) * 10 ** (esn0 / 20))
var_vals = 1/(10 ** (esn0 / 10))

src_1 = aff3ct.module.source.Source_random_fast(Kb_1)
enc_n_1 = aff3ct.module.encoder.Encoder_RSC_generic_sys(Kb_1, 2*Kb_1+tail, False, [11, 13])
enc_i_1 = aff3ct.module.encoder.Encoder_RSC_generic_sys(Kb_1, 2*Kb_1+tail, False, [11, 13])
itl_core_1 = aff3ct.tools.interleaver_core.Interleaver_core_LTE(Kb_1)
itl_bit_1  = aff3ct.module.interleaver.Interleaver_int32(itl_core_1)
itl_llr_1  = aff3ct.module.interleaver.Interleaver_float(itl_core_1)
inter_1 = interlacer.interlacer(Kb_1, tail)
rm_1 = LTE_rate_matcher.LTE_rate_matcher(M_1, N_1, matcher_ind_1)
mdm_1 = aff3ct.module.modem.Modem_generic(N_1, cstl_1)
frm_1 = framing.Framing(Ns_1*2, FS)
[chn_p_1, channel_h_1] = config_MCS.get_chn_config(channel_name, Ns_1*2+FS)
gen_1 = aff3ct.tools.Gaussian_noise_generator_implem.FAST
chn_1 = aff3ct.module.channel.Channel_AWGN_LLR(Ns_1*2+FS, gen_1)
b0_1 = B0.Block0(N_1, Q_1)
b1_1 = B1.Block1(N_1, Q_1)
b3_1 = B3_cons_1(N_1, Q_1)
b4_1 = B4_cons_1(N_1, Q_1, ntanh)
b5_1 = B5_anal_fp_fact.Block5(N_1, Q_1)
swi_1 = aff3ct.module.switcher.Switcher(2, N_1, np.float32)
cnt_1 = aff3ct.module.iterator.Iterator(S)
LaZero_1 = np.zeros((1,N_1), np.float32)
x_d_F_1 = np.zeros((1,2*Ns_1), np.float32)
v_d_1 = np.ones((1,1), np.float32)
log_P_1 = np.zeros((1,Ns_1*(2**Q_1)), np.float32)
rd_1 = LTE_rate_dematcher.LTE_rate_dematcher(N_1, M_1, dematcher_ind_1, cyc_shift_1)
trellis_n_1 = enc_n_1.get_trellis()
trellis_i_1 = enc_i_1.get_trellis()
benc_1 = buffer_encoding.BufferEncoding(Kb_1, tail)
dec_n_1 = aff3ct.module.decoder.Decoder_RSC_BCJR_seq_std(Kb_1, trellis_n_1, True)
dec_i_1 = aff3ct.module.decoder.Decoder_RSC_BCJR_seq_std(Kb_1, trellis_i_1, True)
dec_1 = aff3ct.module.decoder.Decoder_turbo_std(Kb_1,M_1,8,dec_n_1,dec_i_1,itl_llr_1, True)
mnt_1 = aff3ct.module.monitor.Monitor_BFER_AR(Kb_1, FE, maxF)


src_2 = aff3ct.module.source.Source_random_fast(Kb_2)
enc_n_2 = aff3ct.module.encoder.Encoder_RSC_generic_sys(Kb_2, 2*Kb_2+tail, False, [11, 13])
enc_i_2 = aff3ct.module.encoder.Encoder_RSC_generic_sys(Kb_2, 2*Kb_2+tail, False, [11, 13])
itl_core_2 = aff3ct.tools.interleaver_core.Interleaver_core_LTE(Kb_2)
itl_bit_2  = aff3ct.module.interleaver.Interleaver_int32(itl_core_2)
itl_llr_2  = aff3ct.module.interleaver.Interleaver_float(itl_core_2)
inter_2 = interlacer.interlacer(Kb_2, tail)
rm_2 = LTE_rate_matcher.LTE_rate_matcher(M_2, N_2, matcher_ind_2)
mdm_2 = aff3ct.module.modem.Modem_generic(N_2, cstl_2)
frm_2 = framing.Framing(Ns_2*2, FS)
[chn_p_2, channel_h_2] = config_MCS.get_chn_config(channel_name, Ns_2*2+FS)
gen_2 = aff3ct.tools.Gaussian_noise_generator_implem.FAST
chn_2 = aff3ct.module.channel.Channel_AWGN_LLR(Ns_2*2+FS, gen_2)
b0_2 = B0.Block0(N_2, Q_2)
b1_2 = B1.Block1(N_2, Q_2)
b3_2 = B3_cons_2(N_2, Q_2)
b4_2 = B4_cons_2(N_2, Q_2, ntanh)
b5_2 = B5_anal_fp_fact.Block5(N_2, Q_2)
swi_2 = aff3ct.module.switcher.Switcher(2, N_2, np.float32)
cnt_2 = aff3ct.module.iterator.Iterator(S)
LaZero_2 = np.zeros((1,N_2), np.float32)
x_d_F_2 = np.zeros((1,2*Ns_2), np.float32)
v_d_2 = np.ones((1,1), np.float32)
log_P_2 = np.zeros((1,Ns_2*(2**Q_2)), np.float32)
rd_2 = LTE_rate_dematcher.LTE_rate_dematcher(N_2, M_2, dematcher_ind_2, cyc_shift_2)
trellis_n_2 = enc_n_2.get_trellis()
trellis_i_2 = enc_i_2.get_trellis()
benc_2 = buffer_encoding.BufferEncoding(Kb_2, tail)
dec_n_2 = aff3ct.module.decoder.Decoder_RSC_BCJR_seq_std(Kb_2, trellis_n_2, True)
dec_i_2 = aff3ct.module.decoder.Decoder_RSC_BCJR_seq_std(Kb_2, trellis_i_2, True)
dec_2 = aff3ct.module.decoder.Decoder_turbo_std(Kb_2,M_2,8,dec_n_2,dec_i_2,itl_llr_2, True)
mnt_2 = aff3ct.module.monitor.Monitor_BFER_AR(Kb_2, FE, maxF)


src_3 = aff3ct.module.source.Source_random_fast(Kb_3)
enc_n_3 = aff3ct.module.encoder.Encoder_RSC_generic_sys(Kb_3, 2*Kb_3+tail, False, [11, 13])
enc_i_3 = aff3ct.module.encoder.Encoder_RSC_generic_sys(Kb_3, 2*Kb_3+tail, False, [11, 13])
itl_core_3 = aff3ct.tools.interleaver_core.Interleaver_core_LTE(Kb_3)
itl_bit_3  = aff3ct.module.interleaver.Interleaver_int32(itl_core_3)
itl_llr_3  = aff3ct.module.interleaver.Interleaver_float(itl_core_3)
inter_3 = interlacer.interlacer(Kb_3, tail)
rm_3 = LTE_rate_matcher.LTE_rate_matcher(M_3, N_3, matcher_ind_3)
mdm_3 = aff3ct.module.modem.Modem_generic(N_3, cstl_3)
frm_3 = framing.Framing(Ns_3*2, FS)
[chn_p_3, channel_h_3] = config_MCS.get_chn_config(channel_name, Ns_3*2+FS)
gen_3 = aff3ct.tools.Gaussian_noise_generator_implem.FAST
chn_3 = aff3ct.module.channel.Channel_AWGN_LLR(Ns_3*2+FS, gen_3)
b0_3 = B0.Block0(N_3, Q_3)
b1_3 = B1.Block1(N_3, Q_3)
b3_3 = B3_cons_3(N_3, Q_3)
b4_3 = B4_cons_3(N_3, Q_3, ntanh)
b5_3 = B5_anal_fp_fact.Block5(N_3, Q_3)
swi_3 = aff3ct.module.switcher.Switcher(2, N_3, np.float32)
cnt_3 = aff3ct.module.iterator.Iterator(S)
LaZero_3 = np.zeros((1,N_3), np.float32)
x_d_F_3 = np.zeros((1,2*Ns_3), np.float32)
v_d_3 = np.ones((1,1), np.float32)
log_P_3 = np.zeros((1,Ns_3*(2**Q_3)), np.float32)
rd_3 = LTE_rate_dematcher.LTE_rate_dematcher(N_3, M_3, dematcher_ind_3, cyc_shift_3)
trellis_n_3 = enc_n_3.get_trellis()
trellis_i_3 = enc_i_3.get_trellis()
benc_3 = buffer_encoding.BufferEncoding(Kb_3, tail)
dec_n_3 = aff3ct.module.decoder.Decoder_RSC_BCJR_seq_std(Kb_3, trellis_n_3, True)
dec_i_3 = aff3ct.module.decoder.Decoder_RSC_BCJR_seq_std(Kb_3, trellis_i_3, True)
dec_3 = aff3ct.module.decoder.Decoder_turbo_std(Kb_3,M_3,8,dec_n_3,dec_i_3,itl_llr_3, True)
mnt_3 = aff3ct.module.monitor.Monitor_BFER_AR(Kb_3, FE, maxF)


src_4 = aff3ct.module.source.Source_random_fast(Kb_4)
enc_n_4 = aff3ct.module.encoder.Encoder_RSC_generic_sys(Kb_4, 2*Kb_4+tail, False, [11, 13])
enc_i_4 = aff3ct.module.encoder.Encoder_RSC_generic_sys(Kb_4, 2*Kb_4+tail, False, [11, 13])
itl_core_4 = aff3ct.tools.interleaver_core.Interleaver_core_LTE(Kb_4)
itl_bit_4  = aff3ct.module.interleaver.Interleaver_int32(itl_core_4)
itl_llr_4  = aff3ct.module.interleaver.Interleaver_float(itl_core_4)
inter_4 = interlacer.interlacer(Kb_4, tail)
rm_4 = LTE_rate_matcher.LTE_rate_matcher(M_4, N_4, matcher_ind_4)
mdm_4 = aff3ct.module.modem.Modem_generic(N_4, cstl_4)
frm_4 = framing.Framing(Ns_4*2, FS)
[chn_p_4, channel_h_4] = config_MCS.get_chn_config(channel_name, Ns_4*2+FS)
gen_4 = aff3ct.tools.Gaussian_noise_generator_implem.FAST
chn_4 = aff3ct.module.channel.Channel_AWGN_LLR(Ns_4*2+FS, gen_4)
b0_4 = B0.Block0(N_4, Q_4)
b1_4 = B1.Block1(N_4, Q_4)
b3_4 = B3_cons_4(N_4, Q_4)
b4_4 = B4_cons_4(N_4, Q_4, ntanh)
b5_4 = B5_anal_fp_fact.Block5(N_4, Q_4)
swi_4 = aff3ct.module.switcher.Switcher(2, N_4, np.float32)
cnt_4 = aff3ct.module.iterator.Iterator(S)
LaZero_4 = np.zeros((1,N_4), np.float32)
x_d_F_4 = np.zeros((1,2*Ns_4), np.float32)
v_d_4 = np.ones((1,1), np.float32)
log_P_4 = np.zeros((1,Ns_4*(2**Q_4)), np.float32)
rd_4 = LTE_rate_dematcher.LTE_rate_dematcher(N_4, M_4, dematcher_ind_4, cyc_shift_4)
trellis_n_4 = enc_n_4.get_trellis()
trellis_i_4 = enc_i_4.get_trellis()
benc_4 = buffer_encoding.BufferEncoding(Kb_4, tail)
dec_n_4 = aff3ct.module.decoder.Decoder_RSC_BCJR_seq_std(Kb_4, trellis_n_4, True)
dec_i_4 = aff3ct.module.decoder.Decoder_RSC_BCJR_seq_std(Kb_4, trellis_i_4, True)
dec_4 = aff3ct.module.decoder.Decoder_turbo_std(Kb_4,M_4,8,dec_n_4,dec_i_4,itl_llr_4, True)
mnt_4 = aff3ct.module.monitor.Monitor_BFER_AR(Kb_4, FE, maxF)


src_5 = aff3ct.module.source.Source_random_fast(Kb_5)
enc_n_5 = aff3ct.module.encoder.Encoder_RSC_generic_sys(Kb_5, 2*Kb_5+tail, False, [11, 13])
enc_i_5 = aff3ct.module.encoder.Encoder_RSC_generic_sys(Kb_5, 2*Kb_5+tail, False, [11, 13])
itl_core_5 = aff3ct.tools.interleaver_core.Interleaver_core_LTE(Kb_5)
itl_bit_5  = aff3ct.module.interleaver.Interleaver_int32(itl_core_5)
itl_llr_5  = aff3ct.module.interleaver.Interleaver_float(itl_core_5)
inter_5 = interlacer.interlacer(Kb_5, tail)
rm_5 = LTE_rate_matcher.LTE_rate_matcher(M_5, N_5, matcher_ind_5)
mdm_5 = aff3ct.module.modem.Modem_generic(N_5, cstl_5)
frm_5 = framing.Framing(Ns_5*2, FS)
[chn_p_5, channel_h_5] = config_MCS.get_chn_config(channel_name, Ns_5*2+FS)
gen_5 = aff3ct.tools.Gaussian_noise_generator_implem.FAST
chn_5 = aff3ct.module.channel.Channel_AWGN_LLR(Ns_5*2+FS, gen_5)
b0_5 = B0.Block0(N_5, Q_5)
b1_5 = B1.Block1(N_5, Q_5)
b3_5 = B3_cons_5(N_5, Q_5)
b4_5 = B4_cons_5(N_5, Q_5, ntanh)
b5_5 = B5_anal_fp_fact.Block5(N_5, Q_5)
swi_5 = aff3ct.module.switcher.Switcher(2, N_5, np.float32)
cnt_5 = aff3ct.module.iterator.Iterator(S)
LaZero_5 = np.zeros((1,N_5), np.float32)
x_d_F_5 = np.zeros((1,2*Ns_5), np.float32)
v_d_5 = np.ones((1,1), np.float32)
log_P_5 = np.zeros((1,Ns_5*(2**Q_5)), np.float32)
rd_5 = LTE_rate_dematcher.LTE_rate_dematcher(N_5, M_5, dematcher_ind_5, cyc_shift_5)
trellis_n_5 = enc_n_5.get_trellis()
trellis_i_5 = enc_i_5.get_trellis()
benc_5 = buffer_encoding.BufferEncoding(Kb_5, tail)
dec_n_5 = aff3ct.module.decoder.Decoder_RSC_BCJR_seq_std(Kb_5, trellis_n_5, True)
dec_i_5 = aff3ct.module.decoder.Decoder_RSC_BCJR_seq_std(Kb_5, trellis_i_5, True)
dec_5 = aff3ct.module.decoder.Decoder_turbo_std(Kb_5,M_5,8,dec_n_5,dec_i_5,itl_llr_5, True)
mnt_5 = aff3ct.module.monitor.Monitor_BFER_AR(Kb_5, FE, maxF)


src_6 = aff3ct.module.source.Source_random_fast(Kb_6)
enc_n_6 = aff3ct.module.encoder.Encoder_RSC_generic_sys(Kb_6, 2*Kb_6+tail, False, [11, 13])
enc_i_6 = aff3ct.module.encoder.Encoder_RSC_generic_sys(Kb_6, 2*Kb_6+tail, False, [11, 13])
itl_core_6 = aff3ct.tools.interleaver_core.Interleaver_core_LTE(Kb_6)
itl_bit_6  = aff3ct.module.interleaver.Interleaver_int32(itl_core_6)
itl_llr_6  = aff3ct.module.interleaver.Interleaver_float(itl_core_6)
inter_6 = interlacer.interlacer(Kb_6, tail)
rm_6 = LTE_rate_matcher.LTE_rate_matcher(M_6, N_6, matcher_ind_6)
mdm_6 = aff3ct.module.modem.Modem_generic(N_6, cstl_6)
frm_6 = framing.Framing(Ns_6*2, FS)
[chn_p_6, channel_h_6] = config_MCS.get_chn_config(channel_name, Ns_6*2+FS)
gen_6 = aff3ct.tools.Gaussian_noise_generator_implem.FAST
chn_6 = aff3ct.module.channel.Channel_AWGN_LLR(Ns_6*2+FS, gen_6)
b0_6 = B0.Block0(N_6, Q_6)
b1_6 = B1.Block1(N_6, Q_6)
b3_6 = B3_cons_6(N_6, Q_6)
b4_6 = B4_cons_6(N_6, Q_6, ntanh)
b5_6 = B5_anal_fp_fact.Block5(N_6, Q_6)
swi_6 = aff3ct.module.switcher.Switcher(2, N_6, np.float32)
cnt_6 = aff3ct.module.iterator.Iterator(S)
LaZero_6 = np.zeros((1,N_6), np.float32)
x_d_F_6 = np.zeros((1,2*Ns_6), np.float32)
v_d_6 = np.ones((1,1), np.float32)
log_P_6 = np.zeros((1,Ns_6*(2**Q_6)), np.float32)
rd_6 = LTE_rate_dematcher.LTE_rate_dematcher(N_6, M_6, dematcher_ind_6, cyc_shift_6)
trellis_n_6 = enc_n_6.get_trellis()
trellis_i_6 = enc_i_6.get_trellis()
benc_6 = buffer_encoding.BufferEncoding(Kb_6, tail)
dec_n_6 = aff3ct.module.decoder.Decoder_RSC_BCJR_seq_std(Kb_6, trellis_n_6, True)
dec_i_6 = aff3ct.module.decoder.Decoder_RSC_BCJR_seq_std(Kb_6, trellis_i_6, True)
dec_6 = aff3ct.module.decoder.Decoder_turbo_std(Kb_6,M_6,8,dec_n_6,dec_i_6,itl_llr_6, True)
mnt_6 = aff3ct.module.monitor.Monitor_BFER_AR(Kb_6, FE, maxF)


mnt_1["check_errors::U"] = src_1["generate::U_K"]
enc_n_1["encode::U_K"] = src_1["generate::U_K"]
itl_bit_1["interleave::nat"] = src_1["generate::U_K"]
enc_i_1["encode::U_K"] = itl_bit_1["interleave::itl"]
inter_1["interlace::x_n"] = enc_n_1["encode::X_N"]
inter_1["interlace::x_i"] = enc_i_1["encode::X_N"]
rm_1["rate_match::x"] = inter_1["interlace::y"]
mdm_1["modulate::X_N1"] = rm_1["rate_match::y"]
frm_1["frame::IN"] = mdm_1["modulate::X_N2"]
chn_p_1["add_noise::X_N"] = frm_1["frame::OUT"]
chn_1["add_noise::X_N"] = chn_p_1["add_noise::Y_N"]
frm_1["deframe::IN"] = chn_1["add_noise::Y_N"]
b0_1["decode::y"] = frm_1["deframe::OUT"]
b1_1["decode::y_F"] = b0_1["decode::y_F"]
b1_1["decode::Phi"] = b0_1["decode::Phi"]
b1_1["decode::Psi"] = b0_1["decode::Psi"]
# b1_1["decode::x_d_F"] = x_d_F_1
# b1_1["decode::v_d"] = v_d_1
b1_1["decode::x_d_F"] = b5_1["decode::x_d_F"]
b1_1["decode::v_d"] = b5_1["decode::v_d"]
b3_1["decode::x_e"] = b1_1["decode::x_e"]
b3_1["decode::v_e"] = b1_1["decode::v_e"]
swi_1["commute::data "] = b3_1["decode::Le"]
rd_1["rate_dematch::x"] = swi_1["commute::data1"]
swi_1["select::data0"] = swi_1["commute::data0"]
cnt_1["iterate"] = swi_1["select::status"]
swi_1["commute::ctrl "] = cnt_1["iterate::out"]
b4_1["decode::Le"] = swi_1["select::data"]
b4_1["decode::v_e"] = b1_1["decode::v_e"]
b5_1["decode::x_e"] = b1_1["decode::x_e"]
b5_1["decode::v_e"] = b1_1["decode::v_e"]
b5_1["decode::mi_d"] = b4_1["decode::mi_d"]
b5_1["decode::Cep"] = b4_1["decode::Cep"]
inter_1["deinterlace::x"] = rd_1["rate_dematch::y"]
benc_1["buf_enc::x"] = inter_1["deinterlace::y"]
dec_1["decode_siho::Y_N"] = benc_1["buf_enc::y"]
mnt_1["check_errors::V"] = dec_1["decode_siho ::V_K"]


mnt_2["check_errors::U"] = src_2["generate::U_K"]
enc_n_2["encode::U_K"] = src_2["generate::U_K"]
itl_bit_2["interleave::nat"] = src_2["generate::U_K"]
enc_i_2["encode::U_K"] = itl_bit_2["interleave::itl"]
inter_2["interlace::x_n"] = enc_n_2["encode::X_N"]
inter_2["interlace::x_i"] = enc_i_2["encode::X_N"]
rm_2["rate_match::x"] = inter_2["interlace::y"]
mdm_2["modulate::X_N1"] = rm_2["rate_match::y"]
frm_2["frame::IN"] = mdm_2["modulate::X_N2"]
chn_p_2["add_noise::X_N"] = frm_2["frame::OUT"]
chn_2["add_noise::X_N"] = chn_p_2["add_noise::Y_N"]
frm_2["deframe::IN"] = chn_2["add_noise::Y_N"]
b0_2["decode::y"] = frm_2["deframe::OUT"]
b1_2["decode::y_F"] = b0_2["decode::y_F"]
b1_2["decode::Phi"] = b0_2["decode::Phi"]
b1_2["decode::Psi"] = b0_2["decode::Psi"]
# b1_2["decode::x_d_F"] = x_d_F_2
# b1_2["decode::v_d"] = v_d_2
b1_2["decode::x_d_F"] = b5_2["decode::x_d_F"]
b1_2["decode::v_d"] = b5_2["decode::v_d"]
b3_2["decode::x_e"] = b1_2["decode::x_e"]
b3_2["decode::v_e"] = b1_2["decode::v_e"]
swi_2["commute::data "] = b3_2["decode::Le"]
rd_2["rate_dematch::x"] = swi_2["commute::data1"]
swi_2["select::data0"] = swi_2["commute::data0"]
cnt_2["iterate"] = swi_2["select::status"]
swi_2["commute::ctrl "] = cnt_2["iterate::out"]
b4_2["decode::Le"] = swi_2["select::data"]
b4_2["decode::v_e"] = b1_2["decode::v_e"]
b5_2["decode::x_e"] = b1_2["decode::x_e"]
b5_2["decode::v_e"] = b1_2["decode::v_e"]
b5_2["decode::mi_d"] = b4_2["decode::mi_d"]
b5_2["decode::Cep"] = b4_2["decode::Cep"]
inter_2["deinterlace::x"] = rd_2["rate_dematch::y"]
benc_2["buf_enc::x"] = inter_2["deinterlace::y"]
dec_2["decode_siho::Y_N"] = benc_2["buf_enc::y"]
mnt_2["check_errors::V"] = dec_2["decode_siho ::V_K"]


mnt_3["check_errors::U"] = src_3["generate::U_K"]
enc_n_3["encode::U_K"] = src_3["generate::U_K"]
itl_bit_3["interleave::nat"] = src_3["generate::U_K"]
enc_i_3["encode::U_K"] = itl_bit_3["interleave::itl"]
inter_3["interlace::x_n"] = enc_n_3["encode::X_N"]
inter_3["interlace::x_i"] = enc_i_3["encode::X_N"]
rm_3["rate_match::x"] = inter_3["interlace::y"]
mdm_3["modulate::X_N1"] = rm_3["rate_match::y"]
frm_3["frame::IN"] = mdm_3["modulate::X_N2"]
chn_p_3["add_noise::X_N"] = frm_3["frame::OUT"]
chn_3["add_noise::X_N"] = chn_p_3["add_noise::Y_N"]
frm_3["deframe::IN"] = chn_3["add_noise::Y_N"]
b0_3["decode::y"] = frm_3["deframe::OUT"]
b1_3["decode::y_F"] = b0_3["decode::y_F"]
b1_3["decode::Phi"] = b0_3["decode::Phi"]
b1_3["decode::Psi"] = b0_3["decode::Psi"]
# b1_3["decode::x_d_F"] = x_d_F_3
# b1_3["decode::v_d"] = v_d_3
b1_3["decode::x_d_F"] = b5_3["decode::x_d_F"]
b1_3["decode::v_d"] = b5_3["decode::v_d"]
b3_3["decode::x_e"] = b1_3["decode::x_e"]
b3_3["decode::v_e"] = b1_3["decode::v_e"]
swi_3["commute::data "] = b3_3["decode::Le"]
rd_3["rate_dematch::x"] = swi_3["commute::data1"]
swi_3["select::data0"] = swi_3["commute::data0"]
cnt_3["iterate"] = swi_3["select::status"]
swi_3["commute::ctrl "] = cnt_3["iterate::out"]
b4_3["decode::Le"] = swi_3["select::data"]
b4_3["decode::v_e"] = b1_3["decode::v_e"]
b5_3["decode::x_e"] = b1_3["decode::x_e"]
b5_3["decode::v_e"] = b1_3["decode::v_e"]
b5_3["decode::mi_d"] = b4_3["decode::mi_d"]
b5_3["decode::Cep"] = b4_3["decode::Cep"]
inter_3["deinterlace::x"] = rd_3["rate_dematch::y"]
benc_3["buf_enc::x"] = inter_3["deinterlace::y"]
dec_3["decode_siho::Y_N"] = benc_3["buf_enc::y"]
mnt_3["check_errors::V"] = dec_3["decode_siho ::V_K"]


mnt_4["check_errors::U"] = src_4["generate::U_K"]
enc_n_4["encode::U_K"] = src_4["generate::U_K"]
itl_bit_4["interleave::nat"] = src_4["generate::U_K"]
enc_i_4["encode::U_K"] = itl_bit_4["interleave::itl"]
inter_4["interlace::x_n"] = enc_n_4["encode::X_N"]
inter_4["interlace::x_i"] = enc_i_4["encode::X_N"]
rm_4["rate_match::x"] = inter_4["interlace::y"]
mdm_4["modulate::X_N1"] = rm_4["rate_match::y"]
frm_4["frame::IN"] = mdm_4["modulate::X_N2"]
chn_p_4["add_noise::X_N"] = frm_4["frame::OUT"]
chn_4["add_noise::X_N"] = chn_p_4["add_noise::Y_N"]
frm_4["deframe::IN"] = chn_4["add_noise::Y_N"]
b0_4["decode::y"] = frm_4["deframe::OUT"]
b1_4["decode::y_F"] = b0_4["decode::y_F"]
b1_4["decode::Phi"] = b0_4["decode::Phi"]
b1_4["decode::Psi"] = b0_4["decode::Psi"]
# b1_4["decode::x_d_F"] = x_d_F_4
# b1_4["decode::v_d"] = v_d_4
b1_4["decode::x_d_F"] = b5_4["decode::x_d_F"]
b1_4["decode::v_d"] = b5_4["decode::v_d"]
b3_4["decode::x_e"] = b1_4["decode::x_e"]
b3_4["decode::v_e"] = b1_4["decode::v_e"]
swi_4["commute::data "] = b3_4["decode::Le"]
rd_4["rate_dematch::x"] = swi_4["commute::data1"]
swi_4["select::data0"] = swi_4["commute::data0"]
cnt_4["iterate"] = swi_4["select::status"]
swi_4["commute::ctrl "] = cnt_4["iterate::out"]
b4_4["decode::Le"] = swi_4["select::data"]
b4_4["decode::v_e"] = b1_4["decode::v_e"]
b5_4["decode::x_e"] = b1_4["decode::x_e"]
b5_4["decode::v_e"] = b1_4["decode::v_e"]
b5_4["decode::mi_d"] = b4_4["decode::mi_d"]
b5_4["decode::Cep"] = b4_4["decode::Cep"]
inter_4["deinterlace::x"] = rd_4["rate_dematch::y"]
benc_4["buf_enc::x"] = inter_4["deinterlace::y"]
dec_4["decode_siho::Y_N"] = benc_4["buf_enc::y"]
mnt_4["check_errors::V"] = dec_4["decode_siho ::V_K"]


mnt_5["check_errors::U"] = src_5["generate::U_K"]
enc_n_5["encode::U_K"] = src_5["generate::U_K"]
itl_bit_5["interleave::nat"] = src_5["generate::U_K"]
enc_i_5["encode::U_K"] = itl_bit_5["interleave::itl"]
inter_5["interlace::x_n"] = enc_n_5["encode::X_N"]
inter_5["interlace::x_i"] = enc_i_5["encode::X_N"]
rm_5["rate_match::x"] = inter_5["interlace::y"]
mdm_5["modulate::X_N1"] = rm_5["rate_match::y"]
frm_5["frame::IN"] = mdm_5["modulate::X_N2"]
chn_p_5["add_noise::X_N"] = frm_5["frame::OUT"]
chn_5["add_noise::X_N"] = chn_p_5["add_noise::Y_N"]
frm_5["deframe::IN"] = chn_5["add_noise::Y_N"]
b0_5["decode::y"] = frm_5["deframe::OUT"]
b1_5["decode::y_F"] = b0_5["decode::y_F"]
b1_5["decode::Phi"] = b0_5["decode::Phi"]
b1_5["decode::Psi"] = b0_5["decode::Psi"]
# b1_5["decode::x_d_F"] = x_d_F_5
# b1_5["decode::v_d"] = v_d_5
b1_5["decode::x_d_F"] = b5_5["decode::x_d_F"]
b1_5["decode::v_d"] = b5_5["decode::v_d"]
b3_5["decode::x_e"] = b1_5["decode::x_e"]
b3_5["decode::v_e"] = b1_5["decode::v_e"]
swi_5["commute::data "] = b3_5["decode::Le"]
rd_5["rate_dematch::x"] = swi_5["commute::data1"]
swi_5["select::data0"] = swi_5["commute::data0"]
cnt_5["iterate"] = swi_5["select::status"]
swi_5["commute::ctrl "] = cnt_5["iterate::out"]
b4_5["decode::Le"] = swi_5["select::data"]
b4_5["decode::v_e"] = b1_5["decode::v_e"]
b5_5["decode::x_e"] = b1_5["decode::x_e"]
b5_5["decode::v_e"] = b1_5["decode::v_e"]
b5_5["decode::mi_d"] = b4_5["decode::mi_d"]
b5_5["decode::Cep"] = b4_5["decode::Cep"]
inter_5["deinterlace::x"] = rd_5["rate_dematch::y"]
benc_5["buf_enc::x"] = inter_5["deinterlace::y"]
dec_5["decode_siho::Y_N"] = benc_5["buf_enc::y"]
mnt_5["check_errors::V"] = dec_5["decode_siho ::V_K"]


mnt_6["check_errors::U"] = src_6["generate::U_K"]
enc_n_6["encode::U_K"] = src_6["generate::U_K"]
itl_bit_6["interleave::nat"] = src_6["generate::U_K"]
enc_i_6["encode::U_K"] = itl_bit_6["interleave::itl"]
inter_6["interlace::x_n"] = enc_n_6["encode::X_N"]
inter_6["interlace::x_i"] = enc_i_6["encode::X_N"]
rm_6["rate_match::x"] = inter_6["interlace::y"]
mdm_6["modulate::X_N1"] = rm_6["rate_match::y"]
frm_6["frame::IN"] = mdm_6["modulate::X_N2"]
chn_p_6["add_noise::X_N"] = frm_6["frame::OUT"]
chn_6["add_noise::X_N"] = chn_p_6["add_noise::Y_N"]
frm_6["deframe::IN"] = chn_6["add_noise::Y_N"]
b0_6["decode::y"] = frm_6["deframe::OUT"]
b1_6["decode::y_F"] = b0_6["decode::y_F"]
b1_6["decode::Phi"] = b0_6["decode::Phi"]
b1_6["decode::Psi"] = b0_6["decode::Psi"]
# b1_6["decode::x_d_F"] = x_d_F_6
# b1_6["decode::v_d"] = v_d_6
b1_6["decode::x_d_F"] = b5_6["decode::x_d_F"]
b1_6["decode::v_d"] = b5_6["decode::v_d"]
b3_6["decode::x_e"] = b1_6["decode::x_e"]
b3_6["decode::v_e"] = b1_6["decode::v_e"]
swi_6["commute::data "] = b3_6["decode::Le"]
rd_6["rate_dematch::x"] = swi_6["commute::data1"]
swi_6["select::data0"] = swi_6["commute::data0"]
cnt_6["iterate"] = swi_6["select::status"]
swi_6["commute::ctrl "] = cnt_6["iterate::out"]
b4_6["decode::Le"] = swi_6["select::data"]
b4_6["decode::v_e"] = b1_6["decode::v_e"]
b5_6["decode::x_e"] = b1_6["decode::x_e"]
b5_6["decode::v_e"] = b1_6["decode::v_e"]
b5_6["decode::mi_d"] = b4_6["decode::mi_d"]
b5_6["decode::Cep"] = b4_6["decode::Cep"]
inter_6["deinterlace::x"] = rd_6["rate_dematch::y"]
benc_6["buf_enc::x"] = inter_6["deinterlace::y"]
dec_6["decode_siho::Y_N"] = benc_6["buf_enc::y"]
mnt_6["check_errors::V"] = dec_6["decode_siho ::V_K"]


sigma = np.ndarray(shape = chn_1["add_noise::CP"][:].shape, dtype = np.float32)
chn_1["add_noise::CP"] = sigma
chn_p_1["add_noise::CP"] = sigma
b0_1["decode::h"] = channel_h_1
b1_1["decode::h"] = channel_h_1

chn_2["add_noise::CP"] = sigma
chn_p_2["add_noise::CP"] = sigma
b0_2["decode::h"] = channel_h_2
b1_2["decode::h"] = channel_h_2

chn_3["add_noise::CP"] = sigma
chn_p_3["add_noise::CP"] = sigma
b0_3["decode::h"] = channel_h_3
b1_3["decode::h"] = channel_h_3

chn_4["add_noise::CP"] = sigma
chn_p_4["add_noise::CP"] = sigma
b0_4["decode::h"] = channel_h_4
b1_4["decode::h"] = channel_h_4

chn_5["add_noise::CP"] = sigma
chn_p_5["add_noise::CP"] = sigma
b0_5["decode::h"] = channel_h_5
b1_5["decode::h"] = channel_h_5

chn_6["add_noise::CP"] = sigma
chn_p_6["add_noise::CP"] = sigma
b0_6["decode::h"] = channel_h_6
b1_6["decode::h"] = channel_h_6


Z_1 = np.zeros  (shape = swi_1["select::data1"][:].shape, dtype = np.float32)
swi_1["select::data1"] = Z_1

Z_2 = np.zeros  (shape = swi_2["select::data1"][:].shape, dtype = np.float32)
swi_2["select::data1"] = Z_2

Z_3 = np.zeros  (shape = swi_3["select::data1"][:].shape, dtype = np.float32)
swi_3["select::data1"] = Z_3

Z_4 = np.zeros  (shape = swi_4["select::data1"][:].shape, dtype = np.float32)
swi_4["select::data1"] = Z_4

Z_5 = np.zeros  (shape = swi_5["select::data1"][:].shape, dtype = np.float32)
swi_5["select::data1"] = Z_5

Z_6 = np.zeros  (shape = swi_6["select::data1"][:].shape, dtype = np.float32)
swi_6["select::data1"] = Z_6

var = np.ndarray(shape = b0_1["decode::Var"][:].shape, dtype = np.float32)
b0_1["decode::Var"] = var
b0_2["decode::Var"] = var
b0_3["decode::Var"] = var
b0_4["decode::Var"] = var
b0_5["decode::Var"] = var
b0_6["decode::Var"] = var


ber_1 = np.zeros(len(esn0))
fer_1 = np.zeros(len(esn0))
ser_1 = np.zeros(len(esn0))
ber_2 = np.zeros(len(esn0))
fer_2 = np.zeros(len(esn0))
ser_2 = np.zeros(len(esn0))
ber_3 = np.zeros(len(esn0))
fer_3 = np.zeros(len(esn0))
ser_3 = np.zeros(len(esn0))
ber_4 = np.zeros(len(esn0))
fer_4 = np.zeros(len(esn0))
ser_4 = np.zeros(len(esn0))
ber_5 = np.zeros(len(esn0))
fer_5 = np.zeros(len(esn0))
ser_5 = np.zeros(len(esn0))
ber_6 = np.zeros(len(esn0))
fer_6 = np.zeros(len(esn0))
ser_6 = np.zeros(len(esn0))
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

print(" Es/NO (dB) | Frame number |   BER 1  |   FER 1  |   BER 2  |   FER 2  |   BER 3  |   FER 3  |   BER 4  |   FER 4  |   BER 5  |   FER 5  |   BER 6  |   FER 6  |  Tpt (Mbps)|  Time (s)")
print("------------|--------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|------------|-----------")
for i in range(len(sigma_vals)):
	sigma[:] = sigma_vals[i]
	var[:] = var_vals[i]
	# SNR[:] = esn0[i]

	t = time.time()
	# seq.exec()
	f_1 = mnt_1.is_done()
	f_2 = mnt_2.is_done()
	f_3 = mnt_3.is_done()
	f_4 = mnt_4.is_done()
	f_5 = mnt_5.is_done()
	f_6 = mnt_6.is_done()
	f_is_done = f_1 and f_2 and f_3 and f_4 and f_5 and f_6
	while not f_is_done:
		if not f_1:
			cnt_1["iterate"].exec()
			src_1["generate"].exec()
			enc_n_1["encode"].exec()
			itl_bit_1["interleave"].exec()
			enc_i_1["encode"].exec()
			inter_1["interlace"].exec()
			rm_1["rate_match"].exec()
			mdm_1["modulate"].exec()
			frm_1["frame"].exec()
			chn_p_1["add_noise"].exec()
			chn_1["add_noise"].exec()
			frm_1["deframe"].exec()
			b0_1["decode"].exec()
			b1_1.rst()
			for s in range(S):
				b1_1["decode"].exec()
				b3_1["decode"].exec()
				swi_1["commute"].exec()
				swi_1["select"].exec()
				cnt_1["iterate"].exec()
				b4_1["decode"].exec()
				b5_1["decode"].exec()
			b1_1["decode"].exec()
			b3_1["decode"].exec()
			swi_1["commute"].exec()
			rd_1["rate_dematch"].exec()
			inter_1["deinterlace"].exec()
			benc_1["buf_enc"].exec()
			dec_1["decode_siho"].exec()
			mnt_1["check_errors"].exec()
			f_1 = mnt_1.is_done()

		if not f_2:
			cnt_2["iterate"].exec()
			src_2["generate"].exec()
			enc_n_2["encode"].exec()
			itl_bit_2["interleave"].exec()
			enc_i_2["encode"].exec()
			inter_2["interlace"].exec()
			rm_2["rate_match"].exec()
			mdm_2["modulate"].exec()
			frm_2["frame"].exec()
			chn_p_2["add_noise"].exec()
			chn_2["add_noise"].exec()
			frm_2["deframe"].exec()
			b0_2["decode"].exec()
			b1_2.rst()
			for s in range(S):
				b1_2["decode"].exec()
				b3_2["decode"].exec()
				swi_2["commute"].exec()
				swi_2["select"].exec()
				cnt_2["iterate"].exec()
				b4_2["decode"].exec()
				b5_2["decode"].exec()
			b1_2["decode"].exec()
			b3_2["decode"].exec()
			swi_2["commute"].exec()
			rd_2["rate_dematch"].exec()
			inter_2["deinterlace"].exec()
			benc_2["buf_enc"].exec()
			dec_2["decode_siho"].exec()
			mnt_2["check_errors"].exec()
			f_2 = mnt_2.is_done()

		if not f_3:
			cnt_3["iterate"].exec()
			src_3["generate"].exec()
			enc_n_3["encode"].exec()
			itl_bit_3["interleave"].exec()
			enc_i_3["encode"].exec()
			inter_3["interlace"].exec()
			rm_3["rate_match"].exec()
			mdm_3["modulate"].exec()
			frm_3["frame"].exec()
			chn_p_3["add_noise"].exec()
			chn_3["add_noise"].exec()
			frm_3["deframe"].exec()
			b0_3["decode"].exec()
			b1_3.rst()
			for s in range(S):
				b1_3["decode"].exec()
				b3_3["decode"].exec()
				swi_3["commute"].exec()
				swi_3["select"].exec()
				cnt_3["iterate"].exec()
				b4_3["decode"].exec()
				b5_3["decode"].exec()
			b1_3["decode"].exec()
			b3_3["decode"].exec()
			swi_3["commute"].exec()
			rd_3["rate_dematch"].exec()
			inter_3["deinterlace"].exec()
			benc_3["buf_enc"].exec()
			dec_3["decode_siho"].exec()
			mnt_3["check_errors"].exec()
			f_3 = mnt_3.is_done()

		if not f_4:
			cnt_4["iterate"].exec()
			src_4["generate"].exec()
			enc_n_4["encode"].exec()
			itl_bit_4["interleave"].exec()
			enc_i_4["encode"].exec()
			inter_4["interlace"].exec()
			rm_4["rate_match"].exec()
			mdm_4["modulate"].exec()
			frm_4["frame"].exec()
			chn_p_4["add_noise"].exec()
			chn_4["add_noise"].exec()
			frm_4["deframe"].exec()
			b0_4["decode"].exec()
			b1_4.rst()
			for s in range(S):
				b1_4["decode"].exec()
				b3_4["decode"].exec()
				swi_4["commute"].exec()
				swi_4["select"].exec()
				cnt_4["iterate"].exec()
				b4_4["decode"].exec()
				b5_4["decode"].exec()
			b1_4["decode"].exec()
			b3_4["decode"].exec()
			swi_4["commute"].exec()
			rd_4["rate_dematch"].exec()
			inter_4["deinterlace"].exec()
			benc_4["buf_enc"].exec()
			dec_4["decode_siho"].exec()
			mnt_4["check_errors"].exec()
			f_4 = mnt_4.is_done()

		if not f_5:
			cnt_5["iterate"].exec()
			src_5["generate"].exec()
			enc_n_5["encode"].exec()
			itl_bit_5["interleave"].exec()
			enc_i_5["encode"].exec()
			inter_5["interlace"].exec()
			rm_5["rate_match"].exec()
			mdm_5["modulate"].exec()
			frm_5["frame"].exec()
			chn_p_5["add_noise"].exec()
			chn_5["add_noise"].exec()
			frm_5["deframe"].exec()
			b0_5["decode"].exec()
			b1_5.rst()
			for s in range(S):
				b1_5["decode"].exec()
				b3_5["decode"].exec()
				swi_5["commute"].exec()
				swi_5["select"].exec()
				cnt_5["iterate"].exec()
				b4_5["decode"].exec()
				b5_5["decode"].exec()
			b1_5["decode"].exec()
			b3_5["decode"].exec()
			swi_5["commute"].exec()
			rd_5["rate_dematch"].exec()
			inter_5["deinterlace"].exec()
			benc_5["buf_enc"].exec()
			dec_5["decode_siho"].exec()
			mnt_5["check_errors"].exec()
			f_5 = mnt_5.is_done()

		if not f_6:
			cnt_6["iterate"].exec()
			src_6["generate"].exec()
			enc_n_6["encode"].exec()
			itl_bit_6["interleave"].exec()
			enc_i_6["encode"].exec()
			inter_6["interlace"].exec()
			rm_6["rate_match"].exec()
			mdm_6["modulate"].exec()
			frm_6["frame"].exec()
			chn_p_6["add_noise"].exec()
			chn_6["add_noise"].exec()
			frm_6["deframe"].exec()
			b0_6["decode"].exec()
			b1_6.rst()
			for s in range(S):
				b1_6["decode"].exec()
				b3_6["decode"].exec()
				swi_6["commute"].exec()
				swi_6["select"].exec()
				cnt_6["iterate"].exec()
				b4_6["decode"].exec()
				b5_6["decode"].exec()
			b1_6["decode"].exec()
			b3_6["decode"].exec()
			swi_6["commute"].exec()
			rd_6["rate_dematch"].exec()
			inter_6["deinterlace"].exec()
			benc_6["buf_enc"].exec()
			dec_6["decode_siho"].exec()
			mnt_6["check_errors"].exec()
			f_6 = mnt_6.is_done()

		f_is_done = f_1 and f_2 and f_3 and f_4 and f_5 and f_6

	elapsed = time.time() - t

	total_fra = mnt_6.get_n_analyzed_fra()
	ber_1[i] = mnt_1.get_ber()
	fer_1[i] = mnt_1.get_fer()
	ber_2[i] = mnt_2.get_ber()
	fer_2[i] = mnt_2.get_fer()
	ber_3[i] = mnt_3.get_ber()
	fer_3[i] = mnt_3.get_fer()
	ber_4[i] = mnt_4.get_ber()
	fer_4[i] = mnt_4.get_fer()
	ber_5[i] = mnt_5.get_ber()
	fer_5[i] = mnt_5.get_fer()
	ber_6[i] = mnt_6.get_ber()
	fer_6[i] = mnt_6.get_fer()

	tpt = total_fra * K * 1e-6/elapsed

	print("%11.2f | %12d | %7.2e | %7.2e | %7.2e | %7.2e | %7.2e | %7.2e | %7.2e | %7.2e | %7.2e | %7.2e | %7.2e | %7.2e | %10.2f | %10.2f"%(esn0[i], total_fra, ber_1[i], fer_1[i], ber_2[i], fer_2[i], ber_3[i], fer_3[i], ber_4[i], fer_4[i], ber_5[i], fer_5[i], ber_6[i], fer_6[i], tpt, elapsed))

	src_1.reset()
	cnt_1.reset()
	mnt_1.reset()
	src_2.reset()
	cnt_2.reset()
	mnt_2.reset()
	src_3.reset()
	cnt_3.reset()
	mnt_3.reset()
	src_4.reset()
	cnt_4.reset()
	mnt_4.reset()
	src_5.reset()
	cnt_5.reset()
	mnt_5.reset()
	src_6.reset()
	cnt_6.reset()
	mnt_6.reset()

	# for m in seq.get_modules_reset():
	# 	m.reset()

	if(ber_1[i] < target_ber):
		break

# seq.show_stats()