clear;
close all;
clc;

%% initialization
N = 32;    % input points
N_f = N+1;
M = 64;

out_length = N+N_f-1;
k = 5;

fs = 20e3;
fc = 14*fs/M;
% fc_b = 10e9;  % Blocker/Jammer 

type_filter = "bandpass";   % bandpass bandstop lowpass highpass
f_l = fc-fs*2/N;
f_u = fc+fs*2/N;

t = 0:1/fs:(N*k-1)/fs;
f = -fs/2:fs/M:fs/2-fs/M;

[S, Xt] = input_signal(fc, fs, N, k);
mtx_non_overlap = toeplitz_matrix(N, M, 0 , fs, f_l, f_u, type_filter, 0, '');

ideal_output = Xt * mtx_non_overlap;

mtx_overlap = toeplitz_matrix(N, M, 0 , fs, f_l, f_u, type_filter, 1, '');
outputs_seq = zeros(1, k * N);
previous_out = zeros(1, M);   
outputs = zeros(k, M);
for i = 1:k
    input_vec = [previous_out(M / 2 + 1:end), Xt(i, :)];
    previous_out = input_vec * mtx_overlap;
    outputs(i, :) = previous_out;
    outputs_seq(1 + (i - 1) * N:i * N) = previous_out(1:N);
end
clear outputs;
clear previous_out;