experinment_out = 'out.csv';
experinment_out = readmatrix(experinment_out);
previous_out = zeros(1, M);   
exp_outputs_seq = zeros(1, k * N);
for i = 1:k
    input_vec = [previous_out(M / 2 + 1:end), Xt(i, :)];
    previous_out = input_vec * mtx_overlap;

    exp_outputs_seq(1 + (i - 1) * N:i * N) = experinment_out(i, :) + [previous_out(1, end-M/2+1:end), zeros(1, M / 2)];
end
clear outputs;
clear previous_out;

load('test_data.mat');

figure;
subplot(1, 2, 1);
sinad(outputs_seq, fs);
title("ideal SNDR");
subplot(1, 2, 2);
sinad(exp_outputs_seq, fs);
title("experinment SNDR");