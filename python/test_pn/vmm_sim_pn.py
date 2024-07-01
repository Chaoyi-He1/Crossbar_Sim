import numpy as np
class VmmSim:
    def __init__(self, float_in, float_weight, v_range, v_bit, v_no, g_range,g_bit, g_no,out_range):
        self.float_in = float_in
        self.float_weight = float_weight
        self.rows, self.cols = float_weight.shape
        self.batch_no, _ = float_in.shape
        self.float_out = np.matmul(self.float_in,self.float_weight)
        self.v_range = v_range
        self.v_bit = v_bit
        self.v_no = v_no
        self.g_range = g_range
        self.g_bit = g_bit
        self.g_no = g_no
        self.out_range = out_range
        print('VmmSim init done')

    def weight_mapping_pn(self):
        float_w = self.float_weight
        float_w[np.abs(float_w)<1e-4] = 0
        if self.g_no == 1:
            int_weight_all = np.zeros(float_w.shape,dtype=np.uint8)
            float_weight_all = np.zeros(float_w.shape)
            w_scale = np.zeros(float_w.shape[1])
            w_shift = np.zeros(float_w.shape[1])
            for j in range(float_w.shape[1]):
                int_weight_all[:,j], float_weight_all[:,j],w_scale[j],w_shift[j] =\
                    self.quantize_vec(float_w[:,j],self.g_range,self.g_bit)
        else:
            pass
        w_test = float_weight_all*w_scale+w_shift
        return int_weight_all, float_weight_all, w_scale, w_shift


    def input_mapping_pn(self):
        input_float_p = np.zeros(self.float_in.shape)
        input_float_n = np.zeros(self.float_in.shape)
        input_float_p[self.float_in > 0]= self.float_in[self.float_in>0]
        input_float_n[self.float_in < 0] = -self.float_in[self.float_in < 0]
        input_test = input_float_p-input_float_n

        input_float_pn = np.concatenate((input_float_p,input_float_n),axis=0)

        int_input_slice, float_input_slice, in_scale, in_shift = self.quantize_data(input_float_pn, self.v_range, self.v_bit,self.v_no)

        int_input_all = int_input_slice.reshape(self.v_no*self.batch_no*2, self.rows)
        float_input_all = float_input_slice.reshape(self.v_no*self.batch_no*2, self.rows)

        return int_input_all, float_input_all, in_scale, in_shift

    def quantize_vmm_pn(self):
        qtz_weight_pn,float_weight_pn, w_scale_pn, w_shift_pn = self.weight_mapping_pn()
        qtz_in_pn, float_in_pn, in_scale_pn, in_shift_pn = self.input_mapping_pn()
        int_out, out_full_pn, output_divisor_pn = self.expect_out(qtz_in_pn, qtz_weight_pn, self.out_range)
        return int_out, output_divisor_pn, qtz_in_pn, qtz_weight_pn, in_scale_pn, in_shift_pn, w_scale_pn, w_shift_pn, float_in_pn, float_weight_pn
    def reverse_vmm_pn(self, output, output_divisor, float_in_ideal, in_scale_pn, in_shift_pn, w_scale, w_shift):
        rescaled_output = output * output_divisor
        deduct_out_pn = np.zeros((self.batch_no,self.cols))
        no_state = np.power(2,self.v_bit)
        for i in [0, 1]:
            a_t = np.ones((self.batch_no*2,self.rows)) * in_scale_pn[i]
            d_t = np.ones((self.rows,self.cols)) * w_shift
            c_t = np.ones((self.rows,self.cols)) * w_scale
            ad_t = np.matmul(a_t*float_in_ideal[self.batch_no*2*i:self.batch_no*2*(i+1),:],d_t)
            ac_t = np.matmul(a_t,c_t)
            # ac_t = np.ones((self.batch_no*2, self.cols)) * w_scale * in_scale_pn[i]
            deduct_out_pn_t = (rescaled_output[self.batch_no*2*i:self.batch_no*2*(i+1),:] - ad_t) / ac_t
            deduct_out_pn = deduct_out_pn + (deduct_out_pn_t[:self.batch_no,:] - deduct_out_pn_t[self.batch_no:,:])/np.power(no_state,i)
        # deduct_out = deduct_out_pn[0:self.batch_no,:] - deduct_out_pn[self.batch_no::,:]
        deduct_out = deduct_out_pn
        max_output = np.max(deduct_out)
        min_output = np.min(deduct_out)
        int8_out = np.round((deduct_out - min_output) / (max_output - min_output) * (self.out_range[1]-self.out_range[0])+self.out_range[0])

        return int8_out.astype(np.uint8)
    def quantize_data(self, data, int_range, bit_no, no_slice):
        data_sliced = np.zeros((no_slice,data.shape[0],data.shape[1]),dtype=np.uint8)
        data_float_point = np.zeros((no_slice, data.shape[0], data.shape[1]))
        no_state = np.power(2, bit_no)
        data_point = np.zeros((no_slice,no_state))
        interval = np.zeros(no_slice)

        data_scale = np.zeros(no_slice)
        data_shift = np.zeros(no_slice)

        data_max = np.max(data)
        data_min = np.min(data)
        data_range = data_max - data_min

        codes = np.arange(0, no_state, 1,dtype=np.uint8)
        code_table = np.linspace(int_range[0], int_range[1], no_state, endpoint=True, dtype=np.uint8)

        for i in range(no_slice):
            if i == 0:
                interval[i] = data_range / no_state
                int_index = np.floor((data - data_min) / data_range * no_state).astype(np.uint8)
                data_point[i,:] = codes * interval[i] + data_min
            else:
                interval[i] = interval[i-1] / no_state
                int_index = np.floor(data / interval[i-1] * no_state).astype(np.uint8)
                data_point[i,:] = codes * interval[i]

            data_point_max = np.max(data_point[i, :])
            data_point_min = np.min(data_point[i, :])
            data_scale[i] = (int_range[1]-int_range[0])/(data_point_max-data_point_min)
            data_shift[i] = int_range[0] - data_scale[i]*data_point_min

            int_index = np.clip(int_index, 0, no_state - 1)
            data_sliced[i,:,:] = code_table[int_index]
            data_float_point[i,:,:] = data_point[i, :][int_index]
            data = data - data_float_point[i,:,:]

        return data_sliced, data_float_point, data_scale, data_shift

    def quantize_vec(self, vec, g_range, g_bit):
        no_state = np.power(2, g_bit)
        vec_max = np.max(vec)
        vec_min = np.min(vec)
        codes = np.arange(0, no_state, 1, dtype=np.uint8)
        code_table = np.linspace(g_range[0], g_range[1], no_state, endpoint=True, dtype=np.uint8)

        if vec_max == vec_min:
            vec_int = np.ones(vec.shape) * code_table[int(no_state/2)]
            vec_qtz_float = np.ones(vec.shape)*vec_min
            if vec_max == 0:
                vec_scale = 1e20
                vec_shift = code_table[int(no_state/2)]
            else:
                vec_scale = 1
                vec_shift = code_table[int(no_state/2)] - vec_min
        else:
            vec_range = vec_max - vec_min
            int_index = np.floor((vec - vec_min) / vec_range * no_state).astype(np.uint8)
            interval = vec_range / no_state
            vec_point = codes * interval + vec_min
            int_index = np.clip(int_index, 0, no_state - 1)
            vec_int = code_table[int_index]
            vec_qtz_float = vec_point[int_index]

            vec_qtz_min = np.min(vec_qtz_float)
            vec_qtz_max = np.max(vec_qtz_float)
            vec_qtz_range = vec_qtz_max-vec_qtz_min
            vec_scale = (g_range[1]-g_range[0])/vec_qtz_range
            vec_shift = g_range[0] - vec_qtz_min*vec_scale

        return vec_int, vec_qtz_float, vec_scale, vec_shift

    def expect_out(self,inputs,weights,out_range):
        full_output = np.matmul(inputs.astype(np.int32), weights.astype(np.int32))
        output_divisor = np.zeros((weights.shape[1]), dtype=np.int32)

        max_out = np.max(full_output,axis=0)
        min_out = np.min(full_output,axis=0)
        for k in range(full_output.shape[1]):
            full_channel = full_output[..., k]
            full_channel_max = np.max(full_channel)
            output_divisor[k] = np.floor((full_channel_max) / (out_range[1] - 1))

        expected_output = np.clip((full_output / output_divisor), 0, 255).astype(np.uint8)
        return expected_output, full_output, output_divisor

    def restore_data(self, data_real, data_sliced, data_float_point):
        no_slice = data_sliced.shape[0]
        data_restored = np.zeros((data_sliced.shape[1],data_sliced.shape[2]))
        for i in range(no_slice):
            data_restored = data_restored + data_real[i,:,:]/data_sliced[i,:,:] * data_float_point[i,:,:]
        return data_restored