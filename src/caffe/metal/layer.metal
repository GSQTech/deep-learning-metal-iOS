//
//  tanh_layer.metal
//  metal_mac
//
//  Created by Tec GSQ on 30/11/2017.
//  Copyright © 2017 Tec GSQ. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;


kernel void
TanHForward(const device int* int__ [[ buffer(0) ]],
            const device float *in [[buffer(1)]],
            device float *out [[buffer(2)]],
            uint2 gid [[ thread_position_in_grid ]],
            uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = tanh(in[index]);
    }
}


kernel void
SigmoidForward(const device int* int__ [[ buffer(0) ]],
            const device float *in [[buffer(1)]],
            device float *out [[buffer(2)]],
            uint2 gid [[ thread_position_in_grid ]],
            uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = 0.5 * tanh(0.5 * in[index]) + 0.5;
    }
}

kernel void
BNLLForward(const device int* int__ [[ buffer(0) ]],
            const device float *in [[buffer(1)]],
            device float *out [[buffer(2)]],
            uint2 gid [[ thread_position_in_grid ]],
            uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = in[index] > 0 ? in[index] + log(1. + exp(-in[index])) : log(1. + exp(in[index]));
    }
}


kernel void
ReLUForward(const device int* int__ [[ buffer(0) ]],
            const device float* float__ [[buffer(1)]],
            const device float *in [[buffer(2)]],
            device float *out [[buffer(3)]],
            uint2 gid [[ thread_position_in_grid ]],
            uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = in[index] > 0 ? in[index] : in[index] * float__[0];
    }
}



kernel void
CropForward(const device int* int__ [[ buffer(0) ]],
                    const device int* src_strides[[ buffer(1) ]],
                    const device int* dest_strides[[ buffer(2) ]],
                    const device int* offsets[[ buffer(3) ]],
                    const device float* src[[ buffer(4) ]],
                    device float* dest[[ buffer(5) ]],
                    uint2 gid [[ thread_position_in_grid ]],
                    uint2 tpg [[ threads_per_grid ]]) {

    for (int index = gid.x; index < int__[0]; index += tpg.x){
        int dest_index = index;
        int src_index = 0;
        for (int i = 0; i < int__[1]; ++i) {
            int coord = dest_index / dest_strides[i];
            dest_index -= coord * dest_strides[i];
            src_index += src_strides[i] * (coord + offsets[i]);
        }
        dest[index] = src[src_index];
    }
}



kernel void
ExpForward(const device int* int__ [[ buffer(0) ]],
            const device float* float__ [[buffer(1)]],
            const device float *in [[buffer(2)]],
            device float *out [[buffer(3)]],
            uint2 gid [[ thread_position_in_grid ]],
            uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = float__[1] * exp(float__[0] * in[index]);
    }
}

kernel void
LogForward(const device int* int__ [[ buffer(0) ]],
           const device float* float__ [[buffer(1)]],
           const device float *in [[buffer(2)]],
           device float *out [[buffer(3)]],
           uint2 gid [[ thread_position_in_grid ]],
           uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = float__[2] * log(float__[0] * in[index] + float__[1]);
    }
}


kernel void
BatchReindexForward(const device int* int__ [[ buffer(0) ]],
               const device float *in [[buffer(1)]],
               device float *out [[buffer(2)]],
               const device float *permut [[buffer(3)]],
               uint2 gid [[ thread_position_in_grid ]],
               uint2 tpg [[ threads_per_grid ]]) {
                   
   for (int index = gid.x; index < int__[0]; index += tpg.x){
       int n = index / (int__[1]);
       int in_n = static_cast<int>(permut[n]);
       out[index] = in[in_n * (int__[1]) + index % (int__[1])];
   }
}

kernel void
LRNFillScale(const device int* int__ [[ buffer(0) ]],
             const device float* float__ [[ buffer(1) ]],
             const device float *in [[buffer(2)]],
             device float* const scale [[buffer(3)]],
             uint2 gid [[ thread_position_in_grid ]],
             uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        // find out the local offset
        const int w = index % int__[3];
        const int h = (index / int__[3]) % int__[2];
        const int n = index / int__[3] / int__[2];
        const int offset = (n * int__[1] * int__[2] + h) * int__[3] + w;
        const int step = int__[2] * int__[3];
        const device float* in_off = in + offset;
        device float* scale_off = scale + offset;
        int head = 0;
        const int pre_pad = (int__[4] - 1) / 2;
        const int post_pad = int__[4] - pre_pad - 1;
        float accum_scale = 0;
        // fill the scale at [n, :, h, w]
        // accumulate values
        while (head < post_pad && head < int__[1]) {
            accum_scale += in_off[head * step] * in_off[head * step];
            ++head;
        }
        // both add and subtract
        while (head < int__[1]) {
            accum_scale += in_off[head * step] * in_off[head * step];
            if (head - int__[4] >= 0) {
                accum_scale -= in_off[(head - int__[4]) * step]
                * in_off[(head - int__[4]) * step];
            }
            scale_off[(head - post_pad) * step] = float__[1] + accum_scale * float__[0];
            ++head;
        }
        // subtract only
        while (head < int__[1] + post_pad) {
            if (head - int__[4] >= 0) {
                accum_scale -= in_off[(head - int__[4]) * step]
                * in_off[(head - int__[4]) * step];
            }
            scale_off[(head - post_pad) * step] = float__[1] + accum_scale * float__[0];
            ++head;
        }
    }
}


kernel void
LRNComputeOutput(const device int* int__ [[ buffer(0) ]],
                 const device int* float__ [[ buffer(1) ]],
                 const device float *in [[buffer(2)]],
                 device float *out [[buffer(3)]],
                 const device float *scale [[buffer(4)]],
                 uint2 gid [[ thread_position_in_grid ]],
                 uint2 tpg [[ threads_per_grid ]]){
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = in[index] * pow(scale[index], float__[0]);
    }
}


kernel void
PReLUForward(const device int* int__ [[ buffer(0) ]],
             const device float *in [[buffer(1)]],
             device float *out [[buffer(2)]],
             const device float *slope_data [[buffer(3)]],
             uint2 gid [[ thread_position_in_grid ]],
             uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        int c = (index / int__[2]) % int__[1] / int__[3];
        out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
    }
}


kernel void
ThresholdForward(const device int* int__ [[ buffer(0) ]],
                 const device int* float__ [[ buffer(1) ]],
                 const device float *in [[buffer(2)]],
                 device float *out [[buffer(3)]],
                 uint2 gid [[ thread_position_in_grid ]],
                 uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = in[index] > float__[0] ? 1 : 0;
    }
}

kernel void
BiasForward(const device int* int__ [[ buffer(0) ]],
                 const device float *in [[buffer(1)]],
                 device float *out [[buffer(2)]],
                 const device float *bias [[buffer(3)]],
                 uint2 gid [[ thread_position_in_grid ]],
                 uint2 tpg [[ threads_per_grid ]]){
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        const int bias_index = (index / int__[2]) % int__[1];
        out[index] = in[index] + bias[bias_index];
    }
}



kernel void
ScaleForward(const device int* int__ [[ buffer(0) ]],
             const device float *in [[buffer(1)]],
             device float *out [[buffer(2)]],
             const device float *scale [[buffer(3)]],
             uint2 gid [[ thread_position_in_grid ]],
             uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        const int scale_index = (index / int__[2]) % int__[1];
        out[index] = in[index] * scale[scale_index];
    }
}

kernel void
ScaleBiasForward(const device int* int__ [[ buffer(0) ]],
                 const device float *in [[buffer(1)]],
                 device float *out [[buffer(2)]],
                 const device float *scale [[buffer(3)]],
                 const device float *bias [[buffer(4)]],
                 uint2 gid [[ thread_position_in_grid ]],
                 uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        const int scale_index = (index / int__[2]) % int__[1];
        out[index] = in[index] * scale[scale_index] + bias[scale_index];
    }
}

kernel void
ELUForward(const device int* int__ [[ buffer(0) ]],
           const device float* float__ [[ buffer(1) ]],
           const device float *in [[buffer(2)]],
           device float *out [[buffer(3)]],
           uint2 gid [[ thread_position_in_grid ]],
           uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = in[index] > 0 ? in[index] : float__[0] * (exp(in[index]) - 1);
    }
}

kernel void
MaxPoolForwardTop(const device int* int__ [[ buffer(0) ]],
                  const device float *bottom_data [[buffer(1)]],
                  device float *top_data [[buffer(2)]],
                  device float *top_mask [[buffer(3)]],
                  uint2 gid [[ thread_position_in_grid ]],
                  uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        const int pw = index % int__[6];
        const int ph = (index / int__[6]) % int__[5];
        const int c = (index / int__[6] / int__[5]) % int__[2];
        const int n = index / int__[6] / int__[5] / int__[2];
        int hstart = ph * int__[9] - int__[11];
        int wstart = pw * int__[10] - int__[12];
        const int hend = min(hstart + int__[7], int__[3]);
        const int wend = min(wstart + int__[8], int__[4]);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        float maxval = -FLT_MAX;
        int maxidx = -1;
        const device float* bottom_slice =
        bottom_data + (n * int__[2] + c) * int__[3] * int__[4];
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                if (bottom_slice[h * int__[4] + w] > maxval) {
                    maxidx = h * int__[4] + w;
                    maxval = bottom_slice[maxidx];
                }
            }
        }
        top_data[index] = maxval;

        top_mask[index] = maxidx;
        
    }
}


kernel void
MaxPoolForwardMask(const device int* int__ [[ buffer(0) ]],
                  const device float *bottom_data [[buffer(1)]],
                  device float *top_data [[buffer(2)]],
                  device int *top_mask [[buffer(3)]],
                  uint2 gid [[ thread_position_in_grid ]],
                  uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        const int pw = index % int__[6];
        const int ph = (index / int__[6]) % int__[5];
        const int c = (index / int__[6] / int__[5]) % int__[2];
        const int n = index / int__[6] / int__[5] / int__[2];
        int hstart = ph * int__[9] - int__[11];
        int wstart = pw * int__[10] - int__[12];
        const int hend = min(hstart + int__[7], int__[3]);
        const int wend = min(wstart + int__[8], int__[4]);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        float maxval = -FLT_MAX;
        int maxidx = -1;
        const device float* bottom_slice =
        bottom_data + (n * int__[2] + c) * int__[3] * int__[4];
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                if (bottom_slice[h * int__[4] + w] > maxval) {
                    maxidx = h * int__[4] + w;
                    maxval = bottom_slice[maxidx];
                }
            }
        }
        top_data[index] = maxval;
        
        top_mask[index] = maxidx;
        
    }
}

kernel void
AvePoolForward(const device int* int__ [[ buffer(0) ]],
               const device float *bottom_data [[buffer(1)]],
               device float *top_data [[buffer(2)]],
               uint2 gid [[ thread_position_in_grid ]],
               uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        const int pw = index % int__[6];
        const int ph = (index / int__[6]) % int__[5];
        const int c = (index / int__[6] / int__[5]) % int__[2];
        const int n = index / int__[6] / int__[5] / int__[2];
        int hstart = ph * int__[9] - int__[11];
        int wstart = pw * int__[10] - int__[12];
        int hend = min(hstart + int__[7], int__[3] + int__[11]);
        int wend = min(wstart + int__[8], int__[4] + int__[12]);
        const int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, int__[3]);
        wend = min(wend, int__[4]);
        float aveval = 0;
        const device float* bottom_slice =
        bottom_data + (n * int__[2] + c) * int__[3] * int__[4];
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                aveval += bottom_slice[h * int__[4] + w];
            }
        }
        top_data[index] = aveval / pool_size;
    }
}

kernel void
StoPoolForwardTest(const device int* int__ [[ buffer(0) ]],
                        const device float *bottom_data [[buffer(1)]],
                        device float *top_data [[buffer(2)]],
                        uint2 gid [[ thread_position_in_grid ]],
                        uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        const int pw = index % int__[6];
        const int ph = (index / int__[6]) % int__[5];
        const int c = (index / int__[6] / int__[5]) % int__[2];
        const int n = index / int__[6] / int__[5] / int__[2];
        const int hstart = ph * int__[9];
        const int hend = min(hstart + int__[7], int__[3]);
        const int wstart = pw * int__[10];
        const int wend = min(wstart + int__[8], int__[4]);
        // We set cumsum to be 0 to avoid divide-by-zero problems
        float cumsum = 0.;
        float cumvalues = 0.;
        const device float* bottom_slice =
        bottom_data + (n * int__[2] + c) * int__[3] * int__[4];
        // First pass: get sum
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                cumsum += bottom_slice[h * int__[4] + w];
                cumvalues += bottom_slice[h * int__[4] + w] * bottom_slice[h * int__[4] + w];
            }
        }
        top_data[index] = (cumsum > 0.) ? cumvalues / cumsum : 0.;
    }
}


kernel void
MaxForward(const device int* int__ [[ buffer(0) ]],
           device float *mask [[buffer(1)]],
           device float *top_data [[buffer(2)]],
           const device float *bottom_data_a [[buffer(3)]],
           const device float *bottom_data_b [[buffer(4)]],
           uint2 gid [[ thread_position_in_grid ]],
           uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        float maxval = -FLT_MAX;
        int maxidx = -1;
        if (bottom_data_a[index] > bottom_data_b[index]) {
            // only update for very first bottom_data blob (int__[1] == 0)
            if (int__[1] == 0) {
                maxval = bottom_data_a[index];
                top_data[index] = maxval;
                maxidx = int__[1];
                mask[index] = maxidx;
            }
        } else {
            maxval = bottom_data_b[index];
            top_data[index] = maxval;
            maxidx = int__[1] + 1;
            mask[index] = maxidx;
        }
    }
}





kernel void kernel_channel_max(const device int* int__ [[ buffer(0) ]],
                               const device float *in [[buffer(1)]],
                               device float *out [[buffer(2)]],
                               uint2 gid [[ thread_position_in_grid ]],
                               uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]*int__[2]; index += tpg.x){
        int n = index / int__[2];
        int s = index % int__[2];
        float maxval = -FLT_MAX;
        for (int c = 0; c < int__[1]; ++c) {
            maxval = max(in[(n * int__[1] + c) * int__[2] + s], maxval);
        }
        out[index] = maxval;
    }
}

kernel void kernel_channel_subtract(
                                    const device int* int__ [[ buffer(0) ]],
                                    const device float *in [[buffer(1)]],
                                    device float *out [[buffer(2)]],
                                    uint2 gid [[ thread_position_in_grid ]],
                                    uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        int n = index / int__[1] / int__[2];
        int s = index % int__[2];
        out[index] -= in[n * int__[2] + s];
    }
}


kernel void
kernel_channel_sum(const device int* int__ [[ buffer(0) ]],
                   const device float *in [[buffer(1)]],
                   device float *out [[buffer(2)]],
                   uint2 gid [[ thread_position_in_grid ]],
                   uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]*int__[2]; index += tpg.x){
        int n = index / int__[2];
        int s = index % int__[2];
        float sum = 0;
        for (int c = 0; c < int__[1]; ++c) {
            sum += in[(n * int__[1] + c) * int__[2] + s];
        }
        out[index] = sum;
    }
}

kernel void
kernel_channel_div(const device int* int__ [[ buffer(0) ]],
                   const device float *in [[buffer(1)]],
                   device float *out [[buffer(2)]],
                   uint2 gid [[ thread_position_in_grid ]],
                   uint2 tpg [[ threads_per_grid ]]) {
for (int index = gid.x; index < int__[0]; index += tpg.x){
        int n = index / int__[1] / int__[2];
        int s = index % int__[2];
        out[index] /= in[n * int__[2] + s];
    }
}

kernel void
Slice(const device int* int__ [[ buffer(0) ]],
      const device float *in [[buffer(1)]],
      device float *out [[buffer(2)]],
      uint2 gid [[ thread_position_in_grid ]],
      uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        const int total_slice_size = int__[2] * int__[4];
        const int slice_num = index / total_slice_size;
        const int slice_index = index % total_slice_size;
        const int bottom_index = slice_index +
        (slice_num * int__[3] + int__[5]) * int__[2];
        if (int__[1]) {
            out[index] = in[bottom_index];
        } else {
            out[bottom_index] = in[index];
        }
    }
}


kernel void
Tile(const device int* int__ [[ buffer(0) ]],
     const device float *in [[buffer(1)]],
     device float *out [[buffer(2)]],
     uint2 gid [[ thread_position_in_grid ]],
     uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        const int d = index % int__[1];
        const int b = (index / int__[1] / int__[2]) % int__[3];
        const int n = index / int__[1] / int__[2] / int__[3];
        const int bottom_index = (n * int__[3] + b) * int__[1] + d;
        out[index] = in[bottom_index];
    }
}

kernel void
EmbedForward(const device int* int__ [[ buffer(0) ]],
             const device float *in [[buffer(1)]],
             device float *out [[buffer(2)]],
             const device float *weight [[buffer(3)]],
             uint2 gid [[ thread_position_in_grid ]],
             uint2 tpg [[ threads_per_grid ]]) {
    for (int top_index = gid.x; top_index < int__[0]; top_index += tpg.x){
        const int n = top_index / int__[1];
        const int d = top_index % int__[1];
        const int index = static_cast<int>(in[n]);
        const int weight_index = index * int__[1] + d;
        out[top_index] = weight[weight_index];
    }
}

kernel void
metal_Saxpy(const device int* int__ [[ buffer(0) ]],
            const device int* float__ [[ buffer(1) ]],
            const device float *in [[buffer(2)]],
            device float *out [[buffer(3)]],
            uint2 gid [[ thread_position_in_grid ]],
            uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = tanh(in[index]);
    }
}


kernel void
metal_Sscal(const device int* int__ [[ buffer(0) ]],
            const device int* float__ [[ buffer(1) ]],
            const device float *in [[buffer(2)]],
            device float *out [[buffer(3)]],
            uint2 gid [[ thread_position_in_grid ]],
            uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = tanh(in[index]);
    }
}
