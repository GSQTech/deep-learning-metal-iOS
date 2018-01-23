//
//  math.metal
//  caffe_metal
//
//  Created by Tec GSQ on 29/12/2017.
//  Copyright © 2017 Tec GSQ. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;


kernel void
exp_kernel(const device int* int__ [[ buffer(0) ]],
            const device float *in [[buffer(1)]],
            device float *out [[buffer(2)]],
            uint2 gid [[ thread_position_in_grid ]],
            uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = exp(in[index]);
    }
}

kernel void
abs_kernel(const device int* int__ [[ buffer(0) ]],
           const device float *in [[buffer(1)]],
           device float *out [[buffer(2)]],
           uint2 gid [[ thread_position_in_grid ]],
           uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = abs(in[index]);
    }
}

kernel void
powx_kernel(const device int* int__ [[ buffer(0) ]],
            const device float* float__ [[ buffer(1) ]],
            const device float *in [[buffer(2)]],
            device float *out [[buffer(3)]],
            uint2 gid [[ thread_position_in_grid ]],
            uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = pow(in[index], float__[0]);
    }
}
kernel void
scale_kernel(const device int* int__ [[ buffer(0) ]],
             const device float* float__ [[ buffer(1) ]],
           const device float *in [[buffer(2)]],
           device float *out [[buffer(3)]],
           uint2 gid [[ thread_position_in_grid ]],
           uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = float__[0] * (in[index]);
    }
}

kernel void
scal_kernel(const device int* int__ [[ buffer(0) ]],
             const device float* float__ [[ buffer(1) ]],
             const device float *in [[buffer(2)]],
             device float *out [[buffer(3)]],
             uint2 gid [[ thread_position_in_grid ]],
             uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = float__[0] * (out[index]);
    }
}

kernel void
sqrt_kernel(const device int* int__ [[ buffer(0) ]],
            const device float *in [[buffer(1)]],
            device float *out [[buffer(2)]],
            uint2 gid [[ thread_position_in_grid ]],
            uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = sqrt(in[index]);
    }
}

kernel void
add_scalar_kernel(const device int* int__ [[ buffer(0) ]],
                  const device float* float__ [[ buffer(1) ]],
                  const device float *in [[buffer(2)]],
                  device float *out [[buffer(3)]],
                  uint2 gid [[ thread_position_in_grid ]],
                  uint2 tpg [[ threads_per_grid ]]) {
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] += float__[0];
    }
}
kernel void
mul_kernel(const device int* int__ [[ buffer(0) ]],
             const device float *in1 [[buffer(1)]],
             device float *out [[buffer(2)]],
             const device float *in2 [[buffer(3)]],
             uint2 gid [[ thread_position_in_grid ]],
             uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = in1[index] * in2[index];
    }
}

kernel void
add_kernel(const device int* int__ [[ buffer(0) ]],
           const device float *in1 [[buffer(1)]],
           device float *out [[buffer(2)]],
           const device float *in2 [[buffer(3)]],
           uint2 gid [[ thread_position_in_grid ]],
           uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = in1[index] + in2[index];
    }
}

kernel void
div_kernel(const device int* int__ [[ buffer(0) ]],
           const device float *in1 [[buffer(1)]],
           device float *out [[buffer(2)]],
           const device float *in2 [[buffer(3)]],
           uint2 gid [[ thread_position_in_grid ]],
           uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = in1[index] / in2[index];
    }
}



kernel void
axpy_kernel(const device int* int__ [[ buffer(0) ]],
             const device float* float__ [[ buffer(1) ]],
             const device float *in [[buffer(2)]],
             device float *out [[buffer(3)]],
             uint2 gid [[ thread_position_in_grid ]],
             uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] += float__[0] * (in[index]);
    }
}

kernel void
axpby_kernel(const device int* int__ [[ buffer(0) ]],
            const device float* float__ [[ buffer(1) ]],
            const device float *in [[buffer(2)]],
            device float *out [[buffer(3)]],
            uint2 gid [[ thread_position_in_grid ]],
            uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = float__[0] * (in[index]) + float__[1] * out[index];
    }
}

kernel void
set_kernel(const device int* int__ [[ buffer(0) ]],
            const device float* float__ [[ buffer(1) ]],
            const device float *in [[buffer(2)]],
            device float *out [[buffer(3)]],
            uint2 gid [[ thread_position_in_grid ]],
            uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = float__[0];
    }
}
/*
kernel void
scale_kernel(const device int* int__ [[ buffer(0) ]],
             const device float *in1 [[buffer(1)]],
             device float *out [[buffer(2)]],
             const device float *in2 [[buffer(3)]],
             uint2 gid [[ thread_position_in_grid ]],
             uint2 tpg [[ threads_per_grid ]])
{
    for (int index = gid.x; index < int__[0]; index += tpg.x){
        out[index] = in1[index] * in2[index];
    }
}*/
