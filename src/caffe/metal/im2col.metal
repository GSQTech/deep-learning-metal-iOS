//
//  im2col.metal
//  metal_mac
//
//  Created by Tec GSQ on 30/11/2017.
//  Copyright © 2017 Tec GSQ. All rights reserved.
//

#include <metal_stdlib>
#include "include/caffe/util/im2col-metal.hpp"
using namespace metal;



kernel void
im2col_gpu_kernel(const device int *param [[ buffer(0) ]],
                  const device float *data_im [[buffer(1)]],
                  device float *data_col [[buffer(2)]],
                  uint2 gid [[ thread_position_in_grid ]],
                  uint2 tpg [[ threads_per_grid ]])
{
    int grid_size = tpg.x * tpg.y;
    int n = param[0];
    int height = param[1];
    int width = param[2];
    int kernel_h = param[3];
    int kernel_w = param[4];
    int pad_h = param[5];
    int pad_w = param[6];
    int stride_h = param[7];
    int stride_w = param[8];
    int dilation_h = param[9];
    int dilation_w = param[10];
    int height_col = param[11];
    int width_col = param[12];
    
    
    for (int index = gid.x;
         index < n;
         index += tpg.x){
        int h_index = index / width_col;
        int h_col = h_index % height_col;
        int w_col = index % width_col;
        int c_im = h_index / height_col;
        int c_col = c_im * kernel_h * kernel_w;
        int h_offset = h_col * stride_h - pad_h;
        int w_offset = w_col * stride_w - pad_w;
        // get the index without pointer computing
        
        int col_ptr = (c_col * height_col + h_col) * width_col + w_col;
        int im_ptr = (c_im * height + h_offset) * width + w_offset;
        float real_data;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h_im = h_offset + i * dilation_h;
                int w_im = w_offset + j * dilation_w;
                uint input_index = im_ptr + i * dilation_h * width + j * dilation_w;
                if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width){
                    real_data = data_im[input_index];
                    data_col[col_ptr] = real_data;
                }else{
                    data_col[col_ptr] = 0.0;
                }
                col_ptr += height_col * width_col;
            }
        }
    }
}



kernel void
col2im_gpu_kernel(const device int *param [[ buffer(0) ]],
                  const device float *data_col [[buffer(1)]],
                  device float *data_im [[buffer(2)]],
                  uint2 gid [[ thread_position_in_grid ]],
                  uint2 tpg [[ threads_per_grid ]])
{
    int grid_size = tpg.x * tpg.y;
    int n = param[0];
    int height = param[1];
    int width = param[2];
    int channel = param[3];
    int kernel_h = param[4];
    int kernel_w = param[5];
    int pad_h = param[6];
    int pad_w = param[7];
    int stride_h = param[8];
    int stride_w = param[9];
    int dilation_h = param[10];
    int dilation_w = param[11];
    int height_col = param[12];
    int width_col = param[13];
    
    
    for (int index = gid.x;
         index < n;
         index += tpg.x){
        
        float val = 0;
        const int w_im = index % width + pad_w;
        const int h_im = (index / width) % height + pad_h;
        const int c_im = index / (width * height);
        int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
        int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
        // compute the start and end of the output
        const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);
        const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);
        // TODO: use LCM of stride and dilation to avoid unnecessary loops
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
                if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                    h_k /= dilation_h;
                    w_k /= dilation_w;
                    int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                          height_col + h_col) * width_col + w_col;
                    val += data_col[data_col_index];
                }
            }
        }
        data_im[index] = val;
    }
}
