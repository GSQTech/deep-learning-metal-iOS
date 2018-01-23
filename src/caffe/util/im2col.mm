//
//  im2col.mm
//  metal_mac
//
//  Created by Tec GSQ on 29/11/2017.
//  Copyright © 2017 Tec GSQ. All rights reserved.
//

#ifdef METAL

#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"
#include <Metal/Metal.h>
#include <MetalPerformanceShaders.h>
#include "caffe/util/im2col-metal.hpp"

namespace caffe {



template <typename Dtype>
void im2col_gpu_metal(const Dtype* data_im, const int data_im_size, const int data_im_offset, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col, const int data_col_size, const int data_col_offset) {
    
    
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;
   
    // Create Metal device, queue and function
    id<MTLDevice>       device   = ((__bridge id<MTLDevice>)(Caffe::Get().metal_device));
    id<MTLLibrary>      library  = [device  newDefaultLibrary];
    id<MTLFunction>     function = [library newFunctionWithName:@"im2col_gpu_kernel"];
    
    int tmp1 = [device currentAllocatedSize];
    //@autoreleasepool{
    // Create and initialize buffers
    int param[13] = {num_kernels, height, width,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        height_col, width_col
    };
    
    id<MTLBuffer> parameter = [device newBufferWithLength:13*4 options:MTLStorageModeShared];
    
    int* p_tmp = (int *)[parameter contents];
    
    for(int i = 0; i< 13; ++i) {
        p_tmp[i] = param[i];
    }
    
    
    
    id<MTLBuffer> data_input  = [device newBufferWithBytesNoCopy:(void *)data_im   length:((data_im_size * 4)+4095)/4096 * 4096  options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> data_output = [device newBufferWithBytesNoCopy:(void *)data_col  length:((data_col_size * 4)+4095)/4096 * 4096  options:MTLStorageModeShared deallocator:nil];
    
    
    id<MTLCommandBuffer> buffer = [((__bridge id<MTLCommandQueue>)(Caffe::Get().metal_commandQueue)) commandBuffer];
    
    //[function encodeToCommandBuffer:buffer param:parameter data_im:data_input data_col:data_output];
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    NSError *errors;
    id<MTLComputePipelineState>  state   = [device newComputePipelineStateWithFunction:function error:&errors];
    
    NSUInteger default_grid_dim = (num_kernels + 512 - 1) / 512;
    
    MTLSize groupsize = {512,1,1};
    MTLSize numgroups = {default_grid_dim,1,1};
    [encoder setComputePipelineState:state];

    
    //[encoder setComputePipelineState:state];
    [encoder setBuffer:parameter   offset:0 atIndex:0];
    [encoder setBuffer:data_input  offset:data_im_offset*4 atIndex:1];
    [encoder setBuffer:data_output offset:data_col_offset*4 atIndex:2];
    
    [encoder dispatchThreadgroups:numgroups threadsPerThreadgroup:groupsize];
    
    [encoder endEncoding];
    
    
    [buffer commit];
    [buffer waitUntilCompleted];
        
   // }
int tmp = [device currentAllocatedSize];
    int a = 1;
  // end of this TODO block
}

// Explicit instantiation
template void im2col_gpu_metal<float>(const float* data_im, const int data_im_size, const int data_im_offset, const int channels, const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, float* data_col, const int data_col_size, const int data_col_offset);
/*template void im2col_gpu_metal<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, double* data_col);*/


// for the time being we do not implement n dimensional image to column

template <typename Dtype>
void im2col_nd_gpu_metal(const Dtype* data_im, const int num_spatial_axes,
    const int num_kernels, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col) {

}

// Explicit instantiation
template void im2col_nd_gpu_metal<float>(const float* data_im,
    const int num_spatial_axes, const int col_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_col);
template void im2col_nd_gpu_metal<double>(const double* data_im,
    const int num_spatial_axes, const int col_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_col);





template <typename Dtype>
void col2im_gpu_metal(const Dtype* data_col, const int data_col_size, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im, const int data_im_size) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;
    
    // Create Metal device, queue and function
    id<MTLDevice>       device   = ((__bridge id<MTLDevice>)(Caffe::Get().metal_device));
    id<MTLLibrary>      library  = [device  newDefaultLibrary];
    id<MTLFunction>     function = [library newFunctionWithName:@"col2im_gpu_kernel"];
    
    // Create and initialize buffers
    int param[14] = {num_kernels, height, width, channels,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        height_col, width_col
    };
    
    id<MTLBuffer> parameter = [device newBufferWithLength:14*4 options:MTLStorageModeShared];
    
    int* p_tmp = (int *)[parameter contents];
    
    for(int i = 0; i< 14; ++i) {
        p_tmp[i] = param[i];
    }
    
    id<MTLBuffer> data_input  = [device newBufferWithBytesNoCopy:(void *)data_col   length:((data_col_size * 4)+4095)/4096*4096   options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> data_output = [device newBufferWithBytesNoCopy:(void *)data_im  length:((data_im_size * 4)+4095)/4096*4096   options:MTLStorageModeShared deallocator:nil];
    
    
    id<MTLCommandBuffer> buffer = [((__bridge id<MTLCommandQueue>)(Caffe::Get().metal_commandQueue)) commandBuffer];
    
    //[function encodeToCommandBuffer:buffer param:parameter data_im:data_input data_col:data_output];
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    NSError *errors;
    id<MTLComputePipelineState>  state   = [device newComputePipelineStateWithFunction:function error:&errors];
    
    NSUInteger default_grid_dim = (num_kernels + 512 - 1) / 512;
    
    MTLSize groupsize = {512,1,1};
    MTLSize numgroups = {default_grid_dim,1,1};
    [encoder setComputePipelineState:state];
    
    
    //[encoder setComputePipelineState:state];
    [encoder setBuffer:parameter   offset:0 atIndex:0];
    [encoder setBuffer:data_input  offset:0 atIndex:1];
    [encoder setBuffer:data_output offset:0 atIndex:2];
    
    [encoder dispatchThreadgroups:numgroups threadsPerThreadgroup:groupsize];
    
    [encoder endEncoding];
    
    
    [buffer commit];
    [buffer waitUntilCompleted];

  // end of this TODO block
}

// Explicit instantiation
    template void col2im_gpu_metal<float>(const float* data_col, const int data_col_size, const int channels,
                                          const int height, const int width, const int kernel_h, const int kernel_w,
                                          const int pad_h, const int pad_w, const int stride_h,
                                          const int stride_w, const int dilation_h, const int dilation_w,
                                          float* data_im, const int data_im_size);
//template void col2im_gpu_metal<double>(const double* data_col, const int channels,
//    const int height, const int width, const int kernel_h, const int kernel_w,
//    const int pad_h, const int pad_w, const int stride_h,
//    const int stride_w, const int dilation_h, const int dilation_w,
//    double* data_im);
//
//// for the time being we do not implement n dimensional image to column
//
//template <typename Dtype>
//void col2im_nd_gpu_metal(const Dtype* data_col, const int num_spatial_axes,
//    const int im_size, const int* im_shape, const int* col_shape,
//    const int* kernel_shape, const int* pad, const int* stride,
//    const int* dilation, Dtype* data_im) {
//}
//
//// Explicit instantiation
//template void col2im_nd_gpu_metal<float>(const float* data_col,
//    const int num_spatial_axes, const int im_size,
//    const int* im_shape, const int* col_shape,
//    const int* kernel_shape, const int* pad, const int* stride,
//    const int* dilation, float* data_im);
//template void col2im_nd_gpu_metal<double>(const double* data_col,
//    const int num_spatial_axes, const int im_size,
//    const int* im_shape, const int* col_shape,
//    const int* kernel_shape, const int* pad, const int* stride,
//    const int* dilation, double* data_im);
//
//
}
#endif // !METAL
