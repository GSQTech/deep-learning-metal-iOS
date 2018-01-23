//
//  math_function.mm
//  metal_mac
//
//  Created by Tec GSQ on 29/11/2017.
//  Copyright © 2017 Tec GSQ. All rights reserved.
//

#ifdef METAL
#include <Metal/Metal.h>
#include <MetalPerformanceShaders.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"




namespace caffe {


#ifdef METAL
void metal_wrapper(std::string funcname_, void* in_data, const int in_data_size, void* out_data,  const int out_data_size, std::vector<int> para_int, std::vector<float> para_float, int count) {

    NSString *funcname = [NSString stringWithCString:funcname_.c_str() encoding:[NSString defaultCStringEncoding]];

    id<MTLDevice>       device   = ((__bridge id<MTLDevice>)(Caffe::Get().metal_device));
    id<MTLLibrary>      library  = [device  newDefaultLibrary];
    id<MTLFunction>     function = [library newFunctionWithName:funcname];


    id<MTLBuffer> data_input  = [device newBufferWithBytesNoCopy:in_data  length:align_size(in_data_size)  options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> data_output = [device newBufferWithBytesNoCopy:out_data length:align_size(out_data_size)  options:MTLStorageModeShared deallocator:nil];

    id<MTLCommandBuffer> buffer = [((__bridge id<MTLCommandQueue>)(Caffe::Get().metal_commandQueue)) commandBuffer];

    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    NSError *errors;
    id<MTLComputePipelineState>  state   = [device newComputePipelineStateWithFunction:function error:&errors];
    NSUInteger default_grid_dim = (count + 512 - 1) / 512;
    MTLSize groupsize = {512,1,1};
    MTLSize numgroups = {default_grid_dim,1,1};
    [encoder setComputePipelineState:state];


    id<MTLBuffer> int__;
    id<MTLBuffer> float__;

    int index = 0;

    if(para_int.size()) {
        int__ = [device newBufferWithLength:para_int.size()*4 options:MTLStorageModeShared];
        int* p_tmp_int = (int *)[int__ contents];
        for(int i = 0; i < para_int.size(); ++i) {
            p_tmp_int[i] = para_int[i];
        }

        [encoder setBuffer:int__       offset:0 atIndex:index++];
    }

    if(para_float.size()) {
        float__ = [device newBufferWithLength:para_float.size()*4 options:MTLStorageModeShared];
        float* p_tmp_float = (float *)[float__ contents];
        for(int i = 0; i < para_float.size(); ++i) {
            p_tmp_float[i] = para_float[i];
        }
        [encoder setBuffer:float__     offset:0 atIndex:index++];
    }
    [encoder setBuffer:data_input  offset:0 atIndex:index++];

    
    [encoder setBuffer:data_output offset:0 atIndex:index++];
    
    [encoder dispatchThreadgroups:numgroups threadsPerThreadgroup:groupsize];
    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];
}
    
    
void metal_wrapper3(std::string funcname_,
                    void* in_data, const int in_data_size,
                    void* out_data,  const int out_data_size,
                    void* inter_data,  const int inter_data_size,
                    std::vector<int> para_int, std::vector<float> para_float, int count) {
    
    NSString *funcname = [NSString stringWithCString:funcname_.c_str() encoding:[NSString defaultCStringEncoding]];
    
    id<MTLDevice>       device   = ((__bridge id<MTLDevice>)(Caffe::Get().metal_device));
    id<MTLLibrary>      library  = [device  newDefaultLibrary];
    id<MTLFunction>     function = [library newFunctionWithName:funcname];
    
    
    id<MTLBuffer> data_input  = [device newBufferWithBytesNoCopy:in_data  length:align_size(in_data_size)  options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> data_output = [device newBufferWithBytesNoCopy:out_data length:align_size(out_data_size)  options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> data_inter = [device newBufferWithBytesNoCopy:inter_data length:align_size(inter_data_size)  options:MTLStorageModeShared deallocator:nil];
    
    id<MTLCommandBuffer> buffer = [((__bridge id<MTLCommandQueue>)(Caffe::Get().metal_commandQueue)) commandBuffer];
    
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    NSError *errors;
    id<MTLComputePipelineState>  state   = [device newComputePipelineStateWithFunction:function error:&errors];
    NSUInteger default_grid_dim = (count + 512 - 1) / 512;
    MTLSize groupsize = {512,1,1};
    MTLSize numgroups = {default_grid_dim,1,1};
    [encoder setComputePipelineState:state];
    
    
    id<MTLBuffer> int__;
    id<MTLBuffer> float__;
    
    int index = 0;
    
    if(para_int.size()) {
        int__ = [device newBufferWithLength:para_int.size()*4 options:MTLStorageModeShared];
        int* p_tmp_int = (int *)[int__ contents];
        for(int i = 0; i < para_int.size(); ++i) {
            p_tmp_int[i] = para_int[i];
        }
        
        [encoder setBuffer:int__       offset:0 atIndex:index++];
    }
    
    if(para_float.size()) {
        float__ = [device newBufferWithLength:para_float.size()*4 options:MTLStorageModeShared];
        float* p_tmp_float = (float *)[float__ contents];
        for(int i = 0; i < para_float.size(); ++i) {
            p_tmp_float[i] = para_float[i];
        }
        [encoder setBuffer:float__     offset:0 atIndex:index++];
    }
    [encoder setBuffer:data_input  offset:0 atIndex:index++];
    
    if(out_data != NULL) {
        [encoder setBuffer:data_output offset:0 atIndex:index++];
    }
    if(inter_data != NULL) {
        [encoder setBuffer:data_inter offset:0 atIndex:index++];
    }
    [encoder dispatchThreadgroups:numgroups threadsPerThreadgroup:groupsize];
    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];
}

void metal_wrapper4(std::string funcname_,
                    void* in_data, const int in_data_size,
                    void* out_data,  const int out_data_size,
                    void* other1_data,  const int other1_data_size,
                    void* other2_data,  const int other2_data_size,
                    std::vector<int> para_int, std::vector<float> para_float, int count) {
    
    NSString *funcname = [NSString stringWithCString:funcname_.c_str() encoding:[NSString defaultCStringEncoding]];
    
    id<MTLDevice>       device   = ((__bridge id<MTLDevice>)(Caffe::Get().metal_device));
    id<MTLLibrary>      library  = [device  newDefaultLibrary];
    id<MTLFunction>     function = [library newFunctionWithName:funcname];
    
    
    id<MTLBuffer> data_input  = [device newBufferWithBytesNoCopy:in_data  length:align_size(in_data_size)  options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> data_output = [device newBufferWithBytesNoCopy:out_data length:align_size(out_data_size)  options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> data_other1 = [device newBufferWithBytesNoCopy:other1_data length:align_size(other1_data_size)  options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> data_other2 = [device newBufferWithBytesNoCopy:other1_data length:align_size(other2_data_size)  options:MTLStorageModeShared deallocator:nil];
    
    id<MTLCommandBuffer> buffer = [((__bridge id<MTLCommandQueue>)(Caffe::Get().metal_commandQueue)) commandBuffer];
    
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    NSError *errors;
    id<MTLComputePipelineState>  state   = [device newComputePipelineStateWithFunction:function error:&errors];
    NSUInteger default_grid_dim = (count + 512 - 1) / 512;
    MTLSize groupsize = {512,1,1};
    MTLSize numgroups = {default_grid_dim,1,1};
    [encoder setComputePipelineState:state];
    
    
    id<MTLBuffer> int__;
    id<MTLBuffer> float__;
    
    int index = 0;
    
    if(para_int.size()) {
        int__ = [device newBufferWithLength:para_int.size()*4 options:MTLStorageModeShared];
        int* p_tmp_int = (int *)[int__ contents];
        for(int i = 0; i < para_int.size(); ++i) {
            p_tmp_int[i] = para_int[i];
        }
        
        [encoder setBuffer:int__       offset:0 atIndex:index++];
    }
    
    if(para_float.size()) {
        float__ = [device newBufferWithLength:para_float.size()*4 options:MTLStorageModeShared];
        float* p_tmp_float = (float *)[float__ contents];
        for(int i = 0; i < para_float.size(); ++i) {
            p_tmp_float[i] = para_float[i];
        }
        [encoder setBuffer:float__     offset:0 atIndex:index++];
    }
    [encoder setBuffer:data_input  offset:0 atIndex:index++];
    
    if(out_data != NULL) {
        [encoder setBuffer:data_output offset:0 atIndex:index++];
    }
    
    [encoder setBuffer:data_other1 offset:0 atIndex:index++];
    [encoder setBuffer:data_other2 offset:0 atIndex:index++];
    
    [encoder dispatchThreadgroups:numgroups threadsPerThreadgroup:groupsize];
    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];
}
#endif

    void delloc_tester(void* pointer, NSUInteger length) {
        std::cout << pointer << "and the length is " << length << std::endl;
    }
    
template <>
void caffe_gpu_gemm_metal<float>(
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const float* A, const int A_offset, const float* B, const int B_offset,
    const double beta, float* C, const int C_offset) {
  // Note that cublas follows fortran order.

    bool metal_TransA = (TransA != CblasNoTrans);
    bool metal_TransB = (TransB != CblasNoTrans);

  int lda = (metal_TransA) ? K : M;
  int nda = (metal_TransA) ? M : K;

  int ldb = (metal_TransB) ? N : K;
  int ndb = (metal_TransB) ? K : N;




    
    
    
    @autoreleasepool{
        
        id<MTLDevice> device = ((__bridge id<MTLDevice>)(Caffe::Get().metal_device));
        
        //int tmp1 = [device currentAllocatedSize];
        
        
        MPSMatrixMultiplication *metal_gemm = [[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:metal_TransA transposeRight:metal_TransB resultRows:M resultColumns:N interiorColumns:K alpha:alpha beta:beta];
        
        MPSMatrixDescriptor * descA = [MPSMatrixDescriptor matrixDescriptorWithRows:lda columns:nda rowBytes:nda*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor * descB = [MPSMatrixDescriptor matrixDescriptorWithRows:ldb columns:ndb rowBytes:ndb*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor * descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N*4 dataType:MPSDataTypeFloat32];
        
        id<MTLBuffer> a_  = [device newBufferWithBytesNoCopy:(void *)A  length:ceil(M*K/1024.0)*4096  options:MTLStorageModeShared deallocator:nil]; //^(void *pointer, NSUInteger length){std::cout << pointer << "and the length is " << length << std::endl;}
        id<MTLBuffer> b_  = [device newBufferWithBytesNoCopy:(void *)B  length:ceil(K*N/1024.0)*4096  options:MTLStorageModeShared deallocator:nil];
        id<MTLBuffer> c_  = [device newBufferWithBytesNoCopy:(void *)C  length:ceil(M*N/1024.0)*4096  options:MTLStorageModeShared deallocator:nil];
        
        MPSMatrix *a = [[MPSMatrix alloc] initWithBuffer:a_ descriptor:descA];
        
        MPSMatrix *b = [[MPSMatrix alloc] initWithBuffer:b_ descriptor:descB];
        
        MPSMatrix *c = [[MPSMatrix alloc] initWithBuffer:c_ descriptor:descC];
        //a = [a initWithBuffer: ((__bridge id<MTLBuffer>)((void *)A)) options:nil];
        //MPSMatrix *b;
        //MPSMatrix *c;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> buffer = [queue commandBufferWithUnretainedReferences];//[((__bridge id<MTLCommandQueue>)(Caffe::Get().metal_commandQueue)) commandBuffer];//WithUnretainedReferences];
        
        [metal_gemm encodeToCommandBuffer:buffer leftMatrix:a rightMatrix:b resultMatrix:c];
        
        
        

        [buffer commit];
        [buffer waitUntilCompleted];
        
        [a_ setPurgeableState:MTLPurgeableStateEmpty];
        [b_ setPurgeableState:MTLPurgeableStateEmpty];
        [c_ setPurgeableState:MTLPurgeableStateEmpty];
        
        a_ = nil;
        b_ = nil;
        c_ = nil;
        
        descA = nil;
        descB = nil;
        descC = nil;
        
        
        a = nil;
        b = nil;
        c = nil;
        metal_gemm = nil;
        queue = nil;
        
        device = nil;
        
    }
    
    
    
    

    
    

  //CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
    //  N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));

  // end of this TODO block
}
    
///*
//template <>
//void caffe_gpu_gemm_metal<double>(const CBLAS_TRANSPOSE TransA,
//    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
//    const double alpha, const double* A, const double* B, const double beta,
//    double* C) {
//  // Note that cublas follows fortran order.
//  int lda = (TransA == CblasNoTrans) ? K : M;
//  int ldb = (TransB == CblasNoTrans) ? N : K;
//
//  // TODO change from cuda to metal
//  cublasOperation_t cuTransA =
//      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
//  cublasOperation_t cuTransB =
//      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
//
//  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
//      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
//
//  // end of this TODO block
//}
//*/
template <>
void caffe_gpu_gemv_metal<float>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const double alpha, const float* A, const float* x, const double beta,
    float* y) {

    id<MTLDevice> device = ((__bridge id<MTLDevice>)(Caffe::Get().metal_device));
    
    bool metal_TransA = (TransA != CblasNoTrans);


    int lda = (metal_TransA) ? N : M;
    int nda = (metal_TransA) ? M : N;

    MPSMatrixVectorMultiplication *metal_gemv = [[MPSMatrixVectorMultiplication alloc] initWithDevice:((__bridge id<MTLDevice>)(Caffe::Get().metal_device)) transpose:metal_TransA rows:lda columns:nda alpha:alpha beta:beta];

    MPSMatrixDescriptor * descA = [MPSMatrixDescriptor matrixDescriptorWithDimensions:M columns:N rowBytes:N*4 dataType:MPSDataTypeFloat32];

    MPSVectorDescriptor * descx = [MPSVectorDescriptor vectorDescriptorWithLength:nda dataType:MPSDataTypeFloat32];

    MPSVectorDescriptor * descy = [MPSVectorDescriptor vectorDescriptorWithLength:lda dataType:MPSDataTypeFloat32];

    id<MTLBuffer> a_  = [device newBufferWithBytesNoCopy:(void *)A  length:ceil(M*N/1024.0)*4096  options:MTLStorageModeShared deallocator:^(void *pointer, NSUInteger length){std::cout << pointer << "and the length is " << length << std::endl;}];
    id<MTLBuffer> x_  = [device newBufferWithBytesNoCopy:(void *)x  length:ceil(nda/1024.0)*4096  options:MTLStorageModeShared deallocator:^(void *pointer, NSUInteger length){std::cout << pointer << "and the length is " << length << std::endl;}];
    id<MTLBuffer> y_  = [device newBufferWithBytesNoCopy:(void *)y  length:ceil(lda/1024.0)*4096  options:MTLStorageModeShared deallocator:^(void *pointer, NSUInteger length){std::cout << pointer << "and the length is " << length << std::endl;}];
    
    MPSMatrix *a = [[MPSMatrix alloc] initWithBuffer:a_ descriptor:descA];

    MPSVector *input =  [[MPSVector alloc] initWithBuffer:x_ descriptor:descx];

    MPSVector *output = [[MPSVector alloc] initWithBuffer:y_ descriptor:descy];


    id<MTLCommandBuffer> buffer = [((__bridge id<MTLCommandQueue>)(Caffe::Get().metal_commandQueue)) commandBuffer];

    [metal_gemv encodeToCommandBuffer:buffer inputMatrix:a inputVector:input resultVector:output];

    [buffer commit];
    [buffer waitUntilCompleted];

}
//
//
//template <>
//void caffe_gpu_axpy_metal<float>(const int N, const float alpha, const float* X, float* Y) {
//    metal_wrapper("metal_Saxpy", (void* )X, (void *)Y, {N}, {alpha}, N);
//}
//
//void caffe_gpu_memcpy_metal(const size_t N, const void* X, void* Y) {
//    if (X != Y) {
//        id<MTLDevice> device = ((__bridge id<MTLDevice>)(Caffe::Get().metal_device));
//        void *x_content = [(__bridge id<MTLBuffer>)X contents];
//        CFRelease(Y);
//        Y = (void *)CFBridgingRetain([device newBufferWithBytes:x_content length:N options:MTLStorageModeShared]);
//    }
//}
//
////template <>
//void caffe_gpu_scal(const int N, const float alpha, float *X) {
//    metal_wrapper("metal_Scal", (void* )X, (void *)X, {N}, {alpha}, N);
//}
//
//void caffe_gpu_axpby(const int N, const float alpha, const float* X,
//                            const float beta, float* Y) {
//    metal_wrapper("metal_Saxpby", (void* )X, (void *)Y, {N}, {alpha, beta}, N);
//}
//
//void caffe_gpu_dot(const int N, const float* X, const float* Y,
//                          float* out) {
//    metal_wrapper("metal_Sdot", (void* )X, (void *)Y, {N}, {}, N);
//}
//
//void caffe_gpu_asum(const int N, const float* X, float* y) {
//    metal_wrapper("metal_Sdot", (void* )X, NULL, {N}, {}, N);
//}
///*
//template <>
//void caffe_gpu_gemv_metal<double>(const CBLAS_TRANSPOSE TransA, const int M,
//    const int N, const double alpha, const double* A, const double* x,
//    const double beta, double* y) {
//
//  // TODO change from cuda to metal
//  cublasOperation_t cuTransA =
//      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
//
//  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
//      A, N, x, 1, &beta, y, 1));
//
//  // end of this TODO block
//}
//*/


}  // namespace caffe


#endif

