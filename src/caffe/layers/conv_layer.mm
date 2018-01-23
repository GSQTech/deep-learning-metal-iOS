//CHECK
#include <vector>
#include <Metal.h>
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}
    
// here if define METAL 
// it means that forward/backward gpu is defined
// in another file call foo.mm



    
#ifdef METAL

    
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
    
    /*const Dtype* weight = this->blobs_[0]->gpu_data();
    for (int i = 0; i < bottom.size(); ++i) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* top_data = top[i]->mutable_gpu_data();
        for (int n = 0; n < this->num_; ++n) {
            this->forward_gpu_gemm(bottom_data, bottom[0]->count(), n * this->bottom_dim_, weight,
                                   top_data, top[0]->count(), n * this->top_dim_);
            if (this->bias_term_) {
                const Dtype* bias = this->blobs_[1]->gpu_data();
                this->forward_gpu_bias(top_data, n * this->top_dim_, bias);
            }
        }
    }*/
    ConvolutionParameter conv_param = this->layer_param().convolution_param();
    const int channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
    
    const int first_spatial_axis = channel_axis_ + 1;
    const int num_axes = bottom[0]->num_axes();
    const int num_spatial_axes_ = num_axes - first_spatial_axis;
    int* kernel_shape_data = new int[2];
    // now we only deal with the data with kernel dimension equal to 2
    
    if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
        CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
        CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
        kernel_shape_data[0] = conv_param.kernel_h();
        kernel_shape_data[1] = conv_param.kernel_w();
    } else {
        const int num_kernel_dims = conv_param.kernel_size_size();
        CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
        for (int i = 0; i < num_spatial_axes_; ++i) {
            kernel_shape_data[i] =
            
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
        }
    }
    
    int* pad_data = new int[2];
    // now we only deal with the data with pad dimension equal to 2
    if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
        CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
        CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
        pad_data[0] = conv_param.pad_h();
        pad_data[1] = conv_param.pad_w();
    } else {
        const int num_pad_dims = conv_param.pad_size();
        CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
              num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
        const int kDefaultPad = 0;
        for (int i = 0; i < num_spatial_axes_; ++i) {
            pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
            
            conv_param.pad((num_pad_dims == 1) ? 0 : i);
        }
    }
    
    int* stride_data = new int[2];
    // now we only deal with the data with stride dimension equal to 2
    
    if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
        CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
        CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
        stride_data[0] = conv_param.stride_h();
        stride_data[1] = conv_param.stride_w();
    } else {
        const int num_stride_dims = conv_param.stride_size();
        CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
              num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
        const int kDefaultStride = 1;
        for (int i = 0; i < num_spatial_axes_; ++i) {
            stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
            conv_param.stride((num_stride_dims == 1) ? 0 : i);
            CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
        }
    }
    
    int* dilation_data = new int[num_spatial_axes_];
    const int kDefaultDilation = 1;
    const int num_dilation_dims = conv_param.dilation_size();
    for (int i = 0; i < num_spatial_axes_; ++i) {
        dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
        conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
    }
    
    std::vector<int> params = {2, static_cast<int>(conv_param.group()), (bottom[0]->count()), (top[0]->count()), (bottom[0]->shape(2)), (top[0]->shape(2)), (bottom[0]->shape(3)), (top[0]->shape(3)), (bottom[0]->shape(2)*bottom[0]->shape(3)), (top[0]->shape(2) * top[0]->shape(3)), kernel_shape_data[0], kernel_shape_data[1], pad_data[0], pad_data[1], stride_data[0], stride_data[1], dilation_data[0], dilation_data[1], (bottom[0]->shape(1)), (top[0]->shape(1)), (top[0]->shape(1)), top[0]->shape(1) / static_cast<int>(conv_param.group()),  (top[0]->shape(2) * top[0]->shape(3)), bottom[0]->shape(1)*kernel_shape_data[0] * kernel_shape_data[1], bottom[0]->shape(1)*kernel_shape_data[0] * kernel_shape_data[1]/static_cast<int>(conv_param.group()), ((((bottom[0]->shape(1)*kernel_shape_data[0] * kernel_shape_data[1]/static_cast<int>(conv_param.group())) - 1)/(8*2) + 1)*2)};
    
    
    
    //MTLFunctionConstantValues* constantValues = [MTLFunctionConstantValues new];
    //[constantValues setConstantValues:&params type:MTLDataTypeInt withRange:NSMakeRange(0, 40)];
    
    
    NSString *funcname;
    //= [NSString stringWithCString:this->layer_param().name().c_str() encoding:[NSString defaultCStringEncoding]];

    if(this->bias_term_){
        funcname = @"conv_bias";
    } else {
        funcname = @"conv_no_bias";
    }
    
    NSError *errors;
    //funcname = [NSString stringWithCString:this->layer_param().name().c_str() encoding:[NSString defaultCStringEncoding]];
    id<MTLDevice>       device   = ((__bridge id<MTLDevice>)(Caffe::Get().metal_device));
    id<MTLLibrary>      library  = [device  newDefaultLibrary];
    id<MTLFunction>     function = //[library newFunctionWithName:funcname];
    [library newFunctionWithName:funcname];
    
    
    
    
    id<MTLBuffer> in_gpu    = [device newBufferWithBytesNoCopy:bottom[0]->mutable_cpu_data()  length:align_size(bottom[0]->count()*4)  options:MTLStorageModeShared deallocator:nil];
    
    id<MTLBuffer> out_gpu    = [device newBufferWithBytesNoCopy:top[0]->mutable_cpu_data()  length:align_size(top[0]->count()*4)  options:MTLStorageModeShared deallocator:nil];
    
    id<MTLBuffer> wg_gpu  = [device newBufferWithBytesNoCopy:this->blobs_[0]->mutable_cpu_data()  length:align_size(this->blobs_[0]->count()*4)  options:MTLStorageModeShared deallocator:nil];
    
    id<MTLBuffer> bias_gpu;
    
    id<MTLBuffer> argu_gpu = [device newBufferWithLength:params.size()*4 options:MTLStorageModeShared];
    int* argu_tmp = (int *)[argu_gpu contents];
    
    for(int i = 0; i < params.size(); ++i) {
        argu_tmp[i] = params[i];
    }
    
    if(this->bias_term_) {
        bias_gpu  = [device newBufferWithBytesNoCopy:this->blobs_[1]->mutable_cpu_data()  length:align_size(this->blobs_[1]->count()*4)  options:MTLStorageModeShared deallocator:nil];
    }
    
    id<MTLCommandBuffer> buffer = [((__bridge id<MTLCommandQueue>)(Caffe::Get().metal_commandQueue)) commandBuffer];
    
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    
    id<MTLComputePipelineState>  state   = [device newComputePipelineStateWithFunction:function error:&errors];
    
    int fw_wptn = 4;
    int fw_wptm = 4;
    int fw_wgs0 = 8;
    int fw_wgs1 = 8;
    int fw_div_N = fw_wptn * fw_wgs0;
    int fw_div_M = fw_wptm * fw_wgs1;
    
    MTLSize groupsize = {static_cast<NSUInteger>(fw_wgs0),static_cast<NSUInteger>(fw_wgs1),1};
    MTLSize numgroups = {static_cast<NSUInteger>((((top[0]->count()/top[0]->shape(1)) - 1) / fw_div_N + 1)),static_cast<NSUInteger>(((top[0]->shape(1) - 1) / fw_div_M + 1)),1};
    [encoder setComputePipelineState:state];
    
    [encoder setBuffer:in_gpu  offset:0 atIndex:0];
    [encoder setBuffer:out_gpu  offset:0 atIndex:1];
    [encoder setBuffer:wg_gpu  offset:0 atIndex:2];
    
    if(this->bias_term_) {
        [encoder setBuffer:bias_gpu  offset:0 atIndex:3];
    }
    
    [encoder setBuffer:argu_gpu  offset:0 atIndex:4];
    
    [encoder dispatchThreadgroups:numgroups threadsPerThreadgroup:groupsize];
    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];
    //Forward_cpu(bottom, top);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}
    
#endif
    
INSTANTIATE_CLASS(ConvolutionLayer);
}  // namespace caffe
