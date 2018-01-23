//SJ
#include <vector>

#include "caffe/layers/power_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PowerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  power_ = this->layer_param_.power_param().power();
  scale_ = this->layer_param_.power_param().scale();
  shift_ = this->layer_param_.power_param().shift();
  diff_scale_ = power_  * scale_;
}

// Compute y = (shift + scale * x)^power
template <typename Dtype>
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  // Special case where we can ignore the input: scale or power is 0.
  if (diff_scale_ == Dtype(0)) {
    Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
    caffe_set(count, value, top_data);
    return;
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  caffe_copy(count, bottom_data, top_data);
  if (scale_ != Dtype(1)) {
    caffe_scal(count, scale_, top_data);
  }
  if (shift_ != Dtype(0)) {
    caffe_add_scalar(count, shift_, top_data);
  }
  if (power_ != Dtype(1)) {
    caffe_powx(count, top_data, power_, top_data);
  }
}

template <typename Dtype>
void PowerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    if (diff_scale_ == Dtype(0) || power_ == Dtype(1)) {
      caffe_set(count, diff_scale_, bottom_diff);
    } else {
      const Dtype* bottom_data = bottom[0]->cpu_data();
      // Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
      //               = diff_scale * y / (shift + scale * x)
      if (power_ == Dtype(2)) {
        // Special case for y = (shift + scale * x)^2
        //     -> dy/dx = 2 * scale * (shift + scale * x)
        //              = diff_scale * shift + diff_scale * scale * x
        caffe_cpu_axpby(count, diff_scale_ * scale_, bottom_data,
            Dtype(0), bottom_diff);
        if (shift_ != Dtype(0)) {
          caffe_add_scalar(count, diff_scale_ * shift_, bottom_diff);
        }
      } else if (shift_ == Dtype(0)) {
        // Special case for y = (scale * x)^power
        //     -> dy/dx = scale * power * (scale * x)^(power - 1)
        //              = scale * power * (scale * x)^power * (scale * x)^(-1)
        //              = power * y / x
        const Dtype* top_data = top[0]->cpu_data();
        caffe_div(count, top_data, bottom_data, bottom_diff);
        caffe_scal(count, power_, bottom_diff);
      } else {
        caffe_copy(count, bottom_data, bottom_diff);
        if (scale_ != Dtype(1)) {
          caffe_scal(count, scale_, bottom_diff);
        }
        if (shift_ != Dtype(0)) {
          caffe_add_scalar(count, shift_, bottom_diff);
        }
        const Dtype* top_data = top[0]->cpu_data();
        caffe_div<Dtype>(count, top_data, bottom_diff, bottom_diff);
        if (diff_scale_ != Dtype(1)) {
          caffe_scal(count, diff_scale_, bottom_diff);
        }
      }
    }
    if (diff_scale_ != Dtype(0)) {
      caffe_mul(count, top_diff, bottom_diff, bottom_diff);
    }
  }
}

#ifdef METAL
template <typename Dtype>
void PowerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
    
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
    // Special case where we can ignore the input: scale or power is 0.
    if (diff_scale_ == Dtype(0)) {
        Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
        metal_wrapper("set_kernel", bottom[0]->mutable_cpu_data(), bottom[0]->count() * 4, top[0]->mutable_cpu_data(), top[0]->count() * 4, {count}, {value}, count);
        return;
    }
    const Dtype* bottom_data = bottom[0]->gpu_data();
    caffe_copy(count, bottom_data, top_data);
    if (scale_ != Dtype(1)) {
        metal_wrapper("scal_kernel", bottom[0]->mutable_cpu_data(), bottom[0]->count() * 4, top[0]->mutable_cpu_data(), top[0]->count() * 4, {count}, {scale_}, count);
    }
    if (shift_ != Dtype(0)) {
        metal_wrapper("add_scalar_kernel", bottom[0]->mutable_cpu_data(), bottom[0]->count() * 4, top[0]->mutable_cpu_data(), top[0]->count() * 4, {count}, {shift_}, count);
    }
    if (power_ != Dtype(1)) {
        metal_wrapper("powx_kernel", bottom[0]->mutable_cpu_data(), bottom[0]->count() * 4, top[0]->mutable_cpu_data(), top[0]->count() * 4, {count}, {power_}, count);
    }
    //Forward_cpu(bottom, top);
}

template <typename Dtype>
void PowerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    
}
#endif

INSTANTIATE_CLASS(PowerLayer);
REGISTER_LAYER_CLASS(Power);

}  // namespace caffe
