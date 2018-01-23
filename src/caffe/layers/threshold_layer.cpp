//SJ
#include <vector>

#include "caffe/layers/threshold_layer.hpp"

namespace caffe {

template <typename Dtype>
void ThresholdLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.threshold_param().threshold();
}

template <typename Dtype>
void ThresholdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = (bottom_data[i] > threshold_) ? Dtype(1) : Dtype(0);
  }
}

#ifdef METAL
template <typename Dtype>
void ThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
    //Forward_cpu(bottom, top);
    metal_wrapper("ThresholdForward", bottom[0]->mutable_gpu_data(), bottom[0]->count()*4, top[0]->mutable_gpu_data(), top[0]->count()*4, {bottom[0]->count()}, {threshold_}, bottom[0]->count());
    
    //Forward_cpu(bottom, top);
}
#endif

INSTANTIATE_CLASS(ThresholdLayer);
REGISTER_LAYER_CLASS(Threshold);

}  // namespace caffe
