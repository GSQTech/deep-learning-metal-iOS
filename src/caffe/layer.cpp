//
//  layer.cpp
//  metal_mac
//
//  Created by Tec GSQ on 4/12/2017.
//  Copyright © 2017 Tec GSQ. All rights reserved.
//

#include "caffe/layer.hpp"

namespace caffe {
    
    template <typename Dtype>
    void Layer<Dtype>::Qiaoge_alloc(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    }
    
    template <typename Dtype>
    void Layer<Dtype>::Qiaoge_free(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    }
    
    INSTANTIATE_CLASS(Layer);
    
}  // namespace caffe
