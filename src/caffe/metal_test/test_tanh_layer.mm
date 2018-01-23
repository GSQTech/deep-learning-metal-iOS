////
////  test_tanh_layer.m
////  metal_mac
////
////  Created by Tec GSQ on 30/11/2017.
////  Copyright © 2017 Tec GSQ. All rights reserved.
////
//
//#import <Foundation/Foundation.h>
//
//
//#include "gtest/gtest.h"
//
//#include "caffe/blob.hpp"
//#include "caffe/common.hpp"
//#include "caffe/filler.hpp"
//#include "caffe/layers/tanh_layer.hpp"
//
//#include "test_caffe_main.hpp"
//
//
//namespace caffe {
//    
//    double tanh_naive(double x) {
//        if (x < -40) {
//            // avoid negative overflow
//            return -1;
//        } else if (x > 40) {
//            // avoid positive overflow
//            return 1;
//        } else {
//            // exact expression for tanh, which is unstable for large x
//            double exp2x = exp(2 * x);
//            return (exp2x - 1.0) / (exp2x + 1.0);
//        }
//    }
//    
//    template <typename TypeParam>
//    class TanHLayerTest : public MultiDeviceTest<TypeParam> {
//        typedef typename TypeParam::Dtype Dtype;
//        
//    protected:
//        TanHLayerTest()
//        : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
//        blob_top_(new Blob<Dtype>()) {
//            Caffe::set_random_seed(1701);
//            FillerParameter filler_param;
//            blob_bottom_vec_.push_back(blob_bottom_);
//            blob_top_vec_.push_back(blob_top_);
//        }
//        virtual ~TanHLayerTest() { delete blob_bottom_; delete blob_top_; }
//        
//        void TestForward(Dtype filler_std) {
//            FillerParameter filler_param;
//            filler_param.set_std(filler_std);
//            GaussianFiller<Dtype> filler(filler_param);
//            filler.Fill(this->blob_bottom_);
//            
//            LayerParameter layer_param;
//            TanHLayer<Dtype> layer(layer_param);
//            layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//            layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//            // Now, check values
//            const Dtype* bottom_data = this->blob_bottom_->cpu_data();
//            const Dtype* top_data = this->blob_top_->cpu_data();
//            const Dtype min_precision = 1e-5;
//            for (int i = 0; i < this->blob_bottom_->count(); ++i) {
//                Dtype expected_value = tanh_naive(bottom_data[i]);
//                Dtype precision = std::max(
//                                           Dtype(std::abs(expected_value * Dtype(1e-4))), min_precision);
//                EXPECT_NEAR(expected_value, top_data[i], precision);
//            }
//        }
//        
//        Blob<Dtype>* const blob_bottom_;
//        Blob<Dtype>* const blob_top_;
//        vector<Blob<Dtype>*> blob_bottom_vec_;
//        vector<Blob<Dtype>*> blob_top_vec_;
//    };
//    
//    TYPED_TEST_CASE(TanHLayerTest, TestDtypesAndDevices);
//    
//    TYPED_TEST(TanHLayerTest, TestTanH) {
//        Caffe::Get().set_mode(Caffe::GPU);
//        this->TestForward(1.0);
//    }
//    
//    TYPED_TEST(TanHLayerTest, TestTanHOverflow) {
//        // this will fail if tanh overflow is not properly handled
//        Caffe::Get().set_mode(Caffe::GPU);
//        this->TestForward(10000.0);
//    }
//    
//}  // namespace caffe

