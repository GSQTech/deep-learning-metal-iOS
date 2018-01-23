//
//  test_convolution_layer.m
//  metal_mac
//
//  Created by Tec GSQ on 3/12/2017.
//  Copyright © 2017 Tec GSQ. All rights reserved.
//

#import <vector>


#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

#include "test_caffe_main.hpp"



int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


