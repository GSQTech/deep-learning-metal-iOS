#ifndef ImageReader_h
#define ImageReader_h

#include "caffe/caffe.hpp"

NSString* FilePathForResourceName(NSString* name, NSString* extension);

bool ReadImageToBlob(NSString *file_name,
                     const std::vector<float> &mean,
                     caffe::Blob<float>* input_layer);

#endif /* ImageReader_h */
