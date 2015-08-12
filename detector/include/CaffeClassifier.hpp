#pragma once
#include "caffe/blob.hpp"
#include "Classifier.hpp"

class CaffeClassifier : public Classifier {
public:
    CaffeClassifier(char *net_proto_path, int device_id);
    virtual ~CaffeClassifier();
    virtual Result classify(cv::Mat& img);
private:
    char *net_proto_path;
    int device_id;
    Net<float> caffe_test_net;
};
