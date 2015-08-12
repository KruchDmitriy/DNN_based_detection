#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <vector>
#include <iostream>

#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using namespace cv;
using namespace std;

static void read_Mat2Datum(Mat& img, Datum* datum) {
    int num_channels = img.channels();
    datum->set_channels(num_channels);
    datum->set_height(img.rows);
    datum->set_width(img.cols);
    datum->clear_data();
    datum->clear_float_data();
    string* datum_string = datum->mutable_data();
    if (num_channels!=1) {
        for (int c = 0; c < num_channels; ++c) {
            for (int h = 0; h < img.rows; ++h) {
                for (int w = 0; w < img.cols; ++w) {
                    datum_string->push_back(
                        static_cast<char>(img.at<Vec3b>(h, w)[c]));
                }
            }
        }
    } else {
        for (int h = 0; h < img.rows; ++h) {
            for (int w = 0; w < img.cols; ++w) {
                datum_string->push_back(
                    static_cast<char>(img.at<uchar>(h, w)));
            }
        }
    }
}

CaffeClassifier::CaffeClassifier(char *net_proto_path, int device_id) {
    Caffe::set_phase(Caffe::TEST);

    // Setting CPU or GPU
    if (on_gpu) {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);
        log += "Using GPU #" + to_string(device_id);
    } else {
        Caffe::set_mode(Caffe::CPU);
        log += "Using CPU";
    }

    // load prototxt
    caffe_test_net(net_proto_path);
    // load trained net
    caffe_test_net.CopyTrainedLayersFrom(net_path);
}

Result CaffeClassifier::classify(Mat& img) {
    Datum datum;
    ReadMatToDatum(img, &datum);

    Blob<float>* blob = new Blob<float>(1, datum.channels(),
        datum.height(), datum.width());

    BlobProto blob_proto;
    blob_proto.set_num(1);
    blob_proto.set_channels(datum.channels());
    blob_proto.set_height(datum.height());
    blob_proto.set_width(datum.width());
    const int data_size = datum.channels() * datum.height() * datum.width();
    int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());

    for (int i = 0; i < size_in_datum; ++i) {
        blob_proto.add_data(0.);
    }
    const string& data = datum.data();
    if (data.size() != 0) {
        for (int i = 0; i < size_in_datum; ++i) {
            blob_proto.set_data(i, blob_proto.data(i) + (uint8_t)data[i]);
        }
    }
    blob->FromProto(blob_proto);

    vector<Blob<float>*> bottom;
    bottom.push_back(blob);
    float type = 0.0;
    const vector<Blob<float>*>& result = caffe_test_net.Forward(bottom, &type);

    float max = 0;
    float max_i = 0;
    for (int i = 0; i < result.size(); ++i) {
        float value = result[0]->cpu_data()[i];
        if (max < value){
            max = value;
            max_i = i;
        }
    }

    Result result;
    result.confidence = max;
    result.label = max_i;

    return result;
}
