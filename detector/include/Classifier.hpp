#pragma once
#include <string>
#include <opencv2/core/core.hpp>

struct Result {
	int label;
	float confidence;
	float confidence2;
};

class Classifier {
public:
    Classifier(char* net_path, bool on_gpu);
    virtual Result classify(cv::Mat &img) = 0;
    void report();
    virtual ~Classifier();
protected:
    string log;
    bool on_gpu;
    char* net_path;
};
