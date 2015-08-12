#pragma once
#include "Classifier.hpp"
#include <vector>
#include <opencv2/core/core.hpp>

const float DETECTOR_THRESHOLD = 0.5f;

class Detector {
public:
    void detect(const cv::Mat &img, std::vector<int> &labels, std::vector<double> &scores,
    			std::vector<cv::Rect> &rects,  cv::Ptr<Classifier> classifier,
    			cv::Size windowSize = cv::Size(20, 20), int dx = 1, int dy = 1, double scale = 1.2,
    			int minNeighbors = 3, bool groupRect = false);
private:
	void preprocessing(cv::Mat &img);
};
