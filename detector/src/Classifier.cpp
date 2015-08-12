#include "Classifier.hpp"

Classifier(char* net_path_, bool on_gpu_)
: net_path(net_path_), on_gpu(on_gpu_) {
    log = "";
}

Classifier::~Classifier() {
    report();
}

void Classifier::report_errors() {
    printf("%s\n", log);
}
