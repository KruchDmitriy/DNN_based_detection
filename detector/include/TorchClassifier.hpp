#pragma once
extern "C" {
	#include "lua.hpp"
	#include "luaT.h"
	#include "TH/TH.h"
}
#include "Classifier.hpp"

class TorchClassifier : public Classifier {
public:
	TorchClassifier(char* net_path, bool on_gpu);
    virtual Result classify(cv::Mat& img);
    virtual ~TorchClassifier();
private:
	void report_Lua_errors(lua_State *L, int status);

	lua_State *L;
};
