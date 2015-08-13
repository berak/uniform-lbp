#ifndef __net_onboard__
#define __net_onboard__
#include "opencv2/core.hpp"

struct PNet
{
    virtual ~PNet() {}

    virtual cv::Mat extract(const cv::Mat &img) const = 0; //that's all it needs here.
};

cv::Ptr<PNet> loadNet(const String &fn);

#endif // __net_onboard__

