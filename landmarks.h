#ifndef __Landmarks_onboard__
#define __Landmarks_onboard__

#include <vector>
#include <opencv2/core.hpp>


struct Landmarks
{
    virtual int extract(const cv::Mat &img, std::vector<cv::Point> &pt) const = 0;
};

// define one of:
// HAVE_ELASTIC
// HAVE_DLIB
// HAVE_FACEX
cv::Ptr<Landmarks> createLandmarks();

void gftt64(const cv::Mat &img, std::vector<cv::KeyPoint> &kp);

#endif // __Landmarks_onboard__

