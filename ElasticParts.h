#ifndef __ElasticParts_onboard__
#define __ElasticParts_onboard__


#include <vector>
#include "opencv2/opencv.hpp"

struct ElasticParts
{
    // io:
    virtual bool read (const cv::String &fn) = 0;
    virtual bool write(const cv::String &fn) = 0;

    // test:
    //  find best points to sample:
    virtual void getPoints(const cv::Mat & img, std::vector<cv::KeyPoint> &kp) const = 0;

    // train:
    //  add initial points & size
    virtual void addPart(cv::Point2f p, int w, int h) = 0;
    //  update (e.g. from dlib's facial landmarks)
    virtual void setPoint(int i, const cv::Point2f &p) = 0;
    //  take samples at given position
    virtual void sample(const cv::Mat &img) = 0;
    //  make mean samples for k poses
    virtual void means() = 0;

    // make an instance:
    static cv::Ptr<ElasticParts> create();
};


#endif // __ElasticParts_onboard__

