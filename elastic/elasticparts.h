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
    virtual double getPoints(const cv::Mat & img, std::vector<cv::Point> &p) const = 0;

    //// train:
    ////  add initial points & size
    //virtual void addPart(cv::Point2f p, int w, int h) = 0;
    //// run on imglist
    //virtual bool train( const std::vector<cv::Mat> &imgs, int search, float lambda, float mu_init, int nsamples, bool visu ) = 0;

    // make an instance:
    static cv::Ptr<ElasticParts> createDiscriminative();
    static cv::Ptr<ElasticParts> createGenerative();
};


#endif // __ElasticParts_onboard__

