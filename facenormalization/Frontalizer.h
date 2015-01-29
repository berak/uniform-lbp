#ifndef __Frontalizer_onboard__
#define __Frontalizer_onboard__

#include "opencv2/core.hpp"



//
//@brief  apply a one-size-fits-all 3d model transformation (POSIT style), also do 2d eye-alignment, if nessecary.
//

struct Frontalizer
{
    virtual cv::Mat align2d(const cv::Mat &imgray) const = 0;
    virtual cv::Mat project3d(const cv::Mat &imgray) const = 0;

    static cv::Ptr<Frontalizer> create(const cv::String &dlib_path, int crop, int symThreshold, double symBlend, bool write);
};


#endif // __Frontalizer_onboard__

