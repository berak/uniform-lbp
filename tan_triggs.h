#ifndef __tan_triggs_onboard__
#define __tan_triggs_onboard__

#include <opencv2/core/core.hpp>

cv::Mat tan_triggs_preprocessing(cv::InputArray src,
        float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1,
        int sigma1 = 2);


#endif // __tan_triggs_onboard__

