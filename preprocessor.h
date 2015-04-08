#ifndef __Preprocessor_onboard__
#define __Preprocessor_onboard__

#include "opencv2/opencv.hpp"
#include <opencv2/bioinspired.hpp>
using namespace cv;

class Preprocessor
{
    int preproc, precrop;
    Ptr<CLAHE> clahe;
    Ptr<bioinspired::Retina> retina;
    int fixed_size;

public:

	Preprocessor(int mode=0, int crop=0, int retsize=110);

    Mat process(const Mat &in) const;

    const char *pps() const;
};


#endif // __Preprocessor_onboard__

