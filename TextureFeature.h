#ifndef __TextureFeature_onboard__
#define __TextureFeature_onboard__

#include "opencv2/core/core.hpp"
using cv::Mat;


namespace TextureFeature	
{
    struct Extractor 
    {
        virtual int extract( const Mat &img, Mat &features ) const = 0;
    };

    struct Classifier 
    {
        virtual int train( const Mat &features, const Mat &labels ) = 0;
        virtual int predict( const Mat &test, Mat &result ) const = 0;
    };
};


#endif // __TextureFeature_onboard__

