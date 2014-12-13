#include "Preprocessor.h"
#include "opencv2/opencv.hpp"
#include <opencv2/bioinspired.hpp>
using namespace cv;

//
// taken from : https://github.com/bytefish/opencv/blob/master/misc/tan_triggs.cpp
//
static Mat tan_triggs_preprocessing(InputArray src, float alpha=0.1, float tau=10.0, float gamma=0.2, int sigma0=1, int sigma1=2)
{
    // Convert to floating point:
    Mat X = src.getMat();
    X.convertTo(X, CV_32FC1);
    // Start preprocessing:
    Mat I;
    pow(X, gamma, I);
    // Calculate the DOG Image:
    {
        Mat gaussian0, gaussian1;
        // Kernel Size:
        int kernel_sz0 = (3*sigma0);
        int kernel_sz1 = (3*sigma1);
        // Make them odd for OpenCV:
        kernel_sz0 += ((kernel_sz0 % 2) == 0) ? 1 : 0;
        kernel_sz1 += ((kernel_sz1 % 2) == 0) ? 1 : 0;
        GaussianBlur(I, gaussian0, Size(kernel_sz0,kernel_sz0), sigma0, sigma0, BORDER_CONSTANT);
        GaussianBlur(I, gaussian1, Size(kernel_sz1,kernel_sz1), sigma1, sigma1, BORDER_CONSTANT);
        subtract(gaussian0, gaussian1, I);
    }

    {
        double meanI = 0.0;
        {
            Mat tmp;
            pow(abs(I), alpha, tmp);
            meanI = mean(tmp).val[0];

        }
        I = I / pow(meanI, 1.0/alpha);
    }

    {
        double meanI = 0.0;
        {
            Mat tmp;
            pow(min(abs(I), tau), alpha, tmp);
            meanI = mean(tmp).val[0];
        }
        I = I / pow(meanI, 1.0/alpha);
    }

    // Squash into the tanh:
    {
        for(int r = 0; r < I.rows; r++) {
            for(int c = 0; c < I.cols; c++) {
                I.at<float>(r,c) = tanh(I.at<float>(r,c) / tau);
            }
        }
        I = tau * I;
    }
    return I;
}




Preprocessor::Preprocessor(int mode, int crop, int retsize)
    : preproc(mode)
    , precrop(crop)
    , clahe(createCLAHE(50))
    , retina(bioinspired::createRetina(Size(retsize,retsize)))
{
    //// (realistic setup)
    bioinspired::Retina::RetinaParameters ret_params;
    ret_params.OPLandIplParvo.horizontalCellsGain = 0.7f;
    ret_params.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity = 0.39f;
    ret_params.OPLandIplParvo.ganglionCellsSensitivity = 0.39f;
    retina->setup(ret_params);
}

Mat Preprocessor::process(const Mat &imgin)  const
{
    Mat imgcropped(imgin, Rect(precrop, precrop, imgin.cols-2*precrop, imgin.rows-2*precrop));

    Mat imgout;
    switch(preproc)
    {
        default:
        case 0: imgout = precrop>0 ? imgcropped.clone() : imgcropped; break;
        case 1: equalizeHist(imgcropped,imgout); break;
        case 2: clahe->apply(imgcropped,imgout); break;
        case 3: retina->run(imgcropped); retina->getParvo(imgout); break;
        case 4: cv::normalize(tan_triggs_preprocessing(imgcropped), imgout, 0, 255, NORM_MINMAX, CV_8UC1); break;
        case 5: resize(imgcropped,imgout,Size(32,32)); break;
    }
    return imgout;
}

const char * Preprocessor::pps() const
{
    static const char *PPS[] = { "none","eqhist","clahe","retina","tan-triggs","crop",0 };
    return PPS[preproc];
}
