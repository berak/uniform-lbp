#include "preprocessor.h"
#include "opencv2/opencv.hpp"
#include <opencv2/bioinspired.hpp>
using namespace cv;
//Mat DoGFilter(InputArray src){
//	
//	Mat g1, g2, dst;
//	GaussianBlur(src, g1, Size(5,5), 0);
//	GaussianBlur(src, g2, Size(9,9), 0);
//	
//	dst = g2 - g1;
//	normalize(dst,dst,0, 255, NORM_MINMAX, CV_8U);
//	
//	return dst;
//}
//
//
//Mat CSDNFilter(InputArray _src){
//	
//	Mat dst;
//	Mat src = _src.getMat();
//	src.convertTo(src, CV_32F);
//	
//	blur(src, dst, Size(16,16));
//	divide(src, dst, dst);
//	normalize(dst, dst, 0, 255, NORM_MINMAX, CV_8U);
//	
//	return dst; 
//}
//
//template <typename _Tp> static
//inline void radonTransform(InputArray _src, OutputArray _dst)
//{
//    //get matrices
//    Mat src = _src.getMat();
//    /// Compute a rotation matrix with respect to the center of the image
//    Point center = Point(src.cols/2, src.rows/2);
//    double angle = 0.0;
//    double scale = 1;
//    int diag = int(sqrt(double(src.rows*src.rows+src.cols*src.cols)));
//    // allocate memory for result
//    _dst.create(diag, 180, CV_32FC1);
//    Mat dst = _dst.getMat();
//    // zero
//    dst.setTo(0);
//    Mat rot_mat(2, 3, CV_32FC1);
//    Mat rotate_dst;
//    Size size = Size(diag, diag);
//    /// Rotate the warped image
//    //warpAffine(src, rotate_dst, rot_mat, size);
//    for(angle=0; angle<180; angle++)
//    {
//        /// Get the rotation matrix with the specifications above
//        rot_mat = getRotationMatrix2D( center, 90-angle, scale );
//        rot_mat.at<double>(0,2) += (diag - src.cols)/2.0;
//        rot_mat.at<double>(1,2) += (diag - src.rows)/2.0;
//        /// Rotate the warped image
//        warpAffine(src, rotate_dst, rot_mat, size);
//        for(int i=0; i<diag; i++)
//        {
//            for(int j=0; j<diag; j++)
//            {
//                dst.at<float>(diag-i-1,angle) += rotate_dst.at<_Tp>(i,j);
//            }
//        }
//    }
//    normalize(dst, dst, 0, 1, NORM_MINMAX, CV_32FC1);
//    //normalize(dst, dst, 0, 1, NORM_MINMAX, CV_32FC1);
//}


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
    , fixed_size(retsize)
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
    Mat imgt;
    Size sz(fixed_size,fixed_size);
    if (imgcropped.size() != sz)
        resize(imgcropped, imgt, sz);
    else
        imgt = imgcropped;

    Mat imgout;
    switch(preproc)
    {
        default:
        case 0: imgout = precrop>0 ? imgt.clone() : imgt; break;
        case 1: equalizeHist(imgt,imgout); break;
        case 2: clahe->apply(imgt,imgout); break;
        case 3: retina->run(imgt); retina->getParvo(imgout); break;
        case 4: cv::normalize(tan_triggs_preprocessing(imgt), imgout, 0, 255, NORM_MINMAX, CV_8UC1); break;
        case 5: imgt.convertTo(imgout,CV_32F,1,1); log(imgout,imgout); imgout.convertTo(imgout,CV_8U);break; // logscale
        //case 6: radonTransform<uchar>(imgt,imgout); imgout.convertTo(imgout,CV_8U); break;
        //case 7: imgout = CSDNFilter(imgt); break;
        //case 8: imgout = DoGFilter(imgt); break;
    }
    return imgout;
}

const char * Preprocessor::pps() const
{
    static const char *PPS[] = { "none","eqhist","clahe","retina","tan-triggs","logscale",0 };
    return PPS[preproc];
}
