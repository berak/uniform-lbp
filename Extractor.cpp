#include "opencv2/opencv.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/utility.hpp"
#include "opencv2/xfeatures2d.hpp"



//
// using regular grids for feature extraction on faces clearly sucks.
// one of the attempts tried here instead is, to use assorted landmark points
// from asmlib, flandmark or such instead.
// dlib's implementation seems to rule here, let's try to use it,
// fall back to a precalculated 'one-size-fits-all' manner(based on the mean lfw image), if not present:
//
#define HAVE_DLIB 

#ifdef HAVE_DLIB
 #include <dlib/image_processing.h>
 #include <dlib/opencv/cv_image.h>
#endif


#include <vector>
using std::vector;
#include <iostream>
using std::cerr;
using std::endl;

#include "TextureFeature.h"
#include "ElasticParts.h"
#include "Profile.h"

using namespace cv;

namespace TextureFeatureImpl
{


//
// this is the most simple one.
//
struct ExtractorPixels : public TextureFeature::Extractor
{
    int resw,resh;

    ExtractorPixels(int resw=0,int resh=0): resw(resw), resh(resh) {}

    // TextureFeature::Extractor
    virtual int extract(const Mat &img, Mat &features) const
    {
        if (resw>0 && resh>0)
            resize(img, features, Size(resw,resh));
        else
            features = img;
        features = features.reshape(1,1);
        return features.total() * features.elemSize();
    }
};


//
// later use gridded histograms the same way as with lbp(h)
//
struct FeatureGrad
{
    int nsec,nrad;
    FeatureGrad(int nsec=45, int nrad=8) : nsec(nsec), nrad(nrad) {}

    int operator() (const Mat &I, Mat &fI) const
    {
        Mat s1, s2, s3(I.size(), CV_32F), s4, s5;
        Sobel(I, s1, CV_32F, 1, 0);
        Sobel(I, s2, CV_32F, 0, 1);
        fastAtan2(s1.ptr<float>(0), s2.ptr<float>(0), s3.ptr<float>(0), I.total(), true);
        fI = s3 / (360/nsec);
        fI.convertTo(fI,CV_8U);

        //magnitude(s1.ptr<float>(0), s2.ptr<float>(0), s3.ptr<float>(0), I.total());
        //normalize(s3,s4,nrad);
        //s4.convertTo(s5,CV_8U);
        //s5 *= nsec;
        //fI += s5;
        //return nrad*nsec;
        return nsec;
    }
};


struct FeatureLbp
{
    int operator() (const Mat &I, Mat &fI) const
    {
        Mat_<uchar> feature(I.size(),0);
        Mat_<uchar> img(I);
        const int m=1;
        for (int r=m; r<img.rows-m; r++)
        {
            for (int c=m; c<img.cols-m; c++)
            {
                uchar v = 0;
                uchar cen = img(r,c);
                v |= (img(r-1,c  ) > cen) << 0;
                v |= (img(r-1,c+1) > cen) << 1;
                v |= (img(r  ,c+1) > cen) << 2;
                v |= (img(r+1,c+1) > cen) << 3;
                v |= (img(r+1,c  ) > cen) << 4;
                v |= (img(r+1,c-1) > cen) << 5;
                v |= (img(r  ,c-1) > cen) << 6;
                v |= (img(r-1,c-1) > cen) << 7;
                feature(r,c) = v;
            }
        }
        fI = feature;
        return 256;
    }
};


//
// "Description of Interest Regions with Center-Symmetric Local Binary Patterns"
// (http://www.ee.oulu.fi/mvg/files/pdf/pdf_750.pdf).
//    (w/o threshold)
//
struct FeatureCsLbp
{
    int operator() (const Mat &I, Mat &fI) const
    {
        Mat_<uchar> feature(I.size(),0);
        Mat_<uchar> img(I);
        const int m=1;
        for (int r=m; r<img.rows-m; r++)
        {
            for (int c=m; c<img.cols-m; c++)
            {
                uchar v = 0;
                v |= (img(r-1,c  ) > img(r+1,c  )) << 0;
                v |= (img(r-1,c+1) > img(r+1,c-1)) << 1;
                v |= (img(r  ,c+1) > img(r  ,c-1)) << 2;
                v |= (img(r+1,c+1) > img(r-1,c-1)) << 3;
                feature(r,c) = v;
            }
        }
        fI = feature;
        return 16;
    }
};


//
// / \
// \ /
//
struct FeatureDiamondLbp
{
    int operator() (const Mat &I, Mat &fI) const
    {
        Mat_<uchar> feature(I.size(),0);
        Mat_<uchar> img(I);
        const int m=1;
        for (int r=m; r<img.rows-m; r++)
        {
            for (int c=m; c<img.cols-m; c++)
            {
                uchar v = 0;
                v |= (img(r-1,c  ) > img(r  ,c+1)) << 0;
                v |= (img(r  ,c+1) > img(r+1,c  )) << 1;
                v |= (img(r+1,c  ) > img(r  ,c-1)) << 2;
                v |= (img(r  ,c-1) > img(r-1,c  )) << 3;
                feature(r,c) = v;
            }
        }
        fI = feature;
        return 16;
    }
};


//  _ _
// |   |
// |_ _|
//
struct FeatureSquareLbp
{
    int operator() (const Mat &I, Mat &fI) const
    {
        Mat_<uchar> feature(I.size(),0);
        Mat_<uchar> img(I);
        const int m=1;
        for (int r=m; r<img.rows-m; r++)
        {
            for (int c=m; c<img.cols-m; c++)
            {
                uchar v = 0;
                v |= (img(r-1,c-1) > img(r-1,c+1)) << 0;
                v |= (img(r-1,c+1) > img(r+1,c+1)) << 1;
                v |= (img(r+1,c+1) > img(r+1,c-1)) << 2;
                v |= (img(r+1,c-1) > img(r-1,c-1)) << 3;
                feature(r,c) = v;
            }
        }
        fI = feature;
        return 16;
    }
};


//
// just run around in a circle (instead of comparing to the center) ..
//
struct FeatureBGC1
{
    int operator () (const Mat &I, Mat &fI) const
    {
        Mat_<uchar> feature(I.size(),0);
        Mat_<uchar> img(I);
        const int m=1;
        for (int r=m; r<img.rows-m; r++)
        {
            for (int c=m; c<img.cols-m; c++)
            {
                uchar v = 0;
                v |= (img(r-1,c  ) > img(r-1,c-1)) << 0;
                v |= (img(r-1,c+1) > img(r-1,c  )) << 1;
                v |= (img(r  ,c+1) > img(r-1,c+1)) << 2;
                v |= (img(r+1,c+1) > img(r  ,c+1)) << 3;
                v |= (img(r+1,c  ) > img(r+1,c+1)) << 4;
                v |= (img(r+1,c-1) > img(r+1,c  )) << 5;
                v |= (img(r  ,c-1) > img(r+1,c-1)) << 6;
                v |= (img(r-1,c-1) > img(r  ,c-1)) << 7;
                feature(r,c) = v;
            }
        }
        fI = feature;
        return 256;
    }
};


//
// Antonio Fernandez, Marcos X. Alvarez, Francesco Bianconi:
// "Texture description through histograms of equivalent patterns"
//
//    basically, this is just 1/2 of the full lbp-circle (4bits / 16 bins only!)
//
struct FeatureMTS
{
    int operator () (const Mat &I, Mat &fI) const
    {
        Mat_<uchar> img(I);
        Mat_<uchar> fea(I.size(), 0);
        const int m=1;
        for (int r=m; r<img.rows-m; r++)
        {
            for (int c=m; c<img.cols-m; c++)
            {
                uchar v = 0;
                uchar cen = img(r,c);
                v |= (img(r-1,c  ) > cen) << 0;
                v |= (img(r-1,c+1) > cen) << 1;
                v |= (img(r  ,c+1) > cen) << 2;
                v |= (img(r+1,c+1) > cen) << 3;
                fea(r,c) = v;
            }
        }
        fI = fea;
        return 16;
    }
};

// left half
struct FeatureMTS2
{
    int operator () (const Mat &I, Mat &fI) const
    {
        Mat_<uchar> img(I);
        Mat_<uchar> fea(I.size(), 0);
        const int m=1;
        for (int r=m; r<img.rows-m; r++)
        {
            for (int c=m; c<img.cols-m; c++)
            {
                uchar v = 0;
                uchar cen = img(r,c);
                v |= (img(r+1,c  ) > cen) << 0;
                v |= (img(r+1,c-1) > cen) << 1;
                v |= (img(r  ,c-1) > cen) << 2;
                v |= (img(r-1,c-1) > cen) << 3;
                fea(r,c) = v;
            }
        }
        fI = fea;
        return 16;
    }
};


//
// Wolf, Hassner, Taigman : "Descriptor Based Methods in the Wild"
// 3.1 Three-Patch LBP Codes
//
struct FeatureTPLbp
{
    int operator () (const Mat &img, Mat &features) const
    {
        Mat_<uchar> I(img);
        Mat_<uchar> fI(I.size(), 0);
        const int border=2;
        for (int r=border; r<I.rows-border; r++)
        {
            for (int c=border; c<I.cols-border; c++)
            {
                uchar v = 0;
                v |= ((I(r,c) - I(r  ,c-2)) > (I(r,c) - I(r-2,c  ))) * 1;
                v |= ((I(r,c) - I(r-1,c-1)) > (I(r,c) - I(r-1,c+1))) * 2;
                v |= ((I(r,c) - I(r-2,c  )) > (I(r,c) - I(r  ,c+2))) * 4;
                v |= ((I(r,c) - I(r-1,c+1)) > (I(r,c) - I(r+1,c+1))) * 8;
                v |= ((I(r,c) - I(r  ,c+2)) > (I(r,c) - I(r+1,c  ))) * 16;
                v |= ((I(r,c) - I(r+1,c+1)) > (I(r,c) - I(r+1,c-1))) * 32;
                v |= ((I(r,c) - I(r+1,c  )) > (I(r,c) - I(r  ,c-2))) * 64;
                v |= ((I(r,c) - I(r+1,c-1)) > (I(r,c) - I(r-1,c-1))) * 128;
                fI(r,c) = v;
            }
        }
        features = fI;
        return 256;
    }
};

struct FeatureTPLbp2
{
    int operator () (const Mat &img, Mat &features) const
    {
        Mat_<uchar> I(img);
        Mat_<uchar> fI(I.size(), 0);
        const int border=2;
        for (int r=border; r<I.rows-border; r++)
        {
            for (int c=border; c<I.cols-border; c++)
            {
                uchar v = 0;
                v |= ((I(r,c) - I(r  ,c-2)) > (I(r,c) - I(r-2,c  ))) * 1;
                v |= ((I(r,c) - I(r-1,c-1)) > (I(r,c) - I(r-1,c+1))) * 2;
                //v |= ((I(r,c) - I(r-2,c  )) > (I(r,c) - I(r  ,c+2))) * 4;
                //v |= ((I(r,c) - I(r-1,c+1)) > (I(r,c) - I(r+1,c+1))) * 8;
                //v |= ((I(r,c) - I(r  ,c+2)) > (I(r,c) - I(r+1,c  ))) * 4;
                //v |= ((I(r,c) - I(r+1,c+1)) > (I(r,c) - I(r+1,c-1))) * 2;
                v |= ((I(r,c) - I(r+1,c  )) > (I(r,c) - I(r  ,c-2))) * 4;
                v |= ((I(r,c) - I(r+1,c-1)) > (I(r,c) - I(r-1,c-1))) * 8;
                fI(r,c) = v;
            }
        }
        features = fI;
        return 16;
    }
};



//
// Wolf, Hassner, Taigman : "Descriptor Based Methods in the Wild"
// 3.2 Four-Patch LBP Codes (4bits / 16bins only !)
//
struct FeatureFPLbp
{
    int operator () (const Mat &img, Mat &features) const
    {
        Mat_<uchar> I(img);
        Mat_<uchar> fI(I.size(), 0);
        const int border=2;
        for (int r=border; r<I.rows-border; r++)
        {
            for (int c=border; c<I.cols-border; c++)
            {
                uchar v = 0;
                v |= ((I(r  ,c+1) - I(r+2,c+2)) > (I(r  ,c-1) - I(r-2,c-2))) * 1;
                v |= ((I(r+1,c+1) - I(r+2,c  )) > (I(r-1,c-1) - I(r-2,c  ))) * 2;
                v |= ((I(r+1,c  ) - I(r+2,c-2)) > (I(r-1,c  ) - I(r-2,c+2))) * 4;
                v |= ((I(r+1,c-1) - I(r  ,c-2)) > (I(r-1,c+1) - I(r  ,c+2))) * 8;
                fI(r,c) = v;
            }
        }
        features = fI;
        return 16;
    }
};





static void hist_patch(const Mat_<uchar> &fI, Mat &histo, int histSize=256)
{
    Mat_<float> h(1, histSize, 0.0f);
    for (int i=0; i<fI.rows; i++)
    {
        for (int j=0; j<fI.cols; j++)
        {
            int v = int(fI(i,j));
            h( v ) += 1.0f;
        }
    }
    histo.push_back(h.reshape(1,1));
}

static void hist_patch_uniform(const Mat_<uchar> &fI, Mat &histo, int histSize=256)
{
    static int uniform[256] = 
    {   // the well known original uniform2 pattern
        0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,
        14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,
        58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,
        58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,
        58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,
        58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,
        58,58,58,50,51,52,58,53,54,55,56,57 
    };

    Mat_<float> h(1, 59, 0.0f);
    for (int i=0; i<fI.rows; i++)
    {
        for (int j=0; j<fI.cols; j++)
        {
            int v = int(fI(i,j));
            h( uniform[v] ) += 1.0f;
        }
    }
    histo.push_back(h.reshape(1,1));
}


struct GriddedHist
{
    int GRIDX,GRIDY;

    GriddedHist(int gridx=8, int gridy=8)
        : GRIDX(gridx)
        , GRIDY(gridy)
    {}

    void hist(const Mat &feature, Mat &histo, int histSize=256) const
    {
        histo.release();
        int sw = feature.cols/GRIDX;
        int sh = feature.rows/GRIDY;
        for (int i=0; i<GRIDX; i++)
        {
            for (int j=0; j<GRIDY; j++)
            {
                Mat patch(feature, Range(j*sh,(j+1)*sh), Range(i*sw,(i+1)*sw));
                hist_patch(patch, histo, histSize);
            }
        }
        normalize(histo.reshape(1,1),histo);
    }
};



struct PyramidGrid
{
    void hist_level(const Mat &feature, Mat &histo, int GRIDX, int GRIDY,int histSize=256) const
    {
        int sw = feature.cols/GRIDX;
        int sh = feature.rows/GRIDY;
        for (int i=0; i<GRIDX; i++)
        {
            for (int j=0; j<GRIDY; j++)
            {
                Mat patch(feature, Range(j*sh,(j+1)*sh), Range(i*sw,(i+1)*sw));
                hist_patch(patch, histo, histSize);
            }
        }
    }

    void hist(const Mat &feature, Mat &histo, int histSize=256) const
    {
        histo.release();
        int levels[] = {5,6,7,8};
        for (int i=0; i<4; i++)
        {
            hist_level(feature,histo,levels[i],levels[i],histSize);
        }
        normalize(histo.reshape(1,1),histo);
    }
};



//
// 64 hardcoded, precalculated gftt keypoints from the 90x90 cropped mean lfw2 img
//
static void gftt64(vector<KeyPoint> &kp)
{
    kp.push_back(KeyPoint(14, 33, 3));        kp.push_back(KeyPoint(29, 77, 3));        kp.push_back(KeyPoint(55, 60, 3));
    kp.push_back(KeyPoint(63, 76, 3));        kp.push_back(KeyPoint(76, 32, 3));        kp.push_back(KeyPoint(35, 60, 3));
    kp.push_back(KeyPoint(69, 21, 3));        kp.push_back(KeyPoint(45, 30, 3));        kp.push_back(KeyPoint(27, 31, 3));
    kp.push_back(KeyPoint(64, 26, 3));        kp.push_back(KeyPoint(21, 22, 3));        kp.push_back(KeyPoint(25, 27, 3));
    kp.push_back(KeyPoint(69, 31, 3));        kp.push_back(KeyPoint(54, 81, 3));        kp.push_back(KeyPoint(62, 30, 3));
    kp.push_back(KeyPoint(20, 32, 3));        kp.push_back(KeyPoint(52, 33, 3));        kp.push_back(KeyPoint(37, 32, 3));
    kp.push_back(KeyPoint(38, 81, 3));        kp.push_back(KeyPoint(36, 82, 3));        kp.push_back(KeyPoint(32, 31, 3));
    kp.push_back(KeyPoint(78, 17, 3));        kp.push_back(KeyPoint(59, 24, 3));        kp.push_back(KeyPoint(30, 24, 3));
    kp.push_back(KeyPoint(11, 18, 3));        kp.push_back(KeyPoint(13, 17, 3));        kp.push_back(KeyPoint(56, 30, 3));
    kp.push_back(KeyPoint(73, 15, 3));        kp.push_back(KeyPoint(19, 15, 3));        kp.push_back(KeyPoint(57, 53, 3));
    kp.push_back(KeyPoint(33, 54, 3));        kp.push_back(KeyPoint(34, 52, 3));        kp.push_back(KeyPoint(49, 25, 3));
    kp.push_back(KeyPoint(66, 33, 3));        kp.push_back(KeyPoint(55, 49, 3));        kp.push_back(KeyPoint(61, 33, 3));
    kp.push_back(KeyPoint(39, 29, 3));        kp.push_back(KeyPoint(60, 46, 3));        kp.push_back(KeyPoint(40, 26, 3));
    kp.push_back(KeyPoint(41, 76, 3));        kp.push_back(KeyPoint(50, 76, 3));        kp.push_back(KeyPoint(53, 41, 3));
    kp.push_back(KeyPoint(44, 23, 3));        kp.push_back(KeyPoint(29, 60, 3));        kp.push_back(KeyPoint(54, 54, 3));
    kp.push_back(KeyPoint(30, 47, 3));        kp.push_back(KeyPoint(45, 50, 3));        kp.push_back(KeyPoint(83, 35, 3));
    kp.push_back(KeyPoint(36, 54, 3));        kp.push_back(KeyPoint(13, 46, 3));        kp.push_back(KeyPoint(36, 44, 3));
    kp.push_back(KeyPoint(83, 38, 3));        kp.push_back(KeyPoint(49, 53, 3));        kp.push_back(KeyPoint(33, 83, 3));
    kp.push_back(KeyPoint(17, 88, 3));        kp.push_back(KeyPoint(31, 63, 3));        kp.push_back(KeyPoint(13, 27, 3));
    kp.push_back(KeyPoint(50, 62, 3));        kp.push_back(KeyPoint(11, 43, 3));        kp.push_back(KeyPoint(45, 55, 3));
    kp.push_back(KeyPoint(45, 56, 3));        kp.push_back(KeyPoint(79, 43, 3));        kp.push_back(KeyPoint(74, 88, 3));
    kp.push_back(KeyPoint(41, 62, 3));
}

static void gftt96(vector<KeyPoint> &kp)
{
    kp.push_back(KeyPoint(14, 33, 3));  kp.push_back(KeyPoint(29, 77, 3));  kp.push_back(KeyPoint(55, 60, 3));  kp.push_back(KeyPoint(63, 76, 3));
    kp.push_back(KeyPoint(76, 32, 3));  kp.push_back(KeyPoint(35, 60, 3));  kp.push_back(KeyPoint(69, 21, 3));  kp.push_back(KeyPoint(45, 30, 3));
    kp.push_back(KeyPoint(27, 31, 3));  kp.push_back(KeyPoint(64, 26, 3));  kp.push_back(KeyPoint(21, 22, 3));  kp.push_back(KeyPoint(25, 27, 3));
    kp.push_back(KeyPoint(69, 31, 3));  kp.push_back(KeyPoint(54, 81, 3));  kp.push_back(KeyPoint(62, 30, 3));  kp.push_back(KeyPoint(20, 32, 3));
    kp.push_back(KeyPoint(52, 33, 3));  kp.push_back(KeyPoint(37, 32, 3));  kp.push_back(KeyPoint(38, 81, 3));  kp.push_back(KeyPoint(36, 82, 3));
    kp.push_back(KeyPoint(32, 31, 3));  kp.push_back(KeyPoint(78, 17, 3));  kp.push_back(KeyPoint(59, 24, 3));  kp.push_back(KeyPoint(30, 24, 3));
    kp.push_back(KeyPoint(11, 18, 3));  kp.push_back(KeyPoint(13, 17, 3));  kp.push_back(KeyPoint(56, 30, 3));  kp.push_back(KeyPoint(73, 15, 3));
    kp.push_back(KeyPoint(19, 15, 3));  kp.push_back(KeyPoint(57, 53, 3));  kp.push_back(KeyPoint(33, 54, 3));  kp.push_back(KeyPoint(34, 52, 3));
    kp.push_back(KeyPoint(49, 25, 3));  kp.push_back(KeyPoint(66, 33, 3));  kp.push_back(KeyPoint(55, 49, 3));  kp.push_back(KeyPoint(61, 33, 3));
    kp.push_back(KeyPoint(39, 29, 3));  kp.push_back(KeyPoint(60, 46, 3));  kp.push_back(KeyPoint(40, 26, 3));  kp.push_back(KeyPoint(41, 76, 3));
    kp.push_back(KeyPoint(50, 76, 3));  kp.push_back(KeyPoint(53, 41, 3));  kp.push_back(KeyPoint(44, 23, 3));  kp.push_back(KeyPoint(29, 60, 3));
    kp.push_back(KeyPoint(54, 54, 3));  kp.push_back(KeyPoint(30, 47, 3));  kp.push_back(KeyPoint(45, 50, 3));  kp.push_back(KeyPoint(83, 35, 3));
    kp.push_back(KeyPoint(36, 54, 3));  kp.push_back(KeyPoint(13, 46, 3));  kp.push_back(KeyPoint(36, 44, 3));  kp.push_back(KeyPoint(83, 38, 3));
    kp.push_back(KeyPoint(49, 53, 3));  kp.push_back(KeyPoint(33, 83, 3));  kp.push_back(KeyPoint(17, 88, 3));  kp.push_back(KeyPoint(31, 63, 3));
    kp.push_back(KeyPoint(13, 27, 3));  kp.push_back(KeyPoint(50, 62, 3));  kp.push_back(KeyPoint(11, 43, 3));  kp.push_back(KeyPoint(45, 55, 3));
    kp.push_back(KeyPoint(79, 43, 3));  kp.push_back(KeyPoint(74, 88, 3));  kp.push_back(KeyPoint(41, 62, 3));  kp.push_back(KeyPoint(24, 15, 3));
    kp.push_back(KeyPoint(7,  40, 3));  kp.push_back(KeyPoint(76, 45, 3));  kp.push_back(KeyPoint(8,  42, 3));  kp.push_back(KeyPoint(62, 14, 3));
    kp.push_back(KeyPoint(21, 83, 3));  kp.push_back(KeyPoint(76, 25, 3));  kp.push_back(KeyPoint(46, 67, 3));  kp.push_back(KeyPoint(31, 13, 3));
    kp.push_back(KeyPoint(59, 67, 3));  kp.push_back(KeyPoint(29, 14, 3));  kp.push_back(KeyPoint(62, 63, 3));  kp.push_back(KeyPoint(24, 66, 3));
    kp.push_back(KeyPoint(20, 58, 3));  kp.push_back(KeyPoint(72, 57, 3));  kp.push_back(KeyPoint(67, 64, 3));  kp.push_back(KeyPoint(18, 76, 3));
    kp.push_back(KeyPoint(46, 78, 3));  kp.push_back(KeyPoint(74,  1, 3));  kp.push_back(KeyPoint(74, 74, 3));  kp.push_back(KeyPoint(16, 60, 3));
    kp.push_back(KeyPoint(26, 69, 3));  kp.push_back(KeyPoint(17, 62, 3));  kp.push_back(KeyPoint(57, 88, 3));  kp.push_back(KeyPoint(81, 24, 3));
    kp.push_back(KeyPoint(69, 54, 3));  kp.push_back(KeyPoint(69, 58, 3));  kp.push_back(KeyPoint(58, 73, 3));  kp.push_back(KeyPoint(44, 71, 3));
    kp.push_back(KeyPoint(76, 63, 3));  kp.push_back(KeyPoint(25, 59, 3));  kp.push_back(KeyPoint(25, 59, 3));  kp.push_back(KeyPoint(75, 61, 3));
}


static void gftt32(vector<KeyPoint> &kp)
{
    kp.push_back(KeyPoint(14,33,3,-1,0,0,-1));    kp.push_back(KeyPoint(29,77,3,-1,0,0,-1));
    kp.push_back(KeyPoint(55,60,3,-1,0,0,-1));    kp.push_back(KeyPoint(63,76,3,-1,0,0,-1));
    kp.push_back(KeyPoint(76,32,3,-1,0,0,-1));    kp.push_back(KeyPoint(35,60,3,-1,0,0,-1));
    kp.push_back(KeyPoint(69,21,3,-1,0,0,-1));    kp.push_back(KeyPoint(45,30,3,-1,0,0,-1));
    kp.push_back(KeyPoint(27,31,3,-1,0,0,-1));    kp.push_back(KeyPoint(64,26,3,-1,0,0,-1));
    kp.push_back(KeyPoint(21,22,3,-1,0,0,-1));    kp.push_back(KeyPoint(25,27,3,-1,0,0,-1));
    kp.push_back(KeyPoint(69,31,3,-1,0,0,-1));    kp.push_back(KeyPoint(54,81,3,-1,0,0,-1));
    kp.push_back(KeyPoint(62,30,3,-1,0,0,-1));    kp.push_back(KeyPoint(20,32,3,-1,0,0,-1));
    kp.push_back(KeyPoint(52,33,3,-1,0,0,-1));    kp.push_back(KeyPoint(37,32,3,-1,0,0,-1));
    kp.push_back(KeyPoint(38,81,3,-1,0,0,-1));    kp.push_back(KeyPoint(36,82,3,-1,0,0,-1));
    kp.push_back(KeyPoint(32,31,3,-1,0,0,-1));    kp.push_back(KeyPoint(78,17,3,-1,0,0,-1));
    kp.push_back(KeyPoint(59,24,3,-1,0,0,-1));    kp.push_back(KeyPoint(30,24,3,-1,0,0,-1));
    kp.push_back(KeyPoint(11,18,3,-1,0,0,-1));    kp.push_back(KeyPoint(13,17,3,-1,0,0,-1));
    kp.push_back(KeyPoint(56,30,3,-1,0,0,-1));    kp.push_back(KeyPoint(73,15,3,-1,0,0,-1));
    kp.push_back(KeyPoint(19,15,3,-1,0,0,-1));    kp.push_back(KeyPoint(57,53,3,-1,0,0,-1));
    kp.push_back(KeyPoint(33,54,3,-1,0,0,-1));    kp.push_back(KeyPoint(34,52,3,-1,0,0,-1));
}

static void kp_manual(vector<KeyPoint> &kp)
{
    //kp.push_back(KeyPoint(10,31,3,-1,0,0,-1));    kp.push_back(KeyPoint(13,37,3,-1,0,0,-1));
    //kp.push_back(KeyPoint(82,31,3,-1,0,0,-1));    kp.push_back(KeyPoint(78,37,3,-1,0,0,-1));
    //kp.push_back(KeyPoint(55,27,3,-1,0,0,-1));    kp.push_back(KeyPoint(58,35,3,-1,0,0,-1));
    //kp.push_back(KeyPoint(35,27,3,-1,0,0,-1));    kp.push_back(KeyPoint(32,36,3,-1,0,0,-1));
    //kp.push_back(KeyPoint(7,21,3,-1,0,0,-1));     kp.push_back(KeyPoint(20,19,3,-1,0,0,-1));
    //kp.push_back(KeyPoint(30,19,3,-1,0,0,-1));    kp.push_back(KeyPoint(83,21,3,-1,0,0,-1));
    //kp.push_back(KeyPoint(70,17,3,-1,0,0,-1));    kp.push_back(KeyPoint(59,18,3,-1,0,0,-1));
    //kp.push_back(KeyPoint(38,61,3,-1,0,0,-1));    kp.push_back(KeyPoint(53,61,3,-1,0,0,-1));
    //kp.push_back(KeyPoint(60,53,3,-1,0,0,-1));    kp.push_back(KeyPoint(32,54,3,-1,0,0,-1));
    //kp.push_back(KeyPoint(27,77,3,-1,0,0,-1));    kp.push_back(KeyPoint(63,77,3,-1,0,0,-1));
    //kp.push_back(KeyPoint(38,45,3,-1,0,0,-1));    kp.push_back(KeyPoint(54,45,3,-1,0,0,-1));

    kp.push_back(KeyPoint(15,19,3,-1,0,0,-1));    kp.push_back(KeyPoint(75,19,3,-1,0,0,-1));
    kp.push_back(KeyPoint(29,20,3,-1,0,0,-1));    kp.push_back(KeyPoint(61,20,3,-1,0,0,-1));
    kp.push_back(KeyPoint(36,24,3,-1,0,0,-1));    kp.push_back(KeyPoint(54,24,3,-1,0,0,-1));
    //kp.push_back(KeyPoint(35,27,3,-1,0,0,-1));    kp.push_back(KeyPoint(32,36,3,-1,0,0,-1));
    kp.push_back(KeyPoint(38,35,3,-1,0,0,-1));    kp.push_back(KeyPoint(52,35,3,-1,0,0,-1));
    kp.push_back(KeyPoint(30,39,3,-1,0,0,-1));    kp.push_back(KeyPoint(60,39,3,-1,0,0,-1));
    kp.push_back(KeyPoint(19,39,3,-1,0,0,-1));    kp.push_back(KeyPoint(71,39,3,-1,0,0,-1));
    kp.push_back(KeyPoint(8 ,38,3,-1,0,0,-1));    kp.push_back(KeyPoint(82,38,3,-1,0,0,-1));
    kp.push_back(KeyPoint(40,64,3,-1,0,0,-1));    kp.push_back(KeyPoint(50,64,3,-1,0,0,-1));
    kp.push_back(KeyPoint(31,75,3,-1,0,0,-1));    kp.push_back(KeyPoint(59,75,3,-1,0,0,-1));
    kp.push_back(KeyPoint(27,81,3,-1,0,0,-1));    kp.push_back(KeyPoint(63,81,3,-1,0,0,-1));

    //kp.push_back(KeyPoint(5,25,3));     kp.push_back(KeyPoint(83,23,3));
    //kp.push_back(KeyPoint(20,19,3));    kp.push_back(KeyPoint(68,17,3));
    //kp.push_back(KeyPoint(37,23,3));    kp.push_back(KeyPoint(52,22,3));
    //kp.push_back(KeyPoint(15,34,3));    kp.push_back(KeyPoint(74,33,3));
    //kp.push_back(KeyPoint(32,35,3));    kp.push_back(KeyPoint(57,34,3));
    //kp.push_back(KeyPoint(27,31,3));    kp.push_back(KeyPoint(63,30,3));
    //kp.push_back(KeyPoint(36,62,3));    kp.push_back(KeyPoint(54,62,3));
    //kp.push_back(KeyPoint(46,74,3));    kp.push_back(KeyPoint(46,64,3));
    //kp.push_back(KeyPoint(28,77,3));    kp.push_back(KeyPoint(64,77,3));
    //kp.push_back(KeyPoint(46,80,3));    kp.push_back(KeyPoint(45,32,3));
}


#ifdef HAVE_DLIB
//
// 20 assorted keypoints extracted from the 68 dlib facial landmarks, based on the
//    Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf
//
struct LandMarkDlib
{
    dlib::shape_predictor sp;

    LandMarkDlib()
    {   // lol, it's only 95mb.
        dlib::deserialize("D:/Temp/dlib-18.10/examples/shape_predictor_68_face_landmarks.dat") >> sp;
    }

    int extract(const Mat &img, vector<KeyPoint> &kp) const
    {
        dlib::rectangle rec(0,0,img.cols,img.rows);
        dlib::full_object_detection shape = sp(dlib::cv_image<uchar>(img), rec);

        int idx[] = {17,26, 19,24, 21,22, 36,45, 39,42, 38,43, 31,35, 51,33, 48,54, 57,27, 0};
        //int idx[] = {18,25, 20,24, 21,22, 27,29, 31,35, 38,43, 51, 0};
        for(int k=0; (k<40) && (idx[k]>0); k++)
            kp.push_back(KeyPoint(shape.part(idx[k]).x(), shape.part(idx[k]).y(), 8));
        //dlib::point p1 = shape.part(31) + (shape.part(39) - shape.part(31)) * 0.5; // left of nose
        //dlib::point p2 = shape.part(35) + (shape.part(42) - shape.part(35)) * 0.5;
        //dlib::point p3 = shape.part(36) + (shape.part(39) - shape.part(36)) * 0.5; // left eye center
        //dlib::point p4 = shape.part(42) + (shape.part(45) - shape.part(42)) * 0.5; // right eye center
        //dlib::point p5 = shape.part(31) + (shape.part(48) - shape.part(31)) * 0.5; // betw.mouth&nose
        //dlib::point p6 = shape.part(35) + (shape.part(54) - shape.part(35)) * 0.5; // betw.mouth&nose
        //kp.push_back(KeyPoint(p1.x(), p1.y(), 8));
        //kp.push_back(KeyPoint(p2.x(), p2.y(), 8));
        //kp.push_back(KeyPoint(p3.x(), p3.y(), 8));
        //kp.push_back(KeyPoint(p4.x(), p4.y(), 8));
        //kp.push_back(KeyPoint(p5.x(), p5.y(), 8));
        //kp.push_back(KeyPoint(p6.x(), p6.y(), 8));

        return (int)kp.size();
    }
};
#endif


struct GfttGrid
{
    int gr;
    GfttGrid(int gr=4) : gr(gr) {} // 8x8 rect by default

    void hist(const Mat &feature, Mat &histo, int histSize=256) const
    {
        vector<KeyPoint> kp;
        gftt64(kp);
        //gftt96(kp);
        //kp_manual(kp);

        histo.release();
        Rect bounds(0,0,90,90);
        for (size_t k=0; k<kp.size(); k++)
        {
            Rect part(int(kp[k].pt.x)-gr, int(kp[k].pt.y)-gr, gr*2, gr*2);
            part &= bounds;
            hist_patch(feature(part), histo, histSize);
        }
        normalize(histo.reshape(1,1),histo);
    }
};



//
//
// layered base for lbph,
//  * calc features on the whole image,
//  * calculate the hist on a set of rectangles
//    (which could come from a grid, or a Rects, or a keypoint based model).
//
template <typename Feature, typename Grid>
struct GenericExtractor : public TextureFeature::Extractor
{
    Feature ext;
    Grid grid;

    GenericExtractor(const Feature &ext, const Grid &grid)
        : ext(ext)
        , grid(grid)
    {}

    // TextureFeature::Extractor
    virtual int extract(const Mat &img, Mat &features) const
    {
        Mat fI;
        int histSize = ext(img, fI);
        grid.hist(fI, features, histSize);
        return features.total() * features.elemSize();
    }
};


//
// instead of adding more bits, concatenate several histograms,
// cslbp + dialbp + sqlbp = 3*16 bins = 12288 feature-bytes.
//
template <typename Grid>
struct CombinedExtractor : public TextureFeature::Extractor
{
    FeatureCsLbp      cslbp;
    FeatureDiamondLbp dialbp;
    FeatureSquareLbp  sqlbp;
    Grid grid;

    CombinedExtractor(const Grid &grid)
        : grid(grid)
    {}

    // TextureFeature::Extractor
    virtual int extract(const Mat &img, Mat &features) const
    {
        Mat fI, f;
        int histSize = cslbp(img, f);
        grid.hist(f, fI, histSize);
        features.push_back(fI.reshape(1,1));

        histSize = dialbp(img, f);
        grid.hist(f, fI, histSize);
        features.push_back(fI.reshape(1,1));

        histSize = sqlbp(img, f);
        grid.hist(f, fI, histSize);
        features.push_back(fI.reshape(1,1));
        features = features.reshape(1,1);

        return features.total() * features.elemSize();
    }
};



template <typename Grid>
struct GradMagExtractor : public TextureFeature::Extractor
{
    Grid grid;
    int nbins;

    GradMagExtractor(const Grid &grid)
        : grid(grid)
        , nbins(45)
    {}

    // TextureFeature::Extractor
    virtual int extract(const Mat &I, Mat &features) const
    {
        Mat fgrad, fmag;
        Mat s1, s2, s3(I.size(), CV_32F), s4(I.size(), CV_32F);
        Sobel(I, s1, CV_32F, 1, 0);
        Sobel(I, s2, CV_32F, 0, 1);

        fastAtan2(s1.ptr<float>(0), s2.ptr<float>(0), s3.ptr<float>(0), I.total(), true);
        fgrad = s3 / (360/nbins);
        fgrad.convertTo(fgrad,CV_8U);
        Mat fg;
        grid.hist(fgrad,fg,nbins+1);
        features.push_back(fg.reshape(1,1));

        magnitude(s1.ptr<float>(0), s2.ptr<float>(0), s4.ptr<float>(0), I.total());
        normalize(s4,fmag);
        fmag.convertTo(fmag,CV_8U,nbins+1);
        Mat fm;
        grid.hist(fmag,fm,nbins+1);
        features.push_back(fm.reshape(1,1));

        features = features.reshape(1,1);
        return features.total() * features.elemSize();
    }
};



//
// concat histograms from lbp-like features generated from a bank of gabor filtered images
//   ( due to memory restrictions, it can't use plain lbp(64k),
//     but has to inherit the 2nd best bed (combined(12k)) )
//
template <typename Grid>
struct ExtractorGabor : public CombinedExtractor<Grid>
{
    Size kernel_size;

    ExtractorGabor(const Grid &grid, int kernel_siz=8)
        : CombinedExtractor<Grid>(grid)
        , kernel_size(kernel_siz, kernel_siz)
    {}

    void gabor(const Mat &src_f, Mat &features,double sigma, double theta, double lambda, double gamma, double psi) const
    {
        Mat dest,dest8u,his;
        cv::filter2D(src_f, dest, CV_32F, getGaborKernel(kernel_size, sigma,theta, lambda, gamma, psi));
        dest.convertTo(dest8u, CV_8U);
        CombinedExtractor<Grid>::extract(dest8u, his);
        features.push_back(his.reshape(1, 1));
    }

    virtual int extract(const Mat &img, Mat &features) const
    {
        Mat src_f;
        img.convertTo(src_f, CV_32F, 1.0/255.0);
        gabor(src_f, features, 8,4,90,15,0);
        gabor(src_f, features, 8,4,45,30,1);
        gabor(src_f, features, 8,4,45,45,0);
        gabor(src_f, features, 8,4,90,60,1);
        features = features.reshape(1,1);
        return features.total() * features.elemSize();
    }
};



//
// grid it into 8x8 image patches, do a dct on each,
//  concat downsampled 4x4(topleft) result to feature vector.
//
struct ExtractorDct : public TextureFeature::Extractor
{
    int grid;

    ExtractorDct() : grid(8) {}

    virtual int extract( const Mat &img, Mat &features ) const
    {
        Mat src;
        img.convertTo(src,CV_32F,1.0/255.0);
        for(int i=0; i<src.rows-grid; i+=grid)
        {
            for(int j=0; j<src.cols-grid; j+=grid)
            {
                Mat d;
                dct(src(Rect(i,j,grid,grid)),d);
                // downsampling is just a ROI operation here, still we need a clone()
                Mat e = d(Rect(0,0,grid/2,grid/2)).clone();
                features.push_back(e.reshape(1,1));
            }
        }
        features = features.reshape(1,1);
        return features.total() * features.elemSize();
    }
};



template < class Descriptor >
struct ExtractorGridFeature2d : public TextureFeature::Extractor
{
    int grid;

    ExtractorGridFeature2d(int g=10) : grid(g) {}

    virtual int extract(const Mat &img, Mat &features) const
    {
        float gw = float(img.cols) / grid;
        float gh = float(img.rows) / grid;
        vector<KeyPoint> kp;
        for (float i=gh/2; i<img.rows-gh; i+=gh)
        {
            for (float j=gw/2; j<img.cols-gw; j+=gw)
            {
                KeyPoint k(j, i, gh);
                kp.push_back(k);
            }
        }
        Ptr<Feature2D> f2d = Descriptor::create();
        f2d->compute(img, kp, features);

        features = features.reshape(1,1);
        return features.total() * features.elemSize();
    }
};
typedef ExtractorGridFeature2d<ORB> ExtractorORBGrid;
typedef ExtractorGridFeature2d<BRISK> ExtractorBRISKGrid;
typedef ExtractorGridFeature2d<xfeatures2d::FREAK> ExtractorFREAKGrid;
typedef ExtractorGridFeature2d<xfeatures2d::SIFT> ExtractorSIFTGrid;
typedef ExtractorGridFeature2d<xfeatures2d::BriefDescriptorExtractor> ExtractorBRIEFGrid;


//template < class Descriptor >
struct ExtractorGfttFeature2d : public TextureFeature::Extractor
{
    Ptr<Feature2D> f2d;
#ifdef HAVE_DLIB
    LandMarkDlib land;
#else
    Ptr<ElasticParts> elastic;
#endif

    ExtractorGfttFeature2d(Ptr<Feature2D> f)
        : f2d(f)
    {
#ifndef HAVE_DLIB
        elastic = ElasticParts::create();
        elastic->read("parts.xml.gz");
#endif
    }

    virtual int extract(const Mat &img, Mat &features) const
    {
       // PROFILEX("extract");

        vector<KeyPoint> kp;
#ifdef HAVE_DLIB
        land.extract(img,kp);
#else
        {// PROFILEX("elastic")
        elastic->getPoints(img, kp);
        }
        //kp_manual(kp);
#endif
        size_t s = kp.size();
        float w=5;
        for (size_t i=0; i<s; i++)
        {
            Point2f p(kp[i].pt);
            kp.push_back(KeyPoint(p.x,p.y-w,w*2));
            kp.push_back(KeyPoint(p.x,p.y+w,w*2));
            kp.push_back(KeyPoint(p.x-w,p.y,w*2));
            kp.push_back(KeyPoint(p.x+w,p.y,w*2));
            //kp.push_back(KeyPoint(p.x-w,p.y-w/2,w*2));
            //kp.push_back(KeyPoint(p.x-w,p.y+w/2,w*2));
            //kp.push_back(KeyPoint(p.x+w,p.y-w/2,w*2));
            //kp.push_back(KeyPoint(p.x+w,p.y+w/2,w*2));
        }
        f2d->compute(img, kp, features);
        // resize(features,features,Size(),0.5,1.0);                  // not good.
        // features = features(Rect(64,0,64,features.rows)).clone();  // amazing.
        //features = features.reshape(1,1);
        normalize(features.reshape(1,1), features);
        return features.total() * features.elemSize();
    }
};

//
// "Review and Implementation of High-Dimensional Local Binary Patterns 
//    and Its Application to Face Recognition"
//    Bor-Chun Chen, Chu-Song Chen, Winston Hsu 
//


struct HighDimLbp : public TextureFeature::Extractor
{
    FeatureFPLbp lbp;

#ifdef HAVE_DLIB
    LandMarkDlib land;
#else
    Ptr<ElasticParts> elastic;
#endif

    HighDimLbp()
    {
#ifndef HAVE_DLIB
        //elastic = ElasticParts::create();
        //elastic->read("parts.xml.gz");
#endif
    }

    virtual int extract(const Mat &img, Mat &features) const
    {
        //PROFILEX("extract");
        int gr=10; // 10 used in paper
        vector<KeyPoint> kp;
#ifdef HAVE_DLIB
        land.extract(img,kp);
#else
        { 
            //PROFILEX("elastic");
        //    elastic->getPoints(img, kp);
        }
        kp_manual(kp);
#endif


        Mat histo;
        //float scale[] = {0.6f, 0.9f, 1.2f, 1.5f, 1.8f, 2.3f};
        float scale[] = {0.75f, 1.06f, 1.5f, 2.2f, 3.0f}; // http://bcsiriuschen.github.io/High-Dimensional-LBP/
        //float offsets_4[] = { 
        //    -0.5f,-0.5f, 0.5f,-0.5f,
        //    -0.5f, 0.5f, 0.5f, 0.5f,
        //};
        //float offsets_9[] = {
        //    -1.0f,-1.0f,   -1.0f,-0.0f,  -1.0f, 0.0f,
        //    -0.0f,-1.0f,   -0.0f,-0.0f,  -0.0f, 0.0f,
        //     1.0f,-1.0f,    1.0f,-0.0f,   1.0f, 0.0f
        //};
        float offsets_16[] = {
            -1.5f,-1.5f, -0.5f,-1.5f, 0.5f,-1.5f, 1.5f,-1.5f,
            -1.5f,-0.5f, -0.5f,-0.5f, 0.5f,-0.5f, 1.5f,-0.5f,
            -1.5f, 0.5f, -0.5f, 0.5f, 0.5f, 0.5f, 1.5f, 0.5f,
            -1.5f, 1.5f, -0.5f, 1.5f, 0.5f, 1.5f, 1.5f, 1.5f
        };
        //float *off_table[] = {
        //    offsets_4,
        //    offsets_9,
        //    offsets_16,
        //    offsets_16,
        //    offsets_16
        //};
        //int off_size[] = {
        //    4,9,16,16,16
        //};
        //int grs[] = {
        //    4,8,8,10,10
        //};
        for (int i=0; i<5; i++)
        {
            float s = scale[i];
            int noff = 16;//off_size[i];
            float *off = offsets_16;//off_table[i];
            //gr = grs[i];
            Mat f1,f2,imgs;
            resize(img,imgs,Size(),s,s);
            int histSize = lbp(imgs,f1);

            for (size_t k=0; k<kp.size(); k++)
            {
                Point2f pt(kp[k].pt);
                for (int o=0; o<noff; o++)
                {   
                    Mat patch;
                    getRectSubPix(f1, Size(gr,gr), Point2f(pt.x*s + off[o*2]*gr, pt.y*s + off[o*2+1]*gr), patch);
                    hist_patch(patch, histo, histSize);
                }
            }
        }
        normalize(histo.reshape(1,1), features);
        //features = histo.reshape(1,1);
        return features.total() * features.elemSize();
    }
};


} // namespace TextureFeatureImpl



//
// 'factory' functions (aka public api)
//
using namespace TextureFeatureImpl;


cv::Ptr<TextureFeature::Extractor> createExtractorPixels(int resw, int resh)
{   return makePtr< ExtractorPixels >(resw, resh); }


cv::Ptr<TextureFeature::Extractor> createExtractorLbp(int gx, int gy)
{   return makePtr< GenericExtractor<FeatureLbp,GriddedHist> >(FeatureLbp(), GriddedHist(gx, gy)); }
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidLbp()
{   return makePtr< GenericExtractor<FeatureLbp,PyramidGrid> >(FeatureLbp(), PyramidGrid()); }


cv::Ptr<TextureFeature::Extractor> createExtractorFPLbp(int gx, int gy)
{   return makePtr< GenericExtractor<FeatureFPLbp,GriddedHist> >(FeatureFPLbp(), GriddedHist(gx, gy)); }
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidFpLbp()
{   return makePtr< GenericExtractor<FeatureFPLbp,PyramidGrid> >(FeatureFPLbp(), PyramidGrid()); }


cv::Ptr<TextureFeature::Extractor> createExtractorTPLbp(int gx, int gy)
{   return makePtr< GenericExtractor<FeatureTPLbp,GriddedHist> >(FeatureTPLbp(), GriddedHist(gx, gy)); }
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidTpLbp()
{   return makePtr< GenericExtractor<FeatureTPLbp,PyramidGrid> >(FeatureTPLbp(), PyramidGrid()); }
cv::Ptr<TextureFeature::Extractor> createExtractorGfttTpLbp()
{   return makePtr< GenericExtractor<FeatureTPLbp,GfttGrid> >(FeatureTPLbp(), GfttGrid()); }
cv::Ptr<TextureFeature::Extractor> createExtractorGfttTpLbp2()
{   return makePtr< GenericExtractor<FeatureTPLbp2,GfttGrid> >(FeatureTPLbp2(), GfttGrid()); }


cv::Ptr<TextureFeature::Extractor> createExtractorMTS(int gx, int gy)
{   return makePtr< GenericExtractor<FeatureMTS,GriddedHist> >(FeatureMTS(), GriddedHist(gx, gy)); }
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidMTS()
{   return makePtr< GenericExtractor<FeatureMTS,PyramidGrid> >(FeatureMTS(), PyramidGrid()); }


cv::Ptr<TextureFeature::Extractor> createExtractorBGC1(int gx, int gy)
{   return makePtr< GenericExtractor<FeatureBGC1,GriddedHist> >(FeatureBGC1(), GriddedHist(gx, gy)); }
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidBGC1()
{   return makePtr< GenericExtractor<FeatureBGC1,PyramidGrid> >(FeatureBGC1(), PyramidGrid()); }


cv::Ptr<TextureFeature::Extractor> createExtractorCombined(int gx, int gy)
{   return makePtr< CombinedExtractor<GriddedHist> >(GriddedHist(gx, gy)); }
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidCombined()
{   return makePtr< CombinedExtractor<PyramidGrid> >(PyramidGrid()); }
cv::Ptr<TextureFeature::Extractor> createExtractorGfttCombined()
{   return makePtr< CombinedExtractor<GfttGrid> >(GfttGrid()); }



cv::Ptr<TextureFeature::Extractor> createExtractorGaborLbp(int gx, int gy, int kernel_siz)
{   return makePtr< ExtractorGabor<GriddedHist> >(GriddedHist(gx, gy), kernel_siz); }


cv::Ptr<TextureFeature::Extractor> createExtractorDct()
{   return makePtr< ExtractorDct >(); }


cv::Ptr<TextureFeature::Extractor> createExtractorORBGrid(int g)
{   return makePtr< ExtractorORBGrid >(g); }
cv::Ptr<TextureFeature::Extractor> createExtractorSIFTGrid(int g)
{   return makePtr< ExtractorSIFTGrid >(g); }
cv::Ptr<TextureFeature::Extractor> createExtractorSIFTGftt()
{   return makePtr< ExtractorGfttFeature2d >(xfeatures2d::SIFT::create()); }


cv::Ptr<TextureFeature::Extractor> createExtractorGrad()
{   return makePtr< GenericExtractor<FeatureGrad,GriddedHist> >(FeatureGrad(),GriddedHist()); }
cv::Ptr<TextureFeature::Extractor> createExtractorGfttGrad()
{   return makePtr< GenericExtractor<FeatureGrad,GfttGrid> >(FeatureGrad(),GfttGrid()); }
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidGrad()
{   return makePtr< GenericExtractor<FeatureGrad,PyramidGrid> >(FeatureGrad(),PyramidGrid()); }

cv::Ptr<TextureFeature::Extractor> createExtractorGfttGradMag()
{   return makePtr< GradMagExtractor<GfttGrid> >(GfttGrid()); }

cv::Ptr<TextureFeature::Extractor> createExtractorHighDimLbp()
{   return makePtr< HighDimLbp >(); }
