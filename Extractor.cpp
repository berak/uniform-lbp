#include <vector>
using std::vector;
#include <iostream>
using std::cerr;
using std::endl;

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
using namespace cv;

#include "TextureFeature.h"


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
    int nbins;
    FeatureGrad(int nbins=45) : nbins(nbins) {}

    int operator() (const Mat &I, Mat &fI) const
    {
        Mat s1, s2, s3(I.size(), CV_32F);
        Sobel(I, s1, CV_32F, 1, 0);
        Sobel(I, s2, CV_32F, 0, 1);
        fastAtan2(s1.ptr<float>(0), s2.ptr<float>(0), s3.ptr<float>(0), I.total(), true);
        fI = s3 / (360/nbins);
        fI.convertTo(fI,CV_8U);
        return nbins;
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





static void hist_patch(const Mat_<uchar> &fI, Mat &histo, int histSize=256, bool uni=false)
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

    Mat_<float> h(1, (uni ? 59 : histSize), 0.0f);
    for (int i=0; i<fI.rows; i++)
    {
        for (int j=0; j<fI.cols; j++)
        {
            int v = int(fI(i,j));
            if (uni)
                h( uniform[v] ) += 1.0f;
            else
                h( v ) += 1.0f;
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


struct OverlapGridHist : public GriddedHist
{
    int over,over2;

    OverlapGridHist(int gridx=8, int gridy=8, int over=0)
        : GriddedHist(gridx, gridy)
        , over(over)
        , over2(over*2)
    { }

    void hist(const Mat &feature, Mat &histo, int histSize=256) const
    {
        histo.release();
        int sw = (feature.cols - over2)/GRIDX;
        int sh = (feature.rows - over2)/GRIDY;
        for (int r=over; r<feature.rows-sh; r+=sh)
        {
            for (int c=over; c<feature.cols-sw; c+=sw)
            {
                Rect patch(c-over, r-over, sw+over2, sh+over2);
                hist_patch(feature(patch), histo, histSize);
            }
        }
        normalize(histo.reshape(1,1),histo);
    }
};



//
// hardcoded to funneled, 90x90 images.
//
//   the (offline) train thing uses a majority vote over rects,
//   the current impl concatenated histograms (the majority scheme seems to play nicer)
//
struct ElasticParts
{
    void hist(const Mat &feature, Mat &histo, int histSize=256) const
    {
        const int nparts = 64;
        Rect parts[nparts] =
        {
            Rect(15,23,48, 5), // 0.701167 // 1.504 // gen5
            Rect(24, 0,22,11),            Rect(24,23,55, 4),            Rect(56,21,34, 7),            Rect(24, 9,25,10),
            Rect(25,23,52, 4),            Rect( 0,52,60, 4),            Rect(40,27,35, 7),            Rect(36,59,31, 8),
            Rect( 5,24,38, 6),            Rect( 5, 0,21,11),            Rect( 4, 2,24,10),            Rect( 1,51,36, 6),
            Rect(25,29,18,13),            Rect(10, 1,26, 9),            Rect(50,27,25,10),            Rect(42,17,17,14),
            Rect( 6,26,30, 8),            Rect(34, 6,13,19),            Rect(65, 1,24,10),            Rect(20,24,37, 6),
            Rect(22,22,41, 6),            Rect(60,22,30, 7),            Rect(53,21,37, 6),            Rect(32,19,13,19),
            Rect(45,17,29, 8),            Rect(30,23,55, 4),            Rect(52,17,30, 8),            Rect(21,27,44, 5),
            Rect(39,27,38, 6),            Rect(53,12,28, 8),            Rect(22,29,21,11),            Rect(16, 6,35, 7),
            Rect(31,20,11,22),            Rect(14,24,55, 4),            Rect(37,15,13,19),            Rect(30,61,38, 6),
            Rect(76,11,14,17),            Rect(38,13,25,10),            Rect(26,30,17,14),            Rect(25,30,20,12),
            Rect( 1, 6,17,14),            Rect( 5, 8,22,11),            Rect(56,11,24,10),            Rect(69,14,20,12),
            Rect(41,20,16,15),            Rect(22,22,43, 5),            Rect(64,58,16,15),            Rect(70,42,13,19),
            Rect(39,14,15,16),            Rect(25,60,30, 8),            Rect(10,64,23,10),            Rect(26, 1,17,14),
            Rect(46,77,20,12),            Rect(56, 8,15,16),            Rect(66,55,19,13),            Rect( 8,64,28, 8),
            Rect(70,53,20,12),            Rect(62, 7,12,20),            Rect( 2,24,56, 4),            Rect(25,48,25,10),
            Rect(44,27,34, 7),            Rect(58,21,31, 8),            Rect(49,80,16,10)
        };

        histo.release();
        for (size_t k=0; k<nparts; k++)
        {
            hist_patch(feature(parts[k]), histo, histSize);
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
    kp.push_back(KeyPoint(14,33,3,-1,0,0,-1));
    kp.push_back(KeyPoint(29,77,3,-1,0,0,-1));
    kp.push_back(KeyPoint(55,60,3,-1,0,0,-1));
    kp.push_back(KeyPoint(63,76,3,-1,0,0,-1));
    kp.push_back(KeyPoint(76,32,3,-1,0,0,-1));
    kp.push_back(KeyPoint(35,60,3,-1,0,0,-1));
    kp.push_back(KeyPoint(69,21,3,-1,0,0,-1));
    kp.push_back(KeyPoint(45,30,3,-1,0,0,-1));
    kp.push_back(KeyPoint(27,31,3,-1,0,0,-1));
    kp.push_back(KeyPoint(64,26,3,-1,0,0,-1));
    kp.push_back(KeyPoint(21,22,3,-1,0,0,-1));
    kp.push_back(KeyPoint(25,27,3,-1,0,0,-1));
    kp.push_back(KeyPoint(69,31,3,-1,0,0,-1));
    kp.push_back(KeyPoint(54,81,3,-1,0,0,-1));
    kp.push_back(KeyPoint(62,30,3,-1,0,0,-1));
    kp.push_back(KeyPoint(20,32,3,-1,0,0,-1));
    kp.push_back(KeyPoint(52,33,3,-1,0,0,-1));
    kp.push_back(KeyPoint(37,32,3,-1,0,0,-1));
    kp.push_back(KeyPoint(38,81,3,-1,0,0,-1));
    kp.push_back(KeyPoint(36,82,3,-1,0,0,-1));
    kp.push_back(KeyPoint(32,31,3,-1,0,0,-1));
    kp.push_back(KeyPoint(78,17,3,-1,0,0,-1));
    kp.push_back(KeyPoint(59,24,3,-1,0,0,-1));
    kp.push_back(KeyPoint(30,24,3,-1,0,0,-1));
    kp.push_back(KeyPoint(11,18,3,-1,0,0,-1));
    kp.push_back(KeyPoint(13,17,3,-1,0,0,-1));
    kp.push_back(KeyPoint(56,30,3,-1,0,0,-1));
    kp.push_back(KeyPoint(73,15,3,-1,0,0,-1));
    kp.push_back(KeyPoint(19,15,3,-1,0,0,-1));
    kp.push_back(KeyPoint(57,53,3,-1,0,0,-1));
    kp.push_back(KeyPoint(33,54,3,-1,0,0,-1));
    kp.push_back(KeyPoint(34,52,3,-1,0,0,-1));
}

static void kp_manual(vector<KeyPoint> &kp)
{
    kp.push_back(KeyPoint(10,31,3,-1,0,0,-1));
    kp.push_back(KeyPoint(13,37,3,-1,0,0,-1));
    kp.push_back(KeyPoint(82,31,3,-1,0,0,-1));
    kp.push_back(KeyPoint(78,37,3,-1,0,0,-1));
    kp.push_back(KeyPoint(55,27,3,-1,0,0,-1));
    kp.push_back(KeyPoint(58,35,3,-1,0,0,-1));
    kp.push_back(KeyPoint(35,27,3,-1,0,0,-1));
    kp.push_back(KeyPoint(32,36,3,-1,0,0,-1));
    kp.push_back(KeyPoint(7,21,3,-1,0,0,-1));
    kp.push_back(KeyPoint(20,19,3,-1,0,0,-1));
    kp.push_back(KeyPoint(30,19,3,-1,0,0,-1));
    kp.push_back(KeyPoint(83,21,3,-1,0,0,-1));
    kp.push_back(KeyPoint(70,17,3,-1,0,0,-1));
    kp.push_back(KeyPoint(59,18,3,-1,0,0,-1));
    kp.push_back(KeyPoint(38,61,3,-1,0,0,-1));
    kp.push_back(KeyPoint(53,61,3,-1,0,0,-1));
    kp.push_back(KeyPoint(60,53,3,-1,0,0,-1));
    kp.push_back(KeyPoint(32,54,3,-1,0,0,-1));
    kp.push_back(KeyPoint(27,77,3,-1,0,0,-1));
    kp.push_back(KeyPoint(63,77,3,-1,0,0,-1));
    kp.push_back(KeyPoint(38,45,3,-1,0,0,-1));
    kp.push_back(KeyPoint(54,45,3,-1,0,0,-1));
}


struct GfttGrid
{
    int gr;
    GfttGrid(int gr=4) : gr(gr) {} // 8x8 rect by default

    void hist(const Mat &feature, Mat &histo, int histSize=256) const
    {
        vector<KeyPoint> kp;
        gftt64(kp);
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
    //FeatureGrad       grad;
    Grid grid;

    CombinedExtractor(const Grid &grid)
        : grid(grid)
        //, grad(16)
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
        //histSize = grad(img, f);
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

    ExtractorGfttFeature2d(Ptr<Feature2D> f)
        : f2d(f)
    {}

    virtual int extract(const Mat &img, Mat &features) const
    {
        vector<KeyPoint> kp;
        gftt96(kp);
        f2d->compute(img, kp, features);
        // resize(features,features,Size(),0.5,1.0);                  // not good.
        // features = features(Rect(64,0,64,features.rows)).clone();  // amazing.
        features = features.reshape(1,1);
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
    //FeatureGrad lbp;
    //FeatureMTS lbp;
    FeatureFPLbp lbp;
    //FeatureCsLbp lbp;
    //FeatureLbp lbp;

    //HighDimLbp() : lbp(16) {}
    virtual int extract(const Mat &img, Mat &features) const
    {
        bool uni=false;
        //bool uni=true;
        int gr=8; // 10 used in paper
        vector<KeyPoint> kp;
        //gftt32(kp);
        kp_manual(kp);

        Mat histo;
        //float scale[] = {0.6f, 0.9f, 1.2f, 1.5f, 1.8f, 2.3f};
        float scale[] = {0.75f, 1.06f, 1.5f, 2.2f, 3.0f}; // http://bcsiriuschen.github.io/High-Dimensional-LBP/
        int offsets[16][2] = {       -1,-1,  -1, 0,
                                      0,-1,   0, 0, 
                             -2,-2,  -2,-1,  -2, 0,  -2, 1,
                             -1,-2,                  -1, 1,
                              0,-2,                   0, 1,
                              1,-2,   1,-1,   1, 0,   1, 1,
        };
        //double offsets[9][2] = { -1.5,-1.5,   -1.5,-0.5,  -1.5, 0.5,
        //                         -0.5,-1.5,   -0.5,-0.5,  -0.5, 0.5,
        //                          1.5,-1.5,    1.5,-0.5,   1.5, 0.5 };
        for (int i=0; i<5; i++)
        {
            float s = scale[i];

            Mat f1,imgs;
            resize(img,imgs,Size(),s,s);
            int histSize = lbp(imgs,f1);

            Rect bounds(0,0,int(90*s),int(90*s));
            for (size_t k=0; k<kp.size(); k++)
            {
                for (int o=0; o<16; o++)
                {
                    Rect part(int(kp[k].pt.x*s)+offsets[o][0]*gr, int(kp[k].pt.y)+offsets[o][1]*gr, gr, gr);
                    part &= bounds;
                    hist_patch(f1(part), histo, histSize, uni);
                }
            }
        }

        features = histo.reshape(1,1);
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
cv::Ptr<TextureFeature::Extractor> createExtractorElasticLbp()
{   return makePtr< GenericExtractor<FeatureLbp,ElasticParts> >(FeatureLbp(), ElasticParts()); }
cv::Ptr<TextureFeature::Extractor> createExtractorOverlapLbp(int gx, int gy, int over)
{   return makePtr< GenericExtractor<FeatureLbp,OverlapGridHist> >(FeatureLbp(), OverlapGridHist(gx, gy, over)); }
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidLbp()
{   return makePtr< GenericExtractor<FeatureLbp,PyramidGrid> >(FeatureLbp(), PyramidGrid()); }


cv::Ptr<TextureFeature::Extractor> createExtractorFPLbp(int gx, int gy)
{   return makePtr< GenericExtractor<FeatureFPLbp,GriddedHist> >(FeatureFPLbp(), GriddedHist(gx, gy)); }
cv::Ptr<TextureFeature::Extractor> createExtractorElasticFpLbp()
{   return makePtr< GenericExtractor<FeatureFPLbp,ElasticParts> >(FeatureFPLbp(), ElasticParts()); }
cv::Ptr<TextureFeature::Extractor> createExtractorOverlapFpLbp(int gx, int gy, int over)
{   return makePtr< GenericExtractor<FeatureFPLbp,OverlapGridHist> >(FeatureFPLbp(), OverlapGridHist(gx, gy, over)); }
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidFpLbp()
{   return makePtr< GenericExtractor<FeatureFPLbp,PyramidGrid> >(FeatureFPLbp(), PyramidGrid()); }


cv::Ptr<TextureFeature::Extractor> createExtractorTPLbp(int gx, int gy)
{   return makePtr< GenericExtractor<FeatureTPLbp,GriddedHist> >(FeatureTPLbp(), GriddedHist(gx, gy)); }
cv::Ptr<TextureFeature::Extractor> createExtractorElasticTpLbp()
{   return makePtr< GenericExtractor<FeatureTPLbp,ElasticParts> >(FeatureTPLbp(), ElasticParts()); }
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidTpLbp()
{   return makePtr< GenericExtractor<FeatureTPLbp,PyramidGrid> >(FeatureTPLbp(), PyramidGrid()); }
cv::Ptr<TextureFeature::Extractor> createExtractorGfttTpLbp()
{   return makePtr< GenericExtractor<FeatureTPLbp,GfttGrid> >(FeatureTPLbp(), GfttGrid()); }
cv::Ptr<TextureFeature::Extractor> createExtractorOverlapTpLbp(int gx, int gy, int over)
{   return makePtr< GenericExtractor<FeatureTPLbp,OverlapGridHist> >(FeatureTPLbp(), OverlapGridHist(gx, gy, over)); }


cv::Ptr<TextureFeature::Extractor> createExtractorMTS(int gx, int gy)
{   return makePtr< GenericExtractor<FeatureMTS,GriddedHist> >(FeatureMTS(), GriddedHist(gx, gy)); }
cv::Ptr<TextureFeature::Extractor> createExtractorElasticMTS()
{   return makePtr< GenericExtractor<FeatureMTS,ElasticParts> >(FeatureMTS(), ElasticParts()); }
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidMTS()
{   return makePtr< GenericExtractor<FeatureMTS,PyramidGrid> >(FeatureMTS(), PyramidGrid()); }
cv::Ptr<TextureFeature::Extractor> createExtractorOverlapMTS(int gx, int gy, int over)
{   return makePtr< GenericExtractor<FeatureMTS,OverlapGridHist> >(FeatureMTS(), OverlapGridHist(gx, gy, over)); }


cv::Ptr<TextureFeature::Extractor> createExtractorBGC1(int gx, int gy)
{   return makePtr< GenericExtractor<FeatureBGC1,GriddedHist> >(FeatureBGC1(), GriddedHist(gx, gy)); }
cv::Ptr<TextureFeature::Extractor> createExtractorElasticBGC1()
{   return makePtr< GenericExtractor<FeatureBGC1,ElasticParts> >(FeatureBGC1(), ElasticParts()); }
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidBGC1()
{   return makePtr< GenericExtractor<FeatureBGC1,PyramidGrid> >(FeatureBGC1(), PyramidGrid()); }
cv::Ptr<TextureFeature::Extractor> createExtractorOverlapBGC1(int gx, int gy, int over)
{   return makePtr< GenericExtractor<FeatureBGC1,OverlapGridHist> >(FeatureBGC1(), OverlapGridHist(gx, gy, over)); }


cv::Ptr<TextureFeature::Extractor> createExtractorCombined(int gx, int gy)
{   return makePtr< CombinedExtractor<GriddedHist> >(GriddedHist(gx, gy)); }
cv::Ptr<TextureFeature::Extractor> createExtractorElasticCombined()
{   return makePtr< CombinedExtractor<ElasticParts> >(ElasticParts()); }
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidCombined()
{   return makePtr< CombinedExtractor<PyramidGrid> >(PyramidGrid()); }
cv::Ptr<TextureFeature::Extractor> createExtractorOverlapCombined(int gx, int gy, int over)
{   return makePtr< CombinedExtractor<OverlapGridHist> >(OverlapGridHist(gx, gy, over)); }
cv::Ptr<TextureFeature::Extractor> createExtractorGfttCombined()
{   return makePtr< CombinedExtractor<GfttGrid> >(GfttGrid()); }



cv::Ptr<TextureFeature::Extractor> createExtractorGaborLbp(int gx, int gy, int kernel_siz)
{   return makePtr< ExtractorGabor<GriddedHist> >(GriddedHist(gx, gy), kernel_siz); }
cv::Ptr<TextureFeature::Extractor> createExtractorElasticGaborLbp(int kernel_siz)
{   return makePtr< ExtractorGabor<ElasticParts> >(ElasticParts(), kernel_siz); }


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
cv::Ptr<TextureFeature::Extractor> createExtractorElasticGrad()
{   return makePtr< GenericExtractor<FeatureGrad,ElasticParts> >(FeatureGrad(),ElasticParts()); }
cv::Ptr<TextureFeature::Extractor> createExtractorGfttGrad()
{   return makePtr< GenericExtractor<FeatureGrad,GfttGrid> >(FeatureGrad(),GfttGrid()); }
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidGrad()
{   return makePtr< GenericExtractor<FeatureGrad,PyramidGrid> >(FeatureGrad(),PyramidGrid()); }

cv::Ptr<TextureFeature::Extractor> createExtractorGfttGradMag()
{   return makePtr< GradMagExtractor<GfttGrid> >(GfttGrid()); }

cv::Ptr<TextureFeature::Extractor> createExtractorHighDimLbp()
{   return makePtr< HighDimLbp >(); }
