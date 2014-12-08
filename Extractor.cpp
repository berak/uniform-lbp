#include <vector>
using std::vector;

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
using namespace cv;

#include "TextureFeature.h"




//
// this is the most simple one.
//
class ExtractorPixels : public TextureFeature::Extractor
{
    int resw,resh;
public:
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
// Antonio Fernandez, Marcos X. Alvarez, Francesco Bianconi:
// "Texture description through histograms of equivalent patterns"
//    unfortunately atm. , this delivers images 3x3 pixels smaller than the input
//    thus breaking the Parts based method.
//
#define SHIFTED_MATS_3x3(I) \
        int M = I.rows; \
        int N = I.cols; \
        Mat I7 = I(Range(1,M-2), Range(1,N-2));\
        Mat I6 = I(Range(1,M-2), Range(2,N-1));\
        Mat I5 = I(Range(1,M-2), Range(3,N  ));\
        Mat I4 = I(Range(2,M-1), Range(3,N  ));\
        Mat I3 = I(Range(3,M  ), Range(3,N  ));\
        Mat I2 = I(Range(3,M  ), Range(2,N-1));\
        Mat I1 = I(Range(3,M  ), Range(1,N-2));\
        Mat I0 = I(Range(2,M-1), Range(1,N-2));\
        Mat IC = I(Range(2,M-1), Range(2,N-1));



struct FeatureLbp
{
    int operator() (const Mat &I, Mat &fI) const
    {
        //SHIFTED_MATS_3x3(I);
        //fI = ((I7>IC)&128) |
        //     ((I6>IC)&64)  |
        //     ((I5>IC)&32)  |
        //     ((I4>IC)&16)  |
        //     ((I3>IC)&8)   |
        //     ((I2>IC)&4)   |
        //     ((I1>IC)&2)   |
        //     ((I0>IC)&1);

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




struct FeatureBGC1
{
    int operator () (const Mat &I, Mat &fI) const
    {
        //SHIFTED_MATS_3x3(I);
        //fI = ((I7>=I0)&128) |
        //     ((I6>=I7)& 64) |
        //     ((I5>=I6)& 32) |
        //     ((I4>=I5)& 16) |
        //     ((I3>=I4)&  8) |
        //     ((I2>=I3)&  4) |
        //     ((I1>=I2)&  2) |
        //     ((I0>=I1)&  1);

        Mat_<uchar> feature(I.size(),0);
        Mat_<uchar> img(I);
        const int m=1;
        for (int r=m; r<img.rows-m; r++)
        {
            for (int c=m; c<img.cols-m; c++)
            {
                uchar v = 0;
                uchar cen = img(r,c);
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


struct FeatureMTS
{
    int operator () (const Mat &I, Mat &fI) const
    {
        //SHIFTED_MATS_3x3(img);
        //fI = ((IC>=I7)&8) | ((IC>=I6)&4) | ((IC>=I5)&2) | ((IC>=I4)&1);

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
// Antonio Fernandez, Marcos X. Alvarez, Francesco Bianconi:
// "Texture description through histograms of equivalent patterns"
//
struct FeatureSTU
{
    int kerP1;
    FeatureSTU(int kp1=8)
        : kerP1(kp1)
    {}

    inline Mat eta1(Mat a, int p)const
    {
        Mat c;
        multiply(a,a,c);
        return c > (p*p);
    }

    int operator () (const Mat &img, Mat &fI) const
    {
        SHIFTED_MATS_3x3(img);

        fI = eta1(abs(I6-IC),kerP1) & 8
           | eta1(abs(I4-IC),kerP1) & 4
           | eta1(abs(I2-IC),kerP1) & 2
           | eta1(abs(I0-IC),kerP1) & 1;

        return 16;
    }
};


//
// Wolf, Hassner, Taigman : "Descriptor Based Methods in the Wild"
// 3.1 Three-Patch LBP Codes
//
struct FeatureTPLbp
{
public:
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
// 3.2 Four-Patch LBP Codes
//
struct FeatureFPLbp
{
    int operator () (const Mat &img, Mat &features) const
    {
        //Patches, v1:
        //SHIFTED_MATS_3x3(img);
        //Mat_<uchar> I = I7/9 + I6/9 + I5/9 + I4/9 + I3/9 + I2/9 + I1/9 + I0/9 + IC/9;

        //Patches, v2:
        //Mat_<uchar> I; resize(img,I,Size(img.cols-3,img.rows-3));

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










static void calc_hist(const Mat_<uchar> &feature, Mat_<float> &histo)
{
    for (int i=0; i<feature.rows; i++)
    {
        for (int j=0; j<feature.cols; j++)
        {
            uchar bin = int(feature(i,j));
            histo( bin ) += 1.0f;
        }
    }
}


struct GriddedHist
{
    int GRIDX,GRIDY;


    GriddedHist(int gridx=8, int gridy=8)
        : GRIDX(gridx)
        , GRIDY(gridy)
    {}

    void hist_patch(const Mat &fi, Mat &histo, int histSize=256) const
    {
        Mat_<float> h(1,histSize,0.0f);
        calc_hist(fi,h);
        histo.push_back(h.reshape(1,1));
    }

    void hist(const Mat &feature, Mat &histo, int histSize=256) const
    {
        histo.release();
        int sw = (feature.cols)/(GRIDX);
        int sh = (feature.rows)/(GRIDY);
        for (int i=0; i<GRIDX-1; i++)
        {
            for (int j=0; j<GRIDY-1; j++)
            {
                Rect patch(i*sw,j*sh,sw,sh);
                hist_patch(feature(patch), histo,histSize);
            }
        }
        normalize(histo.reshape(1,1),histo);
    }
};

struct OverlapGridHist : public GriddedHist
{
    int over;


    OverlapGridHist(int gridx=8, int gridy=8, int over=0)
        : GriddedHist(gridx, gridy)
        , over(over)
    { }

    void hist(const Mat &feature, Mat &histo, int histSize=256) const
    {
        histo.release();
        int sw = (feature.cols)/GRIDX;
        int sh = (feature.rows)/GRIDY;
        for (int r=over; r<feature.rows-sh-2*over; r+=sh)
        {
            for (int c=over; c<feature.cols-sh-2*over; c+=sw)
            {
                Rect patch(c-over,r-over,sw+2*over,sh+2*over);
                hist_patch(feature(patch), histo,histSize);
            }
        }
        normalize(histo.reshape(1,1),histo);
    }
};



//
// hardcoded to funneled, 90x90 images.
//
//   the train thing uses a majority vote over rects,
//   the current impl concatenetd histograms (the majority scheme seems to play nicer)
//
struct ElasticParts
{
    void hist(const Mat &feature, Mat &histo, int histSize=256) const
    {
        const int nparts = 64;
        static struct Part {
            Rect r;
            //double eq,ne,k;
            Part() {}
            //Part(int x,int y,int w,int h,double e=0,double n=0)
            Part(int x,int y,int w,int h)
                : r(x,y,w,h)
                //, eq(e)
                //, ne(n)
                //, k(0)
            {}
        } parts[nparts] = {
            Part(15,23,48, 5), // 0.701167 // 1.504 // gen5
            Part(24, 0,22,11),            Part(24,23,55, 4),            Part(56,21,34, 7),            Part(24, 9,25,10),
            Part(25,23,52, 4),            Part( 0,52,60, 4),            Part(40,27,35, 7),            Part(36,59,31, 8),
            Part( 5,24,38, 6),            Part( 5, 0,21,11),            Part( 4, 2,24,10),            Part( 1,51,36, 6),
            Part(25,29,18,13),            Part(10, 1,26, 9),            Part(50,27,25,10),            Part(42,17,17,14),
            Part( 6,26,30, 8),            Part(34, 6,13,19),            Part(65, 1,24,10),            Part(20,24,37, 6),
            Part(22,22,41, 6),            Part(60,22,30, 7),            Part(53,21,37, 6),            Part(32,19,13,19),
            Part(45,17,29, 8),            Part(30,23,55, 4),            Part(52,17,30, 8),            Part(21,27,44, 5),
            Part(39,27,38, 6),            Part(53,12,28, 8),            Part(22,29,21,11),            Part(16, 6,35, 7),
            Part(31,20,11,22),            Part(14,24,55, 4),            Part(37,15,13,19),            Part(30,61,38, 6),
            Part(76,11,14,17),            Part(38,13,25,10),            Part(26,30,17,14),            Part(25,30,20,12),
            Part( 1, 6,17,14),            Part( 5, 8,22,11),            Part(56,11,24,10),            Part(69,14,20,12),
            Part(41,20,16,15),            Part(22,22,43, 5),            Part(64,58,16,15),            Part(70,42,13,19),
            Part(39,14,15,16),            Part(25,60,30, 8),            Part(10,64,23,10),            Part(26, 1,17,14),
            Part(46,77,20,12),            Part(56, 8,15,16),            Part(66,55,19,13),            Part( 8,64,28, 8),
            Part(70,53,20,12),            Part(62, 7,12,20),            Part( 2,24,56, 4),            Part(25,48,25,10),
            Part(44,27,34, 7),            Part(58,21,31, 8),            Part(49,80,16,10)
        };

        histo.release();
        for (size_t k=0; k<nparts; k++)
        {
            Mat roi(feature, parts[k].r);
            Mat_<float> h(1, histSize, 0.0f);
            calc_hist(roi, h);
            histo.push_back(h.reshape(1,1));
        }
        normalize(histo.reshape(1,1),histo);
    }
};



//
//
// layered base for lbph,
//  * calc features on the whole image,
//  * calculate the hist on a set of rectangles
//    (which could come from a grid, or a Parts based model).
//
template <typename Feature, typename Grid>
struct UniformExtractor : public TextureFeature::Extractor
{
    Feature ext;
    Grid grid;
    enum UniformTable
    {
        UniformNormal,    // 58 + noise
        UniformModified,  // 58
        UniformReduced,   // 16 + noise
        UniformNone = -1  // 256, as-is
    };
    int utable;

    UniformExtractor(const Feature &ext, const Grid &grid, int u_table=-1)
        : grid(grid)
        , ext(ext)
        , utable(u_table)
    {}

    // TextureFeature::Extractor
    virtual int extract(const Mat &img, Mat &features) const
    {
        Mat fI;
        int histSize = ext(img, fI);

        if (utable != UniformNone)
        {
            static int uniform[3][256] = {
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
                58,58,58,50,51,52,58,53,54,55,56,57 },
            {   // 'noise' ones mapped to their closest hamming uniform neighbour
                0,1,2,3,4,1,5,6,7,1,2,3,8,8,9,10,11,1,2,3,4,1,5,6,12,12,12,15,13,13,14,15,16,1,2,
                3,4,1,5,6,7,1,2,3,8,8,9,10,17,17,17,3,17,17,20,21,18,18,18,21,19,19,20,21,22,1,
                2,3,4,1,5,6,7,1,2,3,8,8,9,10,11,1,2,3,4,58,5,6,12,12,12,15,13,13,14,15,23,23,23,
                44,23,23,5,45,23,23,23,28,26,26,27,28,24,24,24,49,24,24,27,28,25,25,25,28,26,26,
                27,28,29,30,2,31,4,30,5,32,7,30,2,31,8,33,9,33,11,30,2,31,4,30,5,32,12,12,12,34,
                13,34,14,34,16,30,2,31,4,30,5,32,7,30,58,31,8,33,9,33,17,48,17,49,17,35,20,35,18,
                52,18,35,19,35,20,35,36,37,36,38,36,37,39,39,36,37,36,38,8,40,40,40,36,37,36,38,
                36,37,39,39,51,52,41,41,54,41,41,41,42,43,42,44,42,43,45,45,42,43,42,44,54,46,46,
                46,47,48,47,49,47,48,50,50,51,52,51,53,54,55,56,57 },
            {   // 'reduced' set (16 + 1 bins)
                16,0,16,16,16,16,1,16,2,16,16,16,16,16,16,16,3,16,16,16,16,16,16,16,16,16,16,16,
                16,16,16,4,5,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
                16,16,16,16,16,16,6,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
                16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
                16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,7,16,8,16,16,16,16,16,16,16,16,16,16,
                16,16,16,16,16,16,9,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
                16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
                16,16,16,16,16,10,16,16,16,11,16,16,16,16,16,16,16,12,16,16,16,16,16,16,16,16,16,
                16,16,16,16,16,16,16,16,16,16,13,16,16,16,16,16,16,16,16,16,16,16,16,16,14,16,15,
                16,16,16,16,16,16,16,16,16,16,16,16 }
            };

            Mat lu(1, 256, CV_8U, uniform[utable]);
            LUT(fI, lu, fI);
            int fsiz[3] = {59, 58, 17};
            histSize = fsiz[utable];
        }

        grid.hist(fI, features, histSize);
        return features.total() * features.elemSize();
    }
};



//
// concat histograms from lbp(u) features generated from a bank of gabor filtered images
//
template <typename Feature, typename Grid>
class ExtractorGabor : public UniformExtractor<Feature,Grid>
{
    Size kernel_size;
public:
    ExtractorGabor(const Feature &ext, const Grid &grid, int u_table=UniformNone, int kernel_siz=8)
        : UniformExtractor(ext, grid, u_table)
        , kernel_size(kernel_siz, kernel_siz)
    {}
    void gabor(const Mat &src_f, Mat &features,double sigma, double theta, double lambda, double gamma, double psi) const
    {
        Mat dest,dest8u,his;
        cv::filter2D(src_f, dest, CV_32F, getGaborKernel(kernel_size, sigma,theta, lambda, gamma, psi));
        dest.convertTo(dest8u, CV_8U);
        UniformExtractor::extract(dest8u, his);
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
class ExtractorDct : public TextureFeature::Extractor
{
    int grid;
public:
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
class ExtractorGridFeature : public TextureFeature::Extractor
{
    int grid;
public:
    ExtractorGridFeature(int g=10) : grid(g) {}
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

typedef ExtractorGridFeature<ORB> ExtractorORBGrid;
typedef ExtractorGridFeature<BRISK> ExtractorBRISKGrid;
typedef ExtractorGridFeature<xfeatures2d::FREAK> ExtractorFREAKGrid;
typedef ExtractorGridFeature<xfeatures2d::SIFT> ExtractorSIFTGrid;
typedef ExtractorGridFeature<xfeatures2d::BriefDescriptorExtractor> ExtractorBRIEFGrid;

//
// 'factory' functions (aka public api)
//

cv::Ptr<TextureFeature::Extractor> createExtractorPixels(int resw, int resh)
{
    return makePtr<ExtractorPixels>(resw, resh);
}

cv::Ptr<TextureFeature::Extractor> createExtractorLbp(int gx, int gy, int utable)
{
    return makePtr< UniformExtractor<FeatureLbp,GriddedHist> >(FeatureLbp(), GriddedHist(gx, gy), utable);
}
cv::Ptr<TextureFeature::Extractor> createExtractorElasticLbp()
{
    return makePtr< UniformExtractor<FeatureLbp,ElasticParts> >(FeatureLbp(), ElasticParts());
}
cv::Ptr<TextureFeature::Extractor> createExtractorOverlapLbp(int gx, int gy, int over)
{
    return makePtr< UniformExtractor<FeatureLbp,OverlapGridHist> >(FeatureLbp(), OverlapGridHist(gx, gy, over));
}

cv::Ptr<TextureFeature::Extractor> createExtractorFPLbp(int gx, int gy)
{
    return makePtr< UniformExtractor<FeatureFPLbp,GriddedHist> >(FeatureFPLbp(), GriddedHist(gx, gy));
}
cv::Ptr<TextureFeature::Extractor> createExtractorElasticFpLbp()
{
    return makePtr< UniformExtractor<FeatureFPLbp,ElasticParts> >(FeatureFPLbp(), ElasticParts());
}
cv::Ptr<TextureFeature::Extractor> createExtractorOverlapFpLbp(int gx, int gy, int over)
{
    return makePtr< UniformExtractor<FeatureFPLbp,OverlapGridHist> >(FeatureFPLbp(), OverlapGridHist(gx, gy, over));
}

cv::Ptr<TextureFeature::Extractor> createExtractorTPLbp(int gx, int gy)
{
    return makePtr< UniformExtractor<FeatureTPLbp,GriddedHist> >(FeatureTPLbp(), GriddedHist(gx, gy));
}
cv::Ptr<TextureFeature::Extractor> createExtractorElasticTpLbp()
{
    return makePtr< UniformExtractor<FeatureTPLbp,ElasticParts> >(FeatureTPLbp(), ElasticParts());
}
cv::Ptr<TextureFeature::Extractor> createExtractorOverlapTpLbp(int gx, int gy, int over)
{
    return makePtr< UniformExtractor<FeatureTPLbp,OverlapGridHist> >(FeatureTPLbp(), OverlapGridHist(gx, gy, over));
}

cv::Ptr<TextureFeature::Extractor> createExtractorMTS(int gx, int gy)
{
    return makePtr< UniformExtractor<FeatureMTS,GriddedHist> >(FeatureMTS(), GriddedHist(gx, gy));
}
cv::Ptr<TextureFeature::Extractor> createExtractorElasticMTS()
{
    return makePtr< UniformExtractor<FeatureMTS,ElasticParts> >(FeatureMTS(), ElasticParts());
}
cv::Ptr<TextureFeature::Extractor> createExtractorOverlapMTS(int gx, int gy, int over)
{
    return makePtr< UniformExtractor<FeatureMTS,OverlapGridHist> >(FeatureMTS(), OverlapGridHist(gx, gy, over));
}

cv::Ptr<TextureFeature::Extractor> createExtractorBGC1(int gx, int gy, int utable)
{
    return makePtr< UniformExtractor<FeatureBGC1,GriddedHist> >(FeatureBGC1(), GriddedHist(gx, gy));
}
cv::Ptr<TextureFeature::Extractor> createExtractorElasticBGC1()
{
    return makePtr< UniformExtractor<FeatureBGC1,ElasticParts> >(FeatureBGC1(), ElasticParts());
}
cv::Ptr<TextureFeature::Extractor> createExtractorOverlapBGC1(int gx, int gy, int over)
{
    return makePtr< UniformExtractor<FeatureBGC1,OverlapGridHist> >(FeatureBGC1(), OverlapGridHist(gx, gy, over));
}


cv::Ptr<TextureFeature::Extractor> createExtractorSTU(int gx, int gy,int kp1)
{
    return makePtr< UniformExtractor<FeatureSTU,GriddedHist> >(FeatureSTU(kp1), GriddedHist(gx, gy));
}

cv::Ptr<TextureFeature::Extractor> createExtractorGaborLbp(int gx, int gy, int u_table, int kernel_siz)
{
    return makePtr< ExtractorGabor<FeatureLbp,GriddedHist> >(FeatureLbp(), GriddedHist(gx, gy), u_table, kernel_siz);
}
cv::Ptr<TextureFeature::Extractor> createExtractorElasticGaborLbp(int u_table, int kernel_siz)
{
    return makePtr< ExtractorGabor<FeatureLbp,ElasticParts> >(FeatureLbp(), ElasticParts(), u_table, kernel_siz);
}

cv::Ptr<TextureFeature::Extractor> createExtractorDct()
{
    return makePtr<ExtractorDct>();
}

cv::Ptr<TextureFeature::Extractor> createExtractorORBGrid(int g)
{
    return makePtr<ExtractorORBGrid>(g);
}

cv::Ptr<TextureFeature::Extractor> createExtractorSIFTGrid(int g)
{
    return makePtr<ExtractorSIFTGrid>(g);
}

