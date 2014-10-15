#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
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
            resize(img, features, Size(resw,resh) );
        else
            features=img;
        features = features.reshape(1,1);
        return features.total() * features.elemSize();
    }
};





//
// gridded hu-moments
//
class ExtractorMoments : public TextureFeature::Extractor
{
    static void mom(const Mat &z, Mat &feature, int i, int j, int w, int h)
    {
        double hu[7];
        Mat roi(z, cv::Rect(i*w,j*h,w,h));
        HuMoments(moments(roi, false), hu);
        feature.push_back(hu[0]);
        feature.push_back(hu[1]);
        feature.push_back(hu[2]);
        feature.push_back(hu[3]);
        feature.push_back(hu[4]);
        feature.push_back(hu[5]);
        feature.push_back(hu[6]);
    }
    static Mat mom(const Mat & z)
    {
        Mat mo;
        int sw = (z.cols)/8;
        int sh = (z.rows)/8;
        for (int i=0; i<8; i++)
        {
            for (int j=0; j<8; j++)
            {
                mom(z,mo,i,j,sw,sh);
            }
        }        
        mo.convertTo(mo,CV_32F);
        normalize(mo,mo);
        return mo;
    }
public:
    // TextureFeature::Extractor
    virtual int extract(const Mat &img, Mat &features) const
    {
        features = mom(img);
        features = features.reshape(1,1);
        return features.total() * features.elemSize() ;
    }
};






// 
// base for lbph, calc features on the whole image, the hist on a grid, 
//   so we avoid to waste border pixels 
//     (with probably the price of pixels shared between patches)
//
struct GriddedHist : public TextureFeature::Extractor
{
protected:
    int GRIDX,GRIDY;
    
    void calc_hist(const Mat_<uchar> &feature, Mat_<float> &histo, int histSize, int histRange=256) const
    {   
        for ( int i=0; i<feature.rows; i++ )
        {
            for ( int j=0; j<feature.cols; j++ )
            {
                uchar bin = int(feature(i,j)) * histSize / histRange;
                histo( bin ) += 1.0f;
            }
        }
    }

    void hist(const Mat &feature, Mat &histo, int histSize=256, int histRange=256) const
    {   
        histo.release();
        //const float range[] = { 0, 256 } ;
        //const float* hist_range[] = { range };
        int sw = (feature.cols)/(GRIDX+1);
        int sh = (feature.rows)/(GRIDY+1);
        for ( int i=0; i<GRIDX; i++ )
        {
            for ( int j=0; j<GRIDY; j++ )
            {  
                Rect patch(i*sw,j*sh,sw,sh);
                Mat fi( feature, patch );
                Mat_<float> h(1,histSize,0.0f);
                //calcHist( &fi, 1, 0, Mat(), h, 1, &histSize, &hist_range, true, false );
                calc_hist(fi,h,histSize,histRange);
                histo.push_back(h.reshape(1,1));
            }
        }
        normalize(histo.reshape(1,1),histo);
    }

public:

    GriddedHist(int gridx=8, int gridy=8) 
        : GRIDX(gridx)
        , GRIDY(gridy) 
    {}
};





//#define range(x,M,k) Range((x),((M)-((k)-(x))))

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
        //int k = 3; \
        //Mat I7 = I(range(0,M,k), range(0,N,k));\
        //Mat I6 = I(range(0,M,k), range(1,N,k));\
        //Mat I5 = I(range(0,M,k), range(2,N,k));\
        //Mat I4 = I(range(1,M,k), range(2,N,k));\
        //Mat I3 = I(range(2,M,k), range(2,N,k));\
        //Mat I2 = I(range(2,M,k), range(1,N,k));\
        //Mat I1 = I(range(2,M,k), range(0,N,k));\
        //Mat I0 = I(range(1,M,k), range(0,N,k));\
        //Mat IC = I(range(1,M,k), range(1,N,k));
//
//#define SHIFTED_MATS_5x5(I) \
//        Mat Ia = I(Range(0,M-3), Range(0,N-3));\
//        Mat Ib = I(Range(0,M-3), Range(1,N-2));\
//        Mat Ic = I(Range(0,M-3), Range(2,N-1));\
//        Mat Id = I(Range(0,M-3), Range(3,N  ));\
//        Mat Ie = I(Range(0,M-3), Range(4,N  ));\
//        Mat If = I(Range(2,M-1), Range(3,N  ));\
//        Mat Ig = I(Range(3,M  ), Range(3,N  ));\
//        Mat Ih = I(Range(3,M  ), Range(2,N-1));\
//        Mat Ii = I(Range(3,M  ), Range(2,N-1));
//        Mat Ih = I(Range(3,M-1), Range(2,N-1));



class ExtractorLbp : public GriddedHist
{
protected:

    int utable;

    //
    // "histogram of equivalence patterns" 
    //
    virtual void hep( const Mat &I, Mat &fI ) const
    {
#if 1
        SHIFTED_MATS_3x3(I);

        fI = ((I7>IC)&128) |
             ((I6>IC)&64)  |
             ((I5>IC)&32)  |
             ((I4>IC)&16)  |
             ((I3>IC)&8)   |
             ((I2>IC)&4)   |
             ((I1>IC)&2)   |
             ((I0>IC)&1);  
#else
        Mat_<uchar> feature(I.size());
        Mat_<uchar> img(I);
        const int m=1;
        for ( int r=m; r<img.rows-m; r++ )
        {
            for ( int c=m; c<img.cols-m; c++ )
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
#endif
    }


public:

    enum UniformTable
    {
        UniformNormal,    // 58 + noise
        UniformModified,  // 58
        UniformReduced,   // 16 + noise
        UniformNone = -1
    };

    ExtractorLbp(int gridx=8, int gridy=8, int u_table=UniformNone) 
        : GriddedHist(gridx,gridy) 
        , utable(u_table)
    {}

    // TextureFeature::Extractor
    virtual int extract(const Mat &img, Mat &features) const
    {
        Mat fI;
        hep(img,fI);

        if (utable == UniformNone)
        {
            hist(fI,features,256,256);
            return features.total() * features.elemSize();
        }

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

        Mat h59;
        Mat lu(1,256,CV_8U, uniform[utable]);
        LUT(fI,lu,h59);

        int histlen[] = {59,58,17};
        hist(h59,features,histlen[utable],histlen[utable]);
        features = features.reshape(1,1);
        return features.total() * features.elemSize();
    }
};



class ExtractorBGC1 : public ExtractorLbp
{
protected:
    virtual void hep( const Mat &I, Mat &fI ) const
    {
        SHIFTED_MATS_3x3(I);

        fI = ((I7>=I0)&128) |
             ((I6>=I7)& 64) |
             ((I5>=I6)& 32) |
             ((I4>=I5)& 16) |
             ((I3>=I4)&  8) |
             ((I2>=I3)&  4) |
             ((I1>=I2)&  2) |
             ((I0>=I1)&  1);
    }

public:
    ExtractorBGC1(int gridx=8, int gridy=8, int u_table=UniformNone) 
        : ExtractorLbp(gridx, gridy, u_table) 
    {}
};


class ExtractorLQP : public GriddedHist
{

public:
    ExtractorLQP(int gridx=8, int gridy=8) 
        : GriddedHist(gridx, gridy) 
    {}
    virtual int extract(const Mat &img, Mat &features) const
    {
        int kerP1=5;
        int kerP2=5;
        Mat fI_2,fI_1,fI1,fI2;

        SHIFTED_MATS_3x3(img);

        Mat Icplus1  = IC+kerP1;
        Mat Icplus2  = IC+kerP2;
        Mat Icminus1 = IC-kerP1;
        Mat Icminus2 = IC-kerP2;
        fI_2 =  ((I7<Icminus2)&128 ) |
                ((I6<Icminus2)& 64 ) |
                ((I5<Icminus2)& 32 ) |
                ((I4<Icminus2)& 16 ) |
                ((I3<Icminus2)&  8 ) |
                ((I2<Icminus2)&  4 ) |
                ((I1<Icminus2)&  2 ) |
                ((I0<Icminus2)&  1 );
        fI_1 =  (((I7>=Icminus2) &(I7<Icminus1))&128 ) |
                (((I6>=Icminus2) &(I6<Icminus1))& 64 ) |
                (((I5>=Icminus2) &(I5<Icminus1))& 32 ) |
                (((I4>=Icminus2) &(I4<Icminus1))& 16 ) |
                (((I3>=Icminus2) &(I3<Icminus1))&  8 ) |
                (((I2>=Icminus2) &(I2<Icminus1))&  4 ) |
                (((I1>=Icminus2) &(I1<Icminus1))&  2 ) |
                (((I0>=Icminus2) &(I0<Icminus1))&  1 );
        fI1 =   (((I7>=Icplus1) &(I7<Icplus2))&128 ) |
                (((I6>=Icplus1) &(I6<Icplus2))& 64 ) |
                (((I5>=Icplus1) &(I5<Icplus2))& 32 ) |
                (((I4>=Icplus1) &(I4<Icplus2))& 16 ) |
                (((I3>=Icplus1) &(I3<Icplus2))&  8 ) |
                (((I2>=Icplus1) &(I2<Icplus2))&  4 ) |
                (((I1>=Icplus1) &(I1<Icplus2))&  2 ) |
                (((I0>=Icplus1) &(I0<Icplus2))&  1 );
        fI2 =   ((I7>=Icplus2)&128 ) |
                ((I6>=Icplus2)& 64 ) |
                ((I5>=Icplus2)& 32 ) |
                ((I4>=Icplus2)& 16 ) |
                ((I3>=Icplus2)&  8 ) |
                ((I2>=Icplus2)&  4 ) |
                ((I1>=Icplus2)&  2 ) |
                ((I0>=Icplus2)&  1 );

        Mat h1,h2,h3,h4,h;
        hist(fI_2,h1,256);
        hist(fI_1,h2,256);
        hist(fI1, h3,256);
        hist(fI2, h4,256);
        h.push_back(h1);
        h.push_back(h2);
        h.push_back(h3);
        h.push_back(h4);
        features = h.reshape(1,1);
        return features.total() * features.elemSize();
    }
};


//
//  A Robust Descriptor based on Weber’s Law (i terribly crippled it)
//
class WLD : public GriddedHist
{
    // my histograms looks like this:
    // [32 bins for zeta][2*64 bins for theta][16 bins for center intensity]
    // since the patches are pretty small(12x12), i can even get away using uchar for the historam bins
    // all those are heuristic/empirical, i.e, i found it works better with only the 1st 2 orientations
    
    // configurable, yet hardcoded values
    enum {
        size_center  = 4,   // num bits from the center
        size_theta_n = 2,   // orientation channels used
        size_theta_w = 8,   // each theta orientation channel is 8*w
        size_zeta    = 32,  // bins for zeta

        size_theta = 8*size_theta_w, 
        size_all = (1<<size_center) + size_zeta + size_theta_n * size_theta
        
        // 176 bytes per patch, * 8 * 8 = 11264 bytes per image.
    };

    int typeflag;

    template <class T>
    void oper(const Mat &src, Mat &hist) const 
    {
        const double CV_PI_4 = CV_PI / 4.0;
        int radius = 1;
        for(int i=radius; i<src.rows-radius; i++) 
        {
            for(int j=radius; j<src.cols-radius; j++) 
            {
                // 7 0 1
                // 6 c 2
                // 5 4 3
                uchar c   = src.at<uchar>(i,j);
                uchar n[8]= 
                {
                    src.at<uchar>(i-1,j),
                    src.at<uchar>(i-1,j+1),
                    src.at<uchar>(i,j+1),
                    src.at<uchar>(i+1,j+1),
                    src.at<uchar>(i+1,j),
                    src.at<uchar>(i+1,j-1),
                    src.at<uchar>(i,j-1),
                    src.at<uchar>(i-1,j-1) 
                };
                int p = n[0]+n[1]+n[2]+n[3]+n[4]+n[5]+n[6]+n[7];
                p -= c*8;
                
                // (7), projected from [-pi/2,pi/2] to [0,size_zeta]
                double zeta = 0;
                if (p!=0) zeta = double(size_zeta) * (atan(double(p)/c) + CV_PI*0.5) / CV_PI;
                hist.at<T>(int(zeta)) += 1;

                // (11), projected from [-pi/2,pi/2] to [0,size_theta]
                for ( int i=0; i<size_theta_n; i++ ) 
                {
                    double a = atan2(double(n[i]-n[(i+4)%8]),double(n[(i+2)%8]-n[(i+6)%8]));
                    double theta = CV_PI_4 * fmod( (a+CV_PI)/CV_PI_4+0.5f, 8 ) * size_theta_w; // (11)
                    hist.at<T>(int(theta)+size_zeta+size_theta * i) += 1;
                }

                // additionally, add some bits of the actual center value (MSB).
                int cen = c>>(8-size_center); 
                hist.at<T>(cen+size_zeta+size_theta * size_theta_n) += 1;
            }
        }
    }

public:

    WLD(int gridx=8, int gridy=8,int typeflag=CV_32F) 
        : GriddedHist(gridx,gridy)
        , typeflag(typeflag)
    {}

    // TextureFeature::Extractor
    virtual int extract(const Mat &img, Mat &features) const
    {
        features = Mat::zeros(WLD::size_all*GRIDX*GRIDY,1,typeflag);
        switch(typeflag)
        {
            case CV_32F:  oper<float>(img,features);  break;
            case CV_8U:   oper<uchar>(img,features);  break;
        }
        features = features.reshape(1,1);
        return features.total() * features.elemSize();
    }
};


class ExtractorMTS : public GriddedHist
{
public:
    ExtractorMTS(int gridx=8, int gridy=8) 
        : GriddedHist(gridx, gridy) 
    {}
    virtual int extract(const Mat &img, Mat &features) const
    {
        SHIFTED_MATS_3x3(img);

        Mat h,fI;

        fI = ((IC>=I7)&8) | ((IC>=I6)&4) | ((IC>=I5)&2) | ((IC>=I4)&1);
        hist(fI,h,16,16);
        features = h.reshape(1,1);
        return features.total() * features.elemSize();
    }
};


// helper
static Mat eta1(Mat a, int p)
{
    Mat c;
    multiply(a,a,c);
    return c > (p*p);
}


class ExtractorSTU : public GriddedHist
{
    int kerP1;
public:
    ExtractorSTU(int gridx=8, int gridy=8, int kp1=8) 
        : GriddedHist(gridx, gridy) 
        , kerP1(kp1)
    {}
    virtual int extract(const Mat &img, Mat &features) const
    {
        SHIFTED_MATS_3x3(img);

        Mat h,fI;

        fI = eta1(abs(I6-IC),kerP1) & (1<<6)
           | eta1(abs(I4-IC),kerP1) & (1<<4) 
           | eta1(abs(I2-IC),kerP1) & (1<<2) 
           | eta1(abs(I0-IC),kerP1) & (1<<1);
        hist(fI,h,64,256);

        features = h.reshape(1,1);
        return features.total() * features.elemSize();
    }
};


class ExtractorGLCM : public GriddedHist
{
public:
    ExtractorGLCM(int gridx=8, int gridy=8) 
        : GriddedHist(gridx, gridy) 
    {}
    virtual int extract(const Mat &img, Mat &features) const
    {
        int M = img.rows; 
        int N = img.cols; 
        // shifted images (special case)
        Mat I7 = img(Range(1,M-1), Range(1,N-2));
        Mat I6 = img(Range(1,M-1), Range(2,N-1));
        Mat I5 = img(Range(1,M-1), Range(3,N  ));
        Mat I4 = img(Range(2,M  ), Range(3,N  ));
        Mat IC = img(Range(2,M  ), Range(2,N-1));
        // Compute and normalize the histograms
        // one pixel displacements in orientations 0є, 45є, 90є and 135є
        Mat h4,h5,h6,h7;
        hist((IC|I4), h4);
        hist((IC|I5), h5);
        hist((IC|I6), h6);
        hist((IC|I7), h7);
        // Average 
        features = (h4+h5+h6+h7)/4;
        return features.total() * features.elemSize();
    }
};


//
// concat histograms from lbp(u) features generated from a bank of gabor filtered images
//
class ExtractorGaborLbp : public ExtractorLbp
{
    Size kernel_size;
public:
    ExtractorGaborLbp(int gridx=8, int gridy=8, int u_table=UniformNone, int kernel_siz=8) 
        : ExtractorLbp(gridx, gridy, u_table) 
        , kernel_size(kernel_siz, kernel_siz)
    {}
    void gabor(const Mat &src_f, Mat &features,double sigma, double theta, double lambda, double gamma, double psi) const
    {
        Mat dest,dest8u,his;
        cv::filter2D(src_f, dest, CV_32F, getGaborKernel(kernel_size, sigma,theta, lambda, gamma, psi));
        dest.convertTo(dest8u, CV_8U);
        ExtractorLbp::extract(dest8u, his);
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




class ExtractorWeinbergHash : public TextureFeature::Extractor
{
    int bits;
    int grid;

    template <class T>
    void _extract( const Mat &img, Mat &features ) const
    {
        static unsigned long crc_32_table[256] = 
        {
            0x000000000L, 0x077073096L, 0x0EE0E612CL, 0x0990951BAL, 0x0076DC419L, 0x0706AF48FL, 0x0E963A535L, 0x09E6495A3L,
            0x00EDB8832L, 0x079DCB8A4L, 0x0E0D5E91EL, 0x097D2D988L, 0x009B64C2BL, 0x07EB17CBDL, 0x0E7B82D07L, 0x090BF1D91L,
            0x01DB71064L, 0x06AB020F2L, 0x0F3B97148L, 0x084BE41DEL, 0x01ADAD47DL, 0x06DDDE4EBL, 0x0F4D4B551L, 0x083D385C7L,
            0x0136C9856L, 0x0646BA8C0L, 0x0FD62F97AL, 0x08A65C9ECL, 0x014015C4FL, 0x063066CD9L, 0x0FA0F3D63L, 0x08D080DF5L,
            0x03B6E20C8L, 0x04C69105EL, 0x0D56041E4L, 0x0A2677172L, 0x03C03E4D1L, 0x04B04D447L, 0x0D20D85FDL, 0x0A50AB56BL,
            0x035B5A8FAL, 0x042B2986CL, 0x0DBBBC9D6L, 0x0ACBCF940L, 0x032D86CE3L, 0x045DF5C75L, 0x0DCD60DCFL, 0x0ABD13D59L,
            0x026D930ACL, 0x051DE003AL, 0x0C8D75180L, 0x0BFD06116L, 0x021B4F4B5L, 0x056B3C423L, 0x0CFBA9599L, 0x0B8BDA50FL,
            0x02802B89EL, 0x05F058808L, 0x0C60CD9B2L, 0x0B10BE924L, 0x02F6F7C87L, 0x058684C11L, 0x0C1611DABL, 0x0B6662D3DL,
            0x076DC4190L, 0x001DB7106L, 0x098D220BCL, 0x0EFD5102AL, 0x071B18589L, 0x006B6B51FL, 0x09FBFE4A5L, 0x0E8B8D433L,
            0x07807C9A2L, 0x00F00F934L, 0x09609A88EL, 0x0E10E9818L, 0x07F6A0DBBL, 0x0086D3D2DL, 0x091646C97L, 0x0E6635C01L,
            0x06B6B51F4L, 0x01C6C6162L, 0x0856530D8L, 0x0F262004EL, 0x06C0695EDL, 0x01B01A57BL, 0x08208F4C1L, 0x0F50FC457L,
            0x065B0D9C6L, 0x012B7E950L, 0x08BBEB8EAL, 0x0FCB9887CL, 0x062DD1DDFL, 0x015DA2D49L, 0x08CD37CF3L, 0x0FBD44C65L,
            0x04DB26158L, 0x03AB551CEL, 0x0A3BC0074L, 0x0D4BB30E2L, 0x04ADFA541L, 0x03DD895D7L, 0x0A4D1C46DL, 0x0D3D6F4FBL,
            0x04369E96AL, 0x0346ED9FCL, 0x0AD678846L, 0x0DA60B8D0L, 0x044042D73L, 0x033031DE5L, 0x0AA0A4C5FL, 0x0DD0D7CC9L,
            0x05005713CL, 0x0270241AAL, 0x0BE0B1010L, 0x0C90C2086L, 0x05768B525L, 0x0206F85B3L, 0x0B966D409L, 0x0CE61E49FL,
            0x05EDEF90EL, 0x029D9C998L, 0x0B0D09822L, 0x0C7D7A8B4L, 0x059B33D17L, 0x02EB40D81L, 0x0B7BD5C3BL, 0x0C0BA6CADL,
            0x0EDB88320L, 0x09ABFB3B6L, 0x003B6E20CL, 0x074B1D29AL, 0x0EAD54739L, 0x09DD277AFL, 0x004DB2615L, 0x073DC1683L,
            0x0E3630B12L, 0x094643B84L, 0x00D6D6A3EL, 0x07A6A5AA8L, 0x0E40ECF0BL, 0x09309FF9DL, 0x00A00AE27L, 0x07D079EB1L,
            0x0F00F9344L, 0x08708A3D2L, 0x01E01F268L, 0x06906C2FEL, 0x0F762575DL, 0x0806567CBL, 0x0196C3671L, 0x06E6B06E7L,
            0x0FED41B76L, 0x089D32BE0L, 0x010DA7A5AL, 0x067DD4ACCL, 0x0F9B9DF6FL, 0x08EBEEFF9L, 0x017B7BE43L, 0x060B08ED5L,
            0x0D6D6A3E8L, 0x0A1D1937EL, 0x038D8C2C4L, 0x04FDFF252L, 0x0D1BB67F1L, 0x0A6BC5767L, 0x03FB506DDL, 0x048B2364BL,
            0x0D80D2BDAL, 0x0AF0A1B4CL, 0x036034AF6L, 0x041047A60L, 0x0DF60EFC3L, 0x0A867DF55L, 0x0316E8EEFL, 0x04669BE79L,
            0x0CB61B38CL, 0x0BC66831AL, 0x0256FD2A0L, 0x05268E236L, 0x0CC0C7795L, 0x0BB0B4703L, 0x0220216B9L, 0x05505262FL,
            0x0C5BA3BBEL, 0x0B2BD0B28L, 0x02BB45A92L, 0x05CB36A04L, 0x0C2D7FFA7L, 0x0B5D0CF31L, 0x02CD99E8BL, 0x05BDEAE1DL,
            0x09B64C2B0L, 0x0EC63F226L, 0x0756AA39CL, 0x0026D930AL, 0x09C0906A9L, 0x0EB0E363FL, 0x072076785L, 0x005005713L,
            0x095BF4A82L, 0x0E2B87A14L, 0x07BB12BAEL, 0x00CB61B38L, 0x092D28E9BL, 0x0E5D5BE0DL, 0x07CDCEFB7L, 0x00BDBDF21L,
            0x086D3D2D4L, 0x0F1D4E242L, 0x068DDB3F8L, 0x01FDA836EL, 0x081BE16CDL, 0x0F6B9265BL, 0x06FB077E1L, 0x018B74777L,
            0x088085AE6L, 0x0FF0F6A70L, 0x066063BCAL, 0x011010B5CL, 0x08F659EFFL, 0x0F862AE69L, 0x0616BFFD3L, 0x0166CCF45L,
            0x0A00AE278L, 0x0D70DD2EEL, 0x04E048354L, 0x03903B3C2L, 0x0A7672661L, 0x0D06016F7L, 0x04969474DL, 0x03E6E77DBL,
            0x0AED16A4AL, 0x0D9D65ADCL, 0x040DF0B66L, 0x037D83BF0L, 0x0A9BCAE53L, 0x0DEBB9EC5L, 0x047B2CF7FL, 0x030B5FFE9L,
            0x0BDBDF21CL, 0x0CABAC28AL, 0x053B39330L, 0x024B4A3A6L, 0x0BAD03605L, 0x0CDD70693L, 0x054DE5729L, 0x023D967BFL,
            0x0B3667A2EL, 0x0C4614AB8L, 0x05D681B02L, 0x02A6F2B94L, 0x0B40BBE37L, 0x0C30C8EA1L, 0x05A05DF1BL, 0x02D02EF8DL
        };
        unsigned long crc = 0;

        Mat_<T> feat(1,(1<<bits),0.0f);
        for (int i=0; i<img.rows; i++)
        for (int j=0; j<img.cols; j++)
        {
            uchar c = img.at<uchar>(i,j);
            c ^= crc;                    // apply current crc to current character
            crc >>= 8;                   // shift old value down
            crc ^= crc_32_table[c];      // xor in new value
            int index = crc & ((1 << bits) - 1);
            // Use the nth bit, zero-indexed, to determine if we add or subtract one from the index.
            int sign = (((crc & (1 << bits)) >> bits) << 1) - 1;
            feat(index) += sign;
        }
        features = feat;
    }

public:

    ExtractorWeinbergHash(int grid=4, int bits=8) : grid(grid), bits(bits) {}

    virtual int extract( const Mat &img, Mat &features ) const 
    {
        for (size_t i=0; i<img.rows-grid; i+=grid)
        for (size_t j=0; j<img.cols-grid; j+=grid)
        {
            Mat feature;
            Mat roi(img,Rect(i,j,grid,grid));
            _extract<float>(roi, feature);
            // normalize(feature,feature,1,1);
            features.push_back(feature);
        }
        features = features.reshape(1,1);
        return features.total() * features.elemSize();
    }
};



//
// 'factory' functions (aka public api)
//

cv::Ptr<TextureFeature::Extractor> createExtractorPixels(int resw=0, int resh=0)
{ 
    return makePtr<ExtractorPixels>(resw, resh); 
}

cv::Ptr<TextureFeature::Extractor> createExtractorMoments()
{ 
    return makePtr<ExtractorMoments>(); 
}

cv::Ptr<TextureFeature::Extractor> createExtractorLbp(int gx=8, int gy=8, int utable=ExtractorLbp::UniformNone)
{ 
    return makePtr<ExtractorLbp>(gx, gy, utable); 
}

cv::Ptr<TextureFeature::Extractor> createExtractorBGC1(int gx=8, int gy=8, int utable=ExtractorLbp::UniformNone)
{ 
    return makePtr<ExtractorBGC1>(gx, gy, utable); 
}

cv::Ptr<TextureFeature::Extractor> createExtractorLQP(int gx=8, int gy=8)
{ 
    return makePtr<ExtractorLQP>(gx, gy); 
}

cv::Ptr<TextureFeature::Extractor> createExtractorMTS(int gx=8, int gy=8)
{ 
    return makePtr<ExtractorMTS>(gx, gy); 
}

cv::Ptr<TextureFeature::Extractor> createExtractorSTU(int gx=8, int gy=8,int kp1=5)
{ 
    return makePtr<ExtractorSTU>();//gx, gy, kp1); 
}

cv::Ptr<TextureFeature::Extractor> createExtractorGLCM(int gx=8, int gy=8)
{ 
    return makePtr<ExtractorGLCM>(gx, gy); 
}

cv::Ptr<TextureFeature::Extractor> createExtractorWLD(int gx=8, int gy=8, int tf=CV_32F)
{ 
    return makePtr<WLD>(gx, gy, tf); 
}

cv::Ptr<TextureFeature::Extractor> createExtractorGaborLbp(int gx=8, int gy=8, int u_table=0, int kernel_siz=8)
{ 
    return makePtr<ExtractorGaborLbp>(gx, gy, u_table, kernel_siz); 
}

cv::Ptr<TextureFeature::Extractor> createExtractorDct()
{ 
    return makePtr<ExtractorDct>(); 
}

cv::Ptr<TextureFeature::Extractor> createExtractorWeinbergHash(int grid=4,int bits=8)
{ 
    return makePtr<ExtractorWeinbergHash>(grid,bits); 
}
