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
        features = mom(img).reshape(1,1);
        return features.total() * features.elemSize() ;
    }
};






// 
// base for lbph, calc features on the whole image, the hist on a grid, 
//   so we avoid to waste border pixels 
//
struct GriddedHist : public TextureFeature::Extractor
{
protected:
    Mat_<float> weights;
    int GRIDX,GRIDY;
    bool doWeight;

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
                if ( doWeight )
                    h *= weights(j,i);
                histo.push_back(h.reshape(1,1));
            }
        }
        normalize(histo.reshape(1,1),histo);
    }

public:

    GriddedHist(int gridx=8, int gridy=8, bool doweight=false) 
        : GRIDX(gridx)
        , GRIDY(gridy) 
        , weights(8,8)
        , doWeight(doweight)
    {
        if (doWeight) // not all patches have the same relevance.
        {
            weights << 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 2, 2, 2, 2, 2, 2, 1,
                       1, 3, 3, 3, 3, 3, 3, 1,
                       1, 3, 3, 3, 3, 3, 3, 1,
                       1, 2, 3, 3, 3, 3, 2, 1,
                       1, 2, 3, 3, 3, 3, 2, 1,
                       1, 2, 3, 3, 3, 3, 2, 1,
                       1, 1, 1, 1, 1, 1, 1, 1;
            if ( GRIDX != weights.rows || GRIDY != weights.cols )
                resize(weights, weights, Size(GRIDX,GRIDY));
            normalize(weights, weights);
        }
    }
};




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



class ExtractorLbp : public GriddedHist
{
protected:

    int utable;

    //
    // "histogram of equivalence patterns" 
    //
    virtual void hep( const Mat &I, Mat &fI ) const
    {
#if 0
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
        Mat_<uchar> feature(I.size(),0);
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
        UniformNone = -1  // 256, as-is
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


inline Mat eta1(Mat a, int p)
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
// four-patch (well, more pixels) lbp
//
class ExtractorFPLbp : public GriddedHist
{
public:
    ExtractorFPLbp(int gridx=8, int gridy=8) 
        : GriddedHist(gridx, gridy) 
    {}
    virtual int extract(const Mat &img, Mat &features) const
    {
        Mat_<uchar> fI(img.size(),0);
        Mat_<uchar> I(img);
        const int m=2;
        for ( int r=m; r<I.rows-m; r++ )
        {
            for ( int c=m; c<I.cols-m; c++ )
            {
                uchar v = 0;
                v |= ((I(r  ,c+1) - I(r+2,c+2)) > (I(r  ,c-1) - I(r-2,c-2))) * 1;
                v |= ((I(r+1,c+1) - I(r+2,c  )) > (I(r-1,c-1) - I(r-2,c  ))) * 2;
                v |= ((I(r+1,c  ) - I(r+2,c-2)) > (I(r-1,c  ) - I(r-2,c+2))) * 4;
                v |= ((I(r+1,c-1) - I(r  ,c-2)) > (I(r-1,c+1) - I(r  ,c+2))) * 8;
                fI(r,c) = v;
            }
        }
        hist(fI,features,16,16);
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
    ExtractorGridFeature() : grid(16) {}
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

cv::Ptr<TextureFeature::Extractor> createExtractorPixels(int resw=0, int resh=0)
{   return makePtr<ExtractorPixels>(resw, resh); }

cv::Ptr<TextureFeature::Extractor> createExtractorMoments()
{   return makePtr<ExtractorMoments>(); }

cv::Ptr<TextureFeature::Extractor> createExtractorLbp(int gx=8, int gy=8, int utable=ExtractorLbp::UniformNone)
{   return makePtr<ExtractorLbp>(gx, gy, utable); }

cv::Ptr<TextureFeature::Extractor> createExtractorFPLbp(int gx=8, int gy=8)
{   return makePtr<ExtractorFPLbp>(gx, gy); }

cv::Ptr<TextureFeature::Extractor> createExtractorBGC1(int gx=8, int gy=8, int utable=ExtractorLbp::UniformNone)
{   return makePtr<ExtractorBGC1>(gx, gy, utable); }

cv::Ptr<TextureFeature::Extractor> createExtractorLQP(int gx=8, int gy=8)
{   return makePtr<ExtractorLQP>(gx, gy); }

cv::Ptr<TextureFeature::Extractor> createExtractorMTS(int gx=8, int gy=8)
{   return makePtr<ExtractorMTS>(gx, gy); }

cv::Ptr<TextureFeature::Extractor> createExtractorSTU(int gx=8, int gy=8,int kp1=5)
{   return makePtr<ExtractorSTU>(); }

cv::Ptr<TextureFeature::Extractor> createExtractorGLCM(int gx=8, int gy=8)
{   return makePtr<ExtractorGLCM>(gx, gy); }

cv::Ptr<TextureFeature::Extractor> createExtractorWLD(int gx=8, int gy=8, int tf=CV_32F)
{   return makePtr<WLD>(gx, gy, tf); }

cv::Ptr<TextureFeature::Extractor> createExtractorGaborLbp(int gx=8, int gy=8, int u_table=0, int kernel_siz=8)
{   return makePtr<ExtractorGaborLbp>(gx, gy, u_table, kernel_siz); }

cv::Ptr<TextureFeature::Extractor> createExtractorDct()
{   return makePtr<ExtractorDct>(); }

cv::Ptr<TextureFeature::Extractor> createExtractorORBGrid()
{   return makePtr<ExtractorORBGrid>(); }

cv::Ptr<TextureFeature::Extractor> createExtractorSIFTGrid()
{   return makePtr<ExtractorSIFTGrid>(); }

