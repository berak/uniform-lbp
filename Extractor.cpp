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
        return features.rows;
    }
};





//
// gridded humoments
//
class ExtractorMoments : public TextureFeature::Extractor
{
    static void mom(const Mat &z, Mat & feature, int i, int j, int w, int h)
    {
        double hu[7];
        Mat roi(z, cv::Rect(i*w,j*h,w,h));
        HuMoments( moments( roi, false), hu);
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
        return features.rows;
    }
};






// 
// base for lbph, calc features on pixels, then calc the grid on that, 
//   thus avoiding to waste border pixels 
//     (with probably the price of pixels shared between patches)
//
struct GriddedHist : public TextureFeature::Extractor
{
protected:
    int GRIDX,GRIDY;
    
    // histogram calculation seems to be the bottleneck.
    void calc_hist(const Mat_<uchar> & feature, Mat_<float> & hist, int histSize, int histRange=256) const
    {   
        for ( int i=0; i<feature.rows; i++ )
        {
            for ( int j=0; j<feature.cols; j++ )
            {
                uchar bin = int(feature(i,j)) * histSize / histRange;
                hist( bin ) += 1.0f;
            }
        }
    }

    void hist(const Mat & feature, Mat & histo, int histSize=256, int histRnange=256) const
    {   
        histo.release();
        const float range[] = { 0, 256 } ;
        const float* histRange[] = { range };
        int sw = (feature.cols)/(GRIDX+1);
        int sh = (feature.rows)/(GRIDY+1);
        for ( int i=0; i<GRIDX; i++ )
        {
            for ( int j=0; j<GRIDY; j++ )
            {  
                Rect patch(i*sw,j*sh,sw,sh);
                Mat fi( feature, patch );
                Mat_<float> h(1,histSize,0.0f);
                //calcHist( &fi, 1, 0, Mat(), h, 1, &histSize, &histRange, true, false );
                calc_hist(fi,h,histSize,histRnange);
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





class ExtractorLbp : public GriddedHist
{
protected:

    void lbp_pix( const Mat &z, Mat & f ) const
    {
        Mat_<uchar> fI(z.size());
        Mat_<uchar> img(z);
        const int m=1;
        for ( int r=m; r<z.rows-m; r++ )
        {
            for ( int c=m; c<z.cols-m; c++ )
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
                fI(r,c) = v;
            }
        }
        f = fI;
    }

public:

    ExtractorLbp(int gridx=8, int gridy=8) 
        : GriddedHist(gridx,gridy) 
    {}

    // TextureFeature::Extractor
    virtual int extract(const Mat &img, Mat &features) const
    {
        Mat fI;
        lbp_pix(img,fI);
        hist(fI,features,256);
        return features.rows;
    }
};





class ExtractorLbpUniform : public ExtractorLbp
{

    enum UniformTable
    {
        UniformNormal,    // 58 + noise
        UniformModified,  // 58
        UniformReduced,   // 16 + noise
        UniFormMax
    };

    int utable;

public:

    ExtractorLbpUniform(int gridx=8, int gridy=8, int u_table=0) 
        : ExtractorLbp(gridx,gridy) 
        , utable(u_table)
    {}

    // TextureFeature::Extractor
    virtual int extract(const Mat &img, Mat &features) const
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
        Mat lu(1,256,CV_8U, uniform[utable]);

        Mat fI;
        lbp_pix(img,fI);

        Mat h59;
        LUT(fI,lu,h59);

        int histlen[] = {59,58,17};
        hist(h59,features,histlen[utable],histlen[utable]);
        return features.rows;
    }
};



//
//  A Robust Descriptor based on Weber’s Law
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

    template < class T>
    void oper(const Mat & src, Mat & hist) const {
        const double CV_PI_4 = CV_PI / 4.0;
        int radius = 1;
        for(int i=radius;i<src.rows-radius;i++) {
            for(int j=radius;j<src.cols-radius;j++) {
                // 7 0 1
                // 6 c 2
                // 5 4 3
                uchar c   = src.at<uchar>(i,j);
                uchar n[8]= {
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
                for ( int i=0; i<size_theta_n; i++ ) {
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
        switch(typeflag)
        {
        case CV_32F:
            features = Mat::zeros(WLD::size_all*GRIDX*GRIDY,1,typeflag);
            oper<float>(img,features);
            break;
        case CV_8U:
            features = Mat::zeros(WLD::size_all*GRIDX*GRIDY,1,typeflag);
            oper<uchar>(img,features);
            break;
        }
        return features.rows;
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
cv::Ptr<TextureFeature::Extractor> createExtractorLbp(int gx=8, int gy=8)
{ 
    return makePtr<ExtractorLbp>(gx, gy); 
}
cv::Ptr<TextureFeature::Extractor> createExtractorLbpUniform(int gx=8, int gy=8, int utable=0)
{ 
    return makePtr<ExtractorLbpUniform>(gx, gy, 0); 
}
cv::Ptr<TextureFeature::Extractor> createExtractorWLD(int gx=8, int gy=8, int tf=CV_32F)
{ 
    return makePtr<WLD>(gx, gy, tf); 
}