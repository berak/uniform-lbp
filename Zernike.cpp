#include <opencv2/core/core.hpp>
//#include <opencv2/core/utility.hpp>

#include <iostream>
using namespace std;
using namespace cv;


#include "factory.h"


//
// this code tries to apply zernike moments to a (patched) image as a 'textron' similarity measure
//


//
// nice explanatory snippet taken from: zernike_radial_polynomials.m
// we will be using the precalculated n m Zernike polynomials below.
//

//
//    Written by Mohammed S. Al-Rawi, 2007-2008
//    Last updated 2011.
//    rawi707@yahoo.com
// 
//  The fast method presented in this code:
//   The method implemented in this work is q-recurseive see Ref.
//   Way, C.-C., Paramesran, R., and Mukandan, R., A comparative analysis of algorithms for fast computation of Zernike moments, Pattern Recognition 36 (2003) 731-742.
//   It uses radial polynomials of fixed order p with a varying index q to
//   compute Zernike moments
//
//
//  What are Zernike polynomials?
//    The radial Zernike polynomials are the radial portion of the
//    Zernike functions, which are an orthogonal basis on the unit
//    circle.  The series representation of the radial Zernike
//    polynomials is
// 
//           (n-m)/2
//             __
//     m      \       s                                          n-2s
//    Z(r) =  /__ (-1)  [(n-s)!/(s!((n-m)/2-s)!((n+m)/2-s)!)] * r
//     n      s=0
// 
//    The following table shows the first 12 polynomials.
// 
//        n    m    Zernike polynomial    Normalization
//        ---------------------------------------------
//        0    0    1                        sqrt(2)
//        1    1    r                           2
//        2    0    2*r^2 - 1                sqrt(6)
//        2    2    r^2                      sqrt(6)
//        3    1    3*r^3 - 2*r              sqrt(8)
//        3    3    r^3                      sqrt(8)
//        4    0    6*r^4 - 6*r^2 + 1        sqrt(10)
//        4    2    4*r^4 - 3*r^2            sqrt(10)
//        4    4    r^4                      sqrt(10)
//        5    1    10*r^5 - 12*r^3 + 3*r    sqrt(12)
//        5    3    5*r^5 - 4*r^3            sqrt(12)
//        5    5    r^5                      sqrt(12)
//        ---------------------------------------------
// 


//
// to calculate the ZernikeMoment of a given patch image, 
// this implementation (loosely) follows the pseudocode example in figure 3 of
//   "Anovel approach to the fast computation of Zernikemoments" [Sun-Kyoo Hwang,Whoi-Yul Kim] 2006
//
// since the radial zernike polynomials as well as rho and theta 
//   are 'constant' for a given patchsize (or say, independant of our image),
//   we can cache a Mat with the (radial*cos(m*theta)) term for each of our 10 or so moments,
//   so calculating the ZernikeMoment of a patch (in a later stage) 
//   boils down to a NxN matrix-mult, and a sum over that
//
// omitting the 1st 2 polynomials above(since they don't add much gain),
//   so this has 10 moments
//

struct Zernike : public FaceRecognizer
{
    //;) don't bother optimizing below private code (at all), this is used to generate lut's, once per Zernike instance 
    static double pseudo_20(double r) { return (3 + 10*r*r - 12*r); }
    static double pseudo_21(double r) { return (5*r*r - 4*r); }
    static double pseudo_22(double r) { return (r*r); }
    static double pseudo_30(double r) { return (-4 + 35*r*r*r - 60*r*r + 30*r); }
    static double pseudo_31(double r) { return (21*r*r*r - 30*r*r+10*r); }
    static double pseudo_32(double r) { return (7*r*r*r - 6*r*r); }
    static double pseudo_33(double r) { return (r*r*r); }
    static double pseudo_40(double r) { return (5 + 126*r*r*r*r - 280*r*r*r + 210*r*r - 60*r); }
    static double pseudo_41(double r) { return (84*r*r*r*r - 168*r*r*r + 105*r*r - 20*r); }
    static double pseudo_42(double r) { return (36*r*r*r*r - 56*r*r*r + 21*r*r); }
    static double pseudo_43(double r) { return (9*r*r*r*r - 8*r*r*r); }
    static double pseudo_44(double r) { return (r*r*r*r); }

    static double radpol_20(double r) { return (2*r*r - 1)*sqrt(2.0); }
    static double radpol_22(double r) { return (r*r)*sqrt(6.0); }
    static double radpol_31(double r) { return (3*r*r*r - 2*r*r)*sqrt(8.0); }
    static double radpol_33(double r) { return (r*r*r)*sqrt(8.0); }
    static double radpol_40(double r) { return (6*r*r*r*r - 6*r*r+1)*sqrt(10.0); }
    static double radpol_42(double r) { return (4*r*r*r*r - 3*r*r)*sqrt(10.0); }
    static double radpol_44(double r) { return (r*r*r*r)*sqrt(10.0); }
    static double radpol_51(double r) { return (10*r*r*r*r - 12*r*r*r + 3*r)*sqrt(12.0); }
    static double radpol_53(double r) { return (5*r*r*r*r - 4*r*r*r)*sqrt(12.0); }
    static double radpol_55(double r) { return (r*r*r*r*r)*sqrt(12.0); }


    //! we only save the real/cos part of the (originally complex) equation here.
    void cos_mat(Mat & zm, double maumau, double(*radicalchic)(double) )
    {
        zm = Mat::zeros(N,N,CV_32F);
        int cnt = 0;
        for ( int i=0; i<N; i++ ) 
        {
            for ( int j=0; j<N; j++ ) 
            {
                //double a(2*i-N+1);
                //double b(N-1-j*2);
                double a(c1*i + c2); //(29) Image description with generalized pseudo-Zernike moments 
                double b(c1*j + c2);
                double rho = sqrt(a*a + b*b);
                double theta = atan(b/a);
                double radial = radicalchic(rho);
                zm.at<float>(i,j) = float(radial * cos(maumau * theta));
            }
        }
        zm /= (N*N); // normalized [-1,1]
    }

    enum {NZERN=7+5};
    Mat zerm[NZERN]; // precalc array, one per feature
    int N;           // patchsize
    int nfeatures;   // you might want to use less than max features
    double c1,c2;

    vector<int> labels;
    Mat features;

public:


    //
    //! precalculate the (radial*cos(m*theta)) term for each of our moments
    //! resultant featuresize will be nfeatures*(w/N)*(h/N)
    //
    Zernike(int n=8, int used=NZERN)
        : N(n)
        , nfeatures(min(used, int(NZERN)))
        , c1(sqrt(2.0) / (N-1))
        , c2(1.0 / sqrt(2.0))
    {
        cos_mat(zerm[0],  0.0, pseudo_20);
        cos_mat(zerm[1],  1.0, pseudo_21);
        cos_mat(zerm[2],  2.0, pseudo_22);
        cos_mat(zerm[3],  0.0, pseudo_30);
        cos_mat(zerm[4],  1.0, pseudo_31);
        cos_mat(zerm[5],  2.0, pseudo_32);
        cos_mat(zerm[6],  3.0, pseudo_33);
        cos_mat(zerm[7],  0.0, pseudo_40);
        cos_mat(zerm[8],  1.0, pseudo_41);
        cos_mat(zerm[9],  2.0, pseudo_42);
        cos_mat(zerm[10], 3.0, pseudo_43);
        cos_mat(zerm[11], 4.0, pseudo_44);
    }


    //
    //! adds nfeatures elems for a NxN patch to a feature Mat
    //! expects single channel float Mats as input
    //
    void compute_patch(const Mat & patch, Mat & _features) const
    {
        for (int i=0; i<nfeatures; i++) 
        {
            Mat c;
            multiply(patch, zerm[i], c); // per element
            _features.push_back(float(sum(c)[0]));
        }
    }

    //
    //! calculates an nfeatures*N*N feature vec per image, 
    //!  the (L2)norm of it will be our distance metrics for comparing images.
    //
    void compute(const Mat & img, Mat & _features) const
    {       
        Mat m;
        if ( img.type() != CV_32F )
            img.convertTo(m, CV_32F);
        else
            m=img;
       // normalize(m,m,255.0);

        //// the trick with the precalculated (radial*cos(m*theta)) term requires a fixed patch size,
        //// so let's try to 'equalize' differently sized images here
        ////   downside: this puts a penalty on (small) input images < NxN ,
        ////      please let me know, if you find something better here.
        //cv::resize(m, m, Size(N*N, N*N));
//        cv::pyrUp(m, m);
      //  cv::resize(m, m, m.size()*4);

        for (int i=0; i<m.rows-N; i+=N) 
        {
            for (int j=0; j<m.cols-N; j+=N) 
            {
                Mat patch = m(Rect(j, i, N, N));
                compute_patch(patch, _features);
            }
        }
    }



    virtual void train(InputArray src, InputArray lbls)    
    {
        features.release();
        labels.clear();

        update(src,lbls);
    }

    virtual void predict(InputArray src, int& label, double & minDist) const    
    {
        Mat zerf;
        compute(src.getMat(),zerf);
        zerf = zerf.reshape(1,1);

        minDist = DBL_MAX;
        int minClass = -1;
        for(int i=0; i<features.rows; i++) 
        {
            Mat f = features.row(i);
            double dist = norm(f, zerf, NORM_L2);
            if(dist < minDist) 
            {
                minDist = dist;
                minClass = labels[i];
            }
        }
        label = minClass;
    }
    virtual int predict(InputArray src) const 
    {
        int pred=-1;
        double conf=-1;
        predict(src,pred,conf);
        return pred;
    }
    virtual void update(InputArrayOfArrays src, InputArray lbls) 
    {
        labels = lbls.getMat();

        vector<Mat> imgs;
        src.getMatVector(imgs);

        for (size_t i=0; i<imgs.size(); ++i)
        {
            compute(imgs[i],features);
        }
        features = features.reshape(1,imgs.size());
    }
    virtual void save(const std::string& filename) const    {}
    virtual void save(FileStorage& fs) const    {}
    virtual void load(const std::string& filename)    {}
    virtual void load(const FileStorage& fs)    {}
};

Ptr<FaceRecognizer> createZernikeFaceRecognizer(int N,int used)
{
    return makePtr<Zernike>(N,used);
}





