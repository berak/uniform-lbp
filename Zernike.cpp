#include <opencv2/core/core.hpp>

#include <iostream>
using namespace std;
using namespace cv;


#include "factory.h"


//
// this code tries to apply zernike moments to a (patched) image as a 'textron' similarity measure
//
//
// the main optimization here is, that the moments for rho and theta 
//   are 'constant' for a given patchsize (or say, independant of our image *content*),
//   we can cache a Mat with the (radial*cos(m*theta)) term for each of our moments.
//    so calculating the ZernikeMoment of a patch (in a later stage) 
//    boils down to a NxN (per element) matrix-mult, and a sum over that
//
//

struct Zernike : public FaceRecognizer
{
    //;) don't bother optimizing below private code (at all), this is used to generate lut's, once per Zernike instance 

    ////static double pseudo_20(double r) { return (3 + 10*r*r - 12*r); }
    ////static double pseudo_21(double r) { return (5*r*r - 4*r); }
    ////static double pseudo_22(double r) { return (r*r); }
    ////static double pseudo_30(double r) { return (-4 + 35*r*r*r - 60*r*r + 30*r); }
    ////static double pseudo_31(double r) { return (21*r*r*r - 30*r*r+10*r); }
    ////static double pseudo_32(double r) { return (7*r*r*r - 6*r*r); }
    ////static double pseudo_33(double r) { return (r*r*r); }
    ////static double pseudo_40(double r) { return (5 + 126*r*r*r*r - 280*r*r*r + 210*r*r - 60*r); }
    ////static double pseudo_41(double r) { return (84*r*r*r*r - 168*r*r*r + 105*r*r - 20*r); }
    ////static double pseudo_42(double r) { return (36*r*r*r*r - 56*r*r*r + 21*r*r); }
    ////static double pseudo_43(double r) { return (9*r*r*r*r - 8*r*r*r); }
    ////static double pseudo_44(double r) { return (r*r*r*r); }

    ////static double radpol_20(double r) { return (2*r*r - 1)*sqrt(2.0); }
    ////static double radpol_22(double r) { return (r*r)*sqrt(6.0); }
    ////static double radpol_31(double r) { return (3*r*r*r - 2*r*r)*sqrt(8.0); }
    ////static double radpol_33(double r) { return (r*r*r)*sqrt(8.0); }
    ////static double radpol_40(double r) { return (6*r*r*r*r - 6*r*r+1)*sqrt(10.0); }
    ////static double radpol_42(double r) { return (4*r*r*r*r - 3*r*r)*sqrt(10.0); }
    ////static double radpol_44(double r) { return (r*r*r*r)*sqrt(10.0); }
    ////static double radpol_51(double r) { return (10*r*r*r*r - 12*r*r*r + 3*r)*sqrt(12.0); }
    ////static double radpol_53(double r) { return (5*r*r*r*r - 4*r*r*r)*sqrt(12.0); }
    ////static double radpol_55(double r) { return (r*r*r*r*r)*sqrt(12.0); }

    ////// let's try the low hanging fruit first: 
    ////// this (loosely) follows the pseudocode example in figure 3 of
    //////   "Anovel approach to the fast computation of Zernikemoments" [Sun-Kyoo Hwang,Whoi-Yul Kim] 2006
    ////// use a precomputed rotational polynom in form of a function ptr
    //////  we only save the real/cos part of the (originally complex) equation here.
    ////void radpol(Mat & zm, double maumau, double(*radicalchic)(double) )
    ////{
    ////    zm = Mat::zeros(N,N,CV_32F);
    ////    const double c3(2.0 / ((N-1)*(N-1)));
    ////    for ( int i=0; i<N; i++ ) 
    ////    {
    ////        for ( int j=0; j<N; j++ ) 
    ////        {
    ////            double a(2*i-N+1); //     "Anovel approach to the fast computation of Zernikemoments"
    ////            double b(N-1-j*2);
    ////            double rho = sqrt(a*a + b*b);
    ////            double theta = atan(b/a);
    ////            double radial = radicalchic(rho);
    ////            zm.at<float>(i,j) = float(c3 * radial * cos(maumau * theta)); 
    ////        }
    ////    }
    ////}



    //
    // implemented below is the gpzm approach
    //  "Image description with generalized pseudo-Zernike moments"
    //
    unsigned fac(int n) 
    { 
        if ( n<1 ) return 1;
        unsigned r=n;
        while(--n>0)  r*=n;
        return r;
    }

    // (7) "Image description with generalized pseudo-Zernike moments"
    unsigned pochhammer(unsigned a, int k) // lower
    {
        assert(a>=0);
        assert(k>=0);
        if ( k<1 ) return a;
        for(int n=1; n<k-1; n++)
            a *= (a+n);
        //for(int n=1; n<=k; n++)
        //    a *= (a-n);
        //while (--k > 0)  a *= (a+k); // x*(x+1)*..(x+k-1)
        // FIXME: wikipedia claims, this is 'upper', 
        // while this would be 'lower' instead:
        //while (k-- >= 1)  a *= (a-k); // x*(x-1)*..(x-k+1)
        //   FIXME2: is above 'one-off' ?
        return a;
    }

    // (10) "Image description with generalized pseudo-Zernike moments"
    //  assumes q = abs(q);
    double R(int a, int p, int q, double r)
    {
        double res = 0.0;  
        double f = double(fac(p+q+1)) / pochhammer(a+1, p+q+1);
        for (int s=0; s<=(p-q); s++)
        {
            double v = pow(-1.0, s) * pochhammer(a+1, 2*p+1-s);
            int   vd = fac(s) * fac(p-q-s) * fac(p+q+1-s);
            res += v * pow(r,(p-s)) / double(vd);
        }
        return f * res;
    }

    // normalization/weight factor used in:
    // (21) "Image description with generalized pseudo-Zernike moments"
    //  assumes q = abs(q);
    double Rweighted(int a, int p, int q, double r)
    {
        double x = double((2*p+a+2)*pochhammer((q+1+p-q), 2*q+1));
        double y = 2.0*CV_PI * double(pochhammer(p-q+1,   2*q+1));  
        return sqrt(x / y) * pow(1.0-r, a/2);
    }

    // (28) "Image description with generalized pseudo-Zernike moments"
    // this is the R_flat(r(x,y)) mat that can get precalced/cached for given values of N,p,q,a
    //   i only save the real(cos) part of the (originally complex) equation here, 
    //     since i'm out only for a similarity measure (for reconstruction, it would need the imaginary/sin part as well).
    void gpzm(Mat & zm, int a, int p, int q)
    {
        // normalization factors
        const double c1(sqrt(2.0) / (N-1));
        const double c2(1.0 / sqrt(2.0));
        const double c3(2.0 / ((N-1)*(N-1)));

        zm = Mat::zeros(N,N,CV_32F);
        for (int i=0; i<N; i++) 
        {
            for (int j=0; j<N; j++) 
            {
                double x(c1*j + c2); //(29) 
                double y(c1*i + c2);
                double rho = sqrt(x*x + y*y);
                double theta = atan(y/x);
                double radial = R(a,p,q,rho) * Rweighted(a,p,q,rho);
                zm.at<float>(i,j) = float(c3 * radial * cos(theta*q)); // ignores negative q
            }
        }
        // cerr << a<< " " << p << " " << q << zm << endl << endl;
    }

    enum {NZERN=7+5};  // only the 1st 7 seem to bring any gain
    vector<Mat> zerm;  // precalc array, one per feature
    int N;             // patchsize
    int nfeatures;     // you might want to use less than max features

    vector<int> labels;
    Mat features;

public:


    //
    //! precalculate the (radial*cos(m*theta)) term for each of our moments
    //! resultant featuresize will be nfeatures*(w/N)*(h/N)
    //
    Zernike(int n=8, int used=NZERN)
        : N(n)
    {
        //(32) "Image description with generalized pseudo-Zernike moments"
        //  V = [Z_20 Z_21 Z_22 Z_30 Z_31 Z_32 Z_33]     

        //radpol(zerm[0],  0.0, pseudo_20);
        //radpol(zerm[1],  1.0, pseudo_21);
        //radpol(zerm[2],  2.0, pseudo_22);
        //radpol(zerm[3],  0.0, pseudo_30);
        //radpol(zerm[4],  1.0, pseudo_31);
        //radpol(zerm[5],  2.0, pseudo_32);
        //radpol(zerm[6],  3.0, pseudo_33);
        if ( used>0 )
        {
            //int a = 8;
            //add(a, 2, 0);
            //add(a, 2, 1);
            //add(a, 2, 2);
            //add(a, 3, 0);
            //add(a, 3, 1);
            //add(a, 3, 2);
            //add(a, 3, 3);
            // 2 1 9, 27 11 14, 22 0 11, 28 5 11, 30 6 12, 6 13 2, 2 6 2, : 0.728
            // 18 12 1, 5 0 9, 9 7 2, 27 1 -1, 2 5 1, 11 3 12, 13 3 1
            add(18,12,1);
            add(5,0,9);
            add(9,7,2);
            add(27,1,0);
            add(2,5,1);
            add(11,3,12);
            add(13,3,1);
        }
    }

    void add(int a, int p, int q)
    {
        Mat m;
        gpzm(m, a, p,q);
        zerm.push_back(m);
    }

    //
    //! adds nfeatures elems for a NxN patch to a feature Mat
    //! expects single channel float Mats as input
    //
    void compute_patch(const Mat & patch, Mat & _features) const
    {
        for (int i=0; i<zerm.size(); i++) 
        {
            Mat c;
            multiply(patch, zerm[i], c); // per element
            _features.push_back(float(sum(c)[0]));
        }
    }

    //
    //! calculates an nfeatures*(w/N)*(h/N) feature vec per image, 
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
    virtual void update(InputArrayOfArrays src, InputArray lbls) 
    {
        Mat l = lbls.getMat();
        labels.insert(l.begin(),l.end());

        vector<Mat> imgs;
        src.getMatVector(imgs);

        for (size_t i=0; i<imgs.size(); ++i)
        {
            compute(imgs[i],features);
        }
        features = features.reshape(1,imgs.size());
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
    virtual void save(const std::string& filename) const    {}
    virtual void save(FileStorage& fs) const    {}
    virtual void load(const std::string& filename)    {}
    virtual void load(const FileStorage& fs)    {}
};

Ptr<FaceRecognizer> createZernikeFaceRecognizer(int N,int used)
{
    return makePtr<Zernike>(N,used);
}





struct ga_params
{
    int a,p,q;
    float score;
    ga_params(int a=0,int p=0,int q=0,float s=FLT_MAX) : a(a),p(p),q(q),score(s) {}
};
struct ga_sorter
{
    bool operator () (const ga_params & a,const ga_params & b) const
    {
        return a.score < b.score;
    }
};
String ga_str(const ga_params & ga)
{
    return format("%d %d %d",ga.a,ga.p,ga.q);
}


#include <opencv2/core/utility.hpp>
void zern_ga(const vector<Mat>& images, const vector<int>& labels, float err)
{
    vector <ga_params> tests;

    int nfeatures = 7;
    RNG rng(cv::getTickCount()+13);
    float e = FLT_MAX;
    int gen=0;
    while( (++gen<2000))
    //while((err < e) && (++gen<2000))
    {
        Zernike zern(10,0);
        vector <ga_params> feat;
        if ( gen < 99 )
        {
            for (int i=0; i<nfeatures; i++)
            {
                int a = rng.uniform(1,32);
                int p = rng.uniform(2,16);
                int q = rng.uniform(0,p+1);
                zern.add(a,p,q);
                feat.push_back(ga_params(a,p,q));
            }
        }
        else
        {
            for (int i=0; i<nfeatures/2; i++)
            {
                int a = rng.uniform(1,32);
                int p = rng.uniform(2,16);
                int q = rng.uniform(0,p+1);
                zern.add(a,p,q);
                feat.push_back(ga_params(a,p,q));
            }
            for (int i=0; i<nfeatures-nfeatures/2; i++)
            {
                int rang = (tests.size()/2);
                int id = rng.uniform(0,rang);
                ga_params f = tests[id];
                f.a += rng.uniform(-1,2);
                f.p += rng.uniform(-1,2);
                f.q += rng.uniform(0,2);
                f.a=abs(f.a);f.p=abs(f.p);f.q=abs(f.q);
                zern.add(f.a,f.p,f.q);
                feat.push_back(f);
            }
        }

        vector<Mat> train_set;
        vector<int> train_labels;
        int test_id = rng.uniform(0,labels.size());
        for( size_t i=0; i<images.size(); ++i)
        {
            if (i != test_id)
            {
                train_set.push_back(images[i]);
                train_labels.push_back(labels[i]);
            }
        }
        zern.train(train_set,train_labels);
        int l;
        double d;
        zern.predict(images[test_id],l,d);
        bool hit = (l == labels[test_id]);
        if ( hit && (d<e) )
        {
            for( size_t i=0; i<feat.size(); ++i)
            {
                bool found = false;
                for(size_t j=0; j<tests.size(); ++j)
                {
                    if ( (feat[i].a==tests[j].a) && (feat[i].p==tests[j].p) && (feat[i].q==tests[j].q) )
                    {
                        if (tests[j].score<d)
                            tests[j].score=d;
                        found = true;
                        break;
                    }
                }
                if ( ! found )
                {
                    feat[i].score = d;
                    tests.push_back(feat[i]);
                }
            }
            e = 1+ d * 4;
        }

        if ( tests.size() > 7 )
        {
            std::sort(tests.begin(),tests.end(),ga_sorter());
            cerr << gen << "\t" << hit << "\t"  << tests.size() <<" : " << e << " " << d << endl;
        }
        if ( tests.size() > 200 )
        {
            tests.erase(tests.end()-1);
            tests.erase(tests.end()-1);
            tests.erase(tests.end()-1);
            tests.erase(tests.end()-1);
        }
    }
    cerr << endl << gen << "\t" << tests.size() <<" : " << e << endl;
    for ( int i=0; i<10; i++)
    {
        cerr << ga_str(tests[i]) << ", ";
    }
}
