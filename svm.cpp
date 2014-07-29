//#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include "fr.h"

using namespace std;
using namespace cv;

bool use_uni2=false;

struct Svm : public FaceRecognizer
{

	CvSVM svm;
    CvSVMParams param;
    int preprocessing; // 0 none(pixels), 1 lbph_u, 2 humoments, 3 hog

    Svm(int pre = 0, bool use_uni2=false, double degree = 0.5,double gamma = 0.8,double coef0 = 0,double C = 0.99, double nu = 0.2, double p = 0.5) 
        : preprocessing(pre)
    {
	    param.kernel_type = CvSVM :: POLY ; // CvSVM :: RBF , CvSVM :: LINEAR...
    	param.degree = degree; // for poly
	    param.gamma = gamma; // for poly / rbf / sigmoid
	    param.coef0 = coef0; // for poly / sigmoid
	    param.C = C; // for CV_SVM_C_SVC , CV_SVM_EPS_SVR and CV_SVM_NU_SVR
	    param.nu = nu; // for CV_SVM_NU_SVC , CV_SVM_ONE_CLASS , and CV_SVM_NU_SVR
    	param.p = p; // for CV_SVM_EPS_SVR
    	param.class_weights = NULL ; // for CV_SVM_C_SVC
    	param.term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS ;
    	param.term_crit.max_iter = 1000;
    	param.term_crit.epsilon = 1e-6;
        param.svm_type = CvSVM::NU_SVC;

        ::use_uni2 = use_uni2;
        //cerr << ::use_uni2;
    }

private:

    // square radius 1
    static uchar lbp(const Mat_<uchar> & img, int x, int y)
    {
        uchar v = 0;
        uchar c = img(y,x);
        v += (img(y-1,x  ) > c) << 0;
        v += (img(y-1,x+1) > c) << 1;
        v += (img(y  ,x+1) > c) << 2;
        v += (img(y+1,x+1) > c) << 3;
        v += (img(y+1,x  ) > c) << 4;
        v += (img(y+1,x-1) > c) << 5;
        v += (img(y  ,x-1) > c) << 6;
        v += (img(y-1,x-1) > c) << 7;
        return v;
    }

    // circle radius 2 , linear interpolated neighbours
    static uchar lbp2(const Mat_<uchar> & img, int x, int y)
    {
        uchar v = 0;
        int c = img(y,x);
        v += (img(y-2,x  ) > c) << 0;
        v += ((img(y-1,x+1)+img(y-1,x+2)+img(y-2,x+1)+img(y-2,x+2)) > c*4) << 1;
        v += (img(y  ,x+2) > c) << 2;
        v += ((img(y+1,x+1)+img(y+1,x+2)+img(y+2,x+1)+img(y+2,x+2)) > c*4) << 3;
        v += (img(y+2,x  ) > c) << 4;
        v += ((img(y+1,x-1)+img(y+1,x-2)+img(y+2,x-1)+img(y+2,x-2)) > c*4) << 5;
        v += (img(y  ,x-2) > c) << 6;
        v += ((img(y-2,x-1)+img(y-2,x-2)+img(y-1,x-1)+img(y-1,x-2)) > c*4) << 7;
        return v;
    }

    static Mat lbp(const Mat & z)
    {
        static int uniform[256] = {
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
        int uni2[256] = {
            0,1,2,3,4,1,5,6,7,1,2,3,8,8,9,10,11,1,2,3,4,1,5,6,12,12,12,15,13,13,14,15,16,1,2,
            3,4,1,5,6,7,1,2,3,8,8,9,10,17,17,17,3,17,17,20,21,18,18,18,21,19,19,20,21,22,1,
            2,3,4,1,5,6,7,1,2,3,8,8,9,10,11,1,2,3,4,58,5,6,12,12,12,15,13,13,14,15,23,23,23,
            44,23,23,5,45,23,23,23,28,26,26,27,28,24,24,24,49,24,24,27,28,25,25,25,28,26,26,
            27,28,29,30,2,31,4,30,5,32,7,30,2,31,8,33,9,33,11,30,2,31,4,30,5,32,12,12,12,34,
            13,34,14,34,16,30,2,31,4,30,5,32,7,30,58,31,8,33,9,33,17,48,17,49,17,35,20,35,18,
            52,18,35,19,35,20,35,36,37,36,38,36,37,39,39,36,37,36,38,8,40,40,40,36,37,36,38,
            36,37,39,39,51,52,41,41,54,41,41,41,42,43,42,44,42,43,45,45,42,43,42,44,54,46,46,
            46,47,48,47,49,47,48,50,50,51,52,51,53,54,55,56,57
        };
        int uni3[256] = {
            0,1,2,3,4,1,5,6,7,1,2,3,8,8,9,10,11,1,2,3,4,58,5,6,12,12,12,15,13,13,14,15,16,1,
            2,3,4,58,5,6,7,58,58,58,8,58,9,10,17,17,17,58,17,58,20,21,18,18,18,21,19,19,20,21,
            22,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,
            58,14,15,23,23,23,44,23,58,58,45,23,58,58,58,26,58,27,28,24,24,24,49,24,58,27,28,
            25,25,25,28,26,26,27,28,29,30,2,31,4,30,5,32,7,30,58,31,8,33,9,33,11,30,58,31,58,
            58,58,32,12,58,58,34,13,34,14,34,16,30,58,31,58,58,58,32,58,58,58,58,58,58,58,
            33,17,48,58,49,58,58,58,35,18,52,58,35,19,35,20,35,36,37,36,38,36,37,39,39,36,37,
            58,38,58,40,40,40,36,37,58,38,58,58,58,39,51,52,58,41,54,41,41,41,42,43,42,44,42,
            43,45,45,42,43,58,44,54,46,46,46,47,48,47,49,47,48,50,50,51,52,51,53,54,55,56,57
        };
        int uni4[256] = {
  //0,1,2,3,4,4,5,6,7,7,7,3,8,8,9,10,11,11,11,3,11,11,5,6,12,12,12,12,13,13,14,15,16,16,16,3,16,16,5,6,16,16,16,3,8,8,9,10,17,17,17,17,17,17,17,6,18,18,18,18,19,19,20,21,22,22,22,3,22,22,5,6,22,22,22,3,8,8,9,10,22,22,22,3,22,58,5,6,12,12,12,58,13,13,14,15,23,23,23,23,23,23,23,6,23,23,23,58,23,58,9,10,24,24,24,24,24,24,24,58,25,25,25,25,26,26,27,28,29,30,29,31,29,30,5,32,29,30,29,31,8,30,9,33,29,30,29,31,29,30,5,32,12,30,12,31,13,13,14,34,29,30,29,31,29,30,5,32,29,30,58,31,8,58,9,33,17,30,17,31,17,58,58,32,18,18,18,58,19,19,20,35,36,37,36,38,36,37,36,39,36,37,36,38,36,37,9,40,36,37,36,38,36,37,58,39,36,37,58,38,13,58,14,41,42,43,42,44,42,43,42,45,42,43,42,44,42,43,58,46,47,48,47,49,47,48,47,50,51,52,51,53,54,55,56,57
  //0,1,2,3,4,4,5,6,7,7,7,7,8,7,9,10,11,11,11,11,11,11,11,5,12,11,11,12,13,12,14,15,16,16,16,16,16,16,16,5,16,16,16,3,16,8,8,9,17,16,16,17,16,17,17,6,18,17,17,18,19,18,20,21,22,22,22,22,22,22,22,5,22,22,22,3,22,8,8,9,22,22,22,3,22,58,5,6,22,12,12,58,12,13,13,14,23,22,22,23,22,23,23,6,22,23,23,58,23,58,9,10,24,23,23,24,23,24,24,58,25,24,24,25,26,25,27,28,29,30,29,31,29,29,29,32,29,29,29,30,29,30,8,33,29,29,29,30,29,30,5,31,29,30,12,31,12,13,13,34,29,29,29,30,29,30,5,31,29,30,58,31,8,58,9,32,29,30,17,31,17,58,58,32,17,18,18,58,18,19,19,35,36,37,29,38,29,36,36,39,29,36,36,37,36,37,9,40,29,36,36,37,36,37,58,38,36,37,58,38,13,58,14,41,42,43,36,44,36,42,42,45,36,42,42,43,42,43,58,46,47,48,42,49,42,47,47,50,51,52,47,53,54,55,56,57
0,1,2,3,4,6,5,6,7,58,9,10,8,10,9,10,11,58,58,58,13,58,14,15,12,58,14,15,13,15,14,
15,16,58,58,58,58,58,58,58,18,58,58,58,19,58,20,21,17,58,58,58,19,58,20,21,18,58,
20,21,19,21,20,21,22,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,24,58,58,58,
58,58,58,58,25,58,58,58,26,58,27,41,23,43,58,44,58,58,58,45,25,58,58,58,26,58,27,
46,24,48,58,49,26,58,27,50,25,52,27,53,26,55,27,28,29,30,31,31,58,32,32,32,58,58,
58,33,58,33,33,33,58,58,58,58,58,58,58,34,58,58,58,34,58,34,34,34,42,43,58,44,
58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,58,58,58,50,51,52,58,53,54,55,56,
35,36,37,38,38,58,39,39,39,58,58,58,40,58,40,40,40,47,48,58,49,58,58,58,50,51,52,
58,53,54,55,56,41,42,43,44,44,58,45,45,45,51,52,58,53,54,55,56,46,47,48,49,49,
54,55,56,50,51,52,56,53,54,55,56,57
        };

int uni_r[256] = 
{
    16,0,16,16,16,16,1,16,2,16,16,16,16,16,16,16,3,16,16,16,16,16,16,16,16,16,16,16,
    16,16,16,4,5,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,6,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,7,16,8,16,16,16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,9,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
    16,16,16,16,16,10,16,16,16,11,16,16,16,16,16,16,16,12,16,16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,16,16,13,16,16,16,16,16,16,16,16,16,16,16,16,16,14,16,15,
    16,16,16,16,16,16,16,16,16,16,16,16
};
        int *uni = use_uni2 ? uni4 : uniform;
        Mat h  = Mat::zeros(1,60*8*8,CV_32F);
        int sw = (z.cols)/7;
        int sh = (z.rows)/7;
        const int m=2;
        for ( int r=m; r<z.rows-m; r++ )
        {
            for ( int c=m; c<z.cols-m; c++ )
            {
                int i = r/sh;
                int j = c/sw;
                uchar v = lbp2(z,r,c);
                //v = uniform[v];
                v = uni[v];
                h.at<float>( 60*(i*8+j) + v ) += 1;
            }
        }        
        normalize(h,h);
        return h;
    }
    static void mom(const Mat & z, Mat & feature, int i, int j, int w, int h)
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
        for ( int i=0; i<8; i++ )
        {
            for ( int j=0; j<8; j++ )
            {
                mom(z,mo,i,j,sw,sh);
            }
        }        
        mo.convertTo(mo,CV_32F);
        normalize(mo,mo);
        return mo;
    }

    static Mat hog(const Mat & z)
    {
        vector<float> descriptors;
        vector< Point > location;
        cv::HOGDescriptor hog( Size(8,8),Size(8,8), Size(8,8), Size(8,8), 9, 1, -1,
                HOGDescriptor::L2Hys, 0.2, true, HOGDescriptor::DEFAULT_NLEVELS );


        hog.compute(z, descriptors, Size( 8, 8 ), Size( 0, 0 ), location);
        Mat hist;
        normalize(Mat(descriptors),hist);

        //int sw = 16; //(z.cols)/8;
        //int sh = 16; //(z.rows)/8;
        //for ( int i=0; i<z.rows-sh; i+=sh )
        //{
        //    CV_Assert(i+sh<z.rows);
        //    for ( int j=0; j<z.cols-sw; j+=sw )
        //    {
        //        CV_Assert(j+sw<z.cols);
        //        Mat m = z(Rect(j,i,sw,sh));
        //        //  cerr << "hog.compute " << i << " " << j << " " << m.rows << " " << m.cols << endl;
        //        hog.compute(m, descriptors, Size( 8, 8 ), Size( 0, 0 ), location);
        //        hist.push_back(Mat(descriptors).clone());
        //    }
        //}        
        //normalize(hist,hist);
        return hist;
    }


    Mat preproc(const Mat & z) const
    {
        switch ( preprocessing )
        { 
            case 1: return lbp(z);
            case 2: return mom(z);
            case 3: return hog(z);
        }
        Mat m = z.reshape(1,1);
        m.convertTo(m,CV_32F,1.0/255);
        return m;
    }

public:

    virtual void train(InputArray src, InputArray lbls)    
    {
        vector<Mat> images;
        src.getMatVector(images);
        vector<int> labels = lbls.getMat();
        Mat trainData;
        for ( size_t i=0; i<images.size(); i++ )
        {
            Mat m = preproc(images[i]);
            trainData.push_back(m);
        }
        trainData = trainData.reshape(1,labels.size());
        svm.train( trainData , Mat(labels) , cv::Mat() , cv::Mat() , param );
    }

    virtual void predict(InputArray src, int& label, double & minDist) const    
    {
        Mat m = src.getMat();
        Mat q = preproc(m);
        float r = svm.predict ( q );
        label = (int)floor(r+0.5);
        //cerr << " " << label << " " << r << endl;
    }
    virtual int predict(InputArray src) const 
    {
        int pred=-1;
        double conf=-1;
        predict(src,pred,conf);
        return pred;
    }
    virtual void update(InputArrayOfArrays src, InputArray labels) {train(src,labels);}
    virtual void save(const std::string& filename) const    {}
    virtual void save(FileStorage& fs) const    {}
    virtual void load(const std::string& filename)    {}
    virtual void load(const FileStorage& fs)    {}


//    //static void xbreed(CvSVMParams & p, const CvSVMParams & q, double u=0.5)
//    //{
//    //    p.degree = p.degree * u + q.degree * (1.0-u);
//    //    p.gamma  = p.gamma  * u + q.gamma  * (1.0-u);
//    //    p.coef0  = p.coef0  * u + q.coef0  * (1.0-u);
//    //    p.C  = p.C  * u + q.C  * (1.0-u);
//    //    p.nu = p.nu * u + q.nu * (1.0-u);
//    //    p.p  = p.p  * u + q.p  * (1.0-u);
//    //}
//
//    static string str(const CvSVMParams & p)
//    {
//        return cv::format("[%3.3f %3.3f %3.3f %3.3f %3.3f %3.3f]", p.degree, p.gamma, p.coef0, p.C, p.nu, p.p);
//    }
//    static void clamp(double &z)
//    {
//        if ( z > 1.0 ) z=1.0;
//        if ( z < 0.0001 ) z=0.0001;
//    }
//    static void mutate( CvSVMParams & p,RNG & rng)
//    {
//        double mutation = 2.0;
//        if ( rng.uniform(0,6)>4) p.degree += rng.uniform(-mutation,mutation), clamp(p.degree);
//        if ( rng.uniform(0,6)>4) p.gamma += rng.uniform(-mutation,mutation), clamp(p.gamma);
//        if ( rng.uniform(0,6)>4) p.coef0 += rng.uniform(-mutation,mutation), clamp(p.coef0);
//        if ( rng.uniform(0,6)>4) p.C += rng.uniform(-mutation,mutation), clamp(p.C);
//        if ( rng.uniform(0,6)>4) p.nu += rng.uniform(-mutation,mutation), (p.nu=p.nu<0.0001?0.0001:p.nu>=1.0?p.nu=0.99:p.nu);
//        if ( rng.uniform(0,6)>4) p.p += rng.uniform(-mutation,mutation), clamp(p.p);
//    }
//    
//    static CvSVMParams train_ga(const vector<Mat>& images, const vector<int>& labels, float err)
//    {
//        //RNG rng(13);
//        RNG rng(cv::getTickCount()+13);
//        //float e = FLT_MAX;
//        int gen=0;
//        CvSVMParams p;
//        p.degree = 0.5;
//        p.gamma = 0.8;
//        p.coef0 = 0;
//        p.C = 0.99;
//        p.nu = 0.2;
//        p.p = 0.5;
//        int ngenes=5;
//        vector<CvSVMParams> genes(ngenes,p);
//        vector<double> score(ngenes,0.0);
//        while( (++gen<100))
//        //while((err < e) && (++gen<2000))
//        {
//            double worst     = 1.0;
//            double best      = 0.0;
//            int    worst_gen = 0;
//            //int    best_gen  = 0;
//            for ( size_t i=0;i<genes.size(); i++)
//            {
//                if ( score[i]>best)
//                {
//                    best = score[i];
//                    //best_gen = i;
//                } else
//                if ( (score[i]<worst) /*&& (score[i]>0)*/)
//                {
//                    worst = score[i];
//                    worst_gen = i;
//                }
//            }
//            //svm_xbreed( genes[worst_gen], genes[best_gen], 0.2 );
//            if ( 1.0 - best < err )
//                break;
//            double ad = abs(best-worst);
//
//            //int sel_gen = rng.uniform(0,ngenes);
//            //int sel_gen = gen%ngenes;
//            int sel_gen = worst_gen;
//            if ( rng.uniform(0,2)) 
//                sel_gen = gen%ngenes;;
////            svm_xbreed( genes[sel_gen],genes[best_gen], 0.2 );
//            if ( sel_gen == best )
//                continue;
//
//            CvSVMParams pr(genes[sel_gen]);
//            mutate(pr,rng);
//
//            vector<Mat> train_set,test_set;
//            vector<int> train_labels,test_labels;
//            //int test_skip = 0;//rng.uniform(0,6);
//            int test_split = 6;// = rng.uniform(0,labels.size());
//            int test_off = rng.uniform(0,5);
//            for( size_t i=test_off; i<images.size(); ++i)
//            {
//                //if ( (test_skip >0) && (i % test_skip==0) )
//                //    continue;
//
//                if (i % test_split==0)
//                {
//                    test_set.push_back(images[i]);
//                    test_labels.push_back(labels[i]);
//                }
//                else
//                {
//                    train_set.push_back(images[i]);
//                    train_labels.push_back(labels[i]);
//                }
//            }
//            try {
//
//                Svm svm_loc(pr.degree,pr.gamma,pr.coef0,pr.C,pr.nu,pr.p);
//                svm_loc.train_im(train_set,train_labels);
//
//                int hits = 0;
//                for ( size_t i=0;i<test_set.size(); i++)
//                {
//                    int l; double d;
//                    svm_loc.predict(test_set[i],l,d);
//                    hits += (l == test_labels[i]);
//                }
//                double sc = double(hits)/test_set.size(); 
//                char plus='-';
//                if ( sc > score[sel_gen] )
//                {
//                    genes[sel_gen] = pr;
//                    score[sel_gen] = sc;
//                    plus='+';
//                }
//                cerr << format("%-2d %2d %c  %2.7f  %2.7f  %2.7f  ",gen,sel_gen, plus, score[sel_gen],best,ad) << str(pr) << endl;
//            } catch( Exception &e ) {
//                //cerr << e.what() << endl;
//            }
//        }
//        double best = 0.0;
//        int    best_gen = 0;
//        for ( size_t i=0;i<genes.size(); i++)
//        {
//            if ( score[i]>best)
//            {
//                best = score[i];
//                best_gen = i;
//            } 
//            cerr << format("%2.6f ",score[i]) << str(genes[i]) << endl;
//        }
//
//        CvSVMParams prm = genes[best_gen];
//        cerr << format("choice %4d %2d  %2.8f  ",gen, best_gen, best) << str(prm) << endl;
//        return prm;
//    }
};

Ptr<FaceRecognizer> createSvmFaceRecognizer(int pre, bool use_uni2=false)
{
    return makePtr<Svm>(pre,use_uni2);
}

//
//#include <opencv2/core/utility.hpp>
//void svm_ga(const vector<Mat>& images, const vector<int>& labels, float err)
//{
//    CvSVMParams p = Svm::train_ga(images,labels,0.002f);
//    cerr << Svm::str(p) << endl;
//}
