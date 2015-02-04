

#include "opencv2/opencv.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/datasets/fr_lfw.hpp"

//#include <dlib/image_processing.h>
//#include <dlib/opencv/cv_image.h>

#include "elasticparts.h"

#if 0
 #include "../profile.h"
#else
 #define PROFILE ;
 #define PROFILEX(s) ;
#endif

#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <set>

using namespace std;
using namespace cv;

namespace Discriminant
{

static Mat feature_img(const Mat &I)
{
    Mat fI;

    //Mat s1, s2, s3(I.size(), CV_32F), s4, s5;
    //Sobel(I, s1, CV_32F, 1, 0);
    //Sobel(I, s2, CV_32F, 0, 1);
    //fastAtan2(s1.ptr<float>(0), s2.ptr<float>(0), s3.ptr<float>(0), I.total(), true);
    //fI = s3 ;/// (360/nsec);
    
    I.convertTo(fI, CV_32F);
    //fI += 1.0; log(fI,fI);
    return fI;
}


struct Part
{
    enum 
    { 
        w_base      = 24,
        i_size      = 90
    };

    Mat P;
    Point2f p;
    Size size;
   
    Part(const Point2f &p, int w, int h) : p(p),size(w,h) {}
    Part() {init(Point(), w_base, w_base);}
    Part(const Point2f &p) {init(p, w_base, w_base);}

    void init(const Point2f &p, int w, int h) 
    { 
        this->p = p; 
        this->size = Size(w, h);
        this->P = Mat::zeros(w, h, CV_32F);
    }

    Point detect(const Mat &img, double &q, int search=2) const
    {  
        Point2f p2 = p;
        float scale_x=float(img.cols)/i_size;
        float scale_y=float(img.rows)/i_size;
        if (scale_x != 1.0f)
        {
            p2.x = p.x*scale_x;
            p2.y = p.y*scale_y;
        }

        Mat fI; 
        getRectSubPix(img, Size(search*size.width, search*size.height), p2, fI);

        Mat R;
        matchTemplate(fI, P, R, cv::TM_CCOEFF_NORMED);
        //normalize(R, R, 0, 1, NORM_MINMAX);

        Point pM;
        minMaxLoc(R, 0, &q, 0, &pM);
        //imshow("T",R);
        return Point(int(p2.x) + pM.x - size.width/2, int(p2.y) + pM.y - size.height/2);
    }

    bool train(const vector<Mat> &images,
          const int search,
          const float lambda,
          const float mu_init,
          const int nsamples,
          const bool vis)
    {
        Size wsize(search*size.width, search*size.height);

        int N = images.size();
        int n = size.width*size.height;

        //compute desired response map
        int dx(wsize.width  - size.width);
        int dy(wsize.height - size.height);
        Mat_<float> F(dy, dx, 0.0f);
        for(int y=0; y<dy; y++)
        { 
            float vy = float(dy-1)/2 - y;
            for(int x=0; x<dx; x++)
            { 
                float vx = float(dx-1)/2 - x;
                F(y,x) = exp(-0.5f*(vx*vx+vy*vy));
            }
        }
        normalize(F,F,0,1,NORM_MINMAX);

        //allocate memory
        Mat I;//(wsize.height,wsize.width,CV_32F);
        Mat dP(size.height,size.width,CV_32F);
        Mat O = Mat::ones(size.height,size.width,CV_32F)/n;
        P = Mat::zeros(size.height,size.width,CV_32F);

        //optimise using stochastic gradient descent
        RNG rn(getTickCount()); 
        double mu=mu_init,step=pow(1e-8/mu_init,1.0/nsamples);
        for (int sample=0; sample<nsamples; sample++)
        { 
            int i = rn.uniform(0,N);
            getRectSubPix(images[i], wsize, p, I);
            dP = 0.0;
            for(int y = 0; y < dy; y++)
            {
                for(int x = 0; x < dx; x++)
                {
                    Mat Wi = I(Rect(x,y,size.width,size.height)).clone();
                    Wi -= Wi.dot(O); normalize(Wi,Wi);
                    dP += (F(y,x) - P.dot(Wi))*Wi;
                }
            }    
            P += mu*(dP - lambda*P); mu *= step;

            if (vis)
            {
                int t=5;
                if ((sample % 50 == 0 ))
                {
                    Mat R; matchTemplate(I,P,R,cv::TM_CCOEFF_NORMED);
                    Mat PP; normalize(P,PP,0,1,NORM_MINMAX);
                    normalize(dP,dP,0,1,NORM_MINMAX);
                    normalize(R,R,0,1,NORM_MINMAX);
                    imshow("P",PP); imshow("dP",dP); imshow("R",R); 
                    Mat I2;normalize(images[i], I2, 45);
                    rectangle(I2, Rect(int(p.x)-size.width/2, int(p.y)-size.height/2, size.width, size.height), Scalar(0));
                    Mat I3;normalize(I, I3, 16);
                    if (sample > nsamples-100)
                    {
                        double q=0;
                        Point p2 = detect(images[i],q);
                        rectangle(I2,Rect(int(p2.x)-size.width/2, int(p2.y)-size.height/2, size.width, size.height), Scalar(255));
                    }
                    imshow("I2",I2);
                    imshow("I",I3);
                }
                if(waitKey(t) == 27) return false;
            }
        }
        return true;
    }
    void write(FileStorage &fs)
    {
        fs << "{:";
        fs << "pos" << p;
        fs << "siz" << size;
        fs << "PM" << P;
        fs << "}";
    }
    void read(const FileNode &fn)
    {
        fn["pos"] >> p;
        fn["siz"] >> size;
        fn["PM"]  >> P;
    }
    //void draw(Mat &I2, const Point & pt, const Scalar col=Scalar(255)) const
    //{
    //    rectangle(I2,Rect(int(pt.x)-size.width/2, int(pt.y)-size.height/2, size.width, size.height), col);
    //}
    //void draw(Mat &I2, const Scalar col=Scalar(255)) const
    //{
    //    draw(I2,Point(p),col);
    //}
};

struct DiscriminantPartsImpl : public ElasticParts
{
    vector<Part> parts;

    DiscriminantPartsImpl()
    {
    }

    virtual double getPoints(const Mat & img, vector<Point> &kp) const
    {
        Mat I = feature_img(img);
        double Q=0;
        for (size_t k=0; k<parts.size(); k++)
        {
            double q=0;
            Point p = parts[k].detect(I, q);
            if (q < 0.6)
                p = parts[k].p;
            kp.push_back(p);
            Q += q;
        }
        //cerr << endl << endl;
        return Q / parts.size();
    }

    virtual void addPart(cv::Point2f p,int w, int h) 
    {
        parts.push_back(Part(p,w,h));
    }

    bool train( const vector<Mat> &imgs, int search, float lambda, float mu_init, int nsamples, bool visu  )
    {
        for (size_t k=0; k<parts.size(); k++)
        {
            if ( ! parts[k].train(imgs, search, lambda, mu_init, nsamples, visu) )
                return false;
        }
        return true;
    }

    virtual bool write(const String &fn)
    {
        FileStorage fs(fn,FileStorage::WRITE);
        if (! fs.isOpened()) 
        {
            cerr << "could not write to FileStorage : " << fn << "." << endl;
            return false;
        }

        fs << "parts" << "[";
        for (size_t p=0; p<parts.size(); p++)
        {
            parts[p].write(fs);
        }
        fs << "]";

        fs.release();
        return true;
    }
    
    virtual bool read(const String &fn)
    {
        FileStorage fs(fn,FileStorage::READ);
        if (! fs.isOpened()) 
        {
            cerr << "could not read from FileStorage : " << fn << "." << endl;
            return false;
        }

        FileNode pnodes = fs["parts"];
        for (FileNodeIterator it=pnodes.begin(); it!=pnodes.end(); ++it)
        {
            Part p;
            p.read(*it);
            parts.push_back(p);
        }

        fs.release();
        return true;
    }

    void draw (Mat & img, const vector<Point> &pt) const
    {
        normalize(img,img,42);
        for (size_t k=0; k<parts.size(); k++)
        {
            Point p = parts[k].p;
            //p.x *= img.cols/90;
            //p.y *= img.rows/90;
            circle(img,p,4,Scalar(.1));
            p = pt[k];
            //p.x *= img.cols/90;
            //p.y *= img.rows/90;
            circle(img,p,4,Scalar(.81));
        }
    }

};

} //Discriminant
cv::Ptr<ElasticParts> ElasticParts::createDiscriminative()
{
    return makePtr<Discriminant::DiscriminantPartsImpl>();
}

#ifdef ELASTIC_STANDALONE
using namespace Discriminant;
static void kp_manual(vector<KeyPoint> &kp, float scale=1.0f)
{
    kp.push_back(KeyPoint(15,19,3,-1,0,0,-1));    kp.push_back(KeyPoint(75,19,3,-1,0,0,-1));
    kp.push_back(KeyPoint(29,20,3,-1,0,0,-1));    kp.push_back(KeyPoint(61,20,3,-1,0,0,-1));
    kp.push_back(KeyPoint(36,24,3,-1,0,0,-1));    kp.push_back(KeyPoint(54,24,3,-1,0,0,-1));
    kp.push_back(KeyPoint(38,35,3,-1,0,0,-1));    kp.push_back(KeyPoint(52,35,3,-1,0,0,-1));
    kp.push_back(KeyPoint(30,39,3,-1,0,0,-1));    kp.push_back(KeyPoint(60,39,3,-1,0,0,-1));
    kp.push_back(KeyPoint(19,39,3,-1,0,0,-1));    kp.push_back(KeyPoint(71,39,3,-1,0,0,-1));
    kp.push_back(KeyPoint(8 ,38,3,-1,0,0,-1));    kp.push_back(KeyPoint(82,38,3,-1,0,0,-1));
    kp.push_back(KeyPoint(40,64,3,-1,0,0,-1));    kp.push_back(KeyPoint(50,64,3,-1,0,0,-1));
    kp.push_back(KeyPoint(31,75,3,-1,0,0,-1));    kp.push_back(KeyPoint(59,75,3,-1,0,0,-1));
    kp.push_back(KeyPoint(32,49,3,-1,0,0,-1));    kp.push_back(KeyPoint(59,49,3,-1,0,0,-1));
    if (scale!= 1.0f)
    {
        for (size_t i=0; i<kp.size(); i++)
        {
            kp[i].pt.x *= scale;
            kp[i].pt.y *= scale;
        }
    }
}

int main()
{
    const float lambda = 1e-6f;       //regularization weight
    const float mu_init = 1e-3f;      //initial stoch-grad step size
    const int nsamples = 500;         //number of stoch-grad samples
    const int nimages = 4000;          //number of train images
    const int search = 2;             //search radius
    const bool train = 1;
    const bool optimize = 0;

    RNG rn(getTickCount()); 


    std::vector<String> fns;
    glob("lfw-deepfunneled/*.jpg",fns,true);
    if ( fns.empty())
        return 0;
    vector<Mat> images;
    for (size_t i=0; i<nimages; i++)
    {
        Mat im = imread(fns[i],0);
        Mat i2 = feature_img(im(Rect(80,80,90,90)));
        //Mat i2 = feature_img(im);
        images.push_back(i2);
    }

    vector<KeyPoint> kp;
    kp_manual(kp, float(images[0].rows)/90);


    DiscriminantPartsImpl el;
    if (train)
    {
        namedWindow("P",0);namedWindow("dP",0);namedWindow("R",0);

        for (size_t i=0; i<kp.size(); i++)
        {
            el.addPart(kp[i].pt,24,24);
        }

        for (size_t i=0; i<1; i++)
        {
            if ( ! el.train(images,search,lambda,mu_init,nsamples,true) )
                return false;
        }
        el.write("data/disc.xml.gz");
    }
    bool ok = el.read("data/disc.xml.gz");
    if (! ok) return 1;
    
    namedWindow("T",0);
    
    vector<double> qr(el.parts.size(),0);

    for (size_t i=0; i<500; i++)
    {
        size_t r = rn.uniform(0,images.size());
        Mat img = images[r].clone();

        double Q=0;
        vector<Point> pt;
        for (size_t k=0; k<el.parts.size(); k++)
        {
            double q=0;
            Point p = el.parts[k].detect(img, q);
            if (q < 0.6)
                p = el.parts[k].p;
            pt.push_back(p);
            qr[k] += q;
        }
        double q = el.getPoints(img,pt);
        el.draw(img,pt);
        imshow("T",img);
        if (waitKey(optimize ? 2 : 2000)==27)
            break;
    }
    cerr << endl;
    cerr << endl;
    double Q=0;
    for (size_t i=0; i<qr.size(); i++)
    {
        qr[i] /= 500;
        cerr << qr[i] << " " << el.parts[i].p << endl;
        Q += qr[i];
    }
    cerr << "mean " << Q/el.parts.size() << endl;

    if ( ! optimize)
        return 0;


    int tries[20] = {0};
    for (int z=0; z<300; z++)
    {
        double worst=999999999;
        int wi;
        for (size_t i=0; i<qr.size(); i++)
        {
            if (qr[i] < worst && tries[i] < 30)
            {
                wi=i;
                worst=qr[i];
            }
        }

        cerr << "optimize " << wi << " " << qr[wi] << " " << el.parts[wi].p  << el.parts[wi].size;
        Point oldp = el.parts[wi].p;
        Point olds = el.parts[wi].size;
        Point nn[16] = 
        {
            Point(-1,-1), Point(0,-1), Point(1,-1),Point(1,0),Point(1,1),Point(0,1),Point(-1,1),Point(-1,0),
            Point(-2,-2), Point(0,-2), Point(2,-2),Point(2,0),Point(2,2),Point(0,2),Point(-2,2),Point(-2,0)
        };
        int n = rn.uniform(0,16);
        int z = rn.uniform(0,2);
        if (z==0)
        {
            el.parts[wi].p.x = oldp.x + nn[n].x;
            el.parts[wi].p.y = oldp.y + nn[n].y;
        }
        if (z==1)
        {
            el.parts[wi].size.width += nn[n].x;
            el.parts[wi].size.height += nn[n].y;
        }
        el.parts[wi].train(images, search, lambda, mu_init, nsamples, false);
        double Q = 0;
        size_t ntests=1000;
        for (size_t i=0; i<1000; i++)
        {
            size_t r = rn.uniform(0,images.size());
            Mat img = images[r];
            double q=0;
            el.parts[wi].detect(img,q);
            Q += q;
        }
        Q /= ntests;
        cerr << " -> " << (Q>worst?'+':'-') << " " << Q << " "<<  " " << el.parts[wi].p  << el.parts[wi].size << endl;
        if (Q < worst )
        {
            el.parts[wi].p    = oldp;
            el.parts[wi].size =  olds;
            tries[wi] ++;
        } else {
            qr[wi] = Q;
        }
    }
    el.write("data/disc_opt.xml.gz");

    qr = vector<double> (el.parts.size(),0);
    for (size_t i=0; i<500; i++)
    {
        size_t r = rn.uniform(0,images.size());
        Mat img = images[r].clone();

        vector<Point> pt;
        for (size_t k=0; k<el.parts.size(); k++)
        {
            double q=0;
            Point p = el.parts[k].detect(img, q);
            if (q < 0.6)
                p = el.parts[k].p;
            pt.push_back(p);
            qr[k] += q;
        }
        el.draw(img,pt);
        imshow("T",img);
        if (waitKey(2)==27)
            break;
    }
    cerr << endl;
    cerr << endl;
    Q=0;
    for (size_t i=0; i<qr.size(); i++)
    {
        qr[i] /= 500;
        cerr << qr[i] << " " << el.parts[i].p << endl;
        Q += qr[i];
    }
    cerr << "mean " << Q/el.parts.size() << endl;
    return 0;
}

#endif ELASTIC_STANDALONE