#include "elasticparts.h"


#include "opencv2/opencv.hpp"
#include "opencv2/core/utility.hpp"
using namespace cv;

#include <vector>
using std::vector;

#include <iostream>
using std::cerr;
using std::endl;


namespace Generative
{
void feature_img(const Mat_<uchar> &I, Mat &fI)
{
    //fI.convertTo(fI, CV_32F);
    //fI += 1.0; log(fI,fI);

    //Mat_<float> his=Mat_<float>::zeros(1,16*4);
    //const int m=2;
    //for (int r=m; r<I.rows-m; r++)
    //{
    //    for (int c=m; c<I.cols-m; c++)
    //    {
    //        uchar v = 0;
    //        v |= ((I(r  ,c+1) - I(r+2,c+2)) > (I(r  ,c-1) - I(r-2,c-2))) * 1;
    //        v |= ((I(r+1,c+1) - I(r+2,c  )) > (I(r-1,c-1) - I(r-2,c  ))) * 2;
    //        v |= ((I(r+1,c  ) - I(r+2,c-2)) > (I(r-1,c  ) - I(r-2,c+2))) * 4;
    //        v |= ((I(r+1,c-1) - I(r  ,c-2)) > (I(r-1,c+1) - I(r  ,c+2))) * 8;
    //        int or = (r/(I.rows/2));
    //        int oc = (c/(I.cols/2));
    //        int off = 16 * (2*or+oc); 
    //        his(off+v)++;
    //    }
    //}
    //fI = his;


    //Mat d;
    //I.convertTo(d,CV_32F, 1.0/255);
    //dft(d,d);
    //fI = d(Rect(1,1,I.cols/2,I.rows/2));
    //dft(fI,fI,DCT_INVERSE);
    //fI.convertTo(fI,CV_8U);

    //Laplacian(I,fI,CV_32F,3,10);
    //fI.convertTo(fI, CV_8U);

    ////int nsec=90;
    Mat s1, s2, s3(I.size(), CV_32F), s4, s5;
    Sobel(I, s1, CV_32F, 1, 0);
    Sobel(I, s2, CV_32F, 0, 1);
    fastAtan2(s1.ptr<float>(0), s2.ptr<float>(0), s3.ptr<float>(0), I.total(), true);
    fI = s3 ;/// (360/nsec);
    //////fI.convertTo(fI,CV_8U);
}

struct Part
{
    enum 
    { 
        //num_samples = 1,
        scale       = 2,
        step        = 3*scale,
        w_base      = 12,
        w           = w_base*scale,
        h           = w_base*scale,
    };

    Mat f,fAll;
    Point2f p;
    Size size;

    int num_samples;
   
    Part() {}
    Part(const Point2f &p) : p(p), size(w, h), num_samples(0) {}
    Part(const Point2f &p, const Mat &img) 
        : p(p), size(w, h), num_samples(0)
    {
        getRectSubPix(img, size, p, f);
    }

    void write(FileStorage &fs)
    {
        fs << "{:" ;
        fs << "p" << p;
        fs << "f" << f;
        fs << "}";
    }
    void read(const FileNode &fn)
    {
        fn["p"] >> p;
        fn["f"] >> f;
    }

    double distance(const Mat &a, const Mat &b) const
    {
        return norm(a,b);
    }

    double walk(const Mat &img, Point2f &np) const
    {
        double mDist=DBL_MAX;
        Point2f best(np);
        for (int r=-step; r<step; r++)
        {
            for (int c=-step; c<step; c++)
            {
                Point2f rs = Point2f(np.x+c, np.y+r);
                Mat patch;
                getRectSubPix(img, f.size(), rs, patch);
                double d=distance(patch, f);
                if (d<mDist) 
                { 
                    mDist=d;
                    best=rs; 
                }
            }
        }
        np = best;
        return mDist;
    }
    void sample(const Mat &img)
    {
        walk(img,p);

        Mat fI; 
        getRectSubPix(img, size, p, fI);
        if (fAll.empty())
            fAll = Mat(size, CV_32F, Scalar(0));
        accumulate(fI,fAll);
        num_samples ++;
    }

    void means()
    {
        f  = fAll / num_samples;
        fAll.release();
    }
    
};




struct ElasticPartsImpl : public ElasticParts
{
    vector<Part> parts;

    ElasticPartsImpl()
    {
    }
    virtual void addPart(cv::Point2f p) 
    {
        parts.push_back(Part(p*Part::scale));
    }
    virtual void addParts(const vector<cv::Point2f> &kp, const Mat &img) 
    {
        Mat ims,fI;
        resize(img,ims,Size(),Part::scale,Part::scale);
        feature_img(ims,fI);
        for (size_t i=0; i<kp.size(); i++)
            parts.push_back(Part(kp[i]*Part::scale,fI));
    }
    virtual void setPoint(int i, const Point2f &p) 
    {
        parts[i].p = p*Part::scale;
    }

    virtual void sample(const Mat &img)
    {
        Mat ims, fI;
        resize(img,ims,Size(),Part::scale,Part::scale);
        feature_img(ims,fI);
        for (size_t i=0; i<parts.size(); i++)
            parts[i].sample(fI);
    }
    virtual void means()
    {
        for (size_t i=0; i<parts.size(); i++)
            parts[i].means();
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

    virtual double getPoints(const Mat & img, vector<Point> &kp) const
    {
        Mat ims, fI;
        resize(img,ims,Size(),Part::scale,Part::scale);
        feature_img(ims,fI);
        for (size_t k=0; k<parts.size(); k++)
        {
            Point2f p = parts[k].p;
            parts[k].walk(fI, p);
            p /= Part::scale;
            kp.push_back(p);
        }
        return 0;
    }
};

} // Generative

cv::Ptr<ElasticParts> ElasticParts::createGenerative()
{
    return makePtr<Generative::ElasticPartsImpl>();
}
