#include "ElasticParts.h"


#include "opencv2/opencv.hpp"
#include "opencv2/core/utility.hpp"
using namespace cv;

#include <vector>
using std::vector;

#include <iostream>
using std::cerr;
using std::endl;

void feature_img(const Mat_<uchar> &I, Mat &fI)
{
    I.convertTo(fI, CV_32F);

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

    //Laplacian(I,fI,CV_32F,3);
    //fI.convertTo(fI, CV_8U);

    //int nsec=90;
    //Mat s1, s2, s3(I.size(), CV_32F), s4, s5;
    //Sobel(I, s1, CV_32F, 1, 0);
    //Sobel(I, s2, CV_32F, 0, 1);
    //fastAtan2(s1.ptr<float>(0), s2.ptr<float>(0), s3.ptr<float>(0), I.total(), true);
    //fI = s3 ;/// (360/nsec);
    ////fI.convertTo(fI,CV_8U);
}

struct Part
{
    enum 
    { 
        num_scales=1,
        num_samples=1 
    };
    static const float scale[];

    Mat f[num_samples][num_scales];
    Point2f p;
    Size size;

    vector<Mat> samples; // only used for training

    Rect rect(Point2f _p) 
    { 
        return Rect(int(_p.x-size.width/2), int(_p.y-size.height/2), size.width, size.height); 
    }
    
    Part() {}
    Part(const Point2f &p, int w, int h) : p(p), size(w, h) {}

    void write(FileStorage &fs)
    {
        fs << "{:" ;
        fs << "p" << p;
        fs << "s" << size;
        fs << "f" << "[";
        for(int i=0; i<num_samples; i++)
        {
            for ( int lev=0; lev<num_scales; lev++)
            {
                fs << f[i][lev];
            }
        }
        fs << "]" << "}";
    }
    void read(const FileNode &fn) 
    {
        fn["p"] >> p;
        fn["s"] >> size;

        FileNode pnodes = fn["f"];
        FileNodeIterator it=pnodes.begin();
        for(int i=0; i<num_samples; i++)
        {
            for ( int lev=0; lev<num_scales; lev++)
            {
                (*it) >> f[i][lev];
                ++it;
            }
        }
    }

    void sample(const Mat &img)
    {
        Mat fI; 
        getRectSubPix(img, size, p, fI);
        //f[0][0] = fI;
        samples.push_back(fI);
    }

    void means()
    {
        Mat labels,centers,big;
        for (size_t p=0; p<samples.size(); p++)
            big.push_back(samples[p].reshape(1,1));
        if (big.type() != CV_32F)
            big.convertTo(big, CV_32F);

        kmeans(big,num_samples,labels,TermCriteria(),3,KMEANS_PP_CENTERS,centers);

        for(int i=0; i<num_samples; i++)
        {
            Mat patch = centers.row(i).reshape(1,size.height);
            //patch.convertTo(patch,CV_8U);
            for ( int lev=0; lev<num_scales; lev++)
            {
                resize(patch, f[i][lev], Size(), scale[lev], scale[lev]);
            }
        }
    }
    double distance(const Mat &a, const Mat &b) const
    {
        return norm(a,b);
    }

    double walk(const Mat &img, int level, float step, Point2f &np) const
    {
        double mDist=DBL_MAX;
        Point2f best(np);
        for (int r=-step; r<step; r++)
        {
            for (int c=-step; c<step; c++)
            {
                Point2f rs = Point2f(np.x+c, np.y+r);
                Mat patch;
                for (int j=0; j<num_samples; j++)
                {
                    getRectSubPix(img, f[j][0].size(), rs, patch);
                    double d=distance(patch, f[j][0]);
                    if (d<mDist) 
                    { 
                        mDist=d;
                        best=rs; 
                    }
                }
            }
        }
        np = best;
        return mDist;
    }
    //double walk(const Mat &img, int level, float step, Point2f &np) const
    //{
    //    double mDist=DBL_MAX;
    //    Point2f best(np);
    //    //float off[] = {0,0, -step,0, step,0, 0,-step, 0,step};

    //    float off[] = {0,0, -step,-step, 0,-step, step,-step, step,0, step,step, 0,step, -step,step, -step,0};
    //    for (int i=0; i<9; i++)
    //    {
    //        Point2f rs = Point2f(np.x*scale[level] + off[i*2], np.y*scale[level] + off[i*2+1]);
    //        Mat patch;
    //        for (int j=0; j<num_samples; j++)
    //        {
    //            getRectSubPix(img, f[j][level].size(), rs, patch);
    //            double d=distance(patch, f[j][level]);
    //            if (d<mDist) 
    //            { 
    //                mDist=d;
    //                best=rs; 
    //            }
    //        }
    //    }
    //    best.x/=scale[level];
    //    best.y/=scale[level];
    //    np = best;
    //    return mDist;
    //}
};

//const float Part::scale[] = {.25f, .5f, 1.0f}; 
const float Part::scale[] = {1.0f}; 
//const float Part::scale[] = {0.5f, 1.7f, 2.5f, 5.0f}; 
//const float Part::scale[] = {5.f, 10.f, 15.f}; 



struct ElasticPartsImpl : public ElasticParts
{
    vector<Part> parts;

    ElasticPartsImpl()
    {
        //Mat m=imread("lfw2mean.png",0);
        //parts.push_back(Part(m,Point2f(12,34),12,12));
        //parts.push_back(Part(m,Point2f(81,34),12,12));
        //parts.push_back(Part(m,Point2f(27,78),12,12));
        //parts.push_back(Part(m,Point2f(65,78),12,12));
    }
    virtual void addPart(cv::Point2f p, int w, int h) 
    {
        parts.push_back(Part(p,w,h));
    }
    virtual void setPoint(int i, const Point2f &p) 
    {
        parts[i].p = p;
    }

    virtual void sample(const Mat &img)
    {
        Mat fI;
        feature_img(img,fI);
        for (size_t i=0; i<parts.size(); i++)
            parts[i].sample(fI);
    }
    virtual void means()
    {
        for (size_t i=0; i<parts.size(); i++)
            parts[i].means();
    }
    double walk(const Mat &img, vector<KeyPoint> &kp, float step) const
    {
        Mat fea;
        feature_img(img, fea);
        for (size_t p=0; p<parts.size(); p++)
        {
            kp.push_back(KeyPoint(parts[p].p,8));
            parts[p].walk(fea, 0, step, kp[p].pt);
        }
        //for ( int lev=0; lev<Part::num_scales; lev++)
        //{
        //    Mat ms; 
        //    resize(img, ms, Size(), Part::scale[lev], Part::scale[lev]);
        //    for (size_t p=0; p<parts.size(); p++)
        //        parts[p].walk(ms, lev, step, kp[p].pt);
        //}
        return 1;
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

    virtual void getPoints(const Mat & img, vector<KeyPoint> &kp) const
    {
        walk(img,kp,3);
    }
};

cv::Ptr<ElasticParts> ElasticParts::create()
{
	return makePtr<ElasticPartsImpl>();
}
