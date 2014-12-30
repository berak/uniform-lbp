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
    // equalizeHist(I,fI);
    //fI.convertTo(fI, CV_32F);

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

    //int nsec=90;
    Mat s1, s2, s3(I.size(), CV_32F), s4, s5;
    Sobel(I, s1, CV_32F, 1, 0);
    Sobel(I, s2, CV_32F, 0, 1);
    fastAtan2(s1.ptr<float>(0), s2.ptr<float>(0), s3.ptr<float>(0), I.total(), true);
    fI = s3 ;/// (360/nsec);
    ////fI.convertTo(fI,CV_8U);
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

    //Mat f[num_samples];
    Mat f;
    Point2f p;
    Size size;

    int num_samples;
    //vector<Mat> samples; // only used for training
   
    Part() {}
    Part(const Point2f &p) : p(p), size(w, h), num_samples(0) {}

    void write(FileStorage &fs)
    {
        fs << "{:" ;
        fs << "p" << p;
        fs << "f" << f;
        fs << "}";
        //fs << "f" << "[";
        //for(int i=0; i<num_samples; i++)
        //{
        //    fs << f[i];
        //}
        //fs << "]" << "}";
    }
    void read(const FileNode &fn)
    {
        fn["p"] >> p;
        fn["f"] >> f;
        //FileNode pnodes = fn["f"];
        //FileNodeIterator it=pnodes.begin();
        //for(int i=0; i<num_samples; i++)
        //{
        //    (*it) >> f[i];
        //    ++it;
        //}
    }

    void sample(const Mat &img)
    {
        Mat fI; 
        getRectSubPix(img, size, p, fI);
        if (f.empty())
            f = Mat(size, CV_32F, Scalar(0));
        accumulate(fI,f);
        num_samples ++;
        //samples.push_back(fI);
        //f[0] = fI;
    }

    void means()
    {
        //Mat labels,centers,big;
        //for (size_t p=0; p<samples.size(); p++)
        //    big.push_back(samples[p].reshape(1,1));
        //if (big.type() != CV_32F)
        //    big.convertTo(big, CV_32F);

        //kmeans(big,num_samples,labels,TermCriteria(),3,KMEANS_PP_CENTERS,centers);

        //for(int i=0; i<num_samples; i++)
        //    f[i] = centers.row(i).reshape(1,size.height).clone();
        f /= num_samples;
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
                //for (int j=0; j<num_samples; j++)
                {
                    getRectSubPix(img, f.size(), rs, patch);
                    //getRectSubPix(img, f[j].size(), rs, patch);
                    //double d=distance(patch, f[j]);
                    double d=distance(patch, f);
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
};




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
    virtual void addPart(cv::Point2f p) 
    {
        parts.push_back(Part(p*Part::scale));
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

    virtual void getPoints(const Mat & img, vector<KeyPoint> &kp) const
    {
        Mat ims, fI;
        resize(img,ims,Size(),Part::scale,Part::scale);
        feature_img(ims,fI);
        for (size_t p=0; p<parts.size(); p++)
        {
            kp.push_back(KeyPoint(parts[p].p,8));
            parts[p].walk(fI, kp[p].pt);
            kp[p].pt /= Part::scale;
        }
    }
};

cv::Ptr<ElasticParts> ElasticParts::create()
{
	return makePtr<ElasticPartsImpl>();
}
