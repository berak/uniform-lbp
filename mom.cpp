//#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;

#include "fr.h"


class MomReco  : public FaceRecognizer
{
    vector<int> labels;
    Mat histograms;
    int raster;
    int width;

public:

    MomReco(int raster=4,int width=6)
        : raster(raster)
        , width(width)
    {}


    virtual void train(InputArray src, InputArray lbls) ;

    virtual void predict(InputArray src, int& label, double & conf) const;
	virtual int predict(InputArray src) const;

    virtual void save(const cv::String& filename) const;
    virtual void save(FileStorage& fs) const    {}

    virtual void load(const cv::String& filename);
    virtual void load(const FileStorage& fs)    {}

    virtual void update(InputArrayOfArrays src, InputArray labels) {train(src,labels);}

    void process( const Mat & img, Mat & feature ) const;


};


void MomReco::process( const Mat & img, Mat & feature ) const
{
    Mat bin; threshold(img,bin,0,255,cv::THRESH_OTSU);
    int w = bin.cols / raster;
    int h = bin.rows / raster;
    for ( int i=0; i<raster; i++ ) {
        for ( int j=0; j<raster; j++ ) {
            cv::Mat roi=img(cv::Rect(i*w,j*h,w,h));
            Moments m = moments( roi, false);
            /**
            feature.push_back(m.m00);
            feature.push_back(m.m01);
            feature.push_back(m.m02);
            feature.push_back(m.m03);
            feature.push_back(m.m10);
            feature.push_back(m.m11);
            feature.push_back(m.m12);
            feature.push_back(m.m30);
            **/
            feature.push_back(m.mu02);
            feature.push_back(m.mu03);
            feature.push_back(m.mu11);
            feature.push_back(m.mu12);
            feature.push_back(m.mu20);
            feature.push_back(m.mu21);
            feature.push_back(m.mu30);

            feature.push_back(m.nu02);
            feature.push_back(m.nu03);
            feature.push_back(m.nu11);
            feature.push_back(m.nu12);
            feature.push_back(m.nu20);
            feature.push_back(m.nu21);
            feature.push_back(m.nu30);

            double hu[7];
            HuMoments(m,hu);
            feature.push_back(hu[0]);
            feature.push_back(hu[1]);
            feature.push_back(hu[2]);
            feature.push_back(hu[3]);
            feature.push_back(hu[4]);
            feature.push_back(hu[5]);
            feature.push_back(hu[6]);
        }
    }
}

void MomReco::train(InputArray src, InputArray lbls)    {
    vector<Mat> imgs;
    src.getMatVector(imgs);
    labels = lbls.getMat();
    histograms = cv::Mat(0,0,CV_8UC1);
    for ( size_t i=0; i<imgs.size(); i++ ) {
        Mat hist;
        process( imgs[i], hist );
        histograms.push_back(hist);
    }
    histograms = histograms.reshape(0,imgs.size());
}


void MomReco::predict(InputArray src, int& label, double & minDist) const    {
    if ( histograms.empty() || labels.empty() )
        CV_Error(Error::StsBadArg,"must have train data");

    Mat img = src.getMat();
    Mat hist;
    process( img, hist );

    minDist = DBL_MAX;
    for(size_t i = 0; i < labels.size(); i++) {
        double dist = norm(histograms.row(i), hist.reshape(0,1), NORM_L2);
        if(dist < minDist) {
            minDist = dist;
            label = labels[i];
        }
    }
}

int MomReco::predict(InputArray src) const {
    int pred=-1;
    double conf=-1;
    predict(src,pred,conf);
    return pred;
}


void MomReco::load(const cv::String& filename)    {
    histograms = cv::imread(filename,0);
    FileStorage fs(filename+".yml",FileStorage::READ);
    fs["L"] >> labels;
    fs["R"] >> raster;
    fs["W"] >> width;
}
void MomReco::save(const cv::String& filename) const {
    if ( filename.find(".png") > 0 ) {
        std::vector<int> params;
      //  params.push_back(CV_IMWRITE_PNG_COMPRESSION); ///PPP where did CV_IMWRITE_PNG_COMPRESSION go ?
      //  params.push_back(compression);
        cv::imwrite(filename,histograms,params);
        FileStorage fs(filename+".yml",FileStorage::WRITE);
        fs << "L" << labels;
        fs << "R" << raster;
        fs << "W" << width;
        return;
    }
}


Ptr<FaceRecognizer> createMomFaceRecognizer(int raster,int width) {
    return makePtr<MomReco>(raster,width);
}


