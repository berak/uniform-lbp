//#include "opencv2/opencv.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
 

using namespace std;
using namespace cv;


#include "opencv2/contrib.hpp"
//
//
// the pixel's neighbours:
//
// 0 1 2
// 7 * 3
// 6 5 4
//
// * slice the image into local patches, similar to the lbp idea
// * per pixel, collect the index of biggest and the smallest neighbour
//   into a pos and a neg histogram for each patch. 
//   since the patches can be quite small, this will fit even into a 8*uchar array.
// * concat the pos/neg patch histograms into a large one (again, same idea as in lbp)
// * move the window by 1/2 patch size, and repeat
// * use HAMMING2 norm for distance calculation
// * additional bonus, since this is uchar data, 
//   save the histograms as png, fast & small
//
//

class MinMaxReco  : public FaceRecognizer
{
    vector<int> labels;
    Mat histograms;
    int raster;
    int compression;


public:

    MinMaxReco(int raster=10,int compression=0)
        : raster(raster)
        , compression(compression)
    {}


    virtual void train(InputArray src, InputArray lbls) ;

    virtual void predict(InputArray src, int& label, double & conf) const;
	virtual int predict(InputArray src) const;

    virtual void save(const cv::String& filename) const;
    virtual void save(FileStorage& fs) const    {}

    virtual void load(const cv::String& filename);
    virtual void load(const FileStorage& fs)    {}

    virtual void update(InputArrayOfArrays src, InputArray labels) {train(src,labels);}

private:
    void minmax(const cv::Mat &inp, cv::Mat &pos, cv::Mat &neg) const ;
    void minmax(const cv::Mat &inp, cv::Mat & histogram, int raster) const;

    inline
    void minmax(const cv::Mat &inp, int &m, int &M, int &mi, int &Mi, int i, int j,int id) const {
        uchar p = inp.at<uchar>(i,j);
        if ( p < m ) { m=p; mi=id; }
        if ( p > M ) { M=p; Mi=id; }
    }
};


//
// generate histograms per patch/roi
//
void MinMaxReco::minmax(const cv::Mat &inp, cv::Mat &pos, cv::Mat &neg) const {
    for ( int i=1; i<inp.rows-1; i++ ) {
        for ( int j=1; j<inp.cols-1; j++ ) {
            int m=100000,M=-1;
            int mi=-1,Mi=-1;
            minmax(inp,m,M,mi,Mi,i-1,j-1,0);
            minmax(inp,m,M,mi,Mi,i-1,j,  1);
            minmax(inp,m,M,mi,Mi,i-1,j+1,2);
            minmax(inp,m,M,mi,Mi,i,  j-1,3);
            minmax(inp,m,M,mi,Mi,i,  j+1,4);
            minmax(inp,m,M,mi,Mi,i+1,j-1,5);
            minmax(inp,m,M,mi,Mi,i+1,j,  6);
            minmax(inp,m,M,mi,Mi,i+1,j+1,7);

            pos.at<uchar>(Mi) ++;
            neg.at<uchar>(mi) ++;
        }
    }
}


//
// slice image into patches and collect histogram
//
void MinMaxReco::minmax(const cv::Mat &inp, cv::Mat & histogram, int raster) const {
    int w = inp.cols/raster;
    int h = inp.rows/raster;

    for ( int i=0; i<raster-1; i++ ) {
        for ( int j=0; j<raster; j++ ) {
            for ( int m=0; m<2; m++ ) {  // pseudo sliding-window
                cv::Mat pos = cv::Mat::zeros(1,8,histogram.type());
                cv::Mat neg = cv::Mat::zeros(1,8,histogram.type());
                cv::Mat roi=inp(cv::Rect((m*w/2)+i*w,j*h,w,h));
                minmax(roi,pos,neg);
                histogram.push_back(cv::Mat(pos));
                histogram.push_back(cv::Mat(neg));
            }
        }
    }
}


void MinMaxReco::train(InputArray src, InputArray lbls)    {
    vector<Mat> imgs;
    src.getMatVector(imgs);
    labels = lbls.getMat();
    histograms = cv::Mat(0,0,CV_8UC1);
    for ( size_t i=0; i<imgs.size(); i++ ) {
        cv::Mat hist(0,8,CV_8U);
        minmax( imgs[i], hist, raster );
        histograms.push_back(hist);
    }
    histograms = histograms.reshape(0,imgs.size());
}


void MinMaxReco::predict(InputArray src, int& label, double & minDist) const    {
    if ( histograms.empty() || labels.empty() )
        CV_Error(Error::StsBadArg,"must have train data");

    Mat img = src.getMat();
    cv::Mat hist(0,8,CV_8U);
    minmax( img, hist, raster );

    minDist = DBL_MAX;
    for(size_t i = 0; i < labels.size(); i++) {
        double dist = norm(histograms.row(i), hist.reshape(0,1), NORM_HAMMING2);
        if(dist < minDist) {
            minDist = dist;
            label = labels[i];
        }
    }
}

int MinMaxReco::predict(InputArray src) const {
    int pred=-1;
    double conf=-1;
    predict(src,pred,conf);
    return pred;
}


void MinMaxReco::load(const cv::String& filename)    {
    histograms = cv::imread(filename,0);
    FileStorage fs(filename+".yml",FileStorage::READ);
    fs["L"] >> labels;
    fs["R"] >> raster;
    fs["C"] >> compression;
}
void MinMaxReco::save(const cv::String& filename) const {
    if ( filename.find(".png") > 0 ) {
        std::vector<int> params;
      //  params.push_back(CV_IMWRITE_PNG_COMPRESSION); ///PPP where did CV_IMWRITE_PNG_COMPRESSION go ?
      //  params.push_back(compression);
        cv::imwrite(filename,histograms,params);
        FileStorage fs(filename+".yml",FileStorage::WRITE);
        fs << "L" << labels;
        fs << "R" << raster;
        fs << "C" << compression;
        return;
    }
}


Ptr<FaceRecognizer> createMinMaxFaceRecognizer(int raster,int compression) {
    return makePtr<MinMaxReco>(raster,compression);
}


