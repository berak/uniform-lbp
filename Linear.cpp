//#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace std;
using namespace cv;



//
// reference impl
//  compare plain pixels 
//
struct Linear : public FaceRecognizer
{
    vector<int> labels;
    vector<Mat> imgs;
    int normFlag;

    Linear(int NormFlag=NORM_L2) : normFlag(NormFlag) {}

    virtual void train(InputArray src, InputArray lbls)    {
        src.getMatVector(imgs);
        labels = lbls.getMat();
    }

    virtual void predict(InputArray src, int& label, double & minDist) const    {
        Mat q = src.getMat();
        minDist = DBL_MAX;
        int minClass = -1;
        for(size_t i = 0; i < imgs.size(); i++) {
            double dist = norm(imgs[i], q, normFlag);
            if(dist < minDist) {
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
    virtual void update(InputArrayOfArrays src, InputArray labels) {train(src,labels);}
    virtual void save(const std::string& filename) const    {}
    virtual void save(FileStorage& fs) const    {}
    virtual void load(const std::string& filename)    {}
    virtual void load(const FileStorage& fs)    {}
};

Ptr<FaceRecognizer> createLinearFaceRecognizer(int NORM)
{
    return makePtr<Linear>(NORM);
}





