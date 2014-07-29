//#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ml/ml.hpp>

#include <set>

using namespace std;
using namespace cv;

#include "fr.h"

//
// reference impl
//  compare plain pixels 
//
struct Ann : public FaceRecognizer
{
    CvANN_MLP tinklas;
    int npersons;

    virtual void train(InputArray src, InputArray lbls)    
    {
        vector<Mat> imgs;
        src.getMatVector(imgs);
    
        Mat trainData;
        for ( size_t i=0; i<imgs.size(); i++ )
        {
            Mat m = imgs[i].reshape(1,1);
            m.convertTo(m,CV_32F);
            trainData.push_back(m);
        }

        vector<int>labels;
        labels = lbls.getMat();

        // find the number of unique persons, that's the size of our output layer
        std::set<int> pers;
        pers.insert(labels.begin(),labels.end());
        npersons = pers.size();
        Mat flabels;
        Mat(labels).convertTo(flabels,CV_32F);

        int npixels = imgs[0].total();
        int sks[]={ npixels, npersons*10, npersons, 1 };
        //int sks[]={ npixels, npixels/8, npersons*3, 1 };
        cv::Mat sluoksniai= cv::Mat(1,4,CV_32S,sks); 
        tinklas.create(sluoksniai,CvANN_MLP::SIGMOID_SYM,1,1);

	    CvANN_MLP_TrainParams params ;
	    CvTermCriteria criteria ;
	    criteria.max_iter = 2000;
	    criteria.epsilon = 0.00001f;
	    criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS ;
	    params.train_method = CvANN_MLP_TrainParams::BACKPROP ;
	    //params.bp_dw_scale = 0.05f;
	    //params.bp_moment_scale = 0.05f;
	    params.term_crit = criteria ;

    	tinklas.train ( trainData , flabels , cv::Mat() , cv::Mat() , params );
    }

    virtual void predict(InputArray src, int& label, double & minDist) const    
    {
        Mat q = src.getMat();
        q = q.reshape(1,1);
        q.convertTo(q,CV_32F);

        minDist = DBL_MAX;
        Mat response(1,1,CV_32F);

        tinklas.predict(q,response);
        float r = response.at<float>(0,0);
        label = (int)floor(r+0.5);
        cerr << " " << label << " " << r << endl;
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

Ptr<FaceRecognizer> createAnnFaceRecognizer()
{
    return makePtr<Ann>();
}





