#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
using namespace cv;

#include "TextureFeature.h"


static struct _onceonly { _onceonly() { /*initModule_ml();*/ } } yes;


class ClassifierNearest : public TextureFeature::Classifier
{
protected:
    Mat features;
    Mat labels;
    int flag;

public:

    ClassifierNearest(int flag=NORM_L2) : flag(flag) {}

    virtual double distance(const cv::Mat &trainFeature, const cv::Mat &testFeature) const
    {
        return norm(trainFeature, testFeature, flag);
    }
    // TextureFeature::Classifier
    virtual int predict(const cv::Mat &testFeature, cv::Mat &results) const
    {
        double mind=DBL_MAX;
        int best = -1;
        for (int r=0; r<features.rows; r++)
        {
            double d = distance(testFeature, features.row(r));
            if (d < mind)
            {
                mind = d;
                best = r;
            }
        }
        int found = best>-1 ? labels.at<int>(best) : -1;
        results.push_back(float(found));
        results.push_back(float(mind));
        results.push_back(float(best));
        return 3;
    }

    virtual int train(const cv::Mat &trainFeatures, const cv::Mat &trainLabels)
    {
        features = trainFeatures;
        labels = trainLabels;
        return 1;
    }
};



//
// just swap the comparison
//
class ClassifierHist : public ClassifierNearest
{
public:

    ClassifierHist(int flag=HISTCMP_CHISQR) 
        : ClassifierNearest(flag)
    {}

    // ClassifierNearest
    virtual double distance(const cv::Mat &trainFeature, const cv::Mat &testFeature) const
    {
         return compareHist(trainFeature, testFeature, flag);
    }
};




class ClassifierKNN : public TextureFeature::Classifier
{
    Ptr<ml::KNearest> knn; 
    int K;

public:

    ClassifierKNN(int k=1) 
        : knn(ml::KNearest::create())
        , K(k) 
    {}

    virtual int predict(const cv::Mat &testFeature, cv::Mat &results) const
    {
        Mat resp;
        knn->findNearest(testFeature, K, results, resp, Mat());
        //  std::cerr << "resp " << resp << std::endl;
        //for ( int k=0; k<resp.cols; k++ )
        //    results.push_back(resp.at<float>(k));
        return results.rows;
    }
    virtual int train(const cv::Mat &trainFeatures, const cv::Mat &trainLabels)
    {
        knn->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
        return 1;
    }
};

//
//class ClassifierXX : public TextureFeature::Classifier
//{
//    Ptr<ml::StatModel> me; // hmm type errors with ml::EM
//
//public:
//
//    ClassifierXX(Ptr<ml::StatModel> p) 
//        : me(p)
//    {}
//
//    virtual int predict(const cv::Mat &testFeature, cv::Mat &results) const
//    {
//        Mat feature = testFeature;
//        if ( testFeature.type() != CV_32F )
//            testFeature.convertTo(feature,CV_32F);
//        me->predict(feature, results, ml::ROW_SAMPLE);
//        return results.rows;
//    }
//    virtual int train(const cv::Mat &trainFeatures, const cv::Mat &trainLabels)
//    {
//        Mat feature = trainFeatures;
//        if ( trainFeatures.type() != CV_32F )
//            trainFeatures.convertTo(feature,CV_32F);
//        Mat labels = trainLabels;
//        if ( trainLabels.type() != CV_32F )
//            trainLabels.convertTo(labels,CV_32F);
//        me->train(feature, ml::ROW_SAMPLE, labels);
//        return 1;
//    }
//};
//
//
//


class Svm : public TextureFeature::Classifier
{
    Ptr<ml::SVM> svm;
    ml::SVM::Params param;

public:

    Svm(double degree = 0.5,double gamma = 0.8,double coef0 = 0,double C = 0.99, double nu = 0.2, double p = 0.5) 
    {
        param.kernelType = ml::SVM :: POLY ; // CvSVM :: RBF , CvSVM :: LINEAR...
        param.svmType = ml::SVM::NU_SVC;
        param.degree = degree; // for poly
        param.gamma = gamma; // for poly / rbf / sigmoid
        param.coef0 = coef0; // for poly / sigmoid
        param.C = C; // for CV_SVM_C_SVC , CV_SVM_EPS_SVR and CV_SVM_NU_SVR
        param.nu = nu; // for CV_SVM_NU_SVC , CV_SVM_ONE_CLASS , and CV_SVM_NU_SVR
        param.p = p; // for CV_SVM_EPS_SVR
        param.classWeights = NULL ; // for CV_SVM_C_SVC

        param.termCrit.type = TermCriteria::MAX_ITER | TermCriteria::EPS ;
        param.termCrit.maxCount = 1000;
        param.termCrit.epsilon = 1e-6;
        svm = ml::SVM::create(param);
    }

    virtual int train(const Mat &src, const Mat &labels)
    {
        Mat trainData = src.reshape(1,labels.rows);
        if ( trainData.type() != CV_32F )
            trainData.convertTo(trainData,CV_32F);
        svm->train( trainData , ml::ROW_SAMPLE , Mat(labels) );
        return trainData.rows;
    }

    virtual int predict(const Mat &src, Mat &res) const    
    {
        Mat query;
        if ( src.type() != CV_32F )
            src.convertTo(query,CV_32F);
        else
            query=src;
        svm->predict(query, res);
        return res.rows;
    }
};





//
// 'factory' functions (aka public api)
//

cv::Ptr<TextureFeature::Classifier> createClassifierNearest(int norm_flag=NORM_L2)
{ return makePtr<ClassifierNearest>(norm_flag); }

cv::Ptr<TextureFeature::Classifier> createClassifierHist(int flag=HISTCMP_CHISQR)
{ return makePtr<ClassifierHist>(flag); }

cv::Ptr<TextureFeature::Classifier> createClassifierKNN(int k=1)
{ return makePtr<ClassifierKNN>(k); }

cv::Ptr<TextureFeature::Classifier> createClassifierSVM(double degree = 0.5,double gamma = 0.8,double coef0 = 0,double C = 0.99, double nu = 0.2, double p = 0.5)
{ return makePtr<Svm>(degree, gamma, coef0, C, nu, p); }

//cv::Ptr<TextureFeature::Classifier> createClassifierXX(String s)
//{ 
//    //cv::Ptr<ml::StatModel> m = ml::StatModel::create(s);
//    //cv::Ptr<ml::StatModel> m = Algorithm::create<ml::StatModel>(s);
//    //cv::Ptr<ml::StatModel> m = ml::LogisticRegression::create();
//    cv::Ptr<ml::StatModel> m = ml::KNearest::create();
//    return makePtr<ClassifierXX>(m); 
//}
//
