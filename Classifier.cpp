#include <set>
using namespace std;

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



class ClassifierSvm : public TextureFeature::Classifier
{
    Ptr<ml::SVM> svm;
    ml::SVM::Params param;

public:

    ClassifierSvm(double degree = 0.5,double gamma = 0.8,double coef0 = 0,double C = 0.99, double nu = 0.2, double p = 0.5) 
    {
        param.kernelType = ml::SVM::POLY ; // CvSVM :: RBF , CvSVM :: LINEAR...
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
// ref impl of eigen / fisher faces
//   this is basically bytefish's code, 
//   stripped to the bare minimum
//
class ClassifierEigen : public TextureFeature::Classifier
{
protected:
    int _num_components;
    double _threshold;
    vector<Mat> _projections;
    Mat _labels;
    Mat _eigenvectors;
    Mat _mean;

public:

    ClassifierEigen(int num_components = 0, double threshold = DBL_MAX) 
        : _num_components(num_components)
        , _threshold(threshold) 
    {}

    Mat project(const Mat& src) const
    {
        Mat X, Y;
        src.convertTo(X, _eigenvectors.type());
        for(int i=0; i<src.rows; i++) 
        {
            Mat r_i = X.row(i);
            subtract(r_i, _mean.reshape(1,1), r_i);
        }
        gemm(X, _eigenvectors, 1.0, Mat(), 0.0, Y);
        return Y;
    }
    //Mat reconstruct(const Mat& src) const
    //{
    //    Mat X, Y;
    //    src.convertTo(Y, _eigenvectors.type());
    //    gemm(Y, _eigenvectors, 1.0, Mat(), 0.0, X, GEMM_2_T);
    //    for(int i=0; i<src.rows; i++) 
    //    {
    //        Mat r_i = X.row(i);
    //        add(r_i, _mean.reshape(1,1), r_i);
    //    }
    //    return X;
    //}

    void save_projections(const Mat& data) 
    {
        _projections.clear();
        for(int i=0; i<data.rows; i++) 
        {
            _projections.push_back(project(data.row(i)));
        }
    }

    virtual int train(const Mat & data, const Mat & labels) 
    {
        if((_num_components <= 0) || (_num_components > data.rows))
            _num_components = data.rows;

        PCA pca(data, Mat(), cv::PCA::DATA_AS_ROW, _num_components);

        _labels = labels;
        _mean   = pca.mean.reshape(1,1);
        transpose(pca.eigenvectors, _eigenvectors);
        save_projections(data);
        return 1;
    }

    virtual int predict(const cv::Mat &testFeature, cv::Mat &results) const
    {
        Mat q = project(testFeature.reshape(1,1));
        double minDist = DBL_MAX;
        int minClass = -1;
        int minId=-1;
        for(size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) 
        {
            double dist = norm(_projections[sampleIdx], q, NORM_L2);
            if((dist < minDist) && (dist < _threshold))
            {
                minId    = sampleIdx;
                minDist  = dist;
                minClass = _labels.at<int>((int)sampleIdx);
            }
        }
        results.push_back(float(minClass));
        results.push_back(float(minDist));
        results.push_back(float(minId));
        return 3;
    }
};


class ClassifierFisher : public ClassifierEigen
{
public:

    ClassifierFisher(int num_components = 0, double threshold = DBL_MAX) 
        : ClassifierEigen(num_components, threshold) 
    {}

    int unique(const Mat & labels) const 
    {
        set<int> set_elems;
        for (size_t i=0; i<labels.total(); ++i)
            set_elems.insert(labels.at<int>(i));
        return set_elems.size();
    }

    virtual int train(const Mat & data, const Mat & labels)
    {
        int N = data.rows;
        int C = unique(labels);
        if((_num_components <= 0) || (_num_components > (C-1)))
            _num_components = (C-1);

        PCA pca(data, Mat(), cv::PCA::DATA_AS_ROW, (N-C));
        LDA lda(pca.project(data),labels, _num_components);

        Mat leigen; // hmm, it's new, that i have to convert. something changed in LDA ?
        lda.eigenvectors().convertTo(leigen, pca.eigenvectors.type());
        gemm(pca.eigenvectors, leigen, 1.0, Mat(), 0.0, _eigenvectors, GEMM_1_T);

        _labels = labels;
        _mean   = pca.mean.reshape(1,1);
        save_projections(data);
        return 1;
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

cv::Ptr<TextureFeature::Classifier> createClassifierSVM(double degree=0.5, double gamma=0.8, double coef0=0, double C=0.99, double nu=0.2, double p=0.5)
{ return makePtr<ClassifierSvm>(degree, gamma, coef0, C, nu, p); }

//
// reference impl
//
cv::Ptr<TextureFeature::Classifier> createClassifierEigen()
{ return makePtr<ClassifierEigen>(); }

cv::Ptr<TextureFeature::Classifier> createClassifierFisher()
{ return makePtr<ClassifierFisher>(); }

