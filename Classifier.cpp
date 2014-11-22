#include <set>
using namespace std;

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
using namespace cv;

#include "TextureFeature.h"


//
// find the number of unique labels, the class count
//
int unique(const Mat &labels, set<int> &classes=set<int>())
{
    for (size_t i=0; i<labels.total(); ++i)
        classes.insert(labels.at<int>(i));
    return classes.size();
}




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
//   the flag enums are overlapping, so i like to have this in a different class
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


//
// Negated Mahalanobis Cosine Distance
//
class ClassifierCosine : public ClassifierNearest
{
public:
    // ClassifierNearest
    virtual double distance(const cv::Mat &trainFeature, const cv::Mat &testFeature) const
    {
        double a = trainFeature.dot(testFeature);
        double b = trainFeature.dot(trainFeature);
        double c = testFeature.dot(testFeature);
        return -a / sqrt(b*c);
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
        knn->clear();
        knn->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
        return 1;
    }
};



//
// single svm, multi class.
//
class ClassifierSvm : public TextureFeature::Classifier
{
public:
    Ptr<ml::SVM> svm;
    ml::SVM::Params param;


    ClassifierSvm(double degree = 0.5,double gamma = 0.8,double coef0 = 0,double C = 0.99, double nu = 0.002, double p = 0.5) 
    //ClassifierSvm(double degree = 0.5,double gamma = 0.8,double coef0 = 0,double C = 0.99, double nu = 0.2, double p = 0.5) 
    {
        param.kernelType = ml::SVM::POLY ; // CvSVM :: RBF , CvSVM :: LINEAR...
        param.svmType = ml::SVM::NU_SVC; // NU_SVC
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

        svm->clear();
        bool ok = svm->train( trainData , ml::ROW_SAMPLE , Mat(labels) );
        CV_Assert(ok&&"please check the input params(nu)");
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
// single class(one vs. all), multi svm approach
//
class ClassifierSvmMulti : public TextureFeature::Classifier
{
    vector<Ptr<ml::SVM>> svms;
    ml::SVM::Params param;

public:

    ClassifierSvmMulti() 
    {
        // 
        // again, call me helpless on parameterizing this ;[
        //
        param.kernelType = ml::SVM::LINEAR; //, CvSVM::LINEAR...
        //param.svmType = ml::SVM::NU_SVC;
        ////param.degree = degree; // for poly
        param.gamma = 1.0; // for poly / rbf / sigmoid
        param.coef0 = 0.0; // for poly / sigmoid
        param.C = 0.5; // for CV_SVM_C_SVC , CV_SVM_EPS_SVR and CV_SVM_NU_SVR
        param.nu = 0.5; // for CV_SVM_NU_SVC , CV_SVM_ONE_CLASS , and CV_SVM_NU_SVR
        //param.p = p; // for CV_SVM_EPS_SVR
        //param.classWeights = NULL ; // for CV_SVM_C_SVC
        
        param.termCrit.type = TermCriteria::MAX_ITER | TermCriteria::EPS ;
        param.termCrit.maxCount = 100;
        param.termCrit.epsilon = 1e-6;
    }

    virtual int train(const Mat &src, const Mat &labels)
    {
        svms.clear();

        Mat trainData = src.reshape(1,labels.rows);
        if ( trainData.type() != CV_32F )
            trainData.convertTo(trainData,CV_32F);

        //
        // train one svm per class:
        //
        set<int> classes;
        int N = unique(labels,classes);

        for (set<int>::iterator it=classes.begin(); it != classes.end(); ++it)
        {
            Ptr<ml::SVM> svm = ml::SVM::create(param);
            Mat slabels; // you against all others, that's the only difference.
            for ( size_t j=0; j<labels.total(); ++j)
                slabels.push_back( ( *it == labels.at<int>(j) ) ? 1 : -1 );
            svm->train( trainData , ml::ROW_SAMPLE , slabels ); // same data, different labels.
            svms.push_back(svm);
        }
        return trainData.rows;
    }


    virtual int predict(const Mat &src, Mat &res) const    
    {
        Mat query;
        if ( src.type() != CV_32F )
            src.convertTo(query,CV_32F);
        else
            query=src;

        //
        // predict per-class, return best(largest) result
        // hrmm, this assumes, the labels are [0..N]
        //
        float m = -1.0f;
        float mi = 0.0f;
        for ( size_t j=0; j<svms.size(); ++j)
        {
            Mat r;
            svms[j]->predict(query, r);
            float p = r.at<float>(0);
            if ( p > m ) 
            {
                m = p;
                mi = float(j);
            }
        }
        res = (Mat_<float>(1,2) << mi, m);
        return res.rows;
    }
};

// 
// ref impl of eigen / fisher faces
//   this is basically bytefish's code, 
//   (terribly) condensed to the bare minimum
//
class ClassifierEigen : public TextureFeature::Classifier
{
protected:
    vector<Mat> _projections;
    Mat _labels;
    Mat _eigenvectors;
    Mat _mean;
    int _num_components;

public:

    ClassifierEigen(int num_components=0) 
        : _num_components(num_components) // we don't need a threshold here
    {}

    void save_projections(const Mat &data) 
    {
        _projections.clear();
        for(int i=0; i<data.rows; i++) 
        {
            Mat p = LDA::subspaceProject(_eigenvectors, _mean, data.row(i));
            _projections.push_back(p);
        }
    }

    virtual int train(const Mat &data, const Mat &labels) 
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
        Mat query = LDA::subspaceProject(_eigenvectors, _mean, testFeature.reshape(1,1));
        double minDist = DBL_MAX;
        int minClass = -1;
        int minId=-1;
        for (size_t idx=0; idx<_projections.size(); idx++) 
        {
            double dist = norm(_projections[idx], query, NORM_L2);
            if (dist < minDist)
            {
                minId    = idx;
                minDist  = dist;
                minClass = _labels.at<int>((int)idx);
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

    ClassifierFisher(int num_components=0) 
        : ClassifierEigen(num_components) 
    {}


    virtual int train(const Mat &data, const Mat &labels)
    {
        int N = data.rows;
        int C = unique(labels);
        if((_num_components <= 0) || (_num_components > (C-1))) // btw, why C-1 ?
            _num_components = (C-1);

        // step one, do pca on the original(pixel) data:
        PCA pca(data, Mat(), cv::PCA::DATA_AS_ROW, (N-C));
        _mean = pca.mean.reshape(1,1);

        // step two, do lda on data projected to pca space:
        Mat proj = LDA::subspaceProject(pca.eigenvectors.t(), _mean, data);
        LDA lda(proj, labels, _num_components);

        // step three, combine both:
        Mat leigen; 
        lda.eigenvectors().convertTo(leigen, pca.eigenvectors.type());
        gemm(pca.eigenvectors, leigen, 1.0, Mat(), 0.0, _eigenvectors, GEMM_1_T);

        // step four, project training images to lda space for prediction:
        _labels = labels;
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

cv::Ptr<TextureFeature::Classifier> createClassifierCosine()
{ return makePtr<ClassifierCosine>(); }

cv::Ptr<TextureFeature::Classifier> createClassifierKNN(int k=1)
{ return makePtr<ClassifierKNN>(k); }

cv::Ptr<TextureFeature::Classifier> createClassifierSVM(double degree=0.5, double gamma=0.8, double coef0=0, double C=0.99, double nu=0.002, double p=0.5)
{ return makePtr<ClassifierSvm>(degree, gamma, coef0, C, nu, p); }

cv::Ptr<TextureFeature::Classifier> createClassifierSVMMulti()
{ return makePtr<ClassifierSvmMulti>(); }

//
// reference impl
//
cv::Ptr<TextureFeature::Classifier> createClassifierEigen()
{ return makePtr<ClassifierEigen>(); }

cv::Ptr<TextureFeature::Classifier> createClassifierFisher()
{ return makePtr<ClassifierFisher>(); }

