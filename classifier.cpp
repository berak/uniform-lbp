#include <set>
using namespace std;


#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/flann/miniflann.hpp>
using namespace cv;

#include "texturefeature.h"

using namespace TextureFeature;

namespace TextureFeatureImpl
{


static Mat tofloat(const Mat &src)
{
    if ( src.type() == CV_32F )
        return src;

    Mat query;
    src.convertTo(query,CV_32F);
    return query;
}


struct ClassifierNearest : public TextureFeature::Classifier
{
    Mat features;
    Mat labels;
    int flag;

    ClassifierNearest(int flag=NORM_L2) : flag(flag) {}

    virtual double distance(const cv::Mat &testFeature, const cv::Mat &trainFeature) const
    {
        return norm(testFeature, trainFeature, flag);
    }

    template <typename Dist>
    static void nearest(const cv::Mat &testFeature, const cv::Mat &features, int &best, double &mind, const Dist &dis)
    {
        mind=DBL_MAX;
        best = -1;
        for (int r=0; r<features.rows; r++)
        {
            double d = dis.distance(testFeature, features.row(r));
            if (d < mind)
            {
                mind = d;
                best = r;
            }
        }
    }
    // TextureFeature::Classifier
    virtual int predict(const cv::Mat &testFeature, cv::Mat &results) const
    {
        int best = -1;
        double mind=DBL_MAX;
        nearest(testFeature, features, best, mind, *this);

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

    virtual int update(const cv::Mat &trainFeatures, const cv::Mat &trainLabels)
    {
        features.push_back(trainFeatures);
        labels.push_back(trainLabels);
        return 1;
    }

    // Serialize
    virtual bool save(FileStorage &fs) const
    {
        fs << "labels" << labels;
        fs << "features" << features;
        return true;
    }

    virtual bool load(const FileStorage &fs)
    {
        fs["labels"] >> labels;
        fs["features"] >> features;
        return ! features.empty();
    }
};

struct ClassifierNearestFloat : public ClassifierNearest
{
    ClassifierNearestFloat(int flag=NORM_L2) : ClassifierNearest(flag) {}

    // TextureFeature::Classifier
    virtual int predict(const cv::Mat &testFeature, cv::Mat &results) const
    {
        return ClassifierNearest::predict(tofloat(testFeature), results);
    }

    virtual int train(const cv::Mat &trainFeatures, const cv::Mat &trainLabels)
    {
        return ClassifierNearest::train(tofloat(trainFeatures), trainLabels);
    }
};



//
// just swap the comparison
//   the flag enums are overlapping, so i like to have this in a different class
//   HISTCMP_CHISQR is default as in opencv's lbph facereco, though HELLINGER definitely works better.
//
struct ClassifierHist : public ClassifierNearestFloat
{
    ClassifierHist(int flag=HISTCMP_CHISQR)
        : ClassifierNearestFloat(flag)
    {}

    // ClassifierNearest
    virtual double distance(const cv::Mat &testFeature, const cv::Mat &trainFeature) const
    {
         return compareHist(testFeature, trainFeature, flag);
    }
};


//
// Negated Mahalanobis Cosine Distance
//
struct ClassifierCosine : public ClassifierNearest
{
    static double cosdistance(const cv::Mat &testFeature, const cv::Mat &trainFeature)
    {
        double a = trainFeature.dot(testFeature);
        double b = trainFeature.dot(trainFeature);
        double c = testFeature.dot(testFeature);
        return -a / sqrt(b*c);
    }

    virtual double distance(const cv::Mat &testFeature, const cv::Mat &trainFeature) const
    {
        return cosdistance(testFeature, trainFeature);
    }
};


static int unique(const Mat &labels, set<int> &classes)
{
    for (size_t i=0; i<labels.total(); ++i)
        classes.insert(labels.at<int>(i));
    return classes.size();
}


// outsourced to svmkernel.cpp
extern Ptr<ml::SVM::Kernel> customKernel(int id);

//
// single svm, multi class.
//
struct ClassifierSVM : public TextureFeature::Classifier
{
    Ptr<ml::SVM> svm;
    Ptr<ml::SVM::Kernel> krnl;

    ClassifierSVM(int ktype=ml::SVM::POLY, double degree = 0.5,double gamma = 0.8,double coef0 = 0,double C = 0.99, double nu = 0.002, double p = 0.5)
    {
        svm = ml::SVM::create();
        svm->setType(ml::SVM::NU_SVC);
        if (ktype<0)
        {
            krnl = customKernel(ktype);
            ktype=-1;
            svm->setCustomKernel(krnl);
        }
        svm->setKernel(ktype); //SVM::LINEAR;
        svm->setDegree(degree);
        svm->setGamma(gamma);
        svm->setCoef0(coef0);
        svm->setNu(nu);
        svm->setP(p);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 1e-6));
        svm->setC(C);
    }

    virtual int train(const Mat &src, const Mat &labels)
    {
        Mat trainData = tofloat(src.reshape(1,labels.rows));

        svm->clear();
        bool ok = svm->train(trainData , ml::ROW_SAMPLE , Mat(labels));
        // damn thing fails silently, if nu was not acceptable
        CV_Assert(ok&&"please check the input params(nu)");
        return trainData.rows;
    }

    virtual int predict(const Mat &src, Mat &res) const
    {
        svm->predict(tofloat(src), res);
        return res.rows;
    }

    // Serialize
    virtual bool save(FileStorage &fs) const
    {
        if(!fs.isOpened()) return false;
        svm->write(fs);
        return true;
    }

    virtual bool load(const FileStorage &fs)
    {
        if(!fs.isOpened()) return false;
        svm->read(fs.getFirstTopLevelNode());
        return true;
    }
};



//
// single class(one vs. all), multi svm approach
//
struct ClassifierSvmMulti : public TextureFeature::Classifier
{
    vector< Ptr<ml::SVM> > svms;

    virtual int train(const Mat &src, const Mat &labels)
    {
        svms.clear();

        Mat trainData = tofloat(src.reshape(1,labels.rows));
        //
        // train one svm per class:
        //
        set<int> classes;
        unique(labels,classes);

        for (set<int>::iterator it=classes.begin(); it != classes.end(); ++it)
        {
            Ptr<ml::SVM> svm = ml::SVM::create();
            svm->setType(ml::SVM::NU_SVC);
            svm->setKernel(ml::SVM::LINEAR);
            svm->setDegree(0.8);
            svm->setGamma(1.0);
            svm->setCoef0(0.0);
            svm->setNu(0.05);
            svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 1e-6));

            Mat slabels; // you against all others, that's the only difference.
            for ( size_t j=0; j<labels.total(); ++j)
                slabels.push_back( (*it == labels.at<int>(j)) ? 1 : -1 );
            bool ok = svm->train(trainData , ml::ROW_SAMPLE , slabels); // same data, different labels.
            CV_Assert(ok);
            svms.push_back(svm);
        }
        return trainData.rows;
    }


    virtual int predict(const Mat &src, Mat &res) const
    {
        Mat query = tofloat(src);
        //
        // predict per-class, return first positive result
        // hrmm, this assumes, the labels are ordered [0..N]
        //
        float m = 100.0f;
        float mi = 0.0f;
        for (size_t j=0; j<svms.size(); ++j)
        {
            Mat r;
            svms[j]->predict(query, r);
            float p = r.at<float>(0);
            if (p > 0)
            {
                m = p;
                mi = float(j);
                break;
            }
            //
            // see: https://github.com/berak/uniform-lbp/issues/2
            // (unfortunately, no improvement visible)
            //
            //svms[j]->predict(query, r, ml::StatModel::RAW_OUTPUT);
            //float p = r.at<float>(0);
            //if (p < 0 && abs(p) < m)
            //{
            //    m = abs(p);
            //    mi = float(j);
            //}
        }
        res = (Mat_<float>(1,2) << mi, m);
        return res.rows;
    }
};

//
//
// 'Eigenfaces'
//
struct ClassifierPCA : public ClassifierNearestFloat  //ClassifierCosine
{
    Mat eigenvectors;
    Mat mean;
    int num_components;

    ClassifierPCA(int num_components=0)
        : num_components(num_components)
    {}

    inline
    Mat project(const Mat &src) const
    {
        return LDA::subspaceProject(eigenvectors, mean, src);
    }

    virtual int train(const Mat &trainData, const Mat &trainLabels)
    {
        if((num_components <= 0) || (num_components > trainData.rows))
            num_components = trainData.rows;

        PCA pca(tofloat(trainData), Mat(), cv::PCA::DATA_AS_ROW, num_components);

        transpose(pca.eigenvectors, eigenvectors);
        mean = pca.mean.reshape(1,1);
        labels = trainLabels;
        features = project(trainData);
        return 1;
    }

    virtual int predict(const cv::Mat &testFeature, cv::Mat &results) const
    {
        return ClassifierNearestFloat::predict(project(tofloat(testFeature)), results);
        //return ClassifierCosine::predict(project(tofloat(testFeature)), results);
    }

    // Serialize
    virtual bool save(FileStorage &fs) const
    {
        fs << "labels" << labels;
        fs << "features" << features;
        fs << "mean" << mean;
        fs << "eigenvectors" << eigenvectors;
        fs << "num_components" << num_components;
        return true;
    }
    virtual bool load(const FileStorage &fs)
    {
        fs["labels"] >> labels;
        fs["features"] >> features;
        fs["mean"] >> mean;
        fs["eigenvectors"] >> eigenvectors;
        fs["num_components"] >>num_components;
        return ! features.empty();
    }
};


//
// 'Fisherfaces'
//
struct ClassifierPCA_LDA : public ClassifierPCA
{
    bool useL2;
    Mat icovar;
    ClassifierPCA_LDA(int num_components=0, bool useL2=false)
        : ClassifierPCA(num_components)
        , useL2(useL2)
    {}

    virtual int train(const Mat &trainData, const Mat &trainLabels)
    {
        set<int> classes;
        int C = TextureFeatureImpl::unique(trainLabels,classes);
        int N = trainData.rows;
        if((num_components <= 0) || (num_components > (C-1)))
            num_components = (C-1);

        // step one, do pca on the original data:
        PCA pca(tofloat(trainData), Mat(), cv::PCA::DATA_AS_ROW, (N-C));
        mean = pca.mean.reshape(1,1);

        // step two, do lda on data projected to pca space:
        Mat proj = LDA::subspaceProject(pca.eigenvectors.t(), mean, trainData);
        LDA lda(proj, trainLabels, num_components);

        // step three, combine both:
        Mat leigen;
        lda.eigenvectors().convertTo(leigen, pca.eigenvectors.type());
        gemm(pca.eigenvectors, leigen, 1.0, Mat(), 0.0, eigenvectors, GEMM_1_T);

        // step four, keep labels and projected dataset:
        features = project(trainData);
        labels = trainLabels;

        // if we use Mahalanobis, precalculate the inverse covariance matrix:
        if (!useL2)
        {
            Mat _covar, _mean;
            calcCovarMatrix(features, _covar, _mean, CV_COVAR_NORMAL|CV_COVAR_ROWS, CV_32F);
            _covar /= (features.rows-1);
            invert(_covar, icovar, DECOMP_SVD);
        }

        return 1;
    }

    virtual double distance(const cv::Mat &testFeature, const cv::Mat &trainFeature) const
    {
        return Mahalanobis(testFeature, trainFeature, icovar);
    }

    virtual int predict(const cv::Mat &testFeature, cv::Mat &results) const
    {
        Mat q = project(tofloat(testFeature));

        if (useL2)
            return ClassifierNearestFloat::predict(q, results);

        // else use Mahalanobis
        int minId = -1;
        double minDist = 999999999;
        nearest(q, features, minId,minDist, *this);
        results = (Mat_<float>(1,3) << float(labels.at<int>(minId)), float(minDist), float(minId));
        return 1;
    }
};

struct ClassifierLDA : public ClassifierNearestFloat
{
    Ptr<LDA> lda;
    Mat mean;

    virtual int train(const Mat &trainData, const Mat &trainLabels)
    {
        set<int> classes;
        int C = TextureFeatureImpl::unique(trainLabels,classes);
        int N = trainData.rows;
        int num_components = (C-1);

        lda.release();
        lda = makePtr<LDA>(num_components);
        lda->compute(trainData, trainLabels);

        //reduce(trainData,mean,0,cv::REDUCE_AVG,CV_64F);

        Mat projected = lda->project(trainData);
        return ClassifierNearestFloat::train(projected, trainLabels);
    }

    virtual int predict(const Mat &a, Mat &res) const
    {
        Mat pa = lda->project(tofloat(a));
        return ClassifierNearestFloat::predict(pa, res);
    }
};

struct ClassifierMLP : Classifier
{
    Ptr<ml::ANN_MLP> ann;

    static Ptr<ml::ANN_MLP> setup(int ni, int no)
    {
        Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::create();
        Mat_<int> layers(4,1);
        layers(0) = ni;
        layers(1) = no>2 ? no*2 : 128;
        layers(2) = no>2 ? no*8 : 8;
        layers(3) = no;
        ann->setLayerSizes(layers);
        ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM,0,0);
        ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 300, 0.0001));
        ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);
        return ann;
    }

    virtual int train(const Mat &trainData, const Mat &trainLabels)
    {
        set<int> classes;
        int C = TextureFeatureImpl::unique(trainLabels, classes);

        ann = setup(trainData.cols, C);

        Mat trainClasses = Mat::zeros(trainLabels.total(), C, CV_32FC1);
        for(int i=0; i < trainClasses.rows; i++)
        {
            trainClasses.at<float>(i, trainLabels.at<int>(i)) = 1.f;
        }

        return ann->train(tofloat(trainData), ml::ROW_SAMPLE, trainClasses);
    }

    virtual int predict(const cv::Mat &testFeature, cv::Mat &results) const
    {
        float r = ann->predict(tofloat(testFeature),results);
        results = (Mat_<float>(1,1) << r);
        return 1;
    }
};


struct ClassifierKNN : Classifier
{
    cv::Ptr<cv::flann::Index> index;
    Mat_<int> labels;

    static int majority(const Mat_<int> &ind, const Mat_<int> &labels) // re-used in verifier
    {
        map<int,int> maj;
        for (size_t i=0; i<ind.total(); i++)
        {
            int id = labels(ind(i));
            if (maj.find(id) == maj.end())
                maj[id] = 0;
            maj[id] ++;
        }
        int maxv=0;
        int maxi=0;
        map<int,int>::iterator it = maj.begin();
        for (; it != maj.end(); it++)
        {
            if (it->second > maxv)
            {
                maxv = it->second;
                maxi = it->first;
            }
        }
        return maxi;
    }

    static cv::Ptr<cv::flann::Index> train_index(const Mat &trainData)
    {
        if (trainData.type() == CV_8U)
        {
            return makePtr<cv::flann::Index>(trainData, cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_HAMMING);
        }
        return makePtr<cv::flann::Index>(trainData, cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_L2);
    }

    virtual int train(const Mat &trainData, const Mat &trainLabels)
    {
        index  = train_index(trainData);
        labels = trainLabels;
        return 1;
    }

    virtual int predict(const cv::Mat &testFeature, cv::Mat &results) const
    {
        int K=5;
        cv::flann::SearchParams params;
        cv::Mat dists;
        cv::Mat indices;
        index->knnSearch(testFeature, indices, dists, K, params);

        results = (Mat_<float>(1,1) << majority(indices, labels));
        //results = (Mat_<float>(1,1) << labels(indices.at<int>(0)));
        return 1;
    }
};

//------->8-----------------------------------------------------------------------
//
// while for the identification task, we train a classifier on image features,
// the verification task needs to train a metric model for pairwise distance.
// related, but quite different.
//
//------->8-----------------------------------------------------------------------



//
// train a single threshold value
//
struct VerifierNearest : TextureFeature::Verifier
{
    double thresh;
    int flag;

    VerifierNearest(int f=NORM_L2)
        : thresh(0)
        , flag(f)
    {}

    virtual double distance(const Mat &a, const Mat &b) const
    {
        return norm(a,b,flag);
    }

    virtual int train(const Mat &features, const Mat &labels)
    {
        thresh = 0;
        double dSame=0, dNotSame=0;
        int    nSame=0, nNotSame=0;
        for (size_t i=0; i<labels.total()-1; i+=2)
        {
            int j = i+1;
            double d = distance(features.row(i), features.row(j));
            if (labels.at<int>(i) == labels.at<int>(j))
            {
                dSame += d;
                nSame ++;
            }
            else
            {
                dNotSame += d;
                nNotSame ++;
            }
        }
        dSame    = (dSame/nSame);
        dNotSame = (dNotSame/nNotSame);
        double dt = dNotSame - dSame;
        thresh = dSame + dt*0.25; //(dSame + dNotSame) / 2;
        return 1;
    }

    virtual bool same(const Mat &a, const Mat &b) const
    {
        return (distance(a,b) < thresh);
    }
};

//
// similar to the classification task - just change the distance func.
//
struct VerifierHist : VerifierNearest
{
    VerifierHist(int f=HISTCMP_CHISQR)
        : VerifierNearest(f)
    {}

    virtual double distance(const Mat &a, const Mat &b) const
    {
        return compareHist(tofloat(a),tofloat(b),flag);
    }
};

//
// similar to the classification task - just change the distance func.
//
struct VerifierCosine : VerifierNearest
{
    virtual double distance(const Mat &a, const Mat &b) const
    {
        return ClassifierCosine::cosdistance(a, b);
    }
};



//
// Wolf, Hassner, Taigman : "Descriptor Based Methods in the Wild"
//  4.1 Distance thresholding for pair matching
//
struct PairDistance
{
    //
    // xor for binary, L2 for float
    //
    Mat distance_mat(const Mat &a, const Mat &b) const
    {
        Mat d;
        switch(a.type())
        {
            case CV_8U:
                d = a^b;
                break;
            default:
                d = a-b;
                multiply(d,d,d,1,CV_32F);
                cv::sqrt(d,d);
                break;
        }
        return d;
    }

    //
    // make a 'distance' mat from 2 features,
    // and binary(-1,1) labels
    //
    void train_pre(const Mat &features, const Mat &labels, Mat &distances, Mat &binlabels)
    {
        Mat trainData = features.reshape(1, labels.rows);

        for (size_t i=0; i<labels.total()-1; i+=2)
        {
            int j = i+1;
            distances.push_back(distance_mat(trainData.row(i), trainData.row(j)));

            int l = (labels.at<int>(i) == labels.at<int>(j)) ? 1 : -1;
            binlabels.push_back(l);
        }
    }
};

//
//  base class for svm,em,lr
//
struct VerifierPairDistance : public TextureFeature::Verifier, PairDistance
{
    Ptr<ml::StatModel> model;
    float thresh; // prediction threshold for binary response

    VerifierPairDistance(float t=0.0f) : thresh(t) {}

    virtual int train(const Mat &features, const Mat &labels)
    {
        Mat distances, binlabels;
        train_pre(tofloat(features), labels, distances, binlabels);

        model->clear();
        return model->train(ml::TrainData::create(distances, ml::ROW_SAMPLE, binlabels));
    }

    virtual bool same(const Mat &a, const Mat &b) const
    {
        Mat res;
        model->predict(distance_mat(tofloat(a), tofloat(b)), res);
        return (res.at<float>(0) > thresh);
    }
};


//
// binary (2 class) svm, same or not same based on distance
//
struct VerifierSVM : public VerifierPairDistance
{
    Ptr<ml::SVM::Kernel> krnl;

    VerifierSVM(int ktype=ml::SVM::LINEAR)
    {
        Ptr<ml::SVM> svm = ml::SVM::create();
        svm->setType(ml::SVM::NU_SVC);
        if (ktype<0)
        {
            krnl = customKernel(ktype);
            ktype=-1;
            svm->setCustomKernel(krnl);
        }
        svm->setKernel(ktype);
        svm->setDegree(3.52);
        svm->setGamma(4.29);
        svm->setNu(0.52);
        svm->setC(699);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 1e-6));
        model = svm;
    }
};



struct VerifierKNN : public TextureFeature::Verifier, PairDistance
{
    cv::Ptr<cv::flann::Index> index;
    Mat_<int> labels;
    Mat features;

    virtual int train(const Mat &trainData, const Mat &trainLabels)
    {
        Mat distances, binlabels;
        train_pre(trainData, trainLabels, distances, binlabels);

        index = ClassifierKNN::train_index(distances);

        labels = binlabels;
        features = distances; // need a copy here, because flann tries to run away with mat.data pointer !!!
        return 1;
    }

    virtual bool same(const Mat &a, const Mat &b) const
    {
        int K=5;
        cv::flann::SearchParams params;
        cv::Mat dists;
        cv::Mat indices;
        index->knnSearch(distance_mat(a,b), indices, dists, K, params);

        int hit = ClassifierKNN::majority(indices, labels);
        return hit > 0;
    }
};


// WIP !! ;(
struct VerifierMLP : public VerifierPairDistance
{
    VerifierMLP() : VerifierPairDistance(0.5f) {}

    virtual int train(const Mat &trainData, const Mat &trainLabels)
    {
        model = ClassifierMLP::setup(trainData.cols, 1);

        Mat distances, binlabels;
        train_pre(tofloat(trainData), trainLabels, distances, binlabels);

        Mat trainClasses = Mat::zeros(binlabels.total(), 1, CV_32FC1);
        for(int i=0; i < trainClasses.rows; i++)
        {
            if (binlabels.at<int>(i) > 0)
                trainClasses.at<float>(i,0) = 1.f;
        }

        return model->train(ml::TrainData::create(distances, ml::ROW_SAMPLE, trainClasses));
    }
};


} // TextureFeatureImpl


namespace TextureFeature
{
using namespace TextureFeatureImpl;

Ptr<Classifier> createClassifier(int clsfy)
{
    switch(clsfy)
    {
        case CL_NORM_L2:   return makePtr<ClassifierNearest>(NORM_L2); break;
        case CL_NORM_L2SQR:return makePtr<ClassifierNearest>(NORM_L2SQR); break;
        case CL_NORM_L1:   return makePtr<ClassifierNearest>(NORM_L1); break;
        case CL_HIST_HELL: return makePtr<ClassifierHist>(HISTCMP_HELLINGER); break;
        case CL_HIST_CHI:  return makePtr<ClassifierHist>(HISTCMP_CHISQR); break;
        case CL_KLDIV:     return makePtr<ClassifierHist>(HISTCMP_KL_DIV); break;
        case CL_COSINE:    return makePtr<ClassifierCosine>(); break;
        case CL_SVM_LIN:   return makePtr<ClassifierSVM>(int(cv::ml::SVM::LINEAR)); break;
        case CL_SVM_RBF:   return makePtr<ClassifierSVM>(int(cv::ml::SVM::RBF)); break;
        case CL_SVM_POL:   return makePtr<ClassifierSVM>(int(cv::ml::SVM::POLY)); break;
        case CL_SVM_INT:   return makePtr<ClassifierSVM>(int(cv::ml::SVM::INTER)); break;
        case CL_SVM_INT2:  return makePtr<ClassifierSVM>(-5); break;
        case CL_SVM_HEL:   return makePtr<ClassifierSVM>(-1); break;
        case CL_SVM_HELSQ: return makePtr<ClassifierSVM>(-2); break;
        case CL_SVM_LOW:   return makePtr<ClassifierSVM>(-6); break;
        case CL_SVM_LOG:   return makePtr<ClassifierSVM>(-7); break;
        case CL_SVM_KMOD:  return makePtr<ClassifierSVM>(-8); break;
        case CL_SVM_CAUCHY:return makePtr<ClassifierSVM>(-9); break;
        case CL_SVM_MULTI: return makePtr<ClassifierSvmMulti>(); break;
        case CL_PCA:       return makePtr<ClassifierPCA>(); break;
        case CL_PCA_LDA:   return makePtr<ClassifierPCA_LDA>(); break;
        case CL_MLP:       return makePtr<ClassifierMLP>(); break;
        case CL_KNN:       return makePtr<ClassifierKNN>(); break;
        // case CL_MAHALANOBIS:return makePtr<ClassifierMahalanobis>(); break;
        default: cerr << "classification " << clsfy << " is not yet supported." << endl; exit(-1);
    }
    return Ptr<Classifier>();
}


Ptr<Verifier> createVerifier(int clsfy)
{
    switch(clsfy)
    {
        case CL_NORM_L2:   return makePtr<VerifierNearest>(NORM_L2); break;
        case CL_NORM_L2SQR:return makePtr<VerifierNearest>(NORM_L2SQR); break;
        case CL_NORM_L1:   return makePtr<VerifierNearest>(NORM_L1); break;
        case CL_HIST_HELL: return makePtr<VerifierHist>(HISTCMP_HELLINGER); break;
        case CL_HIST_CHI:  return makePtr<VerifierHist>(HISTCMP_CHISQR); break;
        case CL_SVM_LIN:   return makePtr<VerifierSVM>(int(cv::ml::SVM::LINEAR)); break;
        case CL_SVM_RBF:   return makePtr<VerifierSVM>(int(cv::ml::SVM::RBF)); break;
        case CL_SVM_POL:   return makePtr<VerifierSVM>(int(cv::ml::SVM::POLY)); break;
        case CL_SVM_INT:   return makePtr<VerifierSVM>(int(cv::ml::SVM::INTER)); break;
        case CL_SVM_INT2:  return makePtr<VerifierSVM>(-5); break;
        case CL_SVM_HEL:   return makePtr<VerifierSVM>(-1); break;
        case CL_SVM_HELSQ: return makePtr<VerifierSVM>(-2); break;
        case CL_SVM_LOW:   return makePtr<VerifierSVM>(-6); break;
        case CL_SVM_LOG:   return makePtr<VerifierSVM>(-7); break;
        case CL_SVM_KMOD:  return makePtr<VerifierSVM>(-8); break;
        case CL_SVM_CAUCHY:return makePtr<VerifierSVM>(-9); break;
        case CL_COSINE:    return makePtr<VerifierCosine>(); break;
        case CL_KNN:       return makePtr<VerifierKNN>(); break;
        case CL_MLP:       return makePtr<VerifierMLP>(); break;

        default: cerr << "verification " << clsfy << " is not yet supported." << endl; exit(-1);
    }
    return Ptr<Verifier>();
}


} // namespace TextureFeature

