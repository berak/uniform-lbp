#include <set>
using namespace std;

//#define HAVE_SSE

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
using namespace cv;

#include "texturefeature.h"
//#include "pcanet/PCANet.h"

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
//
//struct ClassifierMahalanobis : public ClassifierNearest
//{
//    virtual double distance(const cv::Mat &testFeature, const cv::Mat &trainFeature) const
//    {
//        Mat c[2] = {testFeature,trainFeature},covar,icovar,mean;
//        calcCovarMatrix(c,2,covar,mean,COVAR_NORMAL);
//        invert(covar,icovar, DECOMP_SVD);
//        return cv::Mahalanobis(testFeature, trainFeature, icovar);
//    }
//};



static int unique(const Mat &labels, set<int> &classes)
{
    for (size_t i=0; i<labels.total(); ++i)
        classes.insert(labels.at<int>(i));
    return classes.size();
}



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
            svm->setNu(0.5);
            svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 1e-6));

            Mat slabels; // you against all others, that's the only difference.
            for ( size_t j=0; j<labels.total(); ++j)
                slabels.push_back( (*it == labels.at<int>(j)) ? 1 : -1 ); 
            svm->train(trainData , ml::ROW_SAMPLE , slabels); // same data, different labels.
            svms.push_back(svm);
        }
        return trainData.rows;
    }


    virtual int predict(const Mat &src, Mat &res) const
    {
        Mat query = tofloat(src);
        //
        // predict per-class, return best(largest) result
        // hrmm, this assumes, the labels are [0..N]
        //
        float m = -1.0f;
        float mi = 0.0f;
        for (size_t j=0; j<svms.size(); ++j)
        {
            Mat r;
            svms[j]->predict(query, r);
            float p = r.at<float>(0);
            if (p > m)
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
//
// 'Eigenfaces'
//
struct ClassifierPCA : public ClassifierNearestFloat
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
        features.release();
        for (int i=0; i<trainData.rows; i++)
            features.push_back(project(trainData.row(i)));
        return 1;
    }

    virtual int predict(const cv::Mat &testFeature, cv::Mat &results) const
    {
        return ClassifierNearestFloat::predict(project(tofloat(testFeature)), results);
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
    ClassifierPCA_LDA(int num_components=0)
        : ClassifierPCA(num_components)
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
        features.release(); // TODO: pre-allocate, push_back() is giving ugly mem-spikes !
        for (int i=0; i<trainData.rows; i++)
            features.push_back(project(trainData.row(i)));
        labels = trainLabels;
        return 1;
    }
};


struct ClassifierMLP : Classifier
{
    Ptr<ml::ANN_MLP> ann;

    void setup(int ni, int no)
    {
        ann = ml::ANN_MLP::create(); 
        Mat_<int> layers(4,1);
        layers(0) = ni;
        layers(1) = no*2;
        layers(2) = no*8;
        layers(3) = no;
        ann->setLayerSizes(layers);
        ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM,0,0);
        ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 300, 0.0001));
        ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);
    }

    virtual int train(const Mat &trainData, const Mat &trainLabels)
    {
        set<int> classes;
        int C = TextureFeatureImpl::unique(trainLabels, classes);

        if (ann.empty())
        {
            setup(trainData.cols, C);
        }
    
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
    int dist_flag;

    PairDistance(int df=2)
        : dist_flag(df)
    {}

    Mat distance_mat(const Mat &a, const Mat &b) const
    {
        Mat d;
        switch(dist_flag)
        {
            case 0: absdiff(a,b,d); break;
            case 1: d = a-b; multiply(d,d,d,1,CV_32F); break;
            case 2: d = a-b; multiply(d,d,d,1,CV_32F); cv::sqrt(d,d); break;
            case 3: d = a^b; break;
        }
        return d;
    }

    void train_pre(const Mat &features, const Mat &labels, Mat &distances, Mat &binlabels)
    {
        Mat trainData = tofloat(features.reshape(1, labels.rows));

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

    VerifierPairDistance(int df=2)
        : PairDistance(df)
    {}

    virtual int train(const Mat &features, const Mat &labels)
    {      
        Mat distances, binlabels;
        train_pre(features, labels, distances, binlabels);

        model->clear();
        return model->train(ml::TrainData::create(distances, ml::ROW_SAMPLE, binlabels));
    }

    virtual bool same(const Mat &a, const Mat &b) const
    {
        Mat res;
        model->predict(distance_mat(tofloat(a), tofloat(b)), res);
        int r = res.at<int>(0);
        return  r > 0;
    }
};


//
// binary (2 class) svm, same or not same based on distance
//
struct VerifierSVM : public VerifierPairDistance
{
    Ptr<ml::SVM::Kernel> krnl;

    VerifierSVM(int ktype=ml::SVM::LINEAR, int distFlag=2)
        : VerifierPairDistance(distFlag)
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


//
//struct VerifierRTree : public VerifierPairDistance
//{
//    VerifierRTree() 
//    {
//        Ptr<ml::Boost> cl = ml::Boost::create();
//        //Ptr<ml::RTrees> cl = ml::RTrees::create();
//        //cl->setMaxCategories(2);
//        //cl->setMaxDepth(2);
//        //cl->setMinSampleCount(2);
//        //cl->setCVFolds(0);
//        model = cl;
//    }
//};
//



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
        case CL_NORM_HAM:  return makePtr<ClassifierNearest>(NORM_HAMMING2); break;
       // case CL_NORM_MIN:  return makePtr<ClassifierNearest>(NORM_INF); break;
        case CL_HIST_HELL: return makePtr<ClassifierHist>(HISTCMP_HELLINGER); break;
        case CL_HIST_CHI:  return makePtr<ClassifierHist>(HISTCMP_CHISQR); break;
        case CL_KLDIV:     return makePtr<ClassifierHist>(HISTCMP_KL_DIV); break;
        case CL_COSINE:    return makePtr<ClassifierCosine>(); break;
        case CL_SVM_LIN:   return makePtr<ClassifierSVM>(int(cv::ml::SVM::LINEAR)); break;
        case CL_SVM_RBF:   return makePtr<ClassifierSVM>(int(cv::ml::SVM::RBF)); break;
        case CL_SVM_POL:   return makePtr<ClassifierSVM>(int(cv::ml::SVM::POLY)); break;
        case CL_SVM_INT:   return makePtr<ClassifierSVM>(int(cv::ml::SVM::INTER)); break;
        case CL_SVM_INT2:  return makePtr<ClassifierSVM>(-5); break;
        case CL_SVM_INT2X: return makePtr<ClassifierSVM>(-5,3); break;
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
        //case CL_MAHALANOBIS: return makePtr<ClassifierMahalanobis>(); break;
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
        case CL_NORM_HAM:  return makePtr<VerifierNearest>(NORM_HAMMING2); break;
        case CL_HIST_HELL: return makePtr<VerifierHist>(HISTCMP_HELLINGER); break;
        case CL_HIST_CHI:  return makePtr<VerifierHist>(HISTCMP_CHISQR); break;
        case CL_SVM_LIN:   return makePtr<VerifierSVM>(int(cv::ml::SVM::LINEAR)); break;
        case CL_SVM_RBF:   return makePtr<VerifierSVM>(int(cv::ml::SVM::RBF)); break;
        case CL_SVM_POL:   return makePtr<VerifierSVM>(int(cv::ml::SVM::POLY)); break;
        case CL_SVM_INT:   return makePtr<VerifierSVM>(int(cv::ml::SVM::INTER)); break;
        case CL_SVM_INT2:  return makePtr<VerifierSVM>(-5); break;
        case CL_SVM_INT2X: return makePtr<VerifierSVM>(-5, 3); break;
        case CL_SVM_HEL:   return makePtr<VerifierSVM>(-1); break;
        case CL_SVM_HELSQ: return makePtr<VerifierSVM>(-2); break;
        case CL_SVM_LOW:   return makePtr<VerifierSVM>(-6); break;
        case CL_SVM_LOG:   return makePtr<VerifierSVM>(-7); break;
        case CL_SVM_KMOD:  return makePtr<VerifierSVM>(-8); break;
        case CL_SVM_CAUCHY:return makePtr<VerifierSVM>(-9); break;
        case CL_COSINE:    return makePtr<VerifierCosine>(); break;
        //case CL_RTREE:     return makePtr<VerifierRTree>(); break;

        default: cerr << "verification " << clsfy << " is not yet supported." << endl; exit(-1);
    }
    return Ptr<Verifier>();
}


} // namespace TextureFeature

