#include <vector>
#include <set>
using namespace std;


#include <opencv2/core/core.hpp>
#include "TextureFeature.h"
using namespace cv;

    
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

        Mat leigen; // hmm, that's new, that i have to convert.
        lda.eigenvectors().convertTo(leigen,pca.eigenvectors.type());
        gemm(pca.eigenvectors, leigen, 1.0, Mat(), 0.0, _eigenvectors, GEMM_1_T);

        _labels = labels;
        _mean   = pca.mean.reshape(1,1); 
        save_projections(data);
        return 1;
    }
};


//
// factory
//
cv::Ptr<TextureFeature::Classifier> createClassifierEigen()
{ return makePtr<ClassifierEigen>(); }
cv::Ptr<TextureFeature::Classifier> createClassifierFisher()
{ return makePtr<ClassifierFisher>(); }

