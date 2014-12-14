#include <set>
using namespace std;

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
using namespace cv;

#include "TextureFeature.h"


struct ReductorNone : public TextureFeature::Reductor
{
    virtual int train(const Mat &features) { return 0; }
    virtual int reduce(const Mat &src, Mat &dest) const  { dest=src; return 0; }
};


struct ReductorPCA : public TextureFeature::Reductor
{
    Mat _eigenvectors;
    Mat _mean;
    int _num_components;
    bool whitening;

    ReductorPCA(int num_components=0, bool whi=false)
        : _num_components(num_components)
        , whitening(whi)
    {}

    int reduce(const Mat &src, Mat &dst) const
    {
        dst = LDA::subspaceProject(_eigenvectors, _mean, src);
        return dst.total();
    }


    virtual int train(const Mat &data)
    {
        if((_num_components <= 0) || (_num_components > data.rows))
            _num_components = data.rows;

        PCA pca(data, Mat(), cv::PCA::DATA_AS_ROW, _num_components);

        if (whitening)
        {
            Mat m2; 
            sqrt(pca.eigenvalues, m2);
            m2 = 1.0 / Mat::diag(m2);
            gemm(m2, pca.eigenvectors, 1, Mat(), 0, _eigenvectors);
        } else {
            transpose(pca.eigenvectors, _eigenvectors);
        }
        _mean = pca.mean.reshape(1,1);
        return 1;
    }
};


cv::Ptr<TextureFeature::Reductor> createReductorNone()
{    return makePtr<ReductorNone>(); }

cv::Ptr<TextureFeature::Reductor> createReductorPCA(int nc, bool whi)
{    return makePtr<ReductorPCA>(nc,whi); }
