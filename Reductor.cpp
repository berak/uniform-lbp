#include <set>
using namespace std;

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
using namespace cv;

#include "TextureFeature.h"


struct ReductorNone : public TextureFeature::Reductor
{
    virtual int train(const Mat &features, const Mat &labels) { return 0; }
    virtual int reduce(const Mat &src, Mat &dest) const  { dest=src; return 0; }
};


struct ReductorPCA : public TextureFeature::Reductor
{
    Mat eigenvectors;
    Mat mean;
    int num_components;
    bool whitening;

    ReductorPCA(int num_components=0, bool whi=false)
        : num_components(num_components)
        , whitening(whi)
    {}

    int reduce(const Mat &src, Mat &dst) const
    {
        dst = LDA::subspaceProject(eigenvectors, mean, src);
        return dst.total()*dst.elemSize();
    }


    virtual int train(const Mat &data, const Mat &labels)
    {
        if((num_components <= 0) || (num_components > data.rows))
            num_components = data.rows;

        PCA pca(data, Mat(), cv::PCA::DATA_AS_ROW, num_components);

        if (whitening)
        {
            Mat m2;
            sqrt(pca.eigenvalues, m2);
            m2 = 1.0 / Mat::diag(m2);
            gemm(m2, pca.eigenvectors, 1, Mat(), 0, eigenvectors);
        } else {
            transpose(pca.eigenvectors, eigenvectors);
        }
        mean = pca.mean.reshape(1,1);
        return 1;
    }
};

struct ReductorPCA_LDA : public ReductorPCA
{
    ReductorPCA_LDA(int num_components=0, bool whi=false)
        : ReductorPCA(num_components,whi)
    {}

    virtual int train(const Mat &data, const Mat &labels)
    {
        // step one, do pca on the original(pixel) data:
        if (num_components<=0)
            num_components = data.rows;
        PCA pca(data, Mat(), cv::PCA::DATA_AS_ROW, num_components);
        mean = pca.mean.reshape(1,1);

        // step two, do lda on data projected to pca space:
        Mat proj = LDA::subspaceProject(pca.eigenvectors.t(), mean, data);
        LDA lda(proj, labels, min(num_components,pca.eigenvectors.rows));

        // step three, combine both:
        Mat leigen;
        lda.eigenvectors().convertTo(leigen, pca.eigenvectors.type());
        gemm(pca.eigenvectors, leigen, 1.0, Mat(), 0.0, eigenvectors, GEMM_1_T);
        return 1;
    }
};

cv::Ptr<TextureFeature::Reductor> createReductorNone()
{    return makePtr<ReductorNone>(); }

cv::Ptr<TextureFeature::Reductor> createReductorPCA(int nc, bool whi)
{    return makePtr<ReductorPCA>(nc,whi); }
cv::Ptr<TextureFeature::Reductor> createReductorPCA_LDA(int nc, bool whi)
{    return makePtr<ReductorPCA_LDA>(nc,whi); }
