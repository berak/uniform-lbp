#include <set>
using namespace std;

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
using namespace cv;

#include "TextureFeature.h"



namespace TextureFeatureImpl
{

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
//
// find the number of unique labels, the class count
//
static int unique(const Mat &labels, set<int> &classes)
{
    for (size_t i=0; i<labels.total(); ++i)
        classes.insert(labels.at<int>(i));
    return classes.size();
}



struct ReductorPCA_LDA : public ReductorPCA
{
    ReductorPCA_LDA(int num_components=0, bool whi=false)
        : ReductorPCA(num_components,whi)
    {}

    virtual int train(const Mat &data, const Mat &labels)
    {
        set<int> classes;
        int C = TextureFeatureImpl::unique(labels,classes);
        // step one, do pca on the original(pixel) data:
        if (num_components<=0)
            num_components = data.rows;
        PCA pca(data, Mat(), cv::PCA::DATA_AS_ROW, std::max(num_components-C,C));
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

struct ReductorWalshHadamard : public TextureFeature::Reductor
{
    template<class T>
    void had(int ndim, int lev, T *in, T *out) const
    {
        int h=lev/2;
        for (int j=0; j<ndim/lev; j++)
        {
	        for(int i=0; i<h; i++)
	        {
		        out[i]   = in[i] + in[i+h];
		        out[i+h] = in[i] - in[i+h];
	        }
            out += lev;
            in  += lev;
        }
    }
    virtual int train(const Mat &features, const Mat &labels) 
    { 
        return 0; 
    }

    virtual int reduce(const Mat &src, Mat &dest) const  
    {
        Mat h; src.convertTo(h, CV_32F);
        Mat h2(h.size(), h.type());
        for (int j=src.total(); j>2; j/=2)
        {
            had(src.total(), j, h.ptr<float>(), h2.ptr<float>());
            cv::swap(h,h2);
        }
        dest=h2;
        return 0; 
    }
};

} // TextureFeatureImpl




cv::Ptr<TextureFeature::Reductor> createReductorNone()
{    return makePtr<TextureFeatureImpl::ReductorNone>(); }

cv::Ptr<TextureFeature::Reductor> createReductorPCA(int nc, bool whi)
{    return makePtr<TextureFeatureImpl::ReductorPCA>(nc,whi); }
cv::Ptr<TextureFeature::Reductor> createReductorPCA_LDA(int nc, bool whi)
{    return makePtr<TextureFeatureImpl::ReductorPCA_LDA>(nc,whi); }

cv::Ptr<TextureFeature::Reductor> createReductorWalshHadamard()
{    return makePtr<TextureFeatureImpl::ReductorWalshHadamard>(); }
