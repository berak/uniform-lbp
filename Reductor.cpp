
#include <opencv2/imgproc.hpp>
using namespace cv;

#include "TextureFeature.h"



namespace TextureFeatureImpl
{


struct ReductorNone : public TextureFeature::Reductor
{
    virtual int reduce(const Mat &src, Mat &dest) const  { dest=src; return 0; }
};



struct ReductorWalshHadamard : public TextureFeature::Reductor
{
    int keep;

    ReductorWalshHadamard(int k=0) : keep(k) {}

    template<class T>
    void fast_had(int ndim, int lev, T *in, T *out) const
    {
        int h = lev/2;
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

    virtual int reduce(const Mat &src, Mat &dest) const
    {
        Mat h; src.convertTo(h,CV_32F);
        Mat h2(h.size(), h.type());
        for (int lev=src.total(); lev>2; lev/=2)
        {
            fast_had(src.total(), lev, h.ptr<float>(), h2.ptr<float>());
            if (lev>4) cv::swap(h,h2);
        }
        if (keep>0)
            dest = h2(Rect(0,0,std::min(keep,h2.cols-1),1));
        else
            dest = h2;
        return 0;
    }
};



struct ReductorDct : public TextureFeature::Reductor
{
    int keep;

    ReductorDct(int k=0) : keep(k) {}

    virtual int reduce(const Mat &src, Mat &dest) const
    {
        Mat h; src.convertTo(h, CV_32F);
        Mat h2(h.size(), h.type());

        dft(h,h2); // dft instead of dct solves pow2 issue

        Mat h3 = (keep>0) ?
                 h2(Rect(0,0,std::min(keep, h2.cols-1),1)) :
                 h2;

        dft(h3, dest, DCT_INVERSE);
        return 0;
    }
};


struct ReductorRandomProjection : public TextureFeature::Reductor
{
    int K;

    ReductorRandomProjection(int k) : K(k) {}

    const Mat & setup(int N) const
    {
        static Mat proj; // else it can't be const ;(
        if (proj.rows==N && proj.cols==K)
            return proj;

        proj = Mat(N, K, CV_32F);

        theRNG().state = 37183927;
        randn(proj, Scalar(0.5), Scalar(0.5));
        //randu(proj, Scalar(0), Scalar(1));

        for (int i=0; i<K; i++)
        {
            normalize(proj.col(i), proj.col(i));
        }
        return proj;
    }

    virtual int reduce(const Mat &src, Mat &dest) const
    {
        const Mat &proj = setup(src.cols);

        Mat s; src.convertTo(s, CV_32F);
        dest = s * proj;
        return 0;
    }
};



//
// hellinger kernel (no reduction)
//
struct ReductorHellinger : public TextureFeature::Reductor
{
    virtual int reduce(const Mat &src, Mat &dest) const
    {
        src.convertTo(dest, CV_32F);
        float eps = 1e-7f;
        dest /= sum(dest)[0] + eps; // L1
        sqrt(dest,dest);
        dest /= norm(dest) + eps; // L2
        return 0;
    }
};


} // TextureFeatureImpl




cv::Ptr<TextureFeature::Reductor> createReductorNone()
{    return makePtr<TextureFeatureImpl::ReductorNone>(); }

cv::Ptr<TextureFeature::Reductor> createReductorWalshHadamard(int keep)
{    return makePtr<TextureFeatureImpl::ReductorWalshHadamard>(keep); }

cv::Ptr<TextureFeature::Reductor> createReductorHellinger()
{    return makePtr<TextureFeatureImpl::ReductorHellinger>(); }

cv::Ptr<TextureFeature::Reductor> createReductorRandomProjection(int k)
{    return makePtr<TextureFeatureImpl::ReductorRandomProjection>(k); }

cv::Ptr<TextureFeature::Reductor> createReductorDct(int k)
{    return makePtr<TextureFeatureImpl::ReductorDct>(k); }
