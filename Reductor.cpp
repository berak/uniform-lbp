
#include <opencv2/imgproc.hpp>
using namespace cv;

#include "TextureFeature.h"


using namespace TextureFeature;

namespace TextureFeatureImpl
{


struct ReductorWalshHadamard : public Reductor
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



struct ReductorDct : public Reductor
{
    int keep;

    ReductorDct(int k=0) : keep(k) {}

    virtual int reduce(const Mat &src, Mat &dest) const
    {
        Mat h; src.convertTo(h, CV_32F);
        Mat h2(h.size(), h.type());

        dft(h,h2, DFT_ROWS); // dft instead of dct solves pow2 issue

        Mat h3 = (keep>0) ?
                 h2(Rect(0,0,std::min(keep, h2.cols-1),1)) :
                 h2;

        dft(h3, dest, DCT_INVERSE | DFT_SCALE | DFT_ROWS);
        return 0;
    }
};


struct ReductorRandomProjection : public Reductor
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
struct ReductorHellinger : public Reductor
{
    virtual int reduce(const Mat &src, Mat &dest) const
    {
        src.convertTo(dest, CV_32F);
        float eps = 1e-7f;
        dest /= (sum(dest)[0] + eps); // L1
        sqrt(dest,dest);
        dest /= (norm(dest) + eps); // L2
        return 0;
    }
};


} // TextureFeatureImpl


namespace TextureFeature
{
using namespace TextureFeatureImpl;


Ptr<Reductor> createReductor(int redu)
{
    switch(redu)
    {
        case RED_NONE:     break; //red = createReductorNone(); break;
        case RED_HELL:     return makePtr<ReductorHellinger>(); break;
        case RED_WHAD:     return makePtr<ReductorWalshHadamard>(8000); break;
        case RED_RP:       return makePtr<ReductorRandomProjection>(8000); break;
        case RED_DCT8:     return makePtr<ReductorDct>(8000); break;
        case RED_DCT12:    return makePtr<ReductorDct>(12000); break;
        case RED_DCT16:    return makePtr<ReductorDct>(16000); break;
        case RED_DCT24:    return makePtr<ReductorDct>(24000); break;
//        default: cerr << "Reductor " << redu << " is not yet supported." << endl; exit(-1);
    }
    return Ptr<Reductor>();
}

} // TextureFeatureImpl

