
//#define HAVE_SSE

#include <opencv2/ml.hpp>
using namespace cv;

#include "texturefeature.h"

using namespace TextureFeature;

namespace TextureFeatureImpl
{

struct CustomKernel : public ml::SVM::Kernel
{
    int K;
    CustomKernel(int k) : K(k) {}

#ifdef HAVE_SSE
    inline float res(const __m128 & s)
    {
        union { __m128 m; float f[4]; } x;
        x.m = s;
        return (x.f[0] + x.f[1] + x.f[2] + x.f[3]);
    }
#endif


    float l2sqr(int var_count, int j, const float *vecs, const float *another)
    {
#ifdef HAVE_SSE
        if (var_count % 4 == 0)
        {
            __m128 c,d, s = _mm_set_ps1(0);
            __m128* ptr_a = (__m128*)another;
            __m128* ptr_b = (__m128*)(&vecs[j*var_count]);
            for(int k=0; k<var_count; k+=4, ptr_a++, ptr_b++)
            {
                c = _mm_sub_ps(*ptr_a, *ptr_b);
                d = _mm_mul_ps(c, c);
                s = _mm_add_ps(s, d);
            }
            return res(s);
        } // else fall back to sw
#endif
        float s = 0;
        float a,b,c;
        const float* sample = &vecs[j*var_count];
        for(int k=0; k<var_count; k++)
        {
            a = sample[k];  b = another[k];
            c = (a-b);
            s += c*c;
        }
        return s;
    }

    float min(int var_count, int j, const float *vecs, const float *another)
    {
#ifdef HAVE_SSE
        if (var_count % 4 == 0)
        { 
            __m128 c,   s = _mm_set_ps1(0);
            __m128* ptr_a = (__m128*)another;
            __m128* ptr_b = (__m128*)(&vecs[j*var_count]);
            for(int k=0; k<var_count; k+=4, ptr_a++, ptr_b++)
            {
                c = _mm_min_ps(*ptr_a, *ptr_b);
                s = _mm_add_ps(s, c);
            }
            return res(s);
        }
#endif
        float s = 0;
        float a,b,c;
        const float* sample = &vecs[j*var_count];
        for(int k=0; k<var_count; k++)
        {
            a = sample[k];  b = another[k];
            c = std::min(a,b);
            s += c*c;
        }
        return s;
    }

    void calc_intersect(int vcount, int var_count, const float* vecs, const float* another, float* results)
    {
        for(int j=0; j<vcount; j++)
        {
            results[j] = min(var_count,j,vecs,another);
        }
    }

    void calc_hellinger(int vcount, int var_count, const float* vecs, const float* another, float* results)
    {
        CV_Assert (var_count<64000);
        //float z[64000]; // there *must* be a better idea than this.
        cv::AutoBuffer<float> buf(var_count);
        float *z = buf;
#ifdef HAVE_SSE
        if (var_count%4 == 0)
        {
            __m128* ptr_out= (__m128*)z;
            __m128* ptr_in = (__m128*)another;
            // cache sqrt(another[k])        
            for(int k=0; k<var_count; k+=4, ptr_in++, ptr_out++)
            {
                *ptr_out = _mm_sqrt_ps(*ptr_in);
            }
            for(int j=0; j<vcount; j++)
            {
                __m128 a,b,c,s = _mm_set_ps1(0);
                __m128* ptr_a = (__m128*)(&vecs[j*var_count]);
                __m128* ptr_b = (__m128*)z;
                for(int k=0; k<var_count; k+=4, ptr_a++, ptr_b++)
                {
                    a = _mm_sqrt_ps(*ptr_a);
                    b = _mm_sub_ps(a, *ptr_b);
                    c = _mm_mul_ps(b, b);
                    s = _mm_add_ps(s, c);
                }
                results[j] = -res(s);
            }
            return;
        }
#endif       
        for(int k=0; k<var_count; k+=4)
        {
            z[k]   = sqrt(another[k]);
            z[k+1] = sqrt(another[k+1]);
            z[k+2] = sqrt(another[k+2]);
            z[k+3] = sqrt(another[k+3]);
        }

        for(int j=0; j<vcount; j++)
        {
            double a,b, s = 0;
            const float* sample = &vecs[j*var_count];
            for(int k=0; k<var_count; k+=4)
            {
                a = sqrt(sample[k]);     b = z[k];    s += (a - b) * (a - b);
                a = sqrt(sample[k+1]);   b = z[k+1];  s += (a - b) * (a - b);
                a = sqrt(sample[k+2]);   b = z[k+2];  s += (a - b) * (a - b);
                a = sqrt(sample[k+3]);   b = z[k+3];  s += (a - b) * (a - b);
            }
            results[j] = (float)(-s);
        }
    }

    // assumes, you did the sqrt before on the input data !    
    void calc_hellinger_sqrt(int vcount, int var_count, const float* vecs, const float* another, float* results)
    {
        for(int j=0; j<vcount; j++)
        {
            float z = l2sqr(var_count,j,vecs,another);
            results[j] = -(z);
        }
    }

    void calc_lowpass(int vcount, int var_count, const float* vecs, const float* another, float* results)
    {
        for(int j=0; j<vcount; j++)
        {
            double s = 0;
            double a1,a2,b1,b2;
            const float* sample = &vecs[j*var_count];
            for(int k=0; k<var_count-1; k++)
            {
                a1 = sample[k];    a2 = sample[k+1];
                b1 = another[k];   b2 = another[k+1];
                s += sqrt((a1+a2) * (b1+b2));
            }
            results[j] = (float)(s);
        }
    }
    void calc_log(int vcount, int var_count, const float* vecs, const float* another, float* results)
    {
        for(int j=0; j<vcount; j++)
        {
            float z = l2sqr(var_count,j,vecs,another);
            results[j] = float(-log(z+1));
        }
    }
    //
    // KMOD-A New Support Vector Machine Kernel With Moderate Decreasing for
    //  Pattern Recognition. Application to Digit Image Recognition.
    //    N.E. Ayat  M. Cheriet  L. Remaki C.Y. Suen
    // 
    //  (4) KMOD(x,y) = K *(exp(gamma / ((||x-y||^2) + (sigma^2))) - 1)
    //
    void calc_kmod(int vcount, int var_count, const float* vecs, const float* another, float* results)
    {
        const float K  = 1.0f;  // normalization constant
        const float s2 = 15.0f; // kernelsize squared
        const float ga = 0.7f;  // decrease speed

        for(int j=0; j<vcount; j++)
        {
            float z = l2sqr(var_count,j,vecs,another);
            results[j] = K * (exp(ga/(z+s2))-1);
        }
    }
    // http://crsouza.blogspot.de/2010/03/kernel-functions-for-machine-learning.html
    // special case for d=2, so it cancels the sqrt
    void calc_rational_quadratic(int vcount, int var_count, const float* vecs, const float* another, float* results)
    {
        const static float C=10.0f;
        for(int j=0; j<vcount; j++)
        {
            float z = l2sqr(var_count,j,vecs,another);
            results[j] = 1.0f - z / (z+C);
        }
    }

    void calc_inv_multiquad(int vcount, int var_count, const float* vecs, const float* another, float* results)
    {
        float C2 = 100;
        for(int j=0; j<vcount; j++)
        {
            float z = l2sqr(var_count,j,vecs,another);
            results[j] = 1.0f/sqrt(z+C2);
        }
    }
    void calc_laplacian(int vcount, int var_count, const float* vecs, const float* another, float* results)
    {
        float sigma = 3;
        for(int j=0; j<vcount; j++)
        {
            float z = l2sqr(var_count,j,vecs,another);
            results[j] = exp(-sqrt(z) / sigma);
        }
    }
    void calc_cauchy(int vcount, int var_count, const float* vecs, const float* another, float* results)
    {
        float sigma2 = 3*3;
        for(int j=0; j<vcount; j++)
        {
            float z = l2sqr(var_count,j,vecs,another);
            results[j] = 1.0f / (1.0f+(z/sigma2));
        }
    }
    void calc(int vcount, int var_count, const float* vecs, const float* another, float* results)
    {
        switch(K)
        {
        case -1: calc_hellinger(vcount, var_count, vecs, another, results); break;
        case -2: calc_hellinger_sqrt(vcount, var_count, vecs, another, results); break;
        //case -2: calc_correl(vcount, var_count, vecs, another, results); break;
        //case -3: calc_cosine(vcount, var_count, vecs, another, results); break;
        //case -4: calc_bhattacharyya(vcount, var_count, vecs, another, results); break;
        case -5: calc_intersect(vcount, var_count, vecs, another, results); break;
        case -6: calc_lowpass(vcount, var_count, vecs, another, results); break;
        case -7: calc_log(vcount, var_count, vecs, another, results); break;
        case -8: calc_kmod(vcount, var_count, vecs, another, results); break;
        case -9: calc_cauchy(vcount, var_count, vecs, another, results); break;
        }
    }
    int getType(void) const
    {
        return 7;
    }
    static Ptr<ml::SVM::Kernel> create(int k)
    {
        return makePtr<CustomKernel>(k);
    }
};

Ptr<ml::SVM::Kernel> customKernel(int id)
{
    return CustomKernel::create(id);
}
} //TextureFeatureImpl

