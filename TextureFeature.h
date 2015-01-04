#ifndef __TextureFeature_onboard__
#define __TextureFeature_onboard__

#include <opencv2/opencv.hpp>
using cv::Mat;
using cv::String;

//
// interfaces
//
namespace TextureFeature
{
    struct Extractor
    {
        virtual int extract(const Mat &img, Mat &features) const = 0;
    };

    struct Reductor
    {
        virtual int reduce(const Mat &src, Mat &dest) const = 0;
    };

    struct Serialize // io
    {
        virtual bool save(const String &fn) const  { return false; }
        virtual bool load(const String &fn)        { return false; }
    };

    struct Classifier : public Serialize // identification
    {
        virtual int train(const Mat &features, const Mat &labels) = 0;
        virtual int predict(const Mat &test, Mat &result) const = 0;
    };

    struct Verifier : public Serialize    // same-notSame
    {
        virtual int train(const Mat &features, const Mat &labels) = 0;
        virtual bool same(const Mat &a, const Mat &b) const = 0;
    };
};



//
// the pipeline is:
//    extractor -> reductor -> classifier (or verifier)
//






//
// supplied impementations:
//

//
// extractors
//
cv::Ptr<TextureFeature::Extractor> createExtractorPixels(int resw=0, int resh=0);
// lbp variants
cv::Ptr<TextureFeature::Extractor> createExtractorLbp(int gridx=8, int gridy=8);
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidLbp();
// four-patch lbp variants
cv::Ptr<TextureFeature::Extractor> createExtractorFPLbp(int gx=8, int gy=8);
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidFpLbp();
// three-patch lbp variants
cv::Ptr<TextureFeature::Extractor> createExtractorTPLbp(int gx=8, int gy=8);
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidTpLbp();
cv::Ptr<TextureFeature::Extractor> createExtractorGfttTpLbp();
cv::Ptr<TextureFeature::Extractor> createExtractorGfttTpLbp2();
// reverse lbp circle
cv::Ptr<TextureFeature::Extractor> createExtractorBGC1(int gx=8, int gy=8);
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidBGC1();
// 1/2 lbp circle
cv::Ptr<TextureFeature::Extractor> createExtractorMTS(int gx=8, int gy=8);
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidMTS();
// linear combination of cslbp,diamond,square (16*3 bins)
cv::Ptr<TextureFeature::Extractor> createExtractorCombined(int gx=8, int gy=8);
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidCombined();
cv::Ptr<TextureFeature::Extractor> createExtractorGfttCombined();
// phase based
cv::Ptr<TextureFeature::Extractor> createExtractorGaborLbp(int gx=8, int gy=8, int kernel_size=8);
// dct based
cv::Ptr<TextureFeature::Extractor> createExtractorDct();
// feature2D abuse
cv::Ptr<TextureFeature::Extractor> createExtractorORBGrid(int g=10);
cv::Ptr<TextureFeature::Extractor> createExtractorSIFTGrid(int g=10);
cv::Ptr<TextureFeature::Extractor> createExtractorSIFTGftt();
// gradients
cv::Ptr<TextureFeature::Extractor> createExtractorGrad();
cv::Ptr<TextureFeature::Extractor> createExtractorGfttGrad();
cv::Ptr<TextureFeature::Extractor> createExtractorPyramidGrad();
cv::Ptr<TextureFeature::Extractor> createExtractorGfttGradMag();

cv::Ptr<TextureFeature::Extractor> createExtractorHighDimLbp();

//
// reductors
//
cv::Ptr<TextureFeature::Reductor> createReductorNone();
cv::Ptr<TextureFeature::Reductor> createReductorWalshHadamard(int keep=0); // 0==all
cv::Ptr<TextureFeature::Reductor> createReductorRandomProjection(int keep=0);
cv::Ptr<TextureFeature::Reductor> createReductorDct(int keep=0);
cv::Ptr<TextureFeature::Reductor> createReductorHellinger();


//
// identification task (get the closest item from a trained db)
//
cv::Ptr<TextureFeature::Classifier> createClassifierNearest(int norm_flag=cv::NORM_L2);
cv::Ptr<TextureFeature::Classifier> createClassifierHist(int flag=cv::HISTCMP_CHISQR);
cv::Ptr<TextureFeature::Classifier> createClassifierCosine();
cv::Ptr<TextureFeature::Classifier> createClassifierKNN(int k=1);
cv::Ptr<TextureFeature::Classifier> createClassifierSVM(double degree = 0.5,double gamma = 0.8,double coef0 = 0,double C = 0.99, double nu = 0.002, double p = 0.5);
cv::Ptr<TextureFeature::Classifier> createClassifierSVMMulti();
cv::Ptr<TextureFeature::Classifier> createClassifierPCA(int n=0);
cv::Ptr<TextureFeature::Classifier> createClassifierPCA_LDA(int n=0);

//
// verification task (same / not same)
//
cv::Ptr<TextureFeature::Verifier> createVerifierNearest(int flag=cv::NORM_L2);
cv::Ptr<TextureFeature::Verifier> createVerifierHist(int flag=cv::HISTCMP_CHISQR);
cv::Ptr<TextureFeature::Verifier> createVerifierSVM(int distfunc=2);
cv::Ptr<TextureFeature::Verifier> createVerifierEM(int distfunc=2);
cv::Ptr<TextureFeature::Verifier> createVerifierLR(int distfunc=2);
cv::Ptr<TextureFeature::Verifier> createVerifierBoost(int distfunc=2);
cv::Ptr<TextureFeature::Verifier> createVerifierKmeans();



#endif // __TextureFeature_onboard__
