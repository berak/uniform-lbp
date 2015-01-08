#ifndef __TextureFeature_onboard__
#define __TextureFeature_onboard__

#include <opencv2/opencv.hpp>
using cv::Mat;
using cv::String;
using cv::FileStorage;

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
        virtual bool save(FileStorage &fs) const  { return false; }
        virtual bool load(const FileStorage &fs)  { return false; }
    };

    struct Classifier : public Serialize // identification
    {
        virtual int train(const Mat &features, const Mat &labels) = 0;
        virtual int predict(const Mat &test, Mat &result) const = 0;
    };

    struct Verifier : public Serialize   // same-notSame
    {
        virtual int train(const Mat &features, const Mat &labels) = 0;
        virtual bool same(const Mat &a, const Mat &b) const = 0;
    };
}



//
// the pipeline is:
//    extractor -> reductor -> classifier (or verifier)
//




namespace TextureFeature
{
    enum EXT {
        EXT_Pixels,
        EXT_Lbp,
        EXT_LBP_P,
        EXT_TPLbp,
        EXT_TPLBP_P,
        EXT_TPLBP_G,
        EXT_TPLbp2_G,
        EXT_FPLbp,
        EXT_FPLBP_P,
        EXT_MTS,
        EXT_MTS_P,
        EXT_BGC1,
        EXT_BGC1_P,
        EXT_COMB,
        EXT_COMB_P,
        EXT_COMB_G,
        EXT_Gabor,
        EXT_Dct,
        EXT_Orb,
        EXT_Sift,
        EXT_Sift_G,
        EXT_Grad,
        EXT_Grad_G,
        EXT_Grad_P,
        EXT_GradMag,
        EXT_HDLBP,
        EXT_MAX
    };
    static const char *EXS[] = {
        "Pixels",
        "Lbp",
        "Lbp_P",
        "TPLbp",
        "TpLbp_P",
        "TpLbp_G",
        "TPLbp2_G",
        "FPLbp",
        "FpLbp_P",
        "MTS",
        "MTS_P",
        "BGC1",
        "BGC1_P",
        "COMB",
        "COMB_P",
        "COMB_G",
        "Gabor",
        "Dct",
        "Orb",
        "Sift",
        "Sift_G",
        "Grad",
        "Grad_G",
        "Grad_P",
        "GradMag",
        "HDLBP",
        0
    };
    enum RED {
        RED_NONE,
        RED_HELL,
        RED_WHAD,
        RED_RP,
        RED_DCT8,
        RED_DCT12,
        RED_DCT16,
        RED_DCT24,
        RED_MAX
    };
    static const char *REDS[] = {
        "none",
        "HELL",
        "WHAD",
        "RP",
        "DCT8",
        "DCT12",
        "DCT16",
        "DCT24",
        0
    };
    enum CLA {
        CL_NORM_L2,
        CL_NORM_L2SQR,
        CL_NORM_L1,
        CL_NORM_HAM,
        CL_HIST_HELL,
        CL_HIST_CHI,
        CL_COSINE,
        CL_SVM_LIN,
        CL_SVM_POL,
        CL_SVM_RBF,
        CL_SVM_INT,
        CL_SVM_INT2,
        CL_SVM_HEL,
        CL_SVM_COR,
        CL_SVM_COS,
        CL_SVM_LOW,
        CL_SVM_MULTI,
        CL_PCA,
        CL_PCA_LDA,
        CL_MAX
    };
    static const char *CLS[] = {
        "N_L2",
        "N_L2SQR",
        "N_L1",
        "N_HAM",
        "H_HELL",
        "H_CHI",
        "COSINE",
        "SVM_LIN",
        "SVM_POL",
        "SVM_RBF",
        "SVM_INT",
        "SVM_INT2",
        "SVM_HEL",
        "SVM_COR",
        "SVM_COS",
        "SVM_LOW",
        "SVM_MULTI",
        "PCA",
        "PCA_LDA",
        0
    };

    cv::Ptr<Extractor>  createExtractor(int ex);
    cv::Ptr<Reductor>   createReductor(int rd);
    cv::Ptr<Classifier> createClassifier(int cl);
    cv::Ptr<Verifier>   createVerifier(int ver);
}

////
//// supplied impementations:
////
//
////
//// extractors
////
//cv::Ptr<TextureFeature::Extractor> createExtractorPixels(int resw=0, int resh=0);
//// lbp variants
//cv::Ptr<TextureFeature::Extractor> createExtractorLbp(int gridx=8, int gridy=8);
//cv::Ptr<TextureFeature::Extractor> createExtractorPyramidLbp();
//// four-patch lbp variants
//cv::Ptr<TextureFeature::Extractor> createExtractorFPLbp(int gx=8, int gy=8);
//cv::Ptr<TextureFeature::Extractor> createExtractorPyramidFpLbp();
//// three-patch lbp variants
//cv::Ptr<TextureFeature::Extractor> createExtractorTPLbp(int gx=8, int gy=8);
//cv::Ptr<TextureFeature::Extractor> createExtractorPyramidTpLbp();
//cv::Ptr<TextureFeature::Extractor> createExtractorGfttTpLbp();
//cv::Ptr<TextureFeature::Extractor> createExtractorGfttTpLbp2();
//// reverse lbp circle
//cv::Ptr<TextureFeature::Extractor> createExtractorBGC1(int gx=8, int gy=8);
//cv::Ptr<TextureFeature::Extractor> createExtractorPyramidBGC1();
//// 1/2 lbp circle
//cv::Ptr<TextureFeature::Extractor> createExtractorMTS(int gx=8, int gy=8);
//cv::Ptr<TextureFeature::Extractor> createExtractorPyramidMTS();
//// linear combination of cslbp,diamond,square (16*3 bins)
//cv::Ptr<TextureFeature::Extractor> createExtractorCombined(int gx=8, int gy=8);
//cv::Ptr<TextureFeature::Extractor> createExtractorPyramidCombined();
//cv::Ptr<TextureFeature::Extractor> createExtractorGfttCombined();
//// phase based
//cv::Ptr<TextureFeature::Extractor> createExtractorGaborLbp(int gx=8, int gy=8, int kernel_size=8);
//// dct based
//cv::Ptr<TextureFeature::Extractor> createExtractorDct();
//// feature2D abuse
//cv::Ptr<TextureFeature::Extractor> createExtractorORBGrid(int g=10);
//cv::Ptr<TextureFeature::Extractor> createExtractorSIFTGrid(int g=10);
//cv::Ptr<TextureFeature::Extractor> createExtractorSIFTGftt();
//// gradients
//cv::Ptr<TextureFeature::Extractor> createExtractorGrad();
//cv::Ptr<TextureFeature::Extractor> createExtractorGfttGrad();
//cv::Ptr<TextureFeature::Extractor> createExtractorPyramidGrad();
//cv::Ptr<TextureFeature::Extractor> createExtractorGfttGradMag();
//
//cv::Ptr<TextureFeature::Extractor> createExtractorHighDimLbp();
//
////
//// reductors
////
//cv::Ptr<TextureFeature::Reductor> createReductorNone();
//cv::Ptr<TextureFeature::Reductor> createReductorWalshHadamard(int keep=0); // 0==all
//cv::Ptr<TextureFeature::Reductor> createReductorRandomProjection(int keep=0);
//cv::Ptr<TextureFeature::Reductor> createReductorDct(int keep=0);
//cv::Ptr<TextureFeature::Reductor> createReductorHellinger();
//
//
////
//// identification task (get the closest item from a trained db)
////
//cv::Ptr<TextureFeature::Classifier> createClassifierNearest(int norm_flag=cv::NORM_L2);
//cv::Ptr<TextureFeature::Classifier> createClassifierHist(int flag=cv::HISTCMP_CHISQR);
//cv::Ptr<TextureFeature::Classifier> createClassifierCosine();
//cv::Ptr<TextureFeature::Classifier> createClassifierKNN(int k=1);
//cv::Ptr<TextureFeature::Classifier> createClassifierSVM(int ktype=cv::ml::SVM::POLY, double degree = 0.5,double gamma = 0.8,double coef0 = 0,double C = 0.99, double nu = 0.002, double p = 0.5);
//cv::Ptr<TextureFeature::Classifier> createClassifierSVMMulti();
//cv::Ptr<TextureFeature::Classifier> createClassifierPCA(int n=0);
//cv::Ptr<TextureFeature::Classifier> createClassifierPCA_LDA(int n=0);
//
////
//// verification task (same / not same)
////
//cv::Ptr<TextureFeature::Verifier> createVerifierNearest(int flag=cv::NORM_L2);
//cv::Ptr<TextureFeature::Verifier> createVerifierHist(int flag=cv::HISTCMP_CHISQR);
//cv::Ptr<TextureFeature::Verifier> createVerifierSVM(int ktype=cv::ml::SVM::LINEAR, int distfunc=2);
//cv::Ptr<TextureFeature::Verifier> createVerifierEM(int distfunc=2);
//cv::Ptr<TextureFeature::Verifier> createVerifierLR(int distfunc=2);
//cv::Ptr<TextureFeature::Verifier> createVerifierBoost(int distfunc=2);
//cv::Ptr<TextureFeature::Verifier> createVerifierKmeans();
//
//

#endif // __TextureFeature_onboard__
