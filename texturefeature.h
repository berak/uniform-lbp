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
        EXT_GradMag_P,
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
        "GradMagP",
        "HDLBP",
        0
    };
    enum RED {
        RED_NONE,
        RED_HELL,
        RED_POW,
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
        "POW",
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
        CL_SVM_INT2, // custom 
        CL_SVM_HEL,  // custom
        CL_SVM_LOW,  // custom
        CL_SVM_LOG,
        CL_SVM_MULTI,
        CL_PCA,
        CL_PCA_LDA,
        CL_EMD,
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
        "SVM_LOW",
        "SVM_LOG",
        "SVM_MULTI",
        "PCA",
        "PCA_LDA",
        "EMD",
        0
    };

    cv::Ptr<Extractor>  createExtractor(int ex);
    cv::Ptr<Reductor>   createReductor(int rd);
    cv::Ptr<Classifier> createClassifier(int cl);
    cv::Ptr<Verifier>   createVerifier(int ver);
}


#endif // __TextureFeature_onboard__
