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

    struct Filter
    {
        virtual int filter(const Mat &src, Mat &dest) const = 0;
    };

    struct Serialize // io
    {
        virtual bool save(FileStorage &fs) const  { return false; }
        virtual bool load(const FileStorage &fs)  { return false; }
    };

    struct Classifier : public Serialize // identification
    {
        virtual int predict(const Mat &test, Mat &result) const = 0;
        virtual int train(const Mat &features, const Mat &labels) = 0;
        virtual int update(const Mat &features, const Mat &labels) 
        {
            throw("not implemented!");
        }
    };

    struct Verifier : public Serialize   // same-notSame
    {
        virtual bool same(const Mat &a, const Mat &b) const = 0;
        virtual int train(const Mat &features, const Mat &labels) = 0;
    };
}



//
// the pipeline is:
//    extractor -> filter -> classifier (or verifier)
//




namespace TextureFeature
{
    enum EXT {
        EXT_Pixels,
        EXT_Lbp,
        EXT_LBP_P,
        EXT_LBPU_P,
        EXT_TPLbp,
        EXT_TPLBP_P,
        EXT_TPLBP_G,
        EXT_FPLbp,
        EXT_FPLBP_P,
        EXT_MTS,
        EXT_MTS_P,
        EXT_BGC1,
        EXT_BGC1_P,
        EXT_COMB,
        EXT_COMB_P,
        EXT_COMB_G,
        EXT_GaborLBP,
        EXT_GaborGB,
        EXT_Dct,
        EXT_Orb,
        EXT_Sift,
        EXT_Sift_G,
        EXT_Grad,
        EXT_Grad_G,
        EXT_Grad_P,
        EXT_GradMag,
        EXT_GradMag_P,
        EXT_GradBin,
        EXT_HDGRAD,
        EXT_HDLBP,
        EXT_HDLBP_PCA,
        EXT_PCASIFT,
        EXT_PCANET,
        EXT_RANDNET,
        EXT_WAVENET,
        EXT_CDIKP,
        EXT_MAX
    };
    static const char *EXS[] = {
        "Pixels",
        "Lbp",
        "Lbp_P",
        "Lbpu_P",
        "TPLbp",
        "TpLbp_P",
        "TpLbp_G",
        "FPLbp",
        "FpLbp_P",
        "MTS",
        "MTS_P",
        "BGC1",
        "BGC1_P",
        "COMB",
        "COMB_P",
        "COMB_G",
        "GaborLBP",
        "GaborGB",
        "Dct",
        "Orb",
        "Sift",
        "Sift_G",
        "Grad",
        "Grad_G",
        "Grad_P",
        "GradMag",
        "GradMagP",
        "GradBin",
        "HDGRAD",
        "HDLBP",
        "HDLBP_PCA",
        "PCASIFT",
        "PCANET",
        "RANDNET",
        "WAVENET",
        "CDIKP",
        0
    };
    enum FIL {
        FIL_NONE,
        FIL_HELL,
        FIL_POW,
        FIL_SQRT,
        FIL_WHAD4,
        FIL_WHAD8,
        FIL_RP,
        FIL_DCT2,
        FIL_DCT4,
        FIL_DCT6,
        FIL_DCT8,
        FIL_DCT12,
        FIL_DCT16,
        FIL_DCT24,
        FIL_MAX
    };
    static const char *FILS[] = {
        "none",
        "HELL",
        "POW",
        "SQRT",
        "WHAD4",
        "WHAD8",
        "RP",
        "DCT2",
        "DCT4",
        "DCT6",
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
        CL_SVM_HELSQ,  // custom
        CL_SVM_LOW,  // custom
        CL_SVM_LOG,  // custom
        CL_SVM_KMOD,  // custom
        CL_SVM_CAUCHY,  // custom
        CL_SVM_MULTI,
        CL_PCA,
        CL_PCA_LDA,
        //CL_RTREE,
        //CL_MAHALANOBIS,
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
        "SVM_HELSQ",
        "SVM_LOW",
        "SVM_LOG",
        "SVM_KMOD",
        "SVM_CAUCHY",
        "SVM_MULTI",
        "PCA",
        "PCA_LDA",
        //"RTREE",
        //"MAHALANOB",
        0
    };

    cv::Ptr<Extractor>  createExtractor(int ext);
    cv::Ptr<Filter>     createFilter(int fil);
    cv::Ptr<Classifier> createClassifier(int cla);
    cv::Ptr<Verifier>   createVerifier(int ver);
}


#endif // __TextureFeature_onboard__
