#ifndef __MyFace_onboard__
#define __MyFace_onboard__
#include "opencv2/opencv.hpp"
using namespace cv;

namespace myface
{
    enum EXT {
        EXT_Pixels,
        EXT_Lbp,
        EXT_LBP_E,
        EXT_LBP_O,
        EXT_LBP_P,
        EXT_TPLbp,
        EXT_TPLBP_E,
        EXT_TPLBP_O,
        EXT_TPLBP_P,
        EXT_TPLBP_G,
        EXT_FPLbp,
        EXT_FPLBP_E,
        EXT_FPLBP_O,
        EXT_FPLBP_P,
        EXT_MTS,
        EXT_MTS_E,
        EXT_MTS_O,
        EXT_MTS_P,
        EXT_BGC1,
        EXT_BGC1_E,
        EXT_BGC1_O,
        EXT_BGC1_P,
        EXT_COMB,
        EXT_COMB_E,
        EXT_COMB_O,
        EXT_COMB_P,
        EXT_COMB_G,
        EXT_Gabor,
        EXT_Gabor_E,
        EXT_Dct,
        EXT_Orb,
        EXT_Sift,
        EXT_Sift_G,
        EXT_Grad,
        EXT_Grad_E,
        EXT_Grad_G,
        EXT_Grad_P,
        EXT_GradMag,
        EXT_HDLBP,
        EXT_MAX
    };
    static const char *EXS[] = {
        "Pixels",
        "Lbp",
        "Lbp_E",
        "Lbp_O",
        "Lbp_P",
        "TPLbp",
        "TpLbp_E",
        "TpLbp_O",
        "TpLbp_P",
        "TpLbp_G",
        "FPLbp",
        "FpLbp_E",
        "FpLbp_O",
        "FpLbp_P",
        "MTS",
        "MTS_E",
        "MTS_O",
        "MTS_P",
        "BGC1",
        "BGC1_E",
        "BGC1_O",
        "BGC1_P",
        "COMB",
        "COMB_E",
        "COMB_O",
        "COMB_P",
        "COMB_G",
        "Gabor",
        "Gabor_E",
        "Dct",
        "Orb",
        "Sift",
        "Sift_G",
        "Grad",
        "Grad_E",
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
        CL_SVM,
        CL_EM,
        CL_LR,
        CL_BOOST,
        CL_KMEANS,
        CL_MAX
    };
    static const char *CLS[] = {
        "N_L2",
        "N_L2SQR",
        "N_L1",
        "N_HAM",
        "H_HELL",
        "H_CHI",
        "SVM",
        "EM",
        "LR",
        "BOOST",
        "KMEANS",
        0
    };
    enum PRE {
        PRE_none,
        PRE_eqhist,
        PRE_clahe,
        PRE_retina,
        PRE_tantriggs,
        PRE_crop,
        PRE_MAX
    };
    static const char *PPS[] = {
        "none",
        "eqhist",
        "clahe",
        "retina",
        "tantriggs",
        "crop",
        0
    };



    struct FaceVerifier
    {
         //virtual void train(InputArrayOfArrays src, InputArray _labels) = 0;
         virtual int addTraining(const Mat & img, int label) = 0;
         virtual bool train() = 0;
         virtual int same(const Mat & a, const Mat &b) const = 0;
    };
}

Ptr<myface::FaceVerifier> createMyFaceVerifier(int extract=0, int redu=0, int clsfy=0, int preproc=0, int precrop=0,bool flip=false);


#endif // __MyFace_onboard__

