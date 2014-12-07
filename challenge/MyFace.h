#ifndef __MyFace_onboard__
#define __MyFace_onboard__
#include "opencv2/opencv.hpp"
using namespace cv;

namespace myface
{
    enum EXT { EXT_Pixels,EXT_Lbp,EXT_Lbpu,EXT_TPLbp,EXT_FPLbp,EXT_MTS,EXT_MTS_E,EXT_LBP_E,EXT_FPLBP_E,EXT_TPLBP_E,EXT_GaborLbp,EXT_Dct,EXT_OrbGrid,EXT_SiftGrid,EXT_MAX };
    enum CLA { CL_NORM_L2,CL_NORM_L2SQR,CL_NORM_L1,CL_NORM_HAM,CL_HIST_HELL,CL_HIST_CHI,CL_SVM,CL_EM,CL_LR,CL_COSINE,CL_FISHER,CL_MAX };
    enum PRE { PRE_none,PRE_eqhist,PRE_clahe,PRE_retina,PRE_tantriggs,PRE_crop,PRE_MAX };

    static const char *EXS[] = { "Pixels","Lbp","Lbpu","TPLbp","FPLbp","MTS","MTS_E","Lbp_E","FpLbp_E","TpLbp_E","GaborLbp","Dct","OrbGrid","SiftGrid",0 };
    static const char *CLS[] = { "NORM_L2","NORM_LSQR","NORM_L1","NORM_HAM","HIST_HELL","HIST_CHI","SVM","EM","LR","COSINE","FISHER",0 };
    static const char *PPS[] = { "none","eqhist","clahe","retina","tantriggs","crop",0 };


    struct FaceVerifier
    {
         virtual void train(InputArrayOfArrays src, InputArray _labels) = 0;
         virtual int same(const Mat & a, const Mat &b) const = 0;
    };
}

Ptr<myface::FaceVerifier> createMyFaceVerifier(int extract=0, int clsfy=0, int preproc=0, int precrop=0,bool flip=false);


#endif // __MyFace_onboard__

