#ifndef __MyFace_onboard__
#define __MyFace_onboard__
#include "opencv2/face.hpp"
using namespace cv;

namespace myface {

    enum EXT { EXT_Pixels,EXT_Lbp,EXT_FPLbp,EXT_MTS,EXT_GaborLbp,EXT_Dct,EXT_OrbGrid,EXT_MAX };
    enum CLA { CL_NORM_L2,CL_NORM_L1,CL_NORM_HAM,CL_HIST_HELL,CL_HIST_ISEC,CL_SVM,CL_SVMMulti,CL_COSINE,CL_MAX };
    enum PRE { PRE_none,PRE_eqhist,PRE_clahe,PRE_retina,PRE_crop,PRE_MAX };

    static const char *EXS[] = { "EXT_Pixels","EXT_Lbp","EXT_FPLbp","EXT_MTS","EXT_GaborLbp","EXT_Dct","EXT_OrbGrid",0 };
    static const char *CLS[] = { "CL_NORM_L2","CL_NORM_L1","CL_NORM_HAM","CL_HIST_HELL","CL_HIST_ISEC","CL_SVM","CL_SVMMulti","CL_COSINE",0 };
    static const char *PPS[] = { "none","eqhist","clahe","retina","crop",0 };
}

Ptr<face::FaceRecognizer> createMyFaceRecognizer(int extract=0, int clsfy=0, int preproc=0, int precrop=0,int psize=250);


#endif // __MyFace_onboard__

