#include "opencv2/opencv.hpp"
using namespace cv;

#include "MyFace.h"
#include "../TextureFeature.h"
#include "../Preprocessor.h"

#include <iostream>
#include <vector>
using namespace std;



namespace myface {

class MyFace : public FaceVerifier
{
    Ptr<TextureFeature::Extractor> ext;
    Ptr<TextureFeature::Verifier> cls;
    Preprocessor pre;
    bool doFlip;

public:

    MyFace(int extract=0, int clsfy=0, int preproc=0, int crop=0, bool flip=false)
        : pre(preproc,crop)
        , doFlip(flip)
    {
        switch(extract)
        {
            case EXT_Pixels:   ext = createExtractorPixels(); break;
            case EXT_Lbp:      ext = createExtractorLbp(); break;
            case EXT_Lbpu:     ext = createExtractorLbp(8,8,0); break;
            case EXT_TPLbp:    ext = createExtractorTPLbp(); break;
            case EXT_FPLbp:    ext = createExtractorFPLbp(); break;
            case EXT_MTS:      ext = createExtractorMTS(); break;
            case EXT_MTS_E:    ext = createExtractorElasticMTS(); break;
            case EXT_LBP_E:    ext = createExtractorElasticLbp(); break;
            case EXT_TPLBP_E:  ext = createExtractorElasticTpLbp(); break;
            case EXT_FPLBP_E:  ext = createExtractorElasticFpLbp(); break;
            case EXT_GaborLbp: ext = createExtractorGaborLbp(); break;
            case EXT_Dct:      ext = createExtractorDct(); break;
            case EXT_OrbGrid:  ext = createExtractorORBGrid(15); break;
            case EXT_SiftGrid: ext = createExtractorSIFTGrid(15); break;
            default: cerr << clsfy << " is not yet supported." << endl; exit(-1);
        }
        switch(clsfy)
        {
            case CL_NORM_L2:   cls = createVerifierNearest(NORM_L2); break;
            case CL_NORM_L2SQR:cls = createVerifierNearest(NORM_L2SQR); break;
            case CL_NORM_L1:   cls = createVerifierNearest(NORM_L1); break;
            case CL_NORM_HAM:  cls = createVerifierNearest(NORM_HAMMING); break;
            case CL_HIST_HELL: cls = createVerifierHist(HISTCMP_HELLINGER); break;
            case CL_HIST_CHI:  cls = createVerifierHist(HISTCMP_CHISQR); break;
            case CL_SVM:       cls = createVerifierSVM(2); break;
            case CL_EM:        cls = createVerifierEM(2, 0.25f); break;
            case CL_LR:        cls = createVerifierLR(2, 0.5f); break;
            case CL_FISHER:    cls = createVerifierFisher(); break;
            default: cerr << clsfy << " is not yet supported." << endl; exit(-1);
            //case CL_SVMMulti:  cls = createClassifierSVMMulti(); break;
            //case CL_COSINE:    cls = createClassifierCosine(); break;
        }
    }

    // Trains a FaceRecognizer.
    virtual void train(InputArrayOfArrays src, InputArray _labels)
    {
        Mat_<int> labels1 = _labels.getMat();
        Mat labels;

        vector<Mat> images;
        src.getMatVector(images);
        int nfeatbytes=0;
        Mat features;
        for (size_t i=0; i<images.size(); i++)
        {
            Mat img = pre.process(images[i]);
            images[i].release();

            Mat feat1;
            nfeatbytes = ext->extract(img, feat1);
            features.push_back(feat1.reshape(1,1));
            labels.push_back(labels1(i));

            if (doFlip) // add a flipped duplicate
            {
                flip(img,img,1);

                Mat feat2;
                ext->extract(img, feat2);
                features.push_back(feat2.reshape(1,1));
                labels.push_back(labels1(i));
            }
        }
        int ok = cls->train(features, labels.reshape(1,features.rows));
        CV_Assert(ok);
        cerr << "trained " << nfeatbytes << " bytes." << '\r';
    }

    
    virtual int same(const Mat & a, const Mat &b) const
    {
        Mat feat1;
        ext->extract(pre.process(a), feat1);

        Mat feat2;
        ext->extract(pre.process(b), feat2);

        return cls->same(feat1,feat2);
    }
};

} // myface

Ptr<myface::FaceVerifier> createMyFaceVerifier(int ex, int cl, int pr, int pc, bool flip)
{  return makePtr<myface::MyFace>(ex, cl, pr, pc, flip);  }

