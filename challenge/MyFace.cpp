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
    Ptr<TextureFeature::Reductor>  red;
    Ptr<TextureFeature::Verifier>  cls;
    Preprocessor pre;
    bool doFlip;

    Mat labels;
    Mat features;

public:

    MyFace(int extract=0, int redu=0, int clsfy=0, int preproc=0, int crop=0, bool flip=false)
        : pre(preproc,crop)
        , doFlip(flip)
    {
        switch(extract)
        {
            case EXT_Pixels:   ext = createExtractorPixels(); break;
            case EXT_Lbp:      ext = createExtractorLbp(); break;
            case EXT_LBP_E:    ext = createExtractorElasticLbp(); break;
            case EXT_LBP_O:    ext = createExtractorOverlapLbp(); break;
            case EXT_LBP_P:    ext = createExtractorPyramidLbp(); break;
            case EXT_TPLbp:    ext = createExtractorTPLbp(); break;
            case EXT_TPLBP_E:  ext = createExtractorElasticTpLbp(); break;
            case EXT_TPLBP_O:  ext = createExtractorOverlapTpLbp(); break;
            case EXT_TPLBP_P:  ext = createExtractorPyramidTpLbp(); break;
            case EXT_TPLBP_G:  ext = createExtractorGfttTpLbp(); break;
            case EXT_FPLbp:    ext = createExtractorFPLbp(); break;
            case EXT_FPLBP_E:  ext = createExtractorElasticFpLbp(); break;
            case EXT_FPLBP_O:  ext = createExtractorOverlapFpLbp(); break;
            case EXT_FPLBP_P:  ext = createExtractorPyramidFpLbp(); break;
            case EXT_MTS:      ext = createExtractorMTS(); break;
            case EXT_MTS_E:    ext = createExtractorElasticMTS(); break;
            case EXT_MTS_O:    ext = createExtractorOverlapMTS(); break;
            case EXT_MTS_P:    ext = createExtractorPyramidMTS(); break;
            case EXT_BGC1:     ext = createExtractorBGC1(); break;
            case EXT_BGC1_E:   ext = createExtractorElasticBGC1(); break;
            case EXT_BGC1_O:   ext = createExtractorOverlapBGC1(); break;
            case EXT_BGC1_P:   ext = createExtractorPyramidBGC1(); break;
            case EXT_COMB:     ext = createExtractorCombined(); break;
            case EXT_COMB_E:   ext = createExtractorElasticCombined(); break;
            case EXT_COMB_O:   ext = createExtractorOverlapCombined(); break;
            case EXT_COMB_P:   ext = createExtractorPyramidCombined(); break;
            case EXT_COMB_G:   ext = createExtractorGfttCombined(); break;
            case EXT_Gabor:    ext = createExtractorGaborLbp(); break;
            case EXT_Gabor_E:  ext = createExtractorElasticGaborLbp(1); break;
            case EXT_Dct:      ext = createExtractorDct(); break;
            case EXT_Orb:      ext = createExtractorORBGrid(15); break;
            case EXT_Sift:     ext = createExtractorSIFTGrid(20); break;
            case EXT_Sift_G:   ext = createExtractorSIFTGftt(); break;
            case EXT_Grad:     ext = createExtractorGrad(); break;
            case EXT_Grad_E:   ext = createExtractorElasticGrad(); break;
            case EXT_Grad_G:   ext = createExtractorGfttGrad(); break;
            case EXT_Grad_P:   ext = createExtractorPyramidGrad(); break;
            case EXT_GradMag:  ext = createExtractorGfttGradMag(); break;
            case EXT_HDLBP:    ext = createExtractorHighDimLbp(); break;
            default: cerr << "extraction " << extract << " is not yet supported." << endl; exit(-1);
        }
        switch(redu)
        {
            case RED_NONE:     break; //red = createReductorNone(); break;
            case RED_HELL:     red = createReductorHellinger(); break;
            case RED_WHAD:     red = createReductorWalshHadamard(8000); break;
            case RED_RP:       red = createReductorRandomProjection(8000); break;
            case RED_DCT8:     red = createReductorDct(8000); break;
            case RED_DCT12:    red = createReductorDct(12000); break;
            case RED_DCT16:    red = createReductorDct(16000); break;
            case RED_DCT24:    red = createReductorDct(24000); break;
            default: cerr << "Reductor " << redu << " is not yet supported." << endl; exit(-1);
        }
        switch(clsfy)
        {
            case CL_NORM_L2:   cls = createVerifierNearest(NORM_L2); break;
            case CL_NORM_L2SQR:cls = createVerifierNearest(NORM_L2SQR); break;
            case CL_NORM_L1:   cls = createVerifierNearest(NORM_L1); break;
            case CL_NORM_HAM:  cls = createVerifierNearest(NORM_HAMMING2); break;
            case CL_HIST_HELL: cls = createVerifierHist(HISTCMP_HELLINGER); break;
            case CL_HIST_CHI:  cls = createVerifierHist(HISTCMP_CHISQR); break;
            case CL_SVM:       cls = createVerifierSVM(); break;
            case CL_EM:        cls = createVerifierEM(); break;
            case CL_LR:        cls = createVerifierLR(); break;
            case CL_BOOST:     cls = createVerifierBoost(); break;
            case CL_KMEANS:    cls = createVerifierKmeans(); break;
            default: cerr << "verification " << clsfy << " is not yet supported." << endl; exit(-1);
        }
    }

    virtual int addTraining(const Mat & img, int label) 
    {
        Mat feat1;
        ext->extract(pre.process(img), feat1);

        Mat fr = feat1.reshape(1,1);
        if (! red.empty())
            red->reduce(fr,fr);

        features.push_back(fr);
        labels.push_back(label);
        cerr <<fr.cols << " i_" << labels.rows << "\r";
        return labels.rows;
    }
    virtual bool train()
    {
        cerr << "." << features.cols << "     ";
        int ok = cls->train(features, labels.reshape(1,features.rows));
        cerr << ".\r";
        CV_Assert(ok);
        features.release();
        labels.release();
        return ok!=0;
    }


    //// Trains a FaceVerifier.
    //virtual void train(InputArrayOfArrays src, InputArray _labels)
    //{
    //    Mat_<int> labels1 = _labels.getMat();
    //    Mat labels;

    //    vector<Mat> images;
    //    src.getMatVector(images);
    //    int nfeatbytes=0;
    //    Mat features;
    //    for (size_t i=0; i<images.size(); i++)
    //    {
    //        Mat img = pre.process(images[i]);

    //        Mat feat1;
    //        nfeatbytes = ext->extract(img, feat1);

    //        Mat fr = feat1.reshape(1,1);
    //        if (! red.empty())
    //            red->reduce(fr,fr);

    //        features.push_back(fr);

    //        labels.push_back(labels1(i));

    //        images[i].release();
    //        cerr << i << "/" << images.size() << "\r";
    //    }
    //    images.clear();
    //    cerr << "." << features.cols;
    //    int ok = cls->train(features, labels.reshape(1,features.rows));
    //    cerr << ".\r";
    //    CV_Assert(ok);
    //    // cerr << "trained " << nfeatbytes << " bytes." << '\r';
    //}


    virtual int same(const Mat & a, const Mat &b) const
    {
        Mat feat1, feat2;
        ext->extract(pre.process(a), feat1);
        ext->extract(pre.process(b), feat2);
        if (! red.empty())
        {
            red->reduce(feat1,feat1);
            red->reduce(feat2,feat2);
        }
        return cls->same(feat1,feat2);
    }
};

} // myface


Ptr<myface::FaceVerifier> createMyFaceVerifier(int ex, int re, int cl, int pr, int pc, bool flip)
{
    return makePtr<myface::MyFace>(ex, re, cl, pr, pc, flip);
}

