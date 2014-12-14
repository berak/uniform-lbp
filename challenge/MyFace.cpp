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
    Ptr<TextureFeature::Verifier>  cls;
    Ptr<TextureFeature::Reductor>  red;
    Preprocessor pre;
    bool doFlip;

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
            case EXT_Sift:     ext = createExtractorSIFTGrid(15); break;
            case EXT_Sift_G:   ext = createExtractorSIFTGftt(); break;
            case EXT_Grad:     ext = createExtractorGrad(); break;
            case EXT_Grad_E:   ext = createExtractorElasticGrad(); break;
            case EXT_Grad_G:   ext = createExtractorGfttGrad(); break;
            case EXT_Grad_P:   ext = createExtractorPyramidGrad(); break;
            case EXT_GradMag:  ext = createExtractorGfttGradMag(); break;
            default: cerr << "extraction " << extract << " is not yet supported." << endl; exit(-1);
        }
        switch(redu)
        {
            case RED_NONE:     break; //red = createReductorNone(); break;
            case RED_PCA:      red = createReductorPCA(); break;
            case RED_PCA64:    red = createReductorPCA(64); break;
            case RED_PCA64_W:  red = createReductorPCA(64, true); break;
            case RED_PCA_LDA:  red = createReductorPCA_LDA(); break;
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
            case CL_SVM:       cls = createVerifierSVM(2); break;
            case CL_EM:        cls = createVerifierEM(2, 0.25f); break;
            case CL_LR:        cls = createVerifierLR(2, 0.5f); break;
            case CL_BOOST:     cls = createVerifierBoost(2); break;
//            case CL_FISHER:    cls = createVerifierFisher(); break;
            default: cerr << "verification " << clsfy << " is not yet supported." << endl; exit(-1);
        }
    }

    // Trains a FaceVerifier.
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
        images.clear();

        if (! red.empty())
        {
            red->train(features,labels);
            Mat f;
            for (int r=0; r<features.rows; ++r)
            {
                Mat fr;
                red->reduce(features.row(r),fr);
                f.push_back(fr);
            }
            features = f;
        }
        int ok = cls->train(features, labels.reshape(1,features.rows));
        CV_Assert(ok);
        // cerr << "trained " << nfeatbytes << " bytes." << '\r';
    }


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

