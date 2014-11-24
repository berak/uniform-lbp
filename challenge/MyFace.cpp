#include "opencv2/opencv.hpp"
#include "opencv2/bioinspired.hpp"
using namespace cv;

#include "MyFace.h"
#include "../TextureFeature.h"

#include <iostream>
#include <vector>
using namespace std;

extern cv::Ptr<TextureFeature::Extractor> createExtractorPixels(int resw=0, int resh=0);
extern cv::Ptr<TextureFeature::Extractor> createExtractorMoments();
extern cv::Ptr<TextureFeature::Extractor> createExtractorLbp(int gridx=8, int gridy=8, int u_table=-1);
extern cv::Ptr<TextureFeature::Extractor> createExtractorFPLbp(int gx=8, int gy=8);
extern cv::Ptr<TextureFeature::Extractor> createExtractorBGC1(int gx=8, int gy=8, int utable=-1);
extern cv::Ptr<TextureFeature::Extractor> createExtractorWLD(int gx=8, int gy=8, int tf=CV_32F);
extern cv::Ptr<TextureFeature::Extractor> createExtractorLQP(int gx=8, int gy=8);
extern cv::Ptr<TextureFeature::Extractor> createExtractorMTS(int gx=8, int gy=8);
extern cv::Ptr<TextureFeature::Extractor> createExtractorSTU(int gx=8, int gy=8, int kp1=5);
extern cv::Ptr<TextureFeature::Extractor> createExtractorGLCM(int gx=8, int gy=8);
extern cv::Ptr<TextureFeature::Extractor> createExtractorGaborLbp(int gx=8, int gy=8, int u_table=0, int kernel_size=8);
extern cv::Ptr<TextureFeature::Extractor> createExtractorDct();
extern cv::Ptr<TextureFeature::Extractor> createExtractorORBGrid(int g=25);
extern cv::Ptr<TextureFeature::Extractor> createExtractorSIFTGrid();

extern cv::Ptr<TextureFeature::Classifier> createClassifierNearest(int norm_flag=NORM_L2);
extern cv::Ptr<TextureFeature::Classifier> createClassifierHist(int flag=HISTCMP_CHISQR);
extern cv::Ptr<TextureFeature::Classifier> createClassifierCosine();
extern cv::Ptr<TextureFeature::Classifier> createClassifierKNN(int n=1);                // TODO: needs a way to get to the k-1 others
extern cv::Ptr<TextureFeature::Classifier> createClassifierSVM(double degree = 0.5,double gamma = 0.8,double coef0 = 0,double C = 0.99, double nu = 0.002, double p = 0.5);
extern cv::Ptr<TextureFeature::Classifier> createClassifierSVMMulti();

extern cv::Ptr<TextureFeature::Classifier> createClassifierEigen();
extern cv::Ptr<TextureFeature::Classifier> createClassifierFisher();


namespace myface {

class MyFace : public face::FaceRecognizer 
{
    Ptr<TextureFeature::Extractor> ext;
    Ptr<TextureFeature::Classifier> cls;
    Ptr<bioinspired::Retina> retina;
    Ptr<CLAHE> clahe;
    int preproc;
    int precrop;
    bool doFlip;
public:

    MyFace(int extract=0, int clsfy=0, int preproc=0, int precrop=0,int psize=250)
        : clahe(createCLAHE(50))
        , retina(bioinspired::createRetina(Size(psize,psize)))
        , preproc(preproc)
        , precrop(precrop)
        , doFlip(false)
    {
        switch(extract) 
        {
            default:
            case EXT_Pixels:   ext = createExtractorPixels(); break;
            case EXT_Lbp:      ext = createExtractorLbp(); break;
            case EXT_FPLbp:    ext = createExtractorFPLbp(); break;
            case EXT_MTS:      ext = createExtractorMTS(); break;
            case EXT_GaborLbp: ext = createExtractorGaborLbp(); break;
            case EXT_Dct:      ext = createExtractorDct(); break;
            case EXT_OrbGrid:  ext = createExtractorORBGrid(15);
        }
        switch(clsfy) 
        {
            default:
            case CL_NORM_L2:   cls = createClassifierNearest(NORM_L2); break;
            case CL_NORM_L1:   cls = createClassifierNearest(NORM_L1); break;
            case CL_NORM_HAM:  cls = createClassifierNearest(NORM_HAMMING); break;
            case CL_HIST_HELL: cls = createClassifierHist(HISTCMP_HELLINGER); break;
            case CL_HIST_ISEC: cls = createClassifierHist(HISTCMP_INTERSECT); break;
            case CL_SVM:       cls = createClassifierSVM(); break;
            case CL_SVMMulti:  cls = createClassifierSVMMulti(); break;
            case CL_COSINE:    cls = createClassifierCosine(); break;
        }

        // (realistic setup)
        bioinspired::Retina::RetinaParameters ret_params;
        ret_params.OPLandIplParvo.horizontalCellsGain = 0.7f;
        ret_params.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity = 0.39f;
        ret_params.OPLandIplParvo.ganglionCellsSensitivity = 0.39f;
        retina->setup(ret_params);
    }

    Mat preprocess(const Mat & imgin) const
    {
        Mat imgcropped(imgin, Rect(precrop, precrop, imgin.cols-2*precrop, imgin.rows-2*precrop));
        Mat imgout;
        switch(preproc)
        {
            default:
            case 0: imgout=imgcropped.clone(); break;
            case 1: equalizeHist(imgcropped,imgout); break;
            case 2: clahe->apply(imgcropped,imgout); break;
            case 3: retina->run(imgcropped); retina->getParvo(imgout); break;
            case 4: resize(imgcropped,imgout,Size(60,60)); break;
        }
        return imgout;
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
        for ( size_t i=0; i<images.size(); i++ )
        {
            Mat img = preprocess(images[i]);

            Mat feat1;
            nfeatbytes = ext->extract(img, feat1);
            features.push_back(feat1.reshape(1,1));
            labels.push_back( labels1(i) );

            if (doFlip) // add a flipped duplicate
            {
                flip(img,img,1);

                Mat feat2;
                ext->extract(img, feat2);
                features.push_back(feat2.reshape(1,1));
                labels.push_back( labels1(i) );
            }
        }
        cls->train( features, labels.reshape(1,features.rows) );
        cerr << "trained " << nfeatbytes << " bytes." << endl;
    }

    // Gets a prediction from a FaceRecognizer.
    virtual int predict(InputArray src) const
    {
        Mat img = src.getMat();

        Mat feat;
        ext->extract(preprocess(img), feat);

        Mat res;
        cls->predict(feat,res);

        return int(res.at<float>(0));
    }



    //
    // dummies ( not needed for this challenge )
    //
    //
    // Updates a FaceRecognizer.
    CV_WRAP virtual void update(InputArrayOfArrays src, InputArray labels)     { /*NO_IMPL*/ }
    // Predicts the label and confidence for a given sample.
    CV_WRAP virtual void predict(InputArray src, CV_OUT int &label, CV_OUT double &confidence) const  { /*NO_IMPL*/ }
    // Serializes this object to a given filename.
    CV_WRAP virtual void save(const String& filename) const { /*NO_IMPL*/ }
    // Deserializes this object from a given filename.
    CV_WRAP virtual void load(const String& filename)  { /*NO_IMPL*/ }
    // Serializes this object to a given cv::FileStorage.
    virtual void save(FileStorage& fs) const  { /*NO_IMPL*/ }
    // Deserializes this object from a given cv::FileStorage.
    virtual void load(const FileStorage& fs) { /*NO_IMPL*/ }
    // Sets additional string info for the label
    virtual void setLabelInfo(int label, const String& strInfo) { /*NO_IMPL*/ }
    // Gets string info by label
    virtual String getLabelInfo(int label) const { return ""; /*NO_IMPL*/ }
    // Gets labels by string
    virtual std::vector<int> getLabelsByString(const String& str) const { return std::vector<int>(); /*NO_IMPL*/ }
};

} // myface

Ptr<face::FaceRecognizer> createMyFaceRecognizer(int ex, int cl, int pr, int pc, int ps)
{
    return makePtr<myface::MyFace>(ex, cl, pr, pc, ps);
}

