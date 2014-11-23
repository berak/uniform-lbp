//
// run k fold crossvalidation train/test on  person db
//

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
using namespace cv;


#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
using namespace std;


#include "TextureFeature.h"
#include "extractDB.h"

bool debug = false;

//
// the current compilation of extractors / classifiers. (should probably go into a header)
//

extern cv::Ptr<TextureFeature::Extractor> createExtractorPixels(int resw=0, int resh=0);
extern cv::Ptr<TextureFeature::Extractor> createExtractorMoments();
extern cv::Ptr<TextureFeature::Extractor> createExtractorLbp(int gridx=8, int gridy=8, int u_table=-1);
extern cv::Ptr<TextureFeature::Extractor> createExtractorBGC1(int gx=8, int gy=8, int utable=-1);
extern cv::Ptr<TextureFeature::Extractor> createExtractorWLD(int gx=8, int gy=8, int tf=CV_32F);
extern cv::Ptr<TextureFeature::Extractor> createExtractorLQP(int gx=8, int gy=8);
extern cv::Ptr<TextureFeature::Extractor> createExtractorMTS(int gx=8, int gy=8);
extern cv::Ptr<TextureFeature::Extractor> createExtractorSTU(int gx=8, int gy=8, int kp1=5);
extern cv::Ptr<TextureFeature::Extractor> createExtractorGLCM(int gx=8, int gy=8);
extern cv::Ptr<TextureFeature::Extractor> createExtractorGaborLbp(int gx=8, int gy=8, int u_table=0, int kernel_size=8);
extern cv::Ptr<TextureFeature::Extractor> createExtractorDct();
extern cv::Ptr<TextureFeature::Extractor> createExtractorORBGrid();
extern cv::Ptr<TextureFeature::Extractor> createExtractorSIFTGrid();
extern cv::Ptr<TextureFeature::Extractor> createExtractorFPLbp(int gx=8, int gy=8);


extern cv::Ptr<TextureFeature::Classifier> createClassifierNearest(int norm_flag=NORM_L2);
extern cv::Ptr<TextureFeature::Classifier> createClassifierHist(int flag=HISTCMP_CHISQR);
extern cv::Ptr<TextureFeature::Classifier> createClassifierCosine();
extern cv::Ptr<TextureFeature::Classifier> createClassifierKNN(int n=1);                // TODO: needs a way to get to the k-1 others
extern cv::Ptr<TextureFeature::Classifier> createClassifierSVM(double degree = 0.5,double gamma = 0.8,double coef0 = 0,double C = 0.99, double nu = 0.002, double p = 0.5);
extern cv::Ptr<TextureFeature::Classifier> createClassifierSVMMulti();

extern cv::Ptr<TextureFeature::Classifier> createClassifierEigen();
extern cv::Ptr<TextureFeature::Classifier> createClassifierFisher();

double ct(int64 t) {    return double(t) / cv::getTickFrequency(); }

using TextureFeature::Extractor;
using TextureFeature::Classifier;


RNG rng(getTickCount());


int crossfoldData( Ptr<Extractor> ext,
                    Mat & trainFeatures,
                    Mat & trainLabels,
                    Mat & testFeatures,
                    Mat & testLabels,
                    const vector<Mat> &images, 
                    const vector<int> &labels, 
                    const vector<vector<int>> &persons,
                    size_t f, size_t fold )
{
    int fsiz=0;

    // split train/test set per person:
    for ( size_t j=0; j<persons.size(); j++ )
    {
        size_t n_per_person = persons[j].size();
        if (n_per_person < fold)
            continue;
        int r = (fold != 0) ? (n_per_person/fold) : -1;
        for ( size_t n=0; n<n_per_person; n++ )
        {
            int index = persons[j][n];

            Mat feature;
            fsiz = ext->extract(images[index],feature);

            // sliding window per fold
            if ( (fold>1) && (n >= f*r) && (n <= (f+1)*r) ) 
            {
                testFeatures.push_back(feature);
                testLabels.push_back(labels[index]);
            }
            else
            {
                trainFeatures.push_back(feature);
                trainLabels.push_back(labels[index]);
            }
        }
    }
    return fsiz;
}


double runtest(string name, Ptr<Extractor> ext, Ptr<Classifier> cls, const vector<Mat> &images, const vector<int> &labels, const vector<vector<int>> &persons, size_t fold=10 ) 
{
    //
    // for each fold, take alternating n/fold items for test, the others for training
    //
    // each test is confused on its own over a lot of folds..
    Mat confusion = Mat::zeros(persons.size(),persons.size(),CV_32F);
    vector<float> tpr;
    vector<float> fnr;

    int64 t0=getTickCount();
    int fsiz=0;
    for ( size_t f=0; f<fold; f++ )
    {
        int64 t1 = cv::getTickCount();
        Mat trainFeatures, trainLabels;
        Mat testFeatures,  testLabels;

        fsiz = crossfoldData(ext,trainFeatures,trainLabels,testFeatures,testLabels,images,labels,persons,f,fold);
       
        cls->train(trainFeatures.reshape(1,trainLabels.rows),trainLabels);

        Mat conf = Mat::zeros(confusion.size(), CV_32F);
        for ( int i=0; i<testFeatures.rows; i++ )
        {
            Mat res;
            cls->predict(testFeatures.row(i).reshape(1,1), res);
    
            int pred = int(res.at<float>(0));
            int ground = testLabels.at<int>(i);
            if ( pred<0 || ground<0 )
            {
                cerr << "neg prediction " << f << " " << i << " " << pred << " " << ground << endl;
                continue;
            }
            conf.at<float>(ground, pred) ++;
        }
        confusion += conf;

        double all = sum(confusion)[0];
        double neg = all - sum(confusion.diag())[0];
        double err = double(neg)/all;
        cout << format("%-13s %-2d %6d %6d %6d %8.3f",name.c_str(),(f+1), fsiz, int(all-neg), int(neg), (1.0-err)) << '\r';
    }


    // evaluate. this is probably all too simple.
    double all = sum(confusion)[0];
    double neg = all - sum(confusion.diag())[0];
    double err = double(neg)/all;
    int64 t1=getTickCount() - t0;
    double t(t1/getTickFrequency());
    cout << format("%-16s %6d %6d %6d %8.3f %8.3f",name.c_str(), fsiz, int(all-neg), int(neg), (1.0-err), t) << endl;
    if (debug) cout << "confusion" << endl << confusion(Range(0,min(20,confusion.rows)), Range(0,min(20,confusion.cols))) << endl;
    return err;
}


//
//
// face att.txt 5     5             1        0     
// face db      fold  reco    preprocessing  debug 
//
// special: reco==0 will run *all* recognizers available on a given db
//
int main(int argc, const char *argv[]) 
{
    vector<Mat> images;
    Mat labels;

    std::string db_path("senthil.txt");
    //std::string db_path("att.txt");
    //std::string db_path("yale.txt");
    if ( argc>1 ) db_path = argv[1];

    size_t fold = 4;
    if ( argc>2 ) fold = atoi(argv[2]);

    int rec = 7;
    if ( argc>3 ) rec = atoi(argv[3]);

    int preproc = 0; // 0-none 1-eqhist 2-tan_triggs 3-clahe 4-retina
    if ( argc>4 ) preproc = atoi(argv[4]);

    if ( argc>5 ) debug = atoi(argv[5])!=0;

    
    extractDB(db_path, images, labels, preproc, 500, 120);

    // per person id lookup
    vector<vector<int>> persons;
    setupPersons( labels, persons );
    fold = std::min(fold,images.size()/persons.size());

    // some diagnostics:
    String dbs = db_path.substr(0,db_path.find_last_of('.')) + ":";
    char *pp[] = { "no preproc", "equalizeHist", "tan-triggs", "CLAHE", "retina" };
    if ( rec==0 )
        cout << "--------------------------------------------------------------" << endl;
    cout << format("%-19s",dbs.c_str()) << fold  << " fold, " << persons.size()  << " classes, " << images.size() << " images, " << pp[preproc] << endl;
    if ( rec==0 ) 
    {
        cout << "--------------------------------------------------------------" << endl;
        cout << "[method]       [f_bytes]  [pos]  [neg]   [hit]   [time]  " << endl;
    }

    // loop through all tests for rec==0, do one test else.
    int n=42;
    if ( rec > 0 ) 
    {
        n = rec+1;
    }
    for ( ; rec<n; rec++ ) 
    {
        switch(rec)
        {
        default: continue;
        case 1:  runtest("pixels_L2",    createExtractorPixels(120,120),   createClassifierNearest(),               images,labels,persons, fold); break;
        case 2:  runtest("pixels_svm",   createExtractorPixels(60,60),     createClassifierSVM(),                   images,labels,persons, fold); break;
        case 3:  runtest("pixels_cosine",createExtractorPixels(120,120),   createClassifierCosine(),                images,labels,persons, fold); break;
        case 4:  runtest("pixels_multi", createExtractorPixels(60,60),     createClassifierSVMMulti(),              images,labels,persons, fold); break;
        case 5:  runtest("lbp_L2",       createExtractorLbp(),             createClassifierNearest(),               images,labels,persons, fold); break;
        case 6:  runtest("lbp_svm",      createExtractorLbp(),             createClassifierSVM(),                   images,labels,persons, fold); break;
        case 7:  runtest("fplbp_svm",    createExtractorFPLbp(),           createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 8:  runtest("lbp_chisqr",   createExtractorLbp(),             createClassifierHist(),                  images,labels,persons, fold); break;
        case 9:  runtest("lbp_hell",     createExtractorLbp(),             createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        //case 10: runtest("lbpu_hell",    createExtractorLbp(),             createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        case 11: runtest("lbpu_red_hell",createExtractorLbp(8,8,2),        createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        case 12: runtest("bgc1_hell",    createExtractorBGC1(),            createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        //case 13: runtest("bgc1_red_hell",createExtractorBGC1(8,8,2),       createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        //case 14: runtest("wld_L1",       createExtractorWLD(8,8,CV_8U),    createClassifierNearest(NORM_L1),        images,labels,persons, fold); break;
        //case 15: runtest("wld_hell",     createExtractorWLD(),             createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        case 16: runtest("mts_svm",      createExtractorMTS(),             createClassifierSVM(),                   images,labels,persons, fold); break;
        case 17: runtest("mts_hell",     createExtractorMTS(),             createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        case 18: runtest("stu_svm",      createExtractorSTU(),             createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 19: runtest("glcm_hell",    createExtractorGLCM(),            createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        case 19: runtest("glcm_svm",     createExtractorGLCM(),            createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 18: runtest("lqp_hell",     createExtractorLQP(),             createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        case 21: runtest("gabor_red",    createExtractorGaborLbp(8,8,2),   createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        case 22: runtest("gabor_svm",    createExtractorGaborLbp(),        createClassifierSVM(),                   images,labels,persons, fold); break;
        case 23: runtest("dct_cosine",   createExtractorDct(),             createClassifierCosine(),                images,labels,persons, fold); break;
        case 24: runtest("dct_L2",       createExtractorDct(),             createClassifierNearest(),               images,labels,persons, fold); break;
        case 26: runtest("dct_svm",      createExtractorDct(),             createClassifierSVM(),                   images,labels,persons, fold); break;
        case 27: runtest("orbgrid_L1",   createExtractorORBGrid(),         createClassifierNearest(NORM_L1),        images,labels,persons, fold); break;
        case 28: runtest("siftgrid_L2",  createExtractorSIFTGrid(),        createClassifierNearest(NORM_L2),        images,labels,persons, fold); break;
        case 40: runtest("eigen",        createExtractorPixels(),          createClassifierEigen(),                 images,labels,persons, fold); break;
        case 41: runtest("fisher",       createExtractorPixels(),          createClassifierFisher(),                images,labels,persons, fold); break;
        }
    }
    return 0;
}



