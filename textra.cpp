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
#include "tan_triggs.h"


//
// the current compilation of extractors / classifiers. (should probably go into a header)
//

extern cv::Ptr<TextureFeature::Extractor> createExtractorPixels(int resw=0, int resh=0);
extern cv::Ptr<TextureFeature::Extractor> createExtractorMoments();
extern cv::Ptr<TextureFeature::Extractor> createExtractorLbp(int gridx=8, int gridy=8);
extern cv::Ptr<TextureFeature::Extractor> createExtractorLbpUniform(int gridx=8, int gridy=8, int u_table=0);
extern cv::Ptr<TextureFeature::Extractor> createExtractorWLD(int gx=8, int gy=8, int tf=CV_32F);

extern cv::Ptr<TextureFeature::Classifier> createClassifierNearest(int norm_flag=NORM_L2);
extern cv::Ptr<TextureFeature::Classifier> createClassifierHist(int flag=HISTCMP_CHISQR);
extern cv::Ptr<TextureFeature::Classifier> createClassifierKNN(int n=1);                // TODO: needs a way to get to the k-1 others
extern cv::Ptr<TextureFeature::Classifier> createClassifierSVM(double degree = 0.5,double gamma = 0.8,double coef0 = 0,double C = 0.99, double nu = 0.2, double p = 0.5);
//extern cv::Ptr<TextureFeature::Classifier> createClassifierBayes();                     // TODO: slow
//extern cv::Ptr<TextureFeature::Classifier> createClassifierRTrees();                    // TODO: broken
//extern cv::Ptr<TextureFeature::Classifier> createClassifierDTree();                     // TODO: broken






//
// read a 'path <blank> label' list
//
int readtxt( const char * fname, std::vector<std::string> & names, std::vector<int> & labels, size_t maxim  ) 
{
    int maxid=-1;
    std::ifstream in(fname);
    while( in.good() && !in.eof() )
    {
        std::string path ;
        in >> path;
        names.push_back(path);

        int label;
        in >> label;
        labels.push_back(label);

        maxid=std::max(maxid,label);
        if ( labels.size() >= maxim ) 
            break;
    }
    return maxid;
}


double ct(int64 t)
{
    return double(t) / cv::getTickFrequency();
}


//
// imglists per person
//
void setupPersons( const vector<int> & labels, vector<vector<int>> & persons )
{
    // find out which index belongs to which person
    //
    persons.resize(1);
    int prvid=0;
    for ( size_t j=0; j<labels.size(); j++ )
    {
        int id = labels[j];
        if (prvid!=id)
        {
            persons.push_back(vector<int>());
            prvid=id;
        }
        persons.back().push_back(j);
    }
}




using TextureFeature::Extractor;
using TextureFeature::Classifier;

//
// train the classifier on features from the extractor
//
void epoch_train(Ptr<Extractor> ext, Ptr<Classifier> cls, const vector<Mat> & img, const Mat & labels)
{
    Mat train;
    for ( size_t i=0; i<img.size(); i++ )
    {
        Mat feature;
        ext->extract(img[i],feature);
        train.push_back(feature);
    }
    cls->train(train.reshape(1,labels.rows),labels);
}


//
// classify test images on features from the extractor, feed the confusion
//
void epoch_test(Ptr<Extractor> ext, Ptr<Classifier> cls, const vector<Mat> & img, const Mat & labels, Mat &confusion)
{
    for ( size_t i=0; i<img.size(); i++ )
    {
        Mat feature;
        ext->extract(img[i], feature);
        Mat res;
        cls->predict(feature.reshape(1,1), res);

        int pred = int(res.at<float>(0));
        int ground  = labels.at<int>(i);
        if ( ground<confusion.rows && pred>=0 && pred<confusion.cols )
            confusion.at<int>(ground, pred) ++;
    }
}


//
// for each fold, take alternating n/fold items for test, the others for training
//
void runtest(string name, Ptr<Extractor> ext, Ptr<Classifier> cls, const vector<Mat>& images, const vector<int>& labels, const vector<vector<int>>& persons, size_t fold=10, bool verbose=false ) 
{
    // each test is confused on its own over a lot of folds..
    Mat confusion = Mat::zeros(persons.size(),persons.size(),CV_32S);
    for ( size_t f=0; f<fold; f++ )
    {
        vector<Mat> trainImages;
        Mat trainLabels;
        vector<Mat> testImages;
        Mat testLabels;

        int64 t1 = cv::getTickCount();
        // split train/test set per person:
        for ( size_t j=0; j<persons.size(); j++ )
        {
            size_t n_per_person = persons[j].size();
            if ( n_per_person < fold ) continue;
            int r = -1;
            if ( fold != 0 )
                r = n_per_person/fold;
            for ( size_t n=0; n<n_per_person; n++ )
            {
                int index = persons[j][n];
                if ( (fold>1) && (n >= f*r) && (n <= (f+1)*r) ) // sliding window per fold
                {
                    testImages.push_back(images[index]);
                    testLabels.push_back(labels[index]);
                }
                else
                {
                    trainImages.push_back(images[index]);
                    trainLabels.push_back(labels[index]);
                }
            }
        }

        epoch_train(ext,cls,trainImages,trainLabels);
        epoch_test (ext,cls,testImages, testLabels, confusion);

        cout << '.';
        double all = sum(confusion)[0], neg = all - sum(confusion.diag())[0];
        if ( verbose ) cerr << format(" %-16s %3d %5d %d",name.c_str(), f, (all-neg), neg) << endl;
    }

    // evaluate. this is probably all too simple.
    double all = sum(confusion)[0];
    double neg = all - sum(confusion.diag())[0];
    double err = double(neg)/all;
    //if ( verbose ) cerr << confusion << endl;
    cout << format(" %-16s %6.1f %6.1f %6.3f",name.c_str(), (all-neg), neg, (1.0-err)) << endl;
}



//
//
// face att.txt 5     5      1         1
// face db      fold  reco  verbose  preprocessing
//
// special: fold==1 will train on all images , save them, and ignore the test step
// special: reco==0 will run *all* recognizers available on a given db
//
int main(int argc, const char *argv[]) 
{
    vector<Mat> images;
    vector<int> labels;
    vector<string> vec;

    std::string imgpath("att.txt");
    //std::string imgpath("yale.txt");
    if ( argc>1 ) imgpath = argv[1];

    size_t fold = 5;
    if ( argc>2 ) fold = atoi(argv[2]);

    int rec = 11;
    if ( argc>3 ) rec = atoi(argv[3]);

    bool verbose = false;
    if ( argc>4 ) verbose = (argv[4][0] != '0');

    int preproc = 0; // 0-none 1-eqhist 2-tantriggs 3-clahe
    if ( argc>5 ) preproc = atoi(argv[5]);

    // read face db
    size_t maxim  = 400; // restrict it.
    int nsubjects = 1 + readtxt(imgpath.c_str(), vec, labels, maxim);


    if ( argc==1 )
    {
        cerr << argv[0] << " " << imgpath << " " << fold << " " << rec << endl;
    }


    //
    // read the images, 
    //   correct labels if empty images are skipped
    //   also apply preprocessing,
    //
    int load_flag = preproc==-1 ? 1 : 0;
    int skipped = 0;
    vector<int> clabels; 
    for ( size_t i=0; i<vec.size(); i++ )
    {
        Mat mm = imread(vec[i], load_flag);
        if ( mm.empty() )
        {
            skipped ++;
            continue;
        }
        Mat m2;
        resize(mm, m2, Size(90,90));
        switch(preproc) 
        {
            default:
            case 0: mm = m2; break;
            case 1: cv::equalizeHist( m2, mm ); break;
            case 2: cv::normalize( tan_triggs_preprocessing(m2), mm, 0, 255, NORM_MINMAX, CV_8UC1); break;
            case 3: 
            {

                Ptr<CLAHE> clahe = createCLAHE();
                clahe->setClipLimit(50); 
                clahe->apply(m2, mm); 
                break;
            }
        }            
        images.push_back(mm);
        clabels.push_back(labels[i]);
        //if ( i%33==0) imshow("i",mm), waitKey(0);
    }
    labels = clabels;

    // per person id lookup
    vector<vector<int>> persons;
    setupPersons( labels, persons );

    fold = std::min(fold,images.size()/nsubjects);

    cout << fold  << " fold, " ;
    cout << nsubjects  << " classes, " ;
    cout << images.size() << " images, ";
    cout << images.size()/nsubjects << " per class, ";
    if ( skipped ) cout << "(" << skipped << " images skipped), ";
    char *pp[] = { "no preproc", "equalizeHist", "tan-triggs", "CLAHE" };
    cout << pp[preproc];
    cout << endl;

    int n=23;
    if ( rec > 0 ) // run through all possibilities for 0, restrict it to the chosen else.
    {
        n = rec+1;
    }
    for ( ; rec<n; rec++ ) 
    {
        switch(rec)
        {
        default:
            continue;
        case 0:
            runtest("pixels",
                createExtractorPixels(120,120),
                createClassifierNearest(), 
                images,labels,persons, fold,verbose);
            break;
        //case 1:
        //    runtest("lbp", createExtractorLbp(), createClassifierNearest(), images,labels,persons, fold,verbose);
        //    break;
        //case 2:
        //    runtest("lbpu", createExtractorLbpUniform(), createClassifierNearest(), images,labels,persons, fold,verbose);
        //    break;
        //case 3:
        //    runtest("lbpu_mod",createExtractorLbpUniform(8,8,1), createClassifierNearest(), images,labels,persons, fold,verbose);
        //    break;
        //case 4:
        //    runtest("lbpu_red",createExtractorLbpUniform(8,8,2), createClassifierNearest(), images,labels,persons, fold,verbose);
        //    break;
        //case 5:
        //    runtest("lbpu_svm",createExtractorLbpUniform(8,8,0), createClassifierSVM(), images,labels,persons, fold,verbose);
        //    break;
        case 6:
            runtest("lbpu_red_svm",
                createExtractorLbpUniform(8,8,2),
                createClassifierSVM(),
                images,labels,persons, fold,verbose);
            break;
        //case 7:
        //    runtest("lbp_chisqr", createExtractorLbp(), createClassifierHist(), images,labels,persons, fold,verbose);
        //    break;
        case 8:
            runtest("lbp_hell",
                createExtractorLbp(), 
                createClassifierHist(HISTCMP_HELLINGER), 
                images,labels,persons, fold,verbose);
            break;
        case 9:
            runtest("lbpu_hell",
                createExtractorLbpUniform(), 
                createClassifierHist(HISTCMP_HELLINGER), 
                images,labels,persons, fold,verbose);
            break;
        case 10:
            runtest("lbpu_red_hell",
                createExtractorLbpUniform(8,8,2), 
                createClassifierHist(HISTCMP_HELLINGER), 
                images,labels,persons, fold,verbose);
            break;
        //case 11:
        //    runtest("wld_L1", createExtractorWLD(8,8,CV_8U), createClassifierNearest(NORM_L1), images,labels,persons, fold,verbose);
        //    break;
        case 13:
            runtest("wld_hell",
                createExtractorWLD(), 
                createClassifierHist(HISTCMP_HELLINGER), 
                images,labels,persons, fold,verbose);
            break;
        }
    }
    return 0;
}
