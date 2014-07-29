//
// run k fold crossvalidation train/test on  person db
//

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

using namespace std;
using namespace cv;

#include "fr.h"
#include "tan_triggs.h"

//
// path label
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


const char *rec_names[] = {
    "fisher",
    "eigen",
    "lbph",
    "lbph2_u",
    "minmax",
    "lbp_comb",
    "lbp_var",
    "ltph",
    "clbpdist",
    "wld",
    "mom",
    "zernike",
    "svm",
    "svm_lbp",
    "svm_lbp_u2",
    "svm_hu",
    "svm_hog",
    "norml2"
};

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

#include "factory.h"

//
// for each fold, take alternating n/fold items for test, the other for training
//
Ptr<FaceRecognizer> runtest( int rec, const vector<Mat>& images, const vector<int>& labels, const vector<vector<int>>& persons, size_t fold=10, bool verbose=false ) 
{
    Ptr<FaceRecognizer> model;

    switch ( rec ) {
        case 0: model = createFisherFaceRecognizer(); break;
        case 1: model = createEigenFaceRecognizer(); break;
        case 2: model = createLBPHFaceRecognizer(1,8,8,8,DBL_MAX); break;
        case 3: model = createLBPHFaceRecognizer2(1,8,8,8,DBL_MAX,true); break;
        case 4: model = createMinMaxFaceRecognizer(10,4); break;
        case 5: model = createCombinedLBPHFaceRecognizer(8,8,DBL_MAX); break;
        case 6: model = createVarLBPFaceRecognizer(8,8,DBL_MAX); break;
        case 7: model = createLTPHFaceRecognizer(25,8,8,DBL_MAX); break;
        case 8: model = createClbpDistFaceRecognizer(DBL_MAX); break;
        case 9: model = createWLDFaceRecognizer(8,8,DBL_MAX); break;
        case 10: model = createMomFaceRecognizer(8,10); break;
        case 11: model = createZernikeFaceRecognizer(4,7); break;
//        case 12: model = createAnnFaceRecognizer(); break;
        case 12: model = createSvmFaceRecognizer(0); break;
        case 13: model = createSvmFaceRecognizer(1); break; // use pixels
        case 14: model = createSvmFaceRecognizer(1,true); break; // use_uni2
        case 15: model = createSvmFaceRecognizer(2); break; // use hu
        case 16: model = createSvmFaceRecognizer(3); break; // use mom
        default: model = createLinearFaceRecognizer(NORM_L2); break;
    }
    //
    // do nfold sliding window train & test runs
    //
    int64 dt1=0, dt2=0;
    double meanSqrError = 0.0;
    double meanError = 0.0;
    Mat confusion = Mat::zeros(persons.size(),persons.size(),CV_32S);
    for ( size_t f=0; f<fold; f++ )
    {
        vector<Mat> trainImages;
        vector<int> trainLabels;
        vector<Mat> testImages;
        vector<int> testLabels;

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

        //  cerr << rec_names[rec] << " " << f << endl;
        model->train(trainImages, trainLabels);

        int64 t2 = cv::getTickCount();

        // test model
        int ntests = testImages.size();
        int misses = 0;
        for ( int i=0; i<ntests; i++ )
        {
            int label = testLabels[i];

            // test positive img:
            double dist = DBL_MAX;
            int predicted = -1;
            // cerr << label << " ";
            model->predict(testImages[i],predicted,dist);
            if ( verbose && (predicted>=0 && size_t(predicted)<persons.size()) ) /// TODO: what if else
                confusion.at<int>(label,predicted) += 1;
            misses += (predicted != label);
        }
        double rror = double(misses)/ntests;
        meanSqrError += rror * rror;
        meanError += rror;// * rror;


        cout << '.';
        int64 t3 = cv::getTickCount();
        dt1 += t2-t1;
        dt2 += t3-t2;

        if ( verbose )
            cout << format(" %-12s %-6.3f (%3d %3d) %3d (%6.3f %6.3f)",rec_names[rec], rror,trainImages.size(), testImages.size(), misses,ct(t2-t1), ct(t3-t2) ) << endl;
    }
    double me = meanError / fold;
    //double me = sqrt(meanSqrError) / fold;
    //if ( verbose )
    //   cerr << confusion << endl;
    cout << format(" %-12s %-10.3f (%6.3f %6.3f",rec_names[rec], (1.0-me), ct(dt1), ct(dt2)) ;
    return model;
}


//
//
// face att.txt 5     5      1         1
// face db      fold  reco  verbose  preprocessing
//
// special: fold==1 will train on all images , save them, and ignore the test step
// special: reco==0 will run *all* recognizers available on a given db
//


extern void svm_ga(const vector<Mat>& images, const vector<int>& labels, float err);


int main(int argc, const char *argv[]) 
{
    //return btest();

    vector<Mat> images;
    vector<int> labels;
    vector<string> vec;

    //std::string imgpath("att.txt");
    std::string imgpath("lfw2fun.txt");
    if ( argc>1 ) imgpath=argv[1];

    size_t fold = 5;
    if ( argc>2 ) fold=atoi(argv[2]);

    int rec = 16;
    if ( argc>3 ) rec=atoi(argv[3]);

    bool verbose = true;
    if ( argc>4 ) verbose=(argv[4][0]=='1');

    bool save = (fold==1);

    int preproc = 0; // 0-none 1-eqhist 2-tantriggs 3-clahe
    if ( argc>5 ) preproc=atoi(argv[5]);

    // read face db
    size_t maxim  = 400;
    int nsubjects = 1 + readtxt(imgpath.c_str(),vec,labels, maxim);


    if ( argc==1 )
    {
        cerr << argv[0] << " " << imgpath << " " << fold << " " << rec << endl;
    }

    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(50); // default:40 (!!)

    // read the images, correct labels if empty image skipped
    int load_flag = preproc==-1 ? 1 : 0;
    int skipped = 0;
    vector<int> clabels; 
    for ( size_t i=0; i<vec.size(); i++ )
    {
        Mat mm = imread(vec[i],load_flag);
        if ( mm.empty() )
        {
            skipped ++;
            continue;
        }
        Mat m2;
        resize(mm,m2,Size(90,90));
        switch(preproc) 
        {
            default:
            case 0: mm = m2; break;
            case 1: cv::equalizeHist( m2, mm ); break;
            case 2: cv::normalize( tan_triggs_preprocessing(m2), mm, 0, 255, NORM_MINMAX, CV_8UC1); break;
            case 3: clahe->apply(m2,mm); break;
        }            
        images.push_back(mm);
        clabels.push_back(labels[i]);
        //if ( i%33==0) imshow("i",mm), waitKey(0);
    }
    labels = clabels;

    ////zern_ga(images, labels, 10.0f);
   // svm_ga(images, labels, 0.005f);
   // return 1;


    // per person id lookup
    vector<vector<int>> persons;
    setupPersons( labels, persons );

    fold = std::min(fold,images.size()/nsubjects);

    cout << fold  << " fold, " ;
    cout << nsubjects  << " classes, " ;
    cout << images.size() << " images, ";
    cout << images.size()/nsubjects << " per class ";
    if ( skipped ) cout << "(" << skipped << " images skipped)";
    cout << endl;

    int n=sizeof(rec_names)/sizeof(char*);
    if ( rec > 0 )
    {
        n = rec+1;
    }
    for ( ; rec<n; rec++ ) 
    {
        Ptr<FaceRecognizer> model = runtest(rec,images,labels,persons, fold,verbose);
        if ( save ) {
            int64 t0 = getTickCount();
            if ( rec==4 )
                model->save(format("%s.png",rec_names[rec]));
            else
                model->save(format("%s.yml",rec_names[rec]));
            int64 t1 = getTickCount();
            cout << format(" %6.3f",ct(t1-t0));
        }
        cout << ")" << endl;
    }
    return 0;
}
